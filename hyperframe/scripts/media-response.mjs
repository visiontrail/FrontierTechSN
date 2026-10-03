import { createReadStream, statSync } from 'node:fs'
import { pipeline, Readable } from 'node:stream'

// Chromium suspends large media downloads once its buffer is full. Serving
// those downloads as unbounded, chunked 200 responses can occupy every HTTP/1
// connection and starve the renderer's extracted-frame image requests.
function mediaTransfer(method, range, size, contentType) {
  const headers = { 'Content-Type': contentType, 'Accept-Ranges': 'bytes' }
  let start = 0
  let end = size - 1
  let status = 200

  // HEAD describes the complete representation, independent of Range.
  if (range && method !== 'HEAD') {
    const match = /^bytes=(\d*)-(\d*)$/.exec(range)
    if (match && (match[1] || match[2])) {
      if (match[1]) {
        start = Number(match[1])
        end = match[2] ? Math.min(Number(match[2]), end) : end
      } else {
        start = Math.max(0, size - Number(match[2]))
      }
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end)
          || start > end || start >= size) {
        return {
          status: 416,
          headers: { ...headers, 'Content-Range': `bytes */${size}`, 'Content-Length': '0' },
          hasBody: false,
        }
      }
      status = 206
      headers['Content-Range'] = `bytes ${start}-${end}/${size}`
    }
    // Ignore malformed/unsupported multipart ranges with a full response.
  }

  headers['Content-Length'] = String(Math.max(0, end - start + 1))
  return { status, headers, start, end, hasBody: method !== 'HEAD' && size > 0 }
}

export function mediaResponse(request, filePath, contentType) {
  const { status, headers, start, end, hasBody } = mediaTransfer(
    request.method, request.headers.get('range'), statSync(filePath).size, contentType,
  )
  const body = hasBody ? Readable.toWeb(createReadStream(filePath, { start, end })) : null
  return new Response(body, { status, headers })
}

// inspect/snapshot use a separate Node HTTP server, not the two Fetch servers.
// pipe with backpressure; pipeline destroys the file stream when a browser
// cancels a buffered media response and never buffers the complete file.
export function nodeMediaResponse(request, response, filePath, contentType) {
  if (request.aborted || response.destroyed) return
  let transfer
  try {
    transfer = mediaTransfer(request.method, request.headers.range, statSync(filePath).size, contentType)
  } catch (error) {
    response.writeHead(error.code === 'ENOENT' ? 404 : 500, { 'Content-Length': '0' })
    response.end()
    return
  }
  response.writeHead(transfer.status, transfer.headers)
  if (!transfer.hasBody) {
    response.end()
    return
  }
  const stream = createReadStream(filePath, { start: transfer.start, end: transfer.end })
  const abort = () => stream.destroy()
  request.once('aborted', abort)
  pipeline(stream, response, () => request.off('aborted', abort))
  return stream
}
