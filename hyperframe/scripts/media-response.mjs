import { createReadStream, statSync } from 'node:fs'
import { Readable } from 'node:stream'

// Chromium suspends large media downloads once its buffer is full. Serving
// those downloads as unbounded, chunked 200 responses can occupy every HTTP/1
// connection and starve the renderer's extracted-frame image requests.
export function mediaResponse(request, filePath, contentType) {
  const size = statSync(filePath).size
  const headers = { 'Content-Type': contentType, 'Accept-Ranges': 'bytes' }
  const range = request.headers.get('range')
  let start = 0
  let end = size - 1
  let status = 200

  // HEAD describes the complete representation, independent of Range.
  if (range && request.method !== 'HEAD') {
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
        return new Response(null, {
          status: 416,
          headers: { ...headers, 'Content-Range': `bytes */${size}`, 'Content-Length': '0' },
        })
      }
      status = 206
      headers['Content-Range'] = `bytes ${start}-${end}/${size}`
    }
    // Ignore malformed/unsupported multipart ranges with a full response.
  }

  headers['Content-Length'] = String(Math.max(0, end - start + 1))
  const body = request.method === 'HEAD' || size === 0
    ? null
    : Readable.toWeb(createReadStream(filePath, { start, end }))
  return new Response(body, { status, headers })
}
