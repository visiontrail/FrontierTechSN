import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { after, test } from 'node:test'
import { mediaResponse } from './media-response.mjs'
import { patchMediaServer } from './patch-media-server.mjs'

const dir = mkdtempSync(join(tmpdir(), 'frontier-media-range-'))
const file = join(dir, 'clip.mp4')
writeFileSync(file, '0123456789')
after(() => rmSync(dir, { recursive: true, force: true }))

for (const [range, status, body, contentRange] of [
  [null, 200, '0123456789', null],
  ['bytes=2-5', 206, '2345', 'bytes 2-5/10'],
  ['bytes=7-', 206, '789', 'bytes 7-9/10'],
  ['bytes=-3', 206, '789', 'bytes 7-9/10'],
  ['bytes=7-999', 206, '789', 'bytes 7-9/10'],
  ['bytes=-99', 206, '0123456789', 'bytes 0-9/10'],
  ['bytes=10-', 416, '', 'bytes */10'],
  ['bytes=5-2', 416, '', 'bytes */10'],
  ['bytes=-0', 416, '', 'bytes */10'],
  ['bytes=9007199254740993-', 416, '', 'bytes */10'],
  ['bytes=0-1,4-5', 200, '0123456789', null],
  ['not-a-range', 200, '0123456789', null],
]) {
  test(`media request ${range}`, async () => {
    const request = new Request('http://localhost/clip.mp4', {
      headers: range ? { Range: range } : {},
    })
    const response = mediaResponse(request, file, 'video/mp4')
    assert.equal(response.status, status)
    assert.equal(response.headers.get('Content-Range'), contentRange)
    assert.equal(response.headers.get('Content-Length'), String(body.length))
    assert.equal(response.headers.get('Accept-Ranges'), 'bytes')
    assert.equal(await response.text(), body)
  })
}

test('HEAD and empty media do not open a body stream', async () => {
  const response = mediaResponse(new Request('http://localhost/clip.mp4', {
    method: 'HEAD', headers: { Range: 'bytes=2-5' },
  }), file, 'audio/wav')
  assert.equal(response.status, 200)
  assert.equal(response.headers.get('Content-Length'), '10')
  assert.equal(response.body, null)
  const empty = join(dir, 'empty.mp4')
  writeFileSync(empty, '')
  const emptyResponse = mediaResponse(new Request('http://localhost/empty.mp4'), empty, 'video/mp4')
  assert.equal(emptyResponse.headers.get('Content-Length'), '0')
  assert.equal(emptyResponse.body, null)
})

test('cancelled downloads release the stream without reading the whole file', async () => {
  const response = mediaResponse(new Request('http://localhost/clip.mp4'), file, 'video/mp4')
  await response.body.cancel()
})

test('installed patch is repeatable and rejects an unknown bundle', () => {
  const installed = readFileSync(new URL('../node_modules/hyperframes/dist/cli.js', import.meta.url), 'utf8')
  const patched = patchMediaServer(installed)
  assert.equal(patchMediaServer(patched), patched)
  assert.equal((patched.match(/return frontierMediaResponse\(c3.req.raw, filePath, contentType\);/g) || []).length, 2)
  assert.throws(() => patchMediaServer('unexpected upstream bundle'), /expected 2 handlers/)
})
