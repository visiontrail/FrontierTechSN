import assert from 'node:assert/strict'
import { createServer, request } from 'node:http'
import { closeSync, existsSync, ftruncateSync, mkdtempSync, openSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { extname, isAbsolute, join, relative, resolve } from 'node:path'
import { after, before, test } from 'node:test'
import { nodeMediaResponse } from './media-response.mjs'
import { patchInspectMediaServer } from './patch-media-server.mjs'

const dir = mkdtempSync(join(tmpdir(), 'frontier-inspect-range-'))
const file = join(dir, 'clip.mp4')
writeFileSync(file, '0123456789')
writeFileSync(join(dir, 'empty.mp4'), '')
writeFileSync(join(dir, 'style.css'), 'body { color: red; }')
const largeSize = 128 * 1024 * 1024
const large = openSync(join(dir, 'large.mp4'), 'w')
ftruncateSync(large, largeSize)
closeSync(large)
const opened = []
let server

before(async () => {
  // Exercise the exact bundled Node handler after an in-memory patch. Never
  // load the executable or mutate the installed CLI during these tests.
  const installed = readFileSync(new URL('../node_modules/hyperframes/dist/cli.js', import.meta.url), 'utf8')
  const patched = patchInspectMediaServer(installed)
  const start = patched.indexOf('async function serveStaticProjectHtml(')
  const end = patched.indexOf('\nvar init_staticProjectServer =', start)
  assert.ok(start > 0 && end > start)
  const serve = new Function(
    'createServer', 'existsSync62', 'readFileSync49', 'resolve42', 'relative13', 'isAbsolute11',
    'getMimeType', 'frontierNodeMediaResponse',
    `${patched.slice(start, end)}; return serveStaticProjectHtml`,
  )(
    createServer, existsSync, readFileSync, resolve, relative, isAbsolute,
    (path) => extname(path) === '.mp4' ? 'video/mp4' : 'text/css',
    (req, res, ...args) => {
      const stream = nodeMediaResponse(req, res, ...args)
      if (stream) {
        const entry = { stream, req, closed: new Promise((done) => stream.once('close', done)) }
        opened.push(entry)
      }
      return stream
    },
  )
  server = await serve(dir, '<html>Fixture composition</html>')
})

after(async () => {
  if (server) await server.close()
  rmSync(dir, { recursive: true, force: true })
})

function fetchFixture(path, { method = 'GET', range } = {}) {
  return new Promise((done, reject) => {
    const req = request(server.url, {
      path, method, agent: false, headers: range ? { Range: range } : {},
    }, (res) => {
      const chunks = []
      res.on('data', (chunk) => chunks.push(chunk))
      res.once('error', reject)
      res.once('end', () => done({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks).toString() }))
    })
    req.once('error', reject)
    req.end()
  })
}

for (const [range, status, body, contentRange] of [
  [undefined, 200, '0123456789', undefined],
  ['bytes=2-5', 206, '2345', 'bytes 2-5/10'],
  ['bytes=7-', 206, '789', 'bytes 7-9/10'],
  ['bytes=-3', 206, '789', 'bytes 7-9/10'],
  ['bytes=7-999', 206, '789', 'bytes 7-9/10'],
  ['bytes=-99', 206, '0123456789', 'bytes 0-9/10'],
  ['bytes=10-', 416, '', 'bytes */10'],
  ['bytes=5-2', 416, '', 'bytes */10'],
  ['bytes=-0', 416, '', 'bytes */10'],
  ['bytes=9007199254740993-', 416, '', 'bytes */10'],
  ['bytes=0-1,4-5', 200, '0123456789', undefined],
  ['not-a-range', 200, '0123456789', undefined],
]) {
  test(`Node inspection media request ${range}`, async () => {
    const result = await fetchFixture('/clip.mp4', { range })
    assert.equal(result.status, status)
    assert.equal(result.body, body)
    assert.equal(result.headers['content-type'], 'video/mp4')
    assert.equal(result.headers['content-range'], contentRange)
    assert.equal(result.headers['content-length'], String(body.length))
    assert.equal(result.headers['accept-ranges'], 'bytes')
    assert.equal(result.headers['transfer-encoding'], undefined)
  })
}

test('Node HEAD, empty and unsatisfiable requests never open media streams', async () => {
  const count = opened.length
  const head = await fetchFixture('/clip.mp4', { method: 'HEAD', range: 'bytes=2-5' })
  assert.equal(head.status, 200)
  assert.equal(head.headers['content-length'], '10')
  assert.equal(head.headers['content-range'], undefined)
  assert.equal(head.body, '')
  const empty = await fetchFixture('/empty.mp4')
  assert.equal(empty.status, 200)
  assert.equal(empty.headers['content-length'], '0')
  const unavailable = await fetchFixture('/empty.mp4', { range: 'bytes=0-' })
  assert.equal(unavailable.status, 416)
  assert.equal(unavailable.headers['content-range'], 'bytes */0')
  assert.equal(opened.length, count)
})

test('Node static handler preserves the document, other assets and path confinement', async () => {
  const count = opened.length
  assert.equal((await fetchFixture('/')).body, '<html>Fixture composition</html>')
  assert.equal((await fetchFixture('/style.css', { range: 'bytes=0-3' })).body, 'body { color: red; }')
  assert.equal((await fetchFixture('/missing.mp4')).status, 404)
  assert.equal((await fetchFixture('/../outside.mp4')).status, 403)
  assert.equal(opened.length, count)
})

test('aborted Node media response closes the file before reading the complete asset', async () => {
  const count = opened.length
  await new Promise((done, reject) => {
    const req = request(server.url, { path: '/large.mp4', agent: false }, (res) => {
      assert.equal(res.statusCode, 200)
      res.once('error', reject)
      res.once('data', () => res.destroy())
      res.once('close', done)
    })
    req.once('error', reject)
    req.end()
  })
  assert.equal(opened.length, count + 1)
  const entry = opened.at(-1)
  await entry.closed
  assert.equal(entry.stream.destroyed, true)
  assert.equal(entry.stream.closed, true)
  assert.ok(entry.stream.bytesRead < largeSize)
  assert.equal(entry.req.listenerCount('aborted'), 0)
})

test('Node static-server patch is complete, repeatable and rejects unknown/partial bundles', () => {
  const cli = new URL('../node_modules/hyperframes/dist/cli.js', import.meta.url)
  const original = readFileSync(cli, 'utf8')
  const patched = patchInspectMediaServer(original)
  assert.equal(patchInspectMediaServer(patched), patched)
  assert.equal((patched.match(/frontierNodeMediaResponse\(req, res, filePath, contentType\);/g) || []).length, 1)
  assert.throws(() => patchInspectMediaServer('unknown bundle'), /expected 1 original anchor/)
  assert.throws(() => patchInspectMediaServer(patched.replace('frontierNodeMediaResponse(req, res, filePath, contentType);', 'changedUpstream();')), /expected 1 original anchor/)
  assert.throws(() => patchInspectMediaServer(patched + patched), /expected 1 original anchor/)
  assert.throws(() => patchInspectMediaServer(patched.replace('import { nodeMediaResponse as frontierNodeMediaResponse }', 'import { unavailable }')), /expected 1 original anchor/)
  assert.equal(readFileSync(cli, 'utf8'), original)
})
