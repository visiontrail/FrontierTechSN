import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { patchRenderRecovery } from './patch-render-recovery.mjs'
import { patchInspectNavigation } from './patch-inspect-navigation.mjs'
import { patchSnapshotMediaTiming } from './patch-snapshot-media-timing.mjs'

const marker = '// FrontierTechSN: range-aware local media responses'
const source = 'import { mediaResponse as frontierMediaResponse } from "../../../scripts/media-response.mjs";'

export function patchMediaServer(code) {
  if (code.includes(marker)) return code
  // HyperFrames 0.6.99 bundles both the engine and producer file servers.
  const pattern = /    const content = (readFileSync(?:\d+)?)\(filePath\);\n    return new Response\(content, \{\n      status: 200,\n      headers: \{ "Content-Type": contentType \}\n    \}\);/g
  const matches = [...code.matchAll(pattern)]
  if (matches.length !== 2) {
    throw new Error(`HyperFrames media-server patch expected 2 handlers, found ${matches.length}; review the installed version before rendering`)
  }
  const patched = code.replace(pattern, (original) => `    if (contentType.startsWith("audio/") || contentType.startsWith("video/")) {
      return frontierMediaResponse(c3.req.raw, filePath, contentType);
    }
${original}`)
  // Preserve the executable's shebang.
  return patched.replace(/^(#![^\n]*\n)?/, (prefix) => `${prefix}${marker}\n${source}\n`)
}

const nodeMarker = '// FrontierTechSN: range-aware Node inspection media responses'
const nodeSource = 'import { nodeMediaResponse as frontierNodeMediaResponse } from "../../../scripts/media-response.mjs";'
// Exact HyperFrames 0.6.99 staticProjectServer anchor, distinct from Hono above.
const nodeBefore = `    if (existsSync62(filePath)) {
      res.writeHead(200, { "Content-Type": getMimeType(filePath) });
      res.end(readFileSync49(filePath));
      return;
    }`
const nodeAfter = `    if (existsSync62(filePath)) {
      const contentType = getMimeType(filePath);
      if (contentType.startsWith("audio/") || contentType.startsWith("video/")) {
        frontierNodeMediaResponse(req, res, filePath, contentType);
        return;
      }
      res.writeHead(200, { "Content-Type": getMimeType(filePath) });
      res.end(readFileSync49(filePath));
      return;
    }`

export function patchInspectMediaServer(code) {
  const markerCount = code.split(nodeMarker).length - 1
  const beforeCount = code.split(nodeBefore).length - 1
  const afterCount = code.split(nodeAfter).length - 1
  const importCount = code.split(nodeSource).length - 1
  if (markerCount === 1 && beforeCount === 0 && afterCount === 1 && importCount === 1) return code
  if (markerCount !== 0 || beforeCount !== 1 || afterCount !== 0 || importCount !== 0) {
    throw new Error(`HyperFrames Node media-server patch expected 1 original anchor or 1 complete patch, found original=${beforeCount}, patched=${afterCount}, markers=${markerCount}, imports=${importCount}; review the installed version before inspecting`)
  }
  return code.replace(nodeBefore, nodeAfter)
    .replace(/^(#![^\n]*\n)?/, (prefix) => `${prefix}${nodeMarker}\n${nodeSource}\n`)
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const cli = new URL('../node_modules/hyperframes/dist/cli.js', import.meta.url)
  const original = readFileSync(cli, 'utf8')
  const patched = patchSnapshotMediaTiming(patchInspectNavigation(patchRenderRecovery(patchInspectMediaServer(patchMediaServer(original)))))
  if (patched !== original) {
    writeFileSync(cli, patched)
    console.log('Applied HyperFrames media Range, render recovery, inspect navigation and snapshot timing fixes')
  }
}
