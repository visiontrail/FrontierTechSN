import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

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

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const cli = new URL('../node_modules/hyperframes/dist/cli.js', import.meta.url)
  const original = readFileSync(cli, 'utf8')
  const patched = patchMediaServer(original)
  if (patched !== original) {
    writeFileSync(cli, patched)
    console.log('Applied HyperFrames media Range response fix')
  }
}
