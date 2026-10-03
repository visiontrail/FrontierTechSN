const marker = '// FrontierTechSN: bounded 60-second layout navigation'

// HyperFrames 0.6.99: --timeout controls timeline readiness, not these two
// page navigations. Keep DOM readiness, settling and all sample checks intact.
const anchors = [
  `  await page.setViewport(size);
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 1e4 });
}
async function runLayoutAudit(projectDir, opts) {`,
  `    const page = await chromeBrowser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    await page.goto(server.url, { waitUntil: "domcontentloaded", timeout: 1e4 });
    await alignViewportToComposition(page, server.url);`,
]

export function patchInspectNavigation(code) {
  const markerCount = code.split(marker).length - 1
  if (markerCount > 1) {
    throw new Error('HyperFrames inspect-navigation patch found duplicate markers; review the installed version before inspecting')
  }
  const alreadyPatched = markerCount === 1
  let patched = code
  for (const [index, before] of anchors.entries()) {
    const after = before.replace('timeout: 1e4', 'timeout: 6e4')
    const beforeCount = code.split(before).length - 1
    const afterCount = code.split(after).length - 1
    if (beforeCount !== (alreadyPatched ? 0 : 1) || afterCount !== (alreadyPatched ? 1 : 0)) {
      throw new Error(`HyperFrames inspect-navigation patch expected 1 ${alreadyPatched ? 'patched' : 'original'} anchor ${index + 1}, found original=${beforeCount}, patched=${afterCount}; review the installed version before inspecting`)
    }
    if (!alreadyPatched) patched = patched.replace(before, after)
  }
  if (alreadyPatched) return code
  // Preserve the executable's shebang and the other persistent CLI patches.
  return patched.replace(/^(#![^\n]*\n)?/, (prefix) => `${prefix}${marker}\n`)
}
