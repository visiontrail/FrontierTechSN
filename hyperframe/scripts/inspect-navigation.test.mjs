import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { patchInspectNavigation } from './patch-inspect-navigation.mjs'

const marker = '// FrontierTechSN: bounded 60-second layout navigation'
const fixture = `#!/usr/bin/env node
async function alignViewportToComposition(page, url) {
  const size = { width: 1920, height: 1080 };
  await page.setViewport(size);
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 1e4 });
}
async function runLayoutAudit(projectDir, opts) {
    const page = await chromeBrowser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    await page.goto(server.url, { waitUntil: "domcontentloaded", timeout: 1e4 });
    await alignViewportToComposition(page, server.url);
    await page.waitForFunction(() => !!window.__timelines, { timeout: opts.timeout });
    await waitForFonts(page, 750);
    const samples = buildLayoutSampleTimes({ duration, samples: opts.samples, at: opts.at });
}
async function unrelatedNavigation(page, url) {
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 1e4 });
}
`

test('change exactly the two layout navigations and preserve waits, samples and unrelated navigation', () => {
  const patched = patchInspectNavigation(fixture)
  const expected = fixture
    .replace('timeout: 1e4', 'timeout: 6e4')
    .replace('page.goto(server.url, { waitUntil: "domcontentloaded", timeout: 1e4 }',
      'page.goto(server.url, { waitUntil: "domcontentloaded", timeout: 6e4 }')
    .replace('#!/usr/bin/env node\n', `#!/usr/bin/env node\n${marker}\n`)
  assert.equal(patched, expected)
  assert.equal((patched.match(/timeout: 6e4/g) || []).length, 2)
  assert.equal((patched.match(/timeout: 1e4/g) || []).length, 1)
  assert.equal(patchInspectNavigation(patched), patched)
})

for (const [label, source] of [
  ['unknown bundle', 'unexpected upstream bundle'],
  ['missing viewport anchor', fixture.replace('page.setViewport(size)', 'page.setViewport(newSize)')],
  ['missing initial navigation anchor', fixture.replace('page.setViewport({ width: 1920, height: 1080 })', 'page.setViewport(defaultSize)')],
  ['duplicated anchors', fixture + fixture],
  ['changed upstream timeout', fixture.replace('timeout: 1e4', 'timeout: 3e4')],
  ['partial unmarked patch', fixture.replace('timeout: 1e4', 'timeout: 6e4')],
  ['marker without patch', `${marker}\n${fixture}`],
  ['partial marked patch', patchInspectNavigation(fixture).replace('timeout: 6e4', 'timeout: 1e4')],
  ['duplicate markers', `${marker}\n${patchInspectNavigation(fixture)}`],
]) {
  test(`reject ${label} before writing a bundle`, () => {
    assert.throws(() => patchInspectNavigation(source), /review the installed version before inspecting/)
  })
}

test('installed bundle matches both supported navigation anchors without modifying its file', () => {
  const cli = new URL('../node_modules/hyperframes/dist/cli.js', import.meta.url)
  const original = readFileSync(cli, 'utf8')
  const patched = patchInspectNavigation(original)
  assert.equal(patchInspectNavigation(patched), patched)
  assert.ok(patched.includes('page.goto(url, { waitUntil: "domcontentloaded", timeout: 6e4 })'))
  assert.ok(patched.includes('page.goto(server.url, { waitUntil: "domcontentloaded", timeout: 6e4 })'))
  assert.equal(readFileSync(cli, 'utf8'), original)
})
