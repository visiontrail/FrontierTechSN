import {ensureBrowser} from '@remotion/renderer'
import path from 'node:path'
import {fileURLToPath} from 'node:url'
// Keep Remotion's managed browser in this package's ignored node_modules cache,
// independent of the caller's working directory. Run in its own process so the
// rendering API never changes a hosting application's process-wide cwd.
process.chdir(path.dirname(path.dirname(fileURLToPath(import.meta.url))))
const browser = await ensureBrowser({logLevel: 'error', onBrowserDownload: () => ({version: null, onProgress: ({percent}) => {
  if (percent === 1) process.stderr.write('Chrome Headless Shell ready\n')
}})})
if (!('path' in browser)) throw new Error(`Browser unavailable: ${browser.type}`)
process.stdout.write(`${JSON.stringify(browser)}\n`)
