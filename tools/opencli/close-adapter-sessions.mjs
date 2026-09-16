import { sendCommand } from './node_modules/@jackwener/opencli/dist/src/browser/daemon-client.js'

const sessions = process.argv.slice(2)

if (sessions.length === 0) {
  console.error('At least one OpenCLI adapter session is required')
  process.exitCode = 2
} else {
  const closed = []
  const errors = []

  for (const session of sessions) {
    try {
      const options = {
        session,
        surface: 'adapter',
        siteSession: 'persistent',
        windowMode: 'background',
      }
      await sendCommand('close-window', options)
      const remaining = await sendCommand('tabs', { ...options, op: 'list' })
      if (!Array.isArray(remaining) || remaining.length !== 0) {
        throw new Error('Adapter session still has tabs, or closure could not be verified')
      }
      closed.push(session)
    } catch (error) {
      errors.push({
        session,
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }

  console.log(JSON.stringify({ closed, errors }))
  if (errors.length > 0) process.exitCode = 1
}
