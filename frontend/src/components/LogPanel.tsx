import { useEffect, useRef, useState } from 'react'
import type { Task } from '../api'
import { logsStreamUrl } from '../api'
import { IconCheck, IconChevronLeft, IconChevronRight, IconCopy } from './Icons'

type TaskStatus = Task['status']

interface LogPanelProps {
  taskId: string
  taskStatus: TaskStatus
  /** Stretch the body to the height of its container instead of capping it. */
  fill?: boolean
  expanded?: boolean
  onExpandedChange?: (expanded: boolean) => void
}

const FINAL_STATUSES: TaskStatus[] = ['complete', 'failed', 'awaiting_review']

/* The Clipboard API needs a secure context *and* write permission; when either
   is missing (http://host, denied prompt) fall back to a throwaway textarea. */
async function writeClipboard(text: string) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return
    }
  } catch {
    /* fall through to the legacy path */
  }
  const area = document.createElement('textarea')
  area.value = text
  area.setAttribute('readonly', '')
  area.style.position = 'fixed'
  area.style.opacity = '0'
  document.body.appendChild(area)
  area.select()
  const ok = document.execCommand('copy')
  document.body.removeChild(area)
  if (!ok) throw new Error('Copy rejected')
}

export default function LogPanel({
  taskId,
  taskStatus,
  fill = false,
  expanded: controlledExpanded,
  onExpandedChange,
}: LogPanelProps) {
  const [internalExpanded, setInternalExpanded] = useState(false)
  const [lines, setLines] = useState<string[]>([])
  const [streamStatus, setStreamStatus] = useState<TaskStatus | null>(null)
  const [connectionState, setConnectionState] = useState<'connecting' | 'live' | 'closed' | 'error'>('connecting')
  const [copied, setCopied] = useState<'idle' | 'done' | 'failed'>('idle')
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const expanded = controlledExpanded ?? internalExpanded

  useEffect(() => {
    const source = new EventSource(logsStreamUrl(taskId))

    source.addEventListener('open', () => {
      setLines([])
      setStreamStatus(null)
      setConnectionState('live')
    })

    source.addEventListener('log', (event) => {
      setLines((current) => [...current, event.data])
    })

    source.addEventListener('status', (event) => {
      const status = event.data as TaskStatus
      setStreamStatus(status)
      setConnectionState('closed')
      source.close()
    })

    source.addEventListener('ping', () => {
      setConnectionState('live')
    })

    source.addEventListener('error', () => {
      if (FINAL_STATUSES.includes(taskStatus)) {
        setConnectionState('closed')
        source.close()
        return
      }
      setConnectionState('error')
    })

    return () => {
      source.close()
    }
  }, [taskId, taskStatus])

  useEffect(() => {
    if (!expanded) return
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [expanded, lines])

  useEffect(() => {
    if (copied === 'idle') return
    const timer = setTimeout(() => setCopied('idle'), 2000)
    return () => clearTimeout(timer)
  }, [copied])

  const copyLogs = async () => {
    try {
      await writeClipboard(lines.join('\n'))
      setCopied('done')
    } catch {
      setCopied('failed')
    }
  }

  const effectiveStatus = streamStatus ?? taskStatus
  const stateLabel =
    connectionState === 'live' && !FINAL_STATUSES.includes(effectiveStatus)
      ? 'Live'
      : connectionState === 'connecting'
        ? 'Connecting'
        : connectionState === 'error'
          ? 'Reconnecting'
          : effectiveStatus

  const toggleExpanded = () => {
    const nextExpanded = !expanded
    if (controlledExpanded === undefined) setInternalExpanded(nextExpanded)
    onExpandedChange?.(nextExpanded)
  }

  return (
    <section className={`log-panel${fill ? ' is-fill' : ''}${expanded ? '' : ' is-collapsed'}`}>
      <div className="log-panel-head">
        <button
          type="button"
          className="log-panel-toggle"
          aria-expanded={expanded}
          aria-label={`${expanded ? 'Collapse' : 'Expand'} pipeline logs`}
          title={`${expanded ? 'Collapse' : 'Expand'} pipeline logs`}
          onClick={toggleExpanded}
        >
          <span className="log-panel-chevron" aria-hidden="true">
            {expanded ? <IconChevronRight /> : <IconChevronLeft />}
          </span>
          <strong>Pipeline Logs</strong>
          <em>{stateLabel}</em>
        </button>
        {expanded && (
          <button
            type="button"
            className={`log-copy-btn${copied === 'done' ? ' is-done' : ''}`}
            disabled={!lines.length}
            title={lines.length ? 'Copy all log lines' : 'Nothing to copy yet'}
            aria-label="Copy logs to clipboard"
            onClick={copyLogs}
          >
            {copied === 'done' ? <IconCheck /> : <IconCopy />}
            <span>
              {copied === 'done'
                ? `Copied ${lines.length}`
                : copied === 'failed'
                  ? 'Copy failed'
                  : 'Copy'}
            </span>
          </button>
        )}
      </div>

      {expanded && (
        <div className="log-panel-body" ref={scrollRef}>
          {lines.length ? (
            lines.map((line, index) => (
              <div className="log-line" key={`${index}-${line}`}>
                {line}
              </div>
            ))
          ) : (
            <div className="log-empty">No logs yet.</div>
          )}
        </div>
      )}
    </section>
  )
}
