import { useState } from 'react'
import type { ExplainerProject, ExplainerStyle } from '../explainerApi'
import { selectStylePreviews, selectStyleVideoPreviews } from '../explainerStylePreview'

type PreviewKind = 'illustration' | 'still' | 'animated'
const PREVIEW_LABELS: Record<PreviewKind, string> = { illustration: 'Illustration', still: 'Still video', animated: 'Animated video' }

export default function ExplainerStylePreview({ style, project, projects, projectId, loading, lookupFailed, onRetryLookup }: {
  style: ExplainerStyle
  project?: ExplainerProject
  projects?: ExplainerProject[]
  projectId?: string
  loading: boolean
  lookupFailed: boolean
  onRetryLookup: () => void
}) {
  const [unavailable, setUnavailable] = useState<string[]>([])
  const [selectedKind, setSelectedKind] = useState<PreviewKind>('illustration')
  const candidates = selectStylePreviews(style, project, projects, projectId)
  const videos = selectStyleVideoPreviews(style, project, projects, projectId)
  const availableKinds = (Object.keys(PREVIEW_LABELS) as PreviewKind[]).filter((kind) => kind === 'illustration'
    ? candidates.length > 0 : videos.some((candidate) => candidate.kind === kind))
  const kind = availableKinds.includes(selectedKind) ? selectedKind : availableKinds[0] ?? 'illustration'
  const illustration = kind === 'illustration' ? candidates.find((candidate) => !unavailable.includes(candidate.imageUrl)) : undefined
  const video = kind !== 'illustration' ? videos.find((candidate) => candidate.kind === kind && !unavailable.includes(candidate.videoUrl)) : undefined
  const preview = illustration ?? video
  const failed = availableKinds.includes(kind) && !preview

  return <figure className="explainer-style-preview" aria-label="Generated style preview">
    <div className="explainer-preview-heading">
      <span className="explainer-preview-caption">GENERATED STYLE</span>
      <span className="explainer-preview-kind">{preview ? preview.isCurrentProject ? 'This project' : 'Matching style' : failed || lookupFailed ? 'Preview unavailable' : 'Awaiting artwork'}</span>
    </div>
    {availableKinds.length > 1 && <div className="explainer-preview-tabs" role="group" aria-label="Generated media">
      {availableKinds.map((option) => <button type="button" key={option} aria-pressed={kind === option} onClick={() => setSelectedKind(option)}>{PREVIEW_LABELS[option]}</button>)}
    </div>}
    {preview ? <>
      {illustration && <a className="explainer-preview-artwork" href={illustration.imageUrl} target="_blank" rel="noreferrer" aria-label="Open full-size generated illustration" style={{ backgroundColor: style.background_color }}>
        <img key={illustration.imageUrl} src={illustration.imageUrl} alt={`Generated illustration from ${illustration.projectTitle}, ${illustration.sceneId}`}
          onError={() => setUnavailable((urls) => [...urls, illustration.imageUrl])} />
      </a>}
      {video && <video className="explainer-preview-video" key={video.videoUrl} src={video.videoUrl} controls playsInline preload="metadata"
        aria-label={`${PREVIEW_LABELS[video.kind]} from ${video.projectTitle}`}
        onError={() => setUnavailable((urls) => [...urls, video.videoUrl])} />}
      <figcaption className="explainer-preview-description">
        <strong>{video ? preview.isCurrentProject ? 'Your rendered video' : 'Rendered with the same visual style'
          : preview.isCurrentProject ? 'The artwork used in your video' : 'Generated with the same visual style'}</strong>
        <p>{video ? 'A previously generated video. Changes to your settings apply to your next generation.' : preview.isCurrentProject
          ? 'The original scene illustration, before animation, titles and captions.'
          : 'Matches your character description, art direction and palette. Your storyboard determines the scene.'}</p>
        <p className="explainer-preview-source">Source: {preview.projectTitle}{illustration ? ` · ${illustration.sceneId}` : ''}</p>
      </figcaption>
    </> : <div className="explainer-preview-empty" role="status">
      <strong>{loading ? 'Loading generated artwork…' : failed ? `The ${kind === 'illustration' ? 'illustration' : 'video'} could not be loaded` : lookupFailed ? 'Could not check for generated artwork' : 'No generated artwork for this style yet'}</strong>
      <p>{loading ? 'Looking for a scene with the same character, art direction and palette.'
        : failed ? 'Retry to load the generated media, or choose another available preview.'
        : lookupFailed ? 'The project library could not be loaded. Retry to find artwork matching these settings.'
        : 'Generate a video with these settings to see its actual illustration here. Changing the character, art direction or palette requires new artwork.'}</p>
      {failed && <button type="button" onClick={() => setUnavailable([])}>Retry preview</button>}
      {!failed && lookupFailed && <button type="button" onClick={onRetryLookup}>Retry lookup</button>}
    </div>}
  </figure>
}
