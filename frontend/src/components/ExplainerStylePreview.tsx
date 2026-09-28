import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { ExplainerProject, ExplainerStyle } from '../explainerApi'
import { selectStylePreviews } from '../explainerStylePreview'

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
  const candidates = selectStylePreviews(style, project, projects, projectId)
  const preview = candidates.find((candidate) => !unavailable.includes(candidate.imageUrl))
  const failed = candidates.length > 0 && !preview

  return <figure className="explainer-style-preview" aria-label="Generated style preview">
    <div className="explainer-preview-heading">
      <span className="explainer-preview-caption">GENERATED STYLE</span>
      <span className="explainer-preview-kind">{preview ? preview.isCurrentProject ? 'This project' : 'Matching style' : failed || lookupFailed ? 'Preview unavailable' : 'Awaiting artwork'}</span>
    </div>
    {preview ? <>
      <a className="explainer-preview-artwork" href={preview.imageUrl} target="_blank" rel="noreferrer" aria-label="Open full-size generated illustration" style={{ backgroundColor: style.background_color }}>
        <img key={preview.imageUrl} src={preview.imageUrl} alt={`Generated illustration from ${preview.projectTitle}, ${preview.sceneId}`}
          onError={() => setUnavailable((urls) => [...urls, preview.imageUrl])} />
      </a>
      <figcaption className="explainer-preview-description">
        <strong>{preview.isCurrentProject ? 'The artwork used in your video' : 'Generated with the same visual style'}</strong>
        <p>{preview.isCurrentProject
          ? 'The original scene illustration, before animation, titles and captions.'
          : 'Matches your character description, art direction and palette. Your storyboard determines the scene.'}</p>
        {!preview.isCurrentProject && <Link to={`/explainer-studio/${preview.projectId}`}>View source project ↗</Link>}
      </figcaption>
    </> : <div className="explainer-preview-empty" role="status">
      <strong>{loading ? 'Loading generated artwork…' : failed ? 'The illustration could not be loaded' : lookupFailed ? 'Could not check for generated artwork' : 'No generated artwork for this style yet'}</strong>
      <p>{loading ? 'Looking for a scene with the same character, art direction and palette.'
        : failed ? 'Retry to load the original scene illustration.'
        : lookupFailed ? 'The project library could not be loaded. Retry to find artwork matching these settings.'
        : 'Generate a video with these settings to see its actual illustration here. Changing the character, art direction or palette requires new artwork.'}</p>
      {failed && <button type="button" onClick={() => setUnavailable([])}>Retry preview</button>}
      {!failed && lookupFailed && <button type="button" onClick={onRetryLookup}>Retry lookup</button>}
    </div>}
  </figure>
}
