import type { ExplainerProject, ExplainerStyle } from './explainerApi'

const COLOR_STYLE_FIELDS = ['background_color', 'ink_color', 'accent_color'] as const

export interface ExplainerStylePreviewCandidate {
  imageUrl: string
  projectId: string
  projectTitle: string
  sceneId: string
  isCurrentProject: boolean
}

export interface ExplainerStyleVideoCandidate {
  videoUrl: string
  projectId: string
  projectTitle: string
  kind: 'still' | 'animated'
  isCurrentProject: boolean
}

export function sameExplainerVisualStyle(left: ExplainerStyle, right: ExplainerStyle): boolean {
  return COLOR_STYLE_FIELDS.every((field) => left[field].toLowerCase() === right[field].toLowerCase())
    && left.character_description === right.character_description
    && left.visual_direction === right.visual_direction
}

function assetUrl(projectId: string, relative: string | undefined, extension: RegExp): string | null {
  if (!relative || /^[a-z][a-z\d+.-]*:/i.test(relative) || relative.includes('\\')) return null
  if (Array.from(relative).some((char) => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) return null
  const parts = relative.split('/')
  if (parts.some((part) => !part || part === '.' || part === '..') || !extension.test(relative)) return null
  return `/api/explainers/${encodeURIComponent(projectId)}/assets/${parts.map(encodeURIComponent).join('/')}`
}

function previewProjects(currentProject?: ExplainerProject, library: ExplainerProject[] = [], currentProjectId = currentProject?.id) {
  // A refreshed project is authoritative over an older copy in the library.
  return [
    ...(currentProject ? [currentProject] : []),
    ...library.filter((project) => project.id !== currentProjectId && project.id !== currentProject?.id),
  ]
}

export function selectStylePreviews(
  style: ExplainerStyle,
  currentProject?: ExplainerProject,
  library: ExplainerProject[] = [],
  currentProjectId = currentProject?.id,
): ExplainerStylePreviewCandidate[] {
  const candidates: ExplainerStylePreviewCandidate[] = []
  const seen = new Set<string>()
  for (const project of previewProjects(currentProject, library, currentProjectId)) {
    if (project.status !== 'completed' || !sameExplainerVisualStyle(style, project.style)) continue
    // Top-level scene_artifacts may survive an edit. Only completed delivery
    // artifacts belong to the saved style shown by this project.
    for (const scene of project.artifacts.scene_artifacts ?? []) {
      const url = assetUrl(project.id, scene.image_path, /\.(png|jpe?g|webp)$/i)
      if (!url || seen.has(url)) continue
      seen.add(url)
      candidates.push({
        imageUrl: url,
        projectId: project.id,
        projectTitle: project.title,
        sceneId: scene.id,
        isCurrentProject: project.id === currentProject?.id,
      })
    }
  }
  return candidates
}

export function selectStyleVideoPreviews(
  style: ExplainerStyle,
  currentProject?: ExplainerProject,
  library: ExplainerProject[] = [],
  currentProjectId = currentProject?.id,
): ExplainerStyleVideoCandidate[] {
  const candidates: ExplainerStyleVideoCandidate[] = []
  const seen = new Set<string>()
  for (const project of previewProjects(currentProject, library, currentProjectId)) {
    if (project.status !== 'completed' || !sameExplainerVisualStyle(style, project.style)) continue
    const supplied = project.artifacts.video_url
    const prefix = `/api/explainers/${encodeURIComponent(project.id)}/assets/`
    if (!supplied?.startsWith(prefix) || /[?#]/.test(supplied)) continue
    let relative: string
    try { relative = decodeURIComponent(supplied.slice(prefix.length)) }
    catch { continue }
    const url = assetUrl(project.id, relative, /\.mp4$/i)
    if (!url || seen.has(url)) continue
    seen.add(url)
    candidates.push({
      videoUrl: url,
      projectId: project.id,
      projectTitle: project.title,
      kind: project.style.motion_mode === 'none' ? 'still' : 'animated',
      isCurrentProject: project.id === currentProject?.id,
    })
  }
  return candidates
}
