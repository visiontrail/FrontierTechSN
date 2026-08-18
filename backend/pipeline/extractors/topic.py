from backend.pipeline.extractors.base import ExtractedContent


async def extract_topic(title: str | None, brief: str | None, log=None) -> ExtractedContent:
    """Turn an editorial brief into the same source contract as file/web inputs."""
    clean_title = (title or "Planned topic").strip()
    clean_brief = (brief or "").strip()
    if len(clean_brief) < 10:
        raise ValueError("A planned topic needs an editorial brief of at least 10 characters")
    if log:
        log(f"Loaded editorial brief for '{clean_title}' ({len(clean_brief.split())} words)")
    return ExtractedContent(
        source_type="topic",
        title=clean_title,
        text=clean_brief,
        metadata={"origin": "content_planning", "research_required": True},
    )
