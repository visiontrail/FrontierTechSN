"""Skills registry — the source of truth for the admin Skills console.

Skills are ``SKILL.md`` bundles under ``.claude/skills`` (most are symlinks into
``.agents/skills``). They give the Claude Agent SDK the composition/render
know-how managed by the Admin console. This module discovers them, parses their
YAML frontmatter (``name`` / ``description``), exposes the markdown body for
viewing and editing, and tracks enabled/disabled state in
``data/admin_state.json``.

The files stay in place when a skill is disabled. The SDK integration consumes
the state through :func:`runtime_skill_names` and blocks disabled Skill tool
calls, so toggling remains reversible without making the flag merely advisory.
"""

from __future__ import annotations

import json
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath

from backend import config

_FRONTMATTER_KEY = re.compile(r"^([A-Za-z0-9_-]+):\s*(.*)$")

SKILLS_DIR = config.PROJECT_ROOT / ".claude" / "skills"
_STATE_PATH = config.PROJECT_ROOT / "data" / "admin_state.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ARCHIVE_FILES = 200
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
_IMPORT_EXTENSIONS = {".skill", ".zip", ".md"}


class SkillImportError(ValueError):
    """Raised when an uploaded file is not a safe, valid skill bundle."""


class SkillAlreadyExistsError(SkillImportError):
    """Raised when an import would replace an installed skill."""


@dataclass
class SkillInfo:
    name: str
    description: str
    path: str  # SKILL.md path relative to project root
    is_symlink: bool
    enabled: bool
    body: str = ""


def _load_state() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _disabled_set() -> set[str]:
    return set(_load_state().get("disabled_skills", []))


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a ``---`` YAML frontmatter block from the markdown body.

    Skill frontmatter only ever carries flat ``key: value`` strings (name,
    description), so a tiny hand parser avoids a YAML dependency and tolerates
    the multi-line descriptions these files use."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end].strip("\n")
    body = text[end + 4:].lstrip("\n")
    meta: dict[str, str] = {}
    current_key: str | None = None
    for line in block.splitlines():
        m = _FRONTMATTER_KEY.match(line)
        if m:
            current_key = m.group(1).strip()
            meta[current_key] = m.group(2).strip()
        elif current_key and line.strip():
            # Continuation of a wrapped value.
            meta[current_key] = f"{meta[current_key]} {line.strip()}".strip()
    return meta, body


def _unquote_frontmatter_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def _validate_skill_document(text: str) -> tuple[str, str]:
    meta, body = _parse_frontmatter(text)
    name = _unquote_frontmatter_value(meta.get("name", ""))
    description = _unquote_frontmatter_value(meta.get("description", ""))
    if not name:
        raise SkillImportError("SKILL.md frontmatter must contain a non-empty 'name'")
    if len(name) > 100 or any(ord(char) < 32 for char in name):
        raise SkillImportError("Skill name is invalid or longer than 100 characters")
    if not description:
        raise SkillImportError("SKILL.md frontmatter must contain a non-empty 'description'")
    if not body.strip():
        raise SkillImportError("SKILL.md must contain instructions below its frontmatter")
    return name, description


def _slug_for(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).casefold()
    slug = re.sub(r"[^\w.-]+", "-", normalized, flags=re.UNICODE).strip("._-")
    if not slug:
        raise SkillImportError("Skill name cannot be converted to a safe directory name")
    return slug[:80].rstrip("._-")


def _decode_skill_document(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SkillImportError("SKILL.md must be UTF-8 encoded") from exc


def _archive_path(raw_name: str) -> PurePosixPath:
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise SkillImportError("ZIP contains an invalid path")
    path = PurePosixPath(raw_name)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise SkillImportError(f"ZIP path traversal is not allowed: {raw_name}")
    return path


def _is_archive_junk(path: PurePosixPath) -> bool:
    return "__MACOSX" in path.parts or path.name in {".DS_Store", "Thumbs.db"}


def _check_archive_member(info: zipfile.ZipInfo) -> PurePosixPath:
    path = _archive_path(info.filename)
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode):
        raise SkillImportError(f"ZIP symbolic links are not allowed: {info.filename}")
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise SkillImportError(f"ZIP special files are not allowed: {info.filename}")
    if info.flag_bits & 0x1:
        raise SkillImportError("Encrypted ZIP files are not supported")
    return path


def _stage_archive(data: bytes, staging: Path) -> tuple[str, str]:
    try:
        archive = zipfile.ZipFile(BytesIO(data))
    except (OSError, zipfile.BadZipFile) as exc:
        raise SkillImportError("The uploaded ZIP archive is invalid") from exc

    with archive:
        checked = [(info, _check_archive_member(info)) for info in archive.infolist()]
        files = [(info, path) for info, path in checked if not info.is_dir()]
        if len(files) > MAX_ARCHIVE_FILES:
            raise SkillImportError(
                f"ZIP contains too many files (maximum {MAX_ARCHIVE_FILES})"
            )
        total_size = sum(info.file_size for info, _ in files)
        if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise SkillImportError(
                "ZIP expands beyond the 50 MB uncompressed size limit"
            )

        meaningful = [
            (info, path)
            for info, path in checked
            if not _is_archive_junk(path)
        ]
        manifests = [
            (info, path)
            for info, path in meaningful
            if not info.is_dir() and path.name.casefold() == "skill.md"
        ]
        if len(manifests) != 1:
            raise SkillImportError(
                "ZIP must contain exactly one SKILL.md (at the root or inside one wrapper directory)"
            )

        manifest_info, manifest_path = manifests[0]
        root_parts = manifest_path.parent.parts
        for _, path in meaningful:
            if path.parts[: len(root_parts)] != root_parts:
                raise SkillImportError(
                    "ZIP files must all be inside the directory that contains SKILL.md"
                )

        try:
            manifest_text = _decode_skill_document(archive.read(manifest_info))
        except (RuntimeError, zipfile.BadZipFile) as exc:
            raise SkillImportError("Could not read SKILL.md from the ZIP archive") from exc
        name, description = _validate_skill_document(manifest_text)

        seen: set[str] = set()
        for info, path in meaningful:
            relative_parts = path.parts[len(root_parts):]
            if not relative_parts:
                continue
            relative_key = "/".join(relative_parts).casefold()
            if relative_key in seen:
                raise SkillImportError(f"ZIP contains a duplicate path: {path}")
            seen.add(relative_key)
            target = staging.joinpath(*relative_parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(info) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
            except (RuntimeError, zipfile.BadZipFile, OSError) as exc:
                raise SkillImportError(f"Could not extract ZIP member: {path}") from exc
            mode = info.external_attr >> 16
            target.chmod(0o755 if mode & stat.S_IXUSR else 0o644)

    return name, description


def _stage_document(data: bytes, staging: Path) -> tuple[str, str]:
    text = _decode_skill_document(data)
    name, description = _validate_skill_document(text)
    (staging / "SKILL.md").write_text(text, encoding="utf-8")
    return name, description


def import_skill(filename: str, data: bytes) -> SkillInfo:
    """Validate and atomically install one uploaded skill bundle.

    ``.zip`` and ZIP-formatted ``.skill`` files may contain one complete bundle,
    including scripts and reference assets. Plain ``.skill``/``.md`` files are
    installed as the bundle's ``SKILL.md``. Existing skills are never replaced.
    """
    safe_filename = Path(filename or "").name
    extension = Path(safe_filename).suffix.casefold()
    if extension not in _IMPORT_EXTENSIONS:
        raise SkillImportError("Upload a .skill, .zip, .md, or SKILL.md file")
    if not data:
        raise SkillImportError("The uploaded skill file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise SkillImportError("The uploaded file exceeds the 10 MB size limit")

    is_archive = zipfile.is_zipfile(BytesIO(data))
    if extension == ".zip" and not is_archive:
        raise SkillImportError("The uploaded .zip file is not a valid ZIP archive")

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".skill-import-", dir=SKILLS_DIR) as temp:
        staging = Path(temp) / "bundle"
        staging.mkdir()
        if is_archive:
            name, _ = _stage_archive(data, staging)
        else:
            name, _ = _stage_document(data, staging)

        slug = _slug_for(name)
        destination = SKILLS_DIR / slug
        if (
            get_skill(name, include_body=False) is not None
            or destination.exists()
            or destination.is_symlink()
        ):
            raise SkillAlreadyExistsError(
                f"A skill named '{name}' is already installed as '{slug}'"
            )
        try:
            staging.rename(destination)
        except FileExistsError as exc:
            raise SkillAlreadyExistsError(
                f"A skill named '{name}' is already installed as '{slug}'"
            ) from exc

    state = _load_state()
    disabled = set(state.get("disabled_skills", []))
    if slug in disabled:
        disabled.discard(slug)
        state["disabled_skills"] = sorted(disabled)
        _save_state(state)

    skill = get_skill(slug, include_body=True)
    if skill is None:
        raise RuntimeError("Skill was installed but could not be loaded")
    return skill


def _skill_md_path(skill_dir) -> "object | None":
    for name in ("SKILL.md", "skill.md"):
        p = skill_dir / name
        if p.exists():
            return p
    return None


def _rel(path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def list_skills(include_body: bool = False) -> list[SkillInfo]:
    if not SKILLS_DIR.exists():
        return []
    disabled = _disabled_set()
    skills: list[SkillInfo] = []
    for entry in sorted(SKILLS_DIR.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith("."):
            continue
        if not entry.is_dir():  # resolves symlinks
            continue
        md = _skill_md_path(entry)
        if md is None:
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(text)
        skills.append(
            SkillInfo(
                name=meta.get("name", entry.name),
                description=meta.get("description", ""),
                path=_rel(md),
                is_symlink=entry.is_symlink(),
                enabled=entry.name not in disabled,
                body=body if include_body else "",
            )
        )
    return skills


def runtime_skill_names() -> tuple[list[str], list[str]]:
    """Return (enabled, disabled) Skill tool names for an SDK invocation."""
    enabled: list[str] = []
    disabled: list[str] = []
    for skill in list_skills(include_body=False):
        target = enabled if skill.enabled else disabled
        target.append(skill.name)
    return enabled, disabled


def get_skill(name: str, include_body: bool = True) -> SkillInfo | None:
    for skill in list_skills(include_body=include_body):
        # Match on the frontmatter name or the directory slug.
        if skill.name == name or skill.path.split("/")[-2] == name:
            return skill
    return None


def _dir_for(name: str):
    """Resolve a skill's directory by frontmatter name or directory slug."""
    if not SKILLS_DIR.exists():
        return None
    direct = SKILLS_DIR / name
    if direct.is_dir():
        return direct
    for entry in SKILLS_DIR.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        md = _skill_md_path(entry)
        if md is None:
            continue
        meta, _ = _parse_frontmatter(md.read_text(encoding="utf-8", errors="replace"))
        if meta.get("name") == name:
            return entry
    return None


def set_enabled(name: str, enabled: bool) -> bool:
    """Toggle a skill's advisory enabled flag. Returns False if unknown."""
    skill_dir = _dir_for(name)
    if skill_dir is None:
        return False
    slug = skill_dir.name
    state = _load_state()
    disabled = set(state.get("disabled_skills", []))
    if enabled:
        disabled.discard(slug)
    else:
        disabled.add(slug)
    state["disabled_skills"] = sorted(disabled)
    _save_state(state)
    return True


def update_body(name: str, content: str) -> bool:
    """Overwrite a skill's SKILL.md while preserving its frontmatter block.

    Only the markdown body below the frontmatter is editable from the console —
    the ``name``/``description`` metadata stays intact. Returns False if unknown.
    """
    skill_dir = _dir_for(name)
    if skill_dir is None:
        return False
    md = _skill_md_path(skill_dir)
    if md is None:
        return False
    original = md.read_text(encoding="utf-8", errors="replace")
    if original.startswith("---"):
        end = original.find("\n---", 3)
        if end != -1:
            frontmatter = original[: end + 4]
            md.write_text(f"{frontmatter}\n\n{content.lstrip()}\n", encoding="utf-8")
            return True
    md.write_text(content, encoding="utf-8")
    return True


def to_dict(skill: SkillInfo, include_body: bool = False) -> dict:
    data = {
        "name": skill.name,
        "slug": skill.path.split("/")[-2] if "/" in skill.path else skill.name,
        "description": skill.description,
        "path": skill.path,
        "is_symlink": skill.is_symlink,
        "enabled": skill.enabled,
    }
    if include_body:
        data["body"] = skill.body
    return data
