import os
import re
import shutil
from collections import defaultdict

# Matches stems like "Album Name 001" or "Some Gallery 42" — space then digits at end
_SUFFIX_RE = re.compile(r"^(.+)\s+(\d+)$")


def tidy_dir(path: str, on_progress=None) -> list[tuple[str, str]]:
    """Recursively walk path and organise numbered file series into subdirs.

    A directory is only reorganised when it contains 2+ distinct groups
    (e.g. "Album One 001-003" and "Album Two 001-002" coexist). If a
    directory holds files from a single series they are already tidy and
    are left alone. Groups with only one file are always ignored.

    This rule makes the operation naturally idempotent: after tidying, every
    directory contains at most one group, so a second run is a no-op.
    """
    if not os.path.isdir(path):
        return []

    # Phase 1: scan the whole tree and plan moves before touching anything.
    planned: list[tuple[str, str]] = []
    for dirpath, _dirnames, filenames in os.walk(path):
        visible = [f for f in filenames if not f.startswith(".")]
        groups: dict[str, list[str]] = defaultdict(list)
        for fname in sorted(visible):
            stem, _ext = os.path.splitext(fname)
            m = _SUFFIX_RE.match(stem)
            if not m:
                continue
            base = m.group(1).strip()
            if base:
                groups[base].append(fname)

        # Groups that have at least 2 files.
        multi = {base: files for base, files in groups.items() if len(files) >= 2}
        if not multi:
            continue

        # Only reorganise when 2+ distinct groups are present.
        if len(multi) < 2:
            continue

        for base, files in multi.items():
            target_dir = os.path.join(dirpath, base)
            for fname in files:
                planned.append((os.path.join(dirpath, fname), os.path.join(target_dir, fname)))

    # Phase 2: execute moves.
    moved = []
    total = len(planned)
    for i, (src, dst) in enumerate(planned):
        if not os.path.exists(src) or os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        moved.append((src, dst))
        if on_progress and total:
            on_progress(i + 1, total)

    return moved
