#!/usr/bin/env python3
"""
Populate <download-dir>/test/ with LFW face data covering all test cases.

Usage:
    python setup_lfw_test.py <download-dir>             # full initial setup (13 cases)
    python setup_lfw_test.py <download-dir> --add       # add later-batch simulation (+4 cases)

The tgz is cached in <download-dir>/lfw-deepfunneled.tgz; re-runs are fast
(only re-extracts the needed people from the archive).
"""

import argparse
import random
import shutil
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

LFW_URLS = [
    "https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",
    "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",
]
SEED = 42

# Curated list of famous people — used instead of pure random.
# Script intersects with whoever actually has 5+ images in the archive.
FAMOUS = [
    # Politicians
    "George_W_Bush",
    "Vladimir_Putin",
    "Tony_Blair",
    "Bill_Clinton",
    "Hillary_Clinton",
    "Colin_Powell",
    # Athletes
    "Tiger_Woods",
    "Serena_Williams",
    "Andre_Agassi",
    "Lance_Armstrong",
    "Roger_Federer",
    "Michael_Schumacher",
    "Lleyton_Hewitt",
    # Actors / Actresses
    "Arnold_Schwarzenegger",
    "Tom_Cruise",
    "Tom_Hanks",
    "Brad_Pitt",
    "Angelina_Jolie",
    "Julia_Roberts",
    "Nicole_Kidman",
    "Will_Smith",
    "George_Clooney",
    "Denzel_Washington",
    "Halle_Berry",
    "Johnny_Depp",
    "Cate_Blanchett",
    "Jim_Carrey",
    # Singers
    "Jennifer_Lopez",
    "Britney_Spears",
    "Christina_Aguilera",
    "Eminem",
    "Beyonce_Knowles",
    "Madonna",
]


# ── Download ──────────────────────────────────────────────────────────────────


def _reporthook(block: int, block_size: int, total: int) -> None:
    if total > 0:
        done = min(block * block_size, total)
        pct = done * 100 // total
        print(f"\r  {done // 1_000_000}MB / {total // 1_000_000}MB  ({pct}%)", end="", flush=True)


def download_tgz(dest: Path) -> None:
    last_err: Exception | None = None
    for url in LFW_URLS:
        try:
            print(f"Downloading LFW deepfunneled (~170 MB)\n  {url}\n  → {dest}")
            urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
            print()
            return
        except Exception as e:
            print(f"\n  failed ({e}), trying next…")
            last_err = e
            if dest.exists():
                dest.unlink()
    print(
        "\nCould not download automatically. Download manually and pass via --tgz:\n"
        "  curl -L -o lfw-deepfunneled.tgz \\\n"
        "    https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz\n"
        "  # or from Kaggle: kaggle datasets download -d jessicali9530/lfw-dataset\n"
        "Then run:\n"
        "  uv run python setup_lfw_test.py <download-dir> --tgz /path/to/lfw-deepfunneled.tgz"
    )
    raise SystemExit(1) from last_err


# ── Archive helpers ───────────────────────────────────────────────────────────


def detect_root(tgz: Path) -> str:
    """Return the top-level directory name inside the archive."""
    with tarfile.open(tgz, "r:gz") as tf:
        for name in tf.getnames():
            parts = Path(name).parts
            if len(parts) >= 1:
                return parts[0]
    raise ValueError("Empty or unrecognised archive")


def count_people(tgz: Path, root: str) -> dict[str, int]:
    """Scan the archive index (no extraction) → {person: image_count}."""
    counts: dict[str, int] = defaultdict(int)
    print("Scanning archive index (no extraction)…")
    with tarfile.open(tgz, "r:gz") as tf:
        for name in tf.getnames():
            parts = Path(name).parts
            if len(parts) >= 3 and parts[0] == root and name.endswith(".jpg"):
                counts[parts[1]] += 1
    return dict(counts)


def extract_person(tgz: Path, root: str, person: str, dest: Path) -> list[Path]:
    """Extract one person's images directly into dest/ (flat, no subdirs)."""
    dest.mkdir(parents=True, exist_ok=True)
    prefix = f"{root}/{person}/"
    out_paths: list[Path] = []
    with tarfile.open(tgz, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name.startswith(prefix) and member.name.endswith(".jpg"):
                fobj = tf.extractfile(member)
                if fobj:
                    out = dest / Path(member.name).name
                    out.write_bytes(fobj.read())
                    out_paths.append(out)
    return sorted(out_paths)


# ── File helpers ──────────────────────────────────────────────────────────────


def copy_series(images: list[Path], dest: Path, count: int, prefix: str = "img") -> None:
    """Copy up to `count` images into dest/ renamed as img_001.jpg …"""
    dest.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(images[:count], 1):
        shutil.copy(src, dest / f"{prefix}_{i:03d}.jpg")


def make_noface_image(path: Path, color: tuple[int, int, int]) -> None:
    """Create a solid-colour JPEG with a rectangle — no human face."""
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (250, 250), color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, 210, 210], outline=(0, 0, 0), width=4)
        draw.line([40, 40, 210, 210], fill=(0, 0, 0), width=2)
        draw.line([210, 40, 40, 210], fill=(0, 0, 0), width=2)
        img.save(path, "JPEG")
    except ImportError:
        # Fallback: minimal valid 1×1 JPEG (no face)
        path.write_bytes(
            bytes.fromhex(
                "ffd8ffe000104a46494600010100000100010000"
                "ffdb004300080606070605080707070909080a0c"
                "140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20"
                "242e2720222c231c1c2837292c30313434341f27"
                "393d38323c2e333432ffffc0000b080001000101"
                "011100ffc4001f00000105010101010101000000"
                "000000000102030405060708090a0bffda000801"
                "01000000f0ffd9"
            )
        )


def make_near_dupe(src: Path, dest: Path) -> None:
    """Re-save at slightly different quality — same face, different bytes/sha256."""
    try:
        from PIL import Image

        img = Image.open(src).convert("RGB")
        w, h = img.size
        # Downscale 2px then upscale back → subtly different pixels, same pHash
        img = img.resize((w - 2, h - 2)).resize((w, h))
        img.save(dest, "JPEG", quality=88)
    except ImportError:
        shutil.copy(src, dest)


def make_fake_video(path: Path) -> None:
    """Write a minimal valid-ish MP4 header so the file has a video extension."""
    # ftyp box: 4-byte size + "ftyp" + "mp42" brand — enough to be recognised
    path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2")


# ── Setup modes ───────────────────────────────────────────────────────────────

# 9 roles — I is only used by --add, but always selected so the seed stays stable
ROLES = [
    "A — clean cluster",
    "B — clean cluster",
    "C — clean cluster",
    "D — merge candidate",
    "E — below threshold (2 imgs)",
    "F — at threshold (3 imgs)",
    "G — special-char folder",
    "H — deep nesting",
    "I — new cluster (add mode)",
]


def run_base(
    test_dir: Path,
    imgs: dict[str, list[Path]],
    chosen: list[str],
    collection_people: list[str] | None = None,
) -> None:
    (
        person_a,
        person_b,
        person_c,
        person_merge,
        person_few,
        person_three,
        person_special,
        person_deep,
        _person_i,  # reserved for --add
    ) = chosen

    # Case 1-3: Three clean clusters
    print("\nCase 1-3  clean clusters (series_alpha / beta / gamma)")
    copy_series(imgs[person_a], test_dir / "series_alpha", 8)
    copy_series(imgs[person_b], test_dir / "series_beta", 8)
    copy_series(imgs[person_c], test_dir / "series_gamma", 8)

    # Case 4: Same person split across two folders → merge suggestion
    print("Case 4    merge candidate (collection_part1 + collection_part2)")
    copy_series(imgs[person_merge], test_dir / "collection_part1", 4)
    copy_series(imgs[person_merge][4:], test_dir / "collection_part2", 4)

    # Case 5: 2 images — below min_images=3, no cluster formed
    print("Case 5    below threshold — tiny_set (2 images)")
    copy_series(imgs[person_few], test_dir / "tiny_set", 2)

    # Case 6: Exactly 3 images — right at default threshold
    print("Case 6    at threshold — borderline_set (3 images)")
    copy_series(imgs[person_three], test_dir / "borderline_set", 3)

    # Case 7: Exact byte-identical duplicate (sha256 hit)
    print("Case 7    exact duplicate — duplicates/")
    dup_dir = test_dir / "duplicates"
    dup_dir.mkdir()
    src7 = imgs[person_a][0]
    shutil.copy(src7, dup_dir / "original.jpg")
    shutil.copy(src7, dup_dir / "duplicate_copy.jpg")

    # Case 8: Near-duplicate (pHash ≤ 8 bits, sha256 differs)
    print("Case 8    near-duplicate — near_dupes/")
    nd_dir = test_dir / "near_dupes"
    nd_dir.mkdir()
    src8 = imgs[person_b][0]
    shutil.copy(src8, nd_dir / "img_001.jpg")
    make_near_dupe(src8, nd_dir / "img_001_resaved.jpg")

    # Case 9: No-face images (solid colour)
    print("Case 9    no-face images — no_faces/")
    nf_dir = test_dir / "no_faces"
    nf_dir.mkdir()
    make_noface_image(nf_dir / "solid_red.jpg", (200, 50, 50))
    make_noface_image(nf_dir / "solid_blue.jpg", (50, 50, 200))
    make_noface_image(nf_dir / "solid_green.jpg", (50, 200, 50))

    # Case 10: Folder name with SQL LIKE wildcards — tests % and _ escaping
    print("Case 10   special-char folder — person_%like_test/")
    copy_series(imgs[person_special], test_dir / "person_%like_test", 5)

    # Case 11: Deep nesting — recursive scan must reach it
    print("Case 11   deep nesting — deep/nested/subdir/")
    copy_series(imgs[person_deep], test_dir / "deep" / "nested" / "subdir", 5)

    # Case 12: Non-media files mixed in — scanner must skip .txt/.json without crashing
    print("Case 12   mixed content — mixed_files/ (images + .txt/.json + fake .mp4)")
    mixed_dir = test_dir / "mixed_files"
    mixed_dir.mkdir()
    copy_series(imgs[person_c][:3], mixed_dir, 3, prefix="photo")
    (mixed_dir / "metadata.txt").write_text("source: test\nauthor: setup_lfw_test\n")
    (mixed_dir / "info.json").write_text('{"collection": "test"}\n')
    make_fake_video(mixed_dir / "clip.mp4")

    # Case 13: Hidden (dot-prefix) files — must be excluded from scan
    # Note: scan_dir skips hidden *directories* via dirs[:]; files starting with "."
    # should also be skipped. This case validates (or exposes) that behaviour.
    print("Case 13   hidden files — has_hidden/ (visible imgs + .dot.jpg files)")
    hid_dir = test_dir / "has_hidden"
    hid_dir.mkdir()
    copy_series(imgs[person_b][:4], hid_dir, 4)
    make_noface_image(hid_dir / ".hidden_a.jpg", (80, 80, 80))
    make_noface_image(hid_dir / ".hidden_b.jpg", (120, 120, 120))

    # Cases 14+: Misc dirs for testing "Add to collection".
    # These are random non-famous people with 1-2 images each — below the
    # face-cluster threshold (default min_images=3), so they won't appear as
    # merge/new-person suggestions.  Use the "Add to collection" button on any
    # suggestion card, or navigate to Suggestions to find these as uncategorized.
    if collection_people:
        print(
            f"\nCases 14-{13 + len(collection_people)}  misc dirs for 'Add to collection' testing"
        )
        for i, person in enumerate(collection_people, 1):
            dest = test_dir / f"misc_person{i}"
            copy_series(imgs.get(person, [])[:4], dest, 4, prefix="photo")
            print(f"  misc_person{i}/  ← {person} (3-4 imgs)")


def run_add(test_dir: Path, imgs: dict[str, list[Path]], chosen: list[str]) -> None:
    """Simulate a second batch of downloads without wiping the test dir."""
    (
        person_a,
        person_b,
        _person_c,
        _person_merge,
        _person_few,
        _person_three,
        _person_special,
        _person_deep,
        person_i,
    ) = chosen

    # Later 1: More images of person A — near-dupes so sha256 differs but face matches.
    # After scan+merge: should suggest "add to categorized/<series_alpha person>".
    print("\nLater 1   later_alpha/ — 3 near-dupe images of person A")
    later_alpha = test_dir / "later_alpha"
    if later_alpha.exists():
        shutil.rmtree(later_alpha)
    later_alpha.mkdir()
    for i, src in enumerate(imgs[person_a][:3], 1):
        make_near_dupe(src, later_alpha / f"photo_{i:03d}.jpg")

    # Later 2: More images of person B
    print("Later 2   later_beta/ — 3 near-dupe images of person B")
    later_beta = test_dir / "later_beta"
    if later_beta.exists():
        shutil.rmtree(later_beta)
    later_beta.mkdir()
    for i, src in enumerate(imgs[person_b][:3], 1):
        make_near_dupe(src, later_beta / f"photo_{i:03d}.jpg")

    # Later 3: Brand-new person never seen before → new cluster suggestion
    print("Later 3   series_delta/ — 5 images of a new person (I)")
    series_delta = test_dir / "series_delta"
    if series_delta.exists():
        shutil.rmtree(series_delta)
    copy_series(imgs[person_i], series_delta, 5)

    # Later 4: Two people mixed in one folder → ambiguous/needs-review suggestion
    print("Later 4   ambiguous_mix/ — 3 imgs of A + 3 imgs of B in same folder")
    amb_dir = test_dir / "ambiguous_mix"
    if amb_dir.exists():
        shutil.rmtree(amb_dir)
    amb_dir.mkdir()
    for i, src in enumerate(imgs[person_a][:3], 1):
        make_near_dupe(src, amb_dir / f"a_{i:03d}.jpg")
    for i, src in enumerate(imgs[person_b][:3], 1):
        make_near_dupe(src, amb_dir / f"b_{i:03d}.jpg")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("download_dir", help="Path to your downloads/ directory")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default 42)")
    parser.add_argument(
        "--tgz",
        metavar="PATH",
        help="Path to an already-downloaded lfw-deepfunneled.tgz (skips download)",
    )
    parser.add_argument(
        "-a",
        "--add",
        action="store_true",
        help=(
            "Simulate a later batch of downloads. Adds later_alpha/, later_beta/, "
            "series_delta/, and ambiguous_mix/ WITHOUT wiping the existing test dir. "
            "Run after the initial setup and a first scan+merge cycle."
        ),
    )
    args = parser.parse_args()

    dl = Path(args.download_dir)
    dl.mkdir(parents=True, exist_ok=True)
    tgz = dl / "lfw-deepfunneled.tgz"

    if args.tgz:
        src = Path(args.tgz)
        if not src.exists():
            sys.exit(f"--tgz path not found: {src}")
        if src != tgz:
            print(f"Copying {src} → {tgz}")
            shutil.copy(src, tgz)
        else:
            print(f"Using {tgz.name}")
    elif not tgz.exists():
        download_tgz(tgz)
    else:
        print(f"Using cached {tgz.name}")

    root = detect_root(tgz)
    print(f"Archive root: {root}/")
    counts = count_people(tgz, root)

    # Prefer FAMOUS people present in the archive with 5+ images.
    famous_eligible = [p for p in FAMOUS if counts.get(p, 0) >= 5]
    print(f"Famous people with 5+ images in archive: {len(famous_eligible)}")
    for p in famous_eligible:
        print(f"  {p} ({counts[p]} imgs)")

    if len(famous_eligible) < len(ROLES):
        # Fall back: supplement with any other eligible person not already picked
        all_eligible = sorted(p for p, c in counts.items() if c >= 5)
        extras = [p for p in all_eligible if p not in famous_eligible]
        random.seed(args.seed)
        supplement = random.sample(extras, len(ROLES) - len(famous_eligible))
        pool = famous_eligible + supplement
        print(f"  (supplemented with {len(supplement)} random people to reach {len(ROLES)})")
    else:
        pool = famous_eligible

    random.seed(args.seed)
    chosen = random.sample(pool, len(ROLES))

    print("\nSelected people:")
    for role, name in zip(ROLES, chosen):
        print(f"  {role:35s}  {name} ({counts[name]} imgs)")

    # Pick non-famous people for "Add to collection" test dirs.
    # Any person with >=1 image who isn't famous and wasn't already chosen.
    non_famous_pool = sorted(
        p for p, c in counts.items() if c >= 1 and p not in set(FAMOUS) and p not in set(chosen)
    )
    random.seed(args.seed + 99)
    coll_people = random.sample(non_famous_pool, min(3, len(non_famous_pool)))
    print("\nCollection test dirs:")
    for i, name in enumerate(coll_people, 1):
        print(f"  misc_person{i}  ← {name} ({counts[name]} imgs in archive, copying up to 4)")

    test_dir = dl / "test"

    if args.add:
        if not test_dir.exists():
            sys.exit("--add requires the test dir to already exist. Run without --add first.")
        # Only re-extract the people needed for add mode
        needed = [chosen[0], chosen[1], chosen[8]]  # A, B, I
        import tempfile

        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)
            imgs: dict[str, list[Path]] = {}
            for person in needed:
                print(f"  extracting {person}…")
                imgs[person] = extract_person(tgz, root, person, tmp / person)
            # Fill in stubs for unused people so run_add can unpack correctly
            for person in chosen:
                if person not in imgs:
                    imgs[person] = []
            run_add(test_dir, imgs, chosen)
    else:
        # Wipe test/ AND categorized/ so stale merged folders from previous runs
        # don't pollute face-similarity scores in the fresh test.
        for stale_dir in (test_dir, dl / "categorized"):
            if stale_dir.exists():
                print(f"Wiping {stale_dir}/")
                shutil.rmtree(stale_dir)
        test_dir.mkdir()
        print(
            "\n⚠  DB reset required: open Settings → 'Reset face data' before scanning,\n"
            "   otherwise old embeddings from previous runs will interfere.\n"
        )

        import tempfile

        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)
            imgs = {}
            for person in chosen + coll_people:
                print(f"  extracting {person}…")
                imgs[person] = extract_person(tgz, root, person, tmp / person)
            run_base(test_dir, imgs, chosen, collection_people=coll_people)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(1 for f in test_dir.rglob("*") if f.is_file())
    print(f"\nDone — {total} files in {test_dir}")
    print()

    if args.add:
        print("Expected behaviour after re-scan + analyse:")
        print("  later_alpha/        → 'add to' or merge with series_alpha person")
        print("  later_beta/         → 'add to' or merge with series_beta person")
        print("  series_delta/       → new clean cluster (5 images, unseen person)")
        print("  ambiguous_mix/      → ambiguous/needs-review (2 people in one folder)")
    else:
        print("Expected behaviour after scan + analyse:")
        print("  series_alpha/beta/gamma    → 3 clean clusters, each 8 images")
        print("  collection_part1+part2     → merge suggestion (same person, 2 folders)")
        print("  tiny_set                   → no cluster (2 imgs < min_images=3)")
        print("  borderline_set             → borderline cluster (exactly 3 imgs)")
        print("  duplicates/                → sha256 exact dup detected")
        print("  near_dupes/                → pHash near-dup detected")
        print("  no_faces/                  → 3 images, face_count=0 each")
        print("  person_%like_test/         → LIKE wildcard escaping exercised")
        print("  deep/nested/subdir/        → recursive scan reaches nested dir")
        print("  mixed_files/               → .txt/.json/.mp4 skipped, 3 jpgs scanned")
        print("  has_hidden/                → .hidden_*.jpg excluded, 4 jpgs scanned")
        for i in range(1, len(coll_people) + 1):
            print(
                f"  misc_person{i}/             → 3-4 imgs — use 'Add to collection' button to test"
            )


if __name__ == "__main__":
    main()
