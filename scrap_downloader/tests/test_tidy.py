import os

from scrap_downloader.tidy import tidy_dir


def make(d, *names):
    os.makedirs(d, exist_ok=True)
    for name in names:
        open(os.path.join(d, name), "w").close()


# ---------------------------------------------------------------------------
# Core grouping rules
# ---------------------------------------------------------------------------


def test_two_groups_are_separated(tmp_path):
    make(
        str(tmp_path),
        "Album One 001.jpg",
        "Album One 002.jpg",
        "Album Two 001.jpg",
        "Album Two 002.jpg",
    )
    moved = tidy_dir(str(tmp_path))
    assert len(moved) == 4
    assert os.path.isfile(tmp_path / "Album One" / "Album One 001.jpg")
    assert os.path.isfile(tmp_path / "Album One" / "Album One 002.jpg")
    assert os.path.isfile(tmp_path / "Album Two" / "Album Two 001.jpg")
    assert os.path.isfile(tmp_path / "Album Two" / "Album Two 002.jpg")


def test_single_group_only_not_moved(tmp_path):
    make(str(tmp_path), "Album 001.jpg", "Album 002.jpg", "Album 003.jpg")
    moved = tidy_dir(str(tmp_path))
    assert moved == []
    assert os.path.isfile(tmp_path / "Album 001.jpg")


def test_single_group_with_other_files_not_moved(tmp_path):
    make(str(tmp_path), "Album 001.jpg", "Album 002.jpg", "cover.jpg", "info.txt")
    moved = tidy_dir(str(tmp_path))
    assert moved == []
    assert os.path.isfile(tmp_path / "Album 001.jpg")
    assert os.path.isfile(tmp_path / "cover.jpg")


def test_lone_numbered_file_ignored(tmp_path):
    # Only one file with this prefix — not a group
    make(str(tmp_path), "Album 001.jpg", "Other 001.jpg", "Other 002.jpg")
    moved = tidy_dir(str(tmp_path))
    assert moved == []


def test_unnumbered_files_never_moved(tmp_path):
    make(
        str(tmp_path),
        "Album One 001.jpg",
        "Album One 002.jpg",
        "Album Two 001.jpg",
        "Album Two 002.jpg",
        "cover.jpg",
        "readme.txt",
    )
    moved = tidy_dir(str(tmp_path))
    assert len(moved) == 4
    assert os.path.isfile(tmp_path / "cover.jpg")
    assert os.path.isfile(tmp_path / "readme.txt")


# ---------------------------------------------------------------------------
# Dot files
# ---------------------------------------------------------------------------


def test_dot_files_ignored_and_not_counted(tmp_path):
    # .DS_Store should not count as an "other file" that triggers grouping
    make(str(tmp_path), "Album 001.jpg", "Album 002.jpg", ".DS_Store", ".gitkeep")
    moved = tidy_dir(str(tmp_path))
    assert moved == []
    assert os.path.isfile(tmp_path / ".DS_Store")


def test_dot_files_ignored_with_two_groups(tmp_path):
    make(
        str(tmp_path),
        "Album One 001.jpg",
        "Album One 002.jpg",
        "Album Two 001.jpg",
        "Album Two 002.jpg",
        ".DS_Store",
    )
    moved = tidy_dir(str(tmp_path))
    assert len(moved) == 4
    assert os.path.isfile(tmp_path / ".DS_Store")


# ---------------------------------------------------------------------------
# Number format variations
# ---------------------------------------------------------------------------


def test_zero_padded_numbers(tmp_path):
    make(str(tmp_path), "A 001.jpg", "A 002.jpg", "B 001.jpg", "B 002.jpg")
    assert len(tidy_dir(str(tmp_path))) == 4


def test_single_digit_numbers(tmp_path):
    make(str(tmp_path), "A 1.jpg", "A 2.jpg", "B 1.jpg", "B 2.jpg")
    assert len(tidy_dir(str(tmp_path))) == 4


def test_no_space_before_number_not_matched(tmp_path):
    # "track1" has no space — should not be grouped
    make(str(tmp_path), "track1.mp3", "track2.mp3", "other1.mp3", "other2.mp3")
    assert tidy_dir(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotent_two_groups(tmp_path):
    make(
        str(tmp_path),
        "Album One 001.jpg",
        "Album One 002.jpg",
        "Album Two 001.jpg",
        "Album Two 002.jpg",
    )
    tidy_dir(str(tmp_path))
    moved2 = tidy_dir(str(tmp_path))
    assert moved2 == []


def test_idempotent_single_group(tmp_path):
    make(str(tmp_path), "Album 001.jpg", "Album 002.jpg")
    tidy_dir(str(tmp_path))
    assert tidy_dir(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# Recursive / nested directories
# ---------------------------------------------------------------------------


def test_walks_nested_structure(tmp_path):
    user_dir = tmp_path / "site.com" / "user"
    make(
        str(user_dir),
        "Album One 001.jpg",
        "Album One 002.jpg",
        "Album Two 001.jpg",
        "Album Two 002.jpg",
    )
    moved = tidy_dir(str(tmp_path))
    assert len(moved) == 4
    assert os.path.isfile(user_dir / "Album One" / "Album One 001.jpg")


def test_nested_single_group_untouched(tmp_path):
    user_dir = tmp_path / "site.com" / "user"
    make(str(user_dir), "Gallery 001.jpg", "Gallery 002.jpg", "Gallery 003.jpg")
    assert tidy_dir(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_nonexistent_path_returns_empty():
    assert tidy_dir("/nonexistent/path/xyz") == []


def test_empty_directory(tmp_path):
    assert tidy_dir(str(tmp_path)) == []


def test_progress_callback(tmp_path):
    make(
        str(tmp_path),
        "A 001.jpg",
        "A 002.jpg",
        "B 001.jpg",
        "B 002.jpg",
    )
    calls = []
    tidy_dir(str(tmp_path), on_progress=lambda done, total: calls.append((done, total)))
    assert len(calls) == 4
    assert calls[-1] == (4, 4)
