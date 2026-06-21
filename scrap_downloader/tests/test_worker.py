import textwrap

from worker import _load_plugin


def test_load_plugin_found(tmp_path, monkeypatch):
    plugin_file = tmp_path / "example.com.py"
    plugin_file.write_text(
        textwrap.dedent("""
            def extract(url):
                return [(url, "video")]
        """)
    )
    monkeypatch.setenv("PLUGINS_DIR", str(tmp_path))
    import worker

    monkeypatch.setattr(worker, "PLUGINS_DIR", str(tmp_path))

    plugin = _load_plugin("example.com")
    assert plugin is not None
    result = plugin.extract("https://example.com/v.mp4")
    assert result == [("https://example.com/v.mp4", "video")]


def test_load_plugin_strips_www(tmp_path, monkeypatch):
    plugin_file = tmp_path / "example.com.py"
    plugin_file.write_text("def extract(url): return []")

    import worker

    monkeypatch.setattr(worker, "PLUGINS_DIR", str(tmp_path))

    plugin = _load_plugin("www.example.com")
    assert plugin is not None


def test_load_plugin_missing(tmp_path, monkeypatch):
    import worker

    monkeypatch.setattr(worker, "PLUGINS_DIR", str(tmp_path))

    plugin = _load_plugin("notfound.com")
    assert plugin is None
