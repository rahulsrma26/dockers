def extract(url: str) -> list[tuple]:
    # yt-dlp handles all YouTube URL formats directly
    return [(url, "video")]
