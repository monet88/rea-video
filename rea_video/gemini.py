"""Google Gemini as the image / video reader, through its own API or a Gemini-compatible gateway.

Set REA_GEMINI_API_KEY (or GEMINI_API_KEY / GOOGLE_API_KEY); the model is REA_GEMINI_MODEL (default gemini-4-hydrogen).
Optional REA_GEMINI_API_BASE points at a Gemini-compatible gateway (e.g. https://gateway.example.com); custom gateways
proxy :generateContent (including YouTube fileUri + startOffset/endOffset) but not the Files API.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .media import duration_s
from .util import LOG

GOOGLE_API = "https://generativelanguage.googleapis.com"
API = GOOGLE_API
DEFAULT_MODEL = "gemini-4-hydrogen"
INLINE_MAX = 14 * 1024 * 1024  # base64 adds 33%: 14 MB raw -> ~18.7 MB body, under Gemini's 20 MB inline limit
CLIP_FPS = ((30.0, 10.0), (120.0, 4.0))  # (clips up to s, frames a second); longer: Gemini's default (1)
_YOUTUBE_RE = re.compile(
    r"^https?://(?:(?:www\.|m\.|music\.)?youtube\.com/(?:watch\?\S*\bv=|shorts/|live/)|youtu\.be/)\S+$",
    re.I,
)
_YOUTUBE_FIND_RE = re.compile(
    r"https?://(?:(?:www\.|m\.|music\.)?youtube\.com/\S+|youtu\.be/\S+)",
    re.I,
)


@dataclass(frozen=True)
class YouTubeClip:
    """Reference to a YouTube video (optionally clipped via videoMetadata.startOffset/endOffset)."""

    url: str
    start: float | None = None
    end: float | None = None


def api_base() -> str:
    return (os.getenv("REA_GEMINI_API_BASE", "") or GOOGLE_API).strip().rstrip("/")


def api_key() -> str:
    return os.getenv("REA_GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")


def configured() -> bool:
    return bool(api_key())


def is_youtube_url(url: str) -> bool:
    return bool(_YOUTUBE_RE.match((url or "").strip()))


def resolve_youtube_url(raw: str | None) -> str | None:
    url = (raw or os.getenv("REA_YOUTUBE_URL", "")).strip() or None
    if url and not is_youtube_url(url):
        raise ValueError(f"invalid YouTube URL: {url}")
    return url


def _clip_fps(dur: float) -> float | None:
    return next((f for lim, f in CLIP_FPS if 0 < dur <= lim), None)


def upload_youtube(video: Path, title: str | None = None) -> str:
    """Upload a local video as unlisted via the global `yt-upload` CLI and return its YouTube URL."""
    exe = shutil.which("yt-upload") or shutil.which("yt-upload.cmd") or shutil.which("yt-upload.ps1")
    if not exe:
        fallback = Path.home() / ".local" / "bin" / "yt-upload.ps1"
        if fallback.is_file():
            exe = str(fallback)
    if not exe:
        raise GeminiError("yt-upload CLI not found on PATH")
    clean_title = re.sub(r"[<>]", "", (title or Path(video).stem).strip())[:90] or "rea-video"
    args = [str(video), "--title", clean_title, "--privacy", "unlisted"]
    if exe.lower().endswith(".ps1"):
        shell = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        cmd = [shell, "-NoProfile", "-File", exe, *args]
    else:
        cmd = [exe, *args]
    LOG.info("uploading %s to YouTube (unlisted) via yt-upload", Path(video).name)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip()[:200]
        raise GeminiError(f"yt-upload failed ({r.returncode}): {detail}")
    urls = _YOUTUBE_FIND_RE.findall(r.stdout or "")
    if not urls:
        raise GeminiError("yt-upload succeeded but no YouTube URL found in stdout")
    return urls[-1]


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    """ask_json(prompt, media) like VLMClient; media = [(path | YouTubeClip, mime)], images and video/mp4 clips."""

    video_ok = True
    youtube_ok = True
    media = "both"  # watches the clip with its sound and reads the key frames

    def __init__(
        self,
        model: str | None = None,
        key: str | None = None,
        timeout: float = 600.0,
        api_base: str | None = None,
        cache_dir: Path | None = None,
    ):
        self.key = key or api_key()
        if not self.key:
            raise GeminiError("REA_GEMINI_API_KEY / GEMINI_API_KEY is not set")
        self.model = model or os.getenv("REA_GEMINI_MODEL", "") or DEFAULT_MODEL
        self.timeout = timeout
        self.api_base = (api_base or globals()["api_base"]()).strip().rstrip("/")
        self.is_gateway = self.api_base != GOOGLE_API
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models = [self.model]
        self._retry_delays = (20.0, 40.0, 60.0)

    def describe(self) -> str:
        if not self.is_gateway:
            return f"Gemini {self.model}"
        host = re.sub(r"^https?://", "", self.api_base).split("/")[0].rsplit("@", 1)[-1]
        return f"Gemini {self.model} @ {host}"

    def _call(self, method: str, url: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 120.0):
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"x-goog-api-key": self.key, "User-Agent": f"rea-video/{__version__}", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return dict(r.headers), r.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()[:300].decode("utf-8", "replace")
            raise GeminiError(f"HTTP {exc.code}: {body}") from None

    def _upload(self, p: Path, mime: str, max_wait: float = 900.0) -> dict:
        size = p.stat().st_size
        h, _ = self._call(
            "POST",
            f"{self.api_base}/upload/v1beta/files",
            json.dumps({"file": {"display_name": "rea-video"}}).encode(),
            {
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime,
                "Content-Type": "application/json",
            },
        )
        url = {k.lower(): v for k, v in h.items()}.get("x-goog-upload-url")
        if not url:
            raise GeminiError("the Files API returned no upload URL")
        _, body = self._call(
            "POST",
            url,
            p.read_bytes(),
            {"Content-Length": str(size), "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"},
            timeout=3600.0,
        )
        info = json.loads(body)["file"]
        waited = 0.0
        try:
            while info.get("state") == "PROCESSING" and waited < max_wait:
                time.sleep(2.0)
                waited += 2.0
                info = json.loads(self._call("GET", f"{self.api_base}/v1beta/{info['name']}")[1])
        except Exception:
            self._delete(info.get("name", ""))
            raise
        if info.get("state") != "ACTIVE":
            self._delete(info.get("name", ""))
            raise GeminiError(f"uploaded file state {info.get('state')}")
        return info

    def _delete(self, name: str) -> None:
        if not name:
            return
        try:
            self._call("DELETE", f"{self.api_base}/v1beta/{name}")
        except Exception as exc:  # noqa: BLE001  cleanup never masks the answer; Google drops it within 48 h
            LOG.warning("could not delete %s from Gemini (%s); it expires within 48 hours", name, str(exc)[:80])

    def _cache_key(self, prompt: str, media: list[tuple[Path | YouTubeClip, str]], max_tokens: int) -> str:
        # Never include the API key in the cache identity.
        request = {
            "api_base": self.api_base,
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        h = hashlib.sha1(json.dumps(request, sort_keys=True, ensure_ascii=False).encode())
        for item, mime in media:
            h.update(mime.encode())
            if isinstance(item, YouTubeClip):
                h.update(f"yt:{item.url}|{item.start}|{item.end}".encode())
            else:
                with Path(item).open("rb") as f:
                    while chunk := f.read(1 << 20):
                        h.update(chunk)
        return h.hexdigest()

    def ask_json(self, prompt: str, media: list[tuple[Path | YouTubeClip, str]], max_tokens: int = 4096) -> dict:
        from .vlm import parse_json

        cached = self.cache_dir / f"{self._cache_key(prompt, media, max_tokens)}.json" if self.cache_dir else None
        if cached and cached.is_file():
            try:
                got = json.loads(cached.read_text(encoding="utf-8"))
                if isinstance(got, dict):
                    return got
            except (OSError, ValueError):
                pass  # a half-written file from an interrupted run: ask again

        local_items = [(Path(p), mime) for p, mime in media if not isinstance(p, YouTubeClip)]
        inline = sum(p.stat().st_size for p, _ in local_items) <= INLINE_MAX
        if self.is_gateway and not inline and any(mime.startswith("video/") for _, mime in local_items):
            raise GeminiError(
                "local video exceeds 14 MB inline limit for custom REA_GEMINI_API_BASE (no Files API); "
                "pass --youtube-url or --yt-upload"
            )

        has_youtube = any(isinstance(p, YouTubeClip) for p, _ in media)
        parts, uploaded = [], []
        try:
            for item, mime in media:
                part: dict
                if isinstance(item, YouTubeClip):
                    part = {"fileData": {"mimeType": mime or "video/mp4", "fileUri": item.url}}
                    meta: dict = {}
                    start_sec = max(0, int(math.floor(item.start))) if item.start is not None else None
                    if start_sec is not None:
                        meta["startOffset"] = f"{start_sec}s"
                    if item.end is not None:
                        end_sec = max((start_sec + 1) if start_sec is not None else 1, int(math.ceil(item.end)))
                        meta["endOffset"] = f"{end_sec}s"
                    if item.start is not None and item.end is not None:
                        fps = _clip_fps(max(0.0, float(item.end) - float(item.start)))
                        if fps:
                            meta["fps"] = fps
                    if meta:
                        part["videoMetadata"] = meta
                else:
                    p = Path(item)
                    if inline or not mime.startswith("video/"):
                        part = {"inlineData": {"mimeType": mime, "data": base64.b64encode(p.read_bytes()).decode()}}
                    else:
                        info = self._upload(p, mime)
                        uploaded.append(info["name"])
                        part = {"fileData": {"mimeType": info.get("mimeType") or mime, "fileUri": info["uri"]}}
                    if mime.startswith("video/"):
                        dur = duration_s(p)
                        if dur <= 0:
                            LOG.warning("could not read the length of %s; Gemini samples it at its default rate", p.name)
                        fps = _clip_fps(dur)
                        if fps:
                            part["videoMetadata"] = {"fps": fps}
                parts.append(part)
            parts.append({"text": prompt})
            body = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2, "responseMimeType": "application/json"},
            }
            payload = json.dumps(body).encode()
            url = f"{self.api_base}/v1beta/models/{self.model}:generateContent"
            delays = list(self._retry_delays) if has_youtube else []
            while True:
                try:
                    _, raw = self._call("POST", url, payload, {"Content-Type": "application/json"}, timeout=self.timeout)
                    break
                except GeminiError as exc:
                    msg = str(exc)
                    if delays and "HTTP 400" in msg and "INVALID_MEDIA" in msg.upper():
                        wait = delays.pop(0)
                        LOG.warning("YouTube media still transcoding (INVALID_MEDIA); retrying in %.0fs", wait)
                        time.sleep(wait)
                        continue
                    raise
        finally:
            for name in uploaded:
                self._delete(name)

        data = json.loads(raw)
        try:
            cand_parts = data["candidates"][0]["content"]["parts"]
            text = "".join(pt.get("text", "") for pt in cand_parts if not pt.get("thought"))
            if not text:
                text = "".join(pt.get("text", "") for pt in cand_parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise GeminiError(f"no answer: {str(data)[:200]}") from exc
        out = parse_json(text)
        if cached and isinstance(out, dict):
            tmp = cached.with_name(f"{cached.name}.{os.getpid()}.{threading.get_ident()}.part")
            tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(cached)
        return out
