"""Tests for GeminiClient gateway support, YouTubeClip payloads, caching, and yt-upload."""
from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from rea_video import gemini


def test_default_model_and_env_precedence(monkeypatch):
    assert gemini.DEFAULT_MODEL == "gemini-4-hydrogen"

    monkeypatch.delenv("REA_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("REA_GEMINI_API_BASE", raising=False)
    monkeypatch.delenv("REA_GEMINI_MODEL", raising=False)

    assert gemini.api_base() == gemini.GOOGLE_API
    assert not gemini.configured()

    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("REA_GEMINI_API_KEY", "rea-key")
    monkeypatch.setenv("REA_GEMINI_API_BASE", "https://gateway.example.com/")

    assert gemini.api_key() == "rea-key"
    assert gemini.api_base() == "https://gateway.example.com"
    assert gemini.configured()

    c = gemini.GeminiClient()
    assert c.model == "gemini-4-hydrogen"
    assert c.is_gateway is True
    assert c.describe() == "Gemini gemini-4-hydrogen @ gateway.example.com"


def test_youtube_clip_serialization_and_thought_filter(tmp_path):
    c = gemini.GeminiClient(
        model="gemini-4-hydrogen",
        key="secret-test-key",
        api_base="https://gateway.example.com",
    )
    calls = []

    def fake_call(method, url, data=None, headers=None, timeout=120.0):
        calls.append((method, url, json.loads(data.decode("utf-8"))))
        resp = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"thought": True, "text": '{"wrong": true}'},
                        {"thoughtSignature": "sig123", "text": '```json\n{"animals": "elephants"}\n```'},
                    ]
                }
            }]
        }
        return {}, json.dumps(resp).encode("utf-8")

    with patch.object(c, "_call", side_effect=fake_call):
        whole = c.ask_json("Describe whole", [(gemini.YouTubeClip("https://youtu.be/jNQXAC9IVRw"), "video/mp4")])
        seg = c.ask_json(
            "Describe seg",
            [(gemini.YouTubeClip("https://www.youtube.com/watch?v=jNQXAC9IVRw", start=12.2, end=12.8), "video/mp4")],
        )

    assert whole == {"animals": "elephants"}
    assert seg == {"animals": "elephants"}
    assert calls[0][1] == "https://gateway.example.com/v1beta/models/gemini-4-hydrogen:generateContent"

    whole_part = calls[0][2]["contents"][0]["parts"][0]
    assert whole_part["fileData"] == {"mimeType": "video/mp4", "fileUri": "https://youtu.be/jNQXAC9IVRw"}
    assert "videoMetadata" not in whole_part

    seg_part = calls[1][2]["contents"][0]["parts"][0]
    assert seg_part["fileData"] == {"mimeType": "video/mp4", "fileUri": "https://www.youtube.com/watch?v=jNQXAC9IVRw"}
    assert seg_part["videoMetadata"] == {"startOffset": "12s", "endOffset": "13s", "fps": 10.0}


def test_gateway_rejects_large_local_video_without_files_api(tmp_path):
    big = tmp_path / "big.mp4"
    big.write_bytes(b"0" * (gemini.INLINE_MAX + 1))

    c = gemini.GeminiClient(
        model="gemini-4-hydrogen",
        key="secret-test-key",
        api_base="https://gateway.example.com",
    )
    with patch.object(c, "_call") as mock_call:
        with pytest.raises(gemini.GeminiError) as exc:
            c.ask_json("Prompt", [(big, "video/mp4")])
        assert "--youtube-url" in str(exc.value) or "--yt-upload" in str(exc.value)
        mock_call.assert_not_called()


def test_disk_cache_hits_and_never_stores_api_key(tmp_path):
    cache_dir = tmp_path / "vlm_cache"
    secret = "sk-never-write-this-key"
    c = gemini.GeminiClient(
        model="gemini-4-hydrogen",
        key=secret,
        api_base="https://gateway.example.com",
        cache_dir=cache_dir,
    )
    n_calls = 0

    def fake_call(method, url, data=None, headers=None, timeout=120.0):
        nonlocal n_calls
        n_calls += 1
        return {}, json.dumps({"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}).encode("utf-8")

    clip1 = gemini.YouTubeClip("https://youtu.be/abc12345678", start=0.0, end=5.0)
    clip2 = gemini.YouTubeClip("https://youtu.be/abc12345678", start=5.0, end=10.0)

    with patch.object(c, "_call", side_effect=fake_call):
        r1 = c.ask_json("Prompt", [(clip1, "video/mp4")])
        r2 = c.ask_json("Prompt", [(clip1, "video/mp4")])
        r3 = c.ask_json("Prompt", [(clip2, "video/mp4")])

    assert r1 == r2 == r3 == {"ok": True}
    assert n_calls == 2

    c_other_model = gemini.GeminiClient(
        model="gemini-3.8-flash",
        key=secret,
        api_base="https://gateway.example.com",
        cache_dir=cache_dir,
    )
    with patch.object(c_other_model, "_call", side_effect=fake_call):
        c_other_model.ask_json("Prompt", [(clip1, "video/mp4")])
    assert n_calls == 3

    files = list(cache_dir.glob("*.json"))
    assert len(files) == 3
    for f in files:
        assert secret not in f.name
        assert secret not in f.read_text(encoding="utf-8")


def test_invalid_media_retries_only_for_youtube_clip():
    c = gemini.GeminiClient(
        model="gemini-4-hydrogen",
        key="secret-test-key",
        api_base="https://gateway.example.com",
    )
    c._retry_delays = (0.0, 0.0, 0.0)
    yt = [(gemini.YouTubeClip("https://youtu.be/abc12345678"), "video/mp4")]

    attempts = 0

    def flaky_call(method, url, data=None, headers=None, timeout=120.0):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise gemini.GeminiError("HTTP 400: INVALID_MEDIA: still processing")
        return {}, json.dumps({"candidates": [{"content": {"parts": [{"text": '{"ready": true}'}]}}]}).encode("utf-8")

    with patch.object(c, "_call", side_effect=flaky_call):
        out = c.ask_json("Prompt", yt)
    assert out == {"ready": True}
    assert attempts == 3

    # Non-INVALID_MEDIA HTTP 400 fails immediately without retry
    attempts_bad = 0

    def bad_call(method, url, data=None, headers=None, timeout=120.0):
        nonlocal attempts_bad
        attempts_bad += 1
        raise gemini.GeminiError("HTTP 400: BAD_REQUEST")

    with patch.object(c, "_call", side_effect=bad_call):
        with pytest.raises(gemini.GeminiError):
            c.ask_json("Prompt", yt)
    assert attempts_bad == 1


def test_is_youtube_url_and_upload_youtube(tmp_path):
    assert gemini.is_youtube_url("https://youtu.be/jNQXAC9IVRw")
    assert gemini.is_youtube_url("https://www.youtube.com/watch?v=jNQXAC9IVRw")
    assert gemini.is_youtube_url("https://youtube.com/shorts/abcDEF12345")
    assert not gemini.is_youtube_url("https://vimeo.com/123456")

    vid = tmp_path / "demo.mp4"
    vid.write_bytes(b"fake")

    with patch("shutil.which", return_value="C:\\bin\\yt-upload.ps1"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Uploading...\nhttps://youtu.be/xyz98765432\n", stderr=""
        )
        url = gemini.upload_youtube(vid)

    assert url == "https://youtu.be/xyz98765432"
    cmd = mock_run.call_args[0][0]
    assert "--privacy" in cmd and "unlisted" in cmd
    assert str(vid) in cmd

    with patch("shutil.which", return_value="C:\\bin\\yt-upload.ps1"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="auth failed")
        with pytest.raises(gemini.GeminiError):
            gemini.upload_youtube(vid)
