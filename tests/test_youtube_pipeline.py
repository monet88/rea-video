"""Tests for YouTube URL / gateway wiring across vlm, pipeline, moments, and cli."""
from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from rea_video import cli, gemini, moments, pipeline, vlm
from rea_video.shots import Shot


def test_make_client_prefer_gemini_overrides_rea_vlm(monkeypatch, tmp_path):
    monkeypatch.setenv("REA_VLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("REA_VLM_API_KEY", "vlm-key")
    monkeypatch.setenv("REA_VLM_MODEL", "gpt-4o")
    monkeypatch.setenv("REA_GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("REA_GEMINI_API_BASE", "https://gateway.example.com")

    c_default = vlm.make_client(tmp_path / "c1")
    assert isinstance(c_default, vlm.VLMClient)

    c_yt = vlm.make_client(tmp_path / "c2", prefer_gemini=True)
    assert isinstance(c_yt, gemini.GeminiClient)
    assert c_yt.cache_dir == tmp_path / "c2"

    monkeypatch.delenv("REA_VLM_BASE_URL")
    monkeypatch.delenv("REA_VLM_API_KEY")
    monkeypatch.delenv("REA_VLM_MODEL")
    c_only_gem = vlm.make_client(tmp_path / "c3")
    assert isinstance(c_only_gem, gemini.GeminiClient)
    assert c_only_gem.cache_dir == tmp_path / "c3"


def test_moments_vlm_asker_uses_youtube_clip_without_cut_clip(tmp_path):
    calls = []

    class FakeClient:
        youtube_ok = True
        video_ok = True

        def ask_json(self, prompt, media, max_tokens=3000):
            calls.append((prompt, media))
            return {"moments": []}

    img = tmp_path / "overview.jpg"
    img.write_bytes(b"jpg")
    asker = moments.vlm_asker(FakeClient(), youtube_url="https://youtu.be/abc12345678")

    with patch("rea_video.media.cut_clip", side_effect=AssertionError("cut_clip should not be called")):
        asker({"id": "find", "instructions": "find", "state": {}, "questions": [], "answer_template": {}}, [img])
        asker(
            {
                "id": "pin",
                "instructions": "pin",
                "state": {"M1": {"times": [10.0, 12.0]}},
                "questions": [],
                "answer_template": {},
            },
            [img],
        )

    assert len(calls) == 2
    assert isinstance(calls[0][1][0][0], gemini.YouTubeClip)
    assert calls[0][1][0][0] == gemini.YouTubeClip("https://youtu.be/abc12345678")
    assert isinstance(calls[1][1][0][0], gemini.YouTubeClip)
    assert calls[1][1][0][0].start == pytest.approx(9.5)
    assert calls[1][1][0][0].end == pytest.approx(12.5)


def test_pipeline_decompose_youtube_url_and_yt_upload(monkeypatch, tmp_path):
    vid = tmp_path / "sample.mp4"
    vid.write_bytes(b"video")
    out = tmp_path / "out"

    info = SimpleNamespace(
        width=1080,
        height=1920,
        fps=30.0,
        duration=6.0,
        vcodec="h264",
        aspect_ratio="9:16",
        has_audio=False,
        rotation=0,
        display_size=(1080, 1920),
    )
    track = SimpleNamespace(
        fps=30.0,
        n=180,
        times=[i / 30.0 for i in range(180)],
        thumbs=np.zeros((180, 12, 8), dtype=np.float32),
    )
    shots = [Shot(index=1, start_frame=0, end_frame=180, fps=30.0, transition_in="start")]

    class FakeClient:
        youtube_ok = True
        media = "both"

        def __init__(self):
            self.calls = []
            self.fail_first_shot = True

        def describe(self):
            return "Gemini gemini-4-hydrogen @ gateway.example.com"

        def ask_json(self, prompt, media, max_tokens=4096):
            self.calls.append(media)
            if self.fail_first_shot and any(isinstance(m[0], gemini.YouTubeClip) and m[0].start is not None for m in media):
                self.fail_first_shot = False
                raise RuntimeError("video part failed")
            return {"label_probs": {"live_action": 1.0}, "evidence": ["ok"], "summary": "s"}

    fake_client = FakeClient()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    smp = SimpleNamespace(frames=[frame, frame], frame_times=[0.0, 3.0])

    monkeypatch.setenv("REA_GEMINI_API_KEY", "test-key")
    with ExitStack() as stack:
        for target, kw in (
            ("rea_video.pipeline.require_tools", {}),
            ("rea_video.pipeline.video_info", {"return_value": info}),
            ("rea_video.pipeline.provenance", {"return_value": {"tags": {}}}),
            ("rea_video.pipeline.build_track", {"return_value": track}),
            ("rea_video.pipeline.detect_shots", {"return_value": shots}),
            ("rea_video.survey.load_timeline", {"return_value": None}),
            ("rea_video.pipeline.collect_all_samples", {"return_value": {1: smp}}),
            ("rea_video.pipeline.key_sheets", {"return_value": {}}),
            ("rea_video.pipeline.extract_frames", {}),
            ("rea_video.pipeline.extract_frame", {}),
            ("rea_video.pipeline.shot_features", {"return_value": {"camera": "static"}}),
            ("rea_video.pipeline.static_overlays", {"return_value": {}}),
            ("rea_video.pipeline.detect_embedded", {"return_value": []}),
            ("rea_video.pipeline.shot_faces", {"return_value": {"presence": 0.0, "frames": []}}),
            ("rea_video.pipeline.aspects.measure_shot", {"return_value": {}}),
            ("rea_video.pipeline.deterministic", {"return_value": ({"live_action": 0.8, "ai_video": 0.2}, [], {})}),
            ("rea_video.pipeline.overlay_layers", {"return_value": []}),
            ("rea_video.pipeline.shot_sheet", {}),
            ("rea_video.pipeline.video_context", {"return_value": {}}),
            ("rea_video.pipeline.cut_clip", {"side_effect": AssertionError("cut_clip should not be called")}),
            ("rea_video.pipeline.finalize", {"side_effect": lambda b, *a, **kw: b}),
        ):
            stack.enter_context(patch(target, **kw))
        mock_make = stack.enter_context(patch("rea_video.pipeline.make_client", return_value=fake_client))
        mock_up = stack.enter_context(patch("rea_video.gemini.upload_youtube", return_value="https://youtu.be/uploaded123"))
        b = pipeline.decompose(
            vid,
            out,
            use_vlm=True,
            ocr=False,
            packaging=False,
            stems=False,
            asr=False,
            speakers=False,
            survey=False,
            moments=False,
            yt_upload=True,
        )

    mock_up.assert_called_once_with(vid.resolve())
    mock_make.assert_called_once_with(out.resolve() / "vlm_cache", prefer_gemini=True)
    assert b["source"]["youtube_url"] == "https://youtu.be/uploaded123"
    # First call: YouTubeClip(0, 6) + 3 keyframes -> failed; second call: frames_only (no mute_clip); third call: whole_vlm YouTubeClip(None, None)
    assert len(fake_client.calls) == 3
    assert isinstance(fake_client.calls[0][0][0], gemini.YouTubeClip)
    assert len(fake_client.calls[0]) == 4
    assert all(not isinstance(m[0], gemini.YouTubeClip) and m[1] == "image/jpeg" for m in fake_client.calls[1])
    assert fake_client.calls[2][0][0] == gemini.YouTubeClip("https://youtu.be/uploaded123")


def test_cli_accepts_youtube_flags_and_rejects_bad_url(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["decompose", "x.mp4", "--vlm", "--youtube-url", "https://not-youtube.example/v"])
    assert exc.value.code != 0

    with pytest.raises(SystemExit) as exc_help:
        cli.main(["--lang", "en", "decompose", "--help"])
    assert exc_help.value.code == 0
    help_txt = capsys.readouterr().out
    assert "--youtube-url" in help_txt
    assert "--yt-upload" in help_txt
