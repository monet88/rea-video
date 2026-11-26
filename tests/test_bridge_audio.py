"""Tests for Bridge Audio Stage (VieNeu-TTS dubbing + stem mix) and Assemble Stage."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rea_video.bridge import BridgeConfig
from rea_video.util import write_json


def _make_breakdown() -> dict:
    return {
        "source": {"path": "source.mp4", "aspect_ratio": "9:16", "duration": 10.0, "has_audio": True},
        "shots": [
            {"index": 1, "start": 0.0, "end": 5.0, "duration": 5.0, "base": {"label": "live_action"}},
            {"index": 2, "start": 5.0, "end": 10.0, "duration": 5.0, "base": {"label": "ai_video"}},
        ],
        "speakers": {
            "count": 2,
            "speakers": [
                {"id": "S1", "name": "Host"},
                {"id": "S2", "name": ""},
            ],
            "turns": [
                {"speaker": "S1", "start": 0.2, "end": 2.4},
                {"speaker": "S2", "start": 5.5, "end": 8.0},
            ],
        },
        "speech": {
            "lines": [
                {"start": 0.2, "end": 2.4, "speaker": "S1", "text": "Hello everyone"},
                {"start": 5.5, "end": 8.0, "speaker": "S2", "text": "Welcome back"},
            ]
        },
        "production_plan": {"units": []},
    }


def test_seed_and_load_or_create_dubbing_script_preserves_user_edits(tmp_path: Path):
    from rea_video.bridge_audio import load_or_create_dubbing_script, seed_dubbing_script

    out_dir = tmp_path / "pkg"
    out_dir.mkdir()
    b = _make_breakdown()
    cfg = BridgeConfig(out_dir=out_dir, tts_voice="Binh", run=True)

    seeded = seed_dubbing_script(b, out_dir, cfg)
    script_path = out_dir / "remake" / "dubbing_script.json"
    assert script_path.is_file()
    assert len(seeded["lines"]) == 2
    assert seeded["lines"][0]["speaker"] == "S1"
    assert seeded["lines"][0]["speaker_label"] == "Host"
    assert seeded["lines"][0]["text"] == "Hello everyone"
    assert seeded["speakers"]["S1"]["label"] == "Host"
    assert seeded["speakers"]["S1"]["voice"] == "Binh"
    assert seeded["speakers"]["S2"]["label"]  # resolved via entities.speaker_label

    # User edits the script on disk and binds a custom voice/ref per speaker
    seeded["lines"][0]["text"] = "Xin chao moi nguoi"
    seeded["speakers"]["S2"]["voice"] = "my_custom_voice"
    write_json(script_path, seeded)

    loaded = load_or_create_dubbing_script(b, out_dir, cfg)
    assert loaded["lines"][0]["text"] == "Xin chao moi nguoi"
    assert loaded["speakers"]["S2"]["voice"] == "my_custom_voice"


def test_compute_atempo_clamps_within_bounds():
    from rea_video.bridge_audio import compute_atempo

    assert compute_atempo(2.0, 2.0) == pytest.approx(1.0)
    assert compute_atempo(4.0, 2.0) == pytest.approx(1.25)
    assert compute_atempo(1.0, 4.0) == pytest.approx(0.85)
    assert compute_atempo(2.2, 2.0) == pytest.approx(1.1)
    assert compute_atempo(0.0, 2.0) == pytest.approx(1.0)


def test_enroll_voice_and_synthesize_line_http_contracts(tmp_path: Path, monkeypatch):
    from rea_video.bridge_audio import enroll_voice, synthesize_line

    monkeypatch.setenv("REA_TTS_BASE_URL", "http://127.0.0.1:9123")
    monkeypatch.setenv("REA_TTS_API_KEY", "tts-secret")

    ref_wav = tmp_path / "sample.wav"
    ref_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    seen_requests = []

    class FakeResp:
        def __init__(self, payload: bytes):
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        seen_requests.append(req)
        if req.full_url.endswith("/v1/voices"):
            return FakeResp(json.dumps({"id": "v1", "name": "ok"}).encode("utf-8"))
        if req.full_url.endswith("/v1/audio/speech"):
            return FakeResp(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        enrolled = enroll_voice(ref_wav, speaker="Host / S1")
        out_wav = tmp_path / "line_001.wav"
        synthesize_line("Xin chao", enrolled, out_wav)

    assert len(seen_requests) == 2
    enroll_req, synth_req = seen_requests

    assert enroll_req.full_url == "http://127.0.0.1:9123/v1/voices"
    assert enroll_req.headers["Authorization"] == "Bearer tts-secret"
    assert "multipart/form-data; boundary=" in enroll_req.headers["Content-type"]
    assert enrolled.startswith("rea_clone_Host_S1_")
    assert len(enrolled) <= 64
    assert enrolled.encode("utf-8") in enroll_req.data
    assert b'name="denoise"' in enroll_req.data

    assert synth_req.full_url == "http://127.0.0.1:9123/v1/audio/speech"
    assert synth_req.headers["Authorization"] == "Bearer tts-secret"
    body = json.loads(synth_req.data.decode("utf-8"))
    assert body["input"] == "Xin chao"
    assert body["voice"] == enrolled
    assert body["response_format"] == "wav"
    assert out_wav.read_bytes().startswith(b"RIFF")


def test_run_audio_stage_clones_synthesizes_aligns_and_mixes(tmp_path: Path):
    from rea_video.bridge_audio import run_audio_stage

    out_dir = tmp_path / "pkg"
    stems = out_dir / "stems"
    stems.mkdir(parents=True)
    (stems / "voice_S1.wav").write_bytes(b"RIFF")
    (stems / "vocals.wav").write_bytes(b"RIFF")
    (stems / "music.wav").write_bytes(b"RIFF")
    (stems / "effects.wav").write_bytes(b"RIFF")

    b = _make_breakdown()
    cfg = BridgeConfig(out_dir=out_dir, stage="audio", run=True, clone_source_voices=True)

    ff_cmds: list[list[str]] = []
    enrolled_names: list[str] = []
    synth_calls: list[dict] = []

    class FakeResp:
        def __init__(self, payload: bytes):
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        if req.full_url.endswith("/v1/voices"):
            # Extract enrolled name from multipart payload
            enrolled_names.append(req.data.decode("utf-8", errors="ignore"))
            return FakeResp(b'{"ok": true}')
        if req.full_url.endswith("/v1/audio/speech"):
            synth_calls.append(json.loads(req.data.decode("utf-8")))
            return FakeResp(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    def fake_subproc(cmd, **_kw):
        ff_cmds.append(list(cmd))
        exe = Path(cmd[0]).name.lower()
        if exe == "ffprobe":
            class P:
                returncode = 0
                stdout = json.dumps({"streams": [{"duration": "3.0"}], "format": {"duration": "3.0"}})
                stderr = ""
            return P()
        if exe == "ffmpeg":
            out_path = Path(cmd[-1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
            class P:
                returncode = 0
                stdout = ""
                stderr = ""
            return P()
        raise AssertionError(f"Unexpected subprocess: {cmd}")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen), patch("subprocess.run", side_effect=fake_subproc):
        res = run_audio_stage(cfg, b, b["production_plan"])

    assert res["stage"] == "audio"
    assert res["synthesized"] == 2
    assert res["skipped"] == 0
    mix_path = out_dir / "remake" / "bridge" / "audio" / "dubbed_mix.wav"
    assert Path(res["mix"]) == mix_path
    assert mix_path.is_file()

    # Verify 5s slicing for S1 (voice_S1.wav) and S2 (fallback vocals.wav)
    slice_cmds = [c for c in ff_cmds if "-t" in c and "5" in c]
    assert len(slice_cmds) == 2
    assert any("voice_S1.wav" in " ".join(c) for c in slice_cmds)
    assert any("vocals.wav" in " ".join(c) for c in slice_cmds)

    # Verify atempo was applied (3.0s actual vs 2.2s/2.5s target)
    atempo_cmds = [c for c in ff_cmds if any("atempo=" in str(arg) for arg in c)]
    assert len(atempo_cmds) == 2

    # Verify amix with music.wav and effects.wav
    mix_cmds = [c for c in ff_cmds if str(mix_path) in c]
    assert len(mix_cmds) == 1
    mix_str = " ".join(mix_cmds[0])
    assert "adelay=" in mix_str
    assert "amix=" in mix_str
    assert "music.wav" in mix_str
    assert "effects.wav" in mix_str

    # Re-run without --force: skips existing aligned line WAVs (no new TTS HTTP calls)
    synth_calls.clear()
    with patch("urllib.request.urlopen", side_effect=fake_urlopen), patch("subprocess.run", side_effect=fake_subproc):
        res2 = run_audio_stage(cfg, b, b["production_plan"])
    assert res2["synthesized"] == 0
    assert res2["skipped"] == 2
    assert len(synth_calls) == 0


def test_run_assemble_stage_auto_selects_dubbed_mix_unless_keep_original_audio(tmp_path: Path):
    from rea_video.bridge_audio import run_assemble_stage

    out_dir = tmp_path / "pkg"
    rdir = out_dir / "remake"
    audio_dir = rdir / "bridge" / "audio"
    audio_dir.mkdir(parents=True)
    b = _make_breakdown()
    write_json(out_dir / "breakdown.json", b)
    mine_results = {"1": {"video": str(out_dir / "shot1.mp4")}}
    write_json(rdir / "mine.json", {"results": mine_results})
    dubbed_mix = audio_dir / "dubbed_mix.wav"
    dubbed_mix.write_bytes(b"RIFF")

    calls: list[dict] = []

    def fake_assemble(breakdown_path, results_path, out_dir=None, overlays=False, code_video=None, extra=None, retime="smooth", audio=None):
        calls.append({
            "breakdown_path": breakdown_path,
            "results_path": results_path,
            "out_dir": out_dir,
            "overlays": overlays,
            "code_video": code_video,
            "extra": extra,
            "retime": retime,
            "audio": audio,
        })
        return json.dumps({"remake": str(rdir / "remake.mp4"), "compare": str(rdir / "compare.mp4"), "legend": []})

    cfg = BridgeConfig(out_dir=out_dir, stage="assemble", run=True, no_overlays=True, retime="even")
    with patch("rea_video.remake.assemble", side_effect=fake_assemble):
        res = run_assemble_stage(cfg, b, b["production_plan"])

    assert len(calls) == 1
    assert calls[0]["extra"] == mine_results
    assert calls[0]["overlays"] is False
    assert calls[0]["retime"] == "even"
    assert calls[0]["audio"] == dubbed_mix
    assert res["audio"] == str(dubbed_mix)

    # With keep_original_audio=True, audio=None is passed
    calls.clear()
    cfg_keep = BridgeConfig(out_dir=out_dir, stage="assemble", run=True, keep_original_audio=True)
    with patch("rea_video.remake.assemble", side_effect=fake_assemble):
        res_keep = run_assemble_stage(cfg_keep, b, b["production_plan"])

    assert len(calls) == 1
    assert calls[0]["audio"] is None
    assert calls[0]["overlays"] is True
    assert res_keep["audio"] is None



