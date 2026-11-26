"""Tests for rea-video bridge CLI (--dry-run / stage framework) and Image Stage."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rea_video import cli
from rea_video.util import write_json


def _make_package(tmp_path: Path) -> Path:
    out = tmp_path / "pkg"
    out.mkdir(parents=True, exist_ok=True)
    src_frame = out / "shot001_first.jpg"
    src_frame.write_bytes(b"fake-jpg")
    breakdown = {
        "source": {"path": "source.mp4", "aspect_ratio": "9:16", "duration": 10.0},
        "shots": [
            {
                "index": 1,
                "start": 0.0,
                "end": 4.0,
                "duration": 4.0,
                "base": {"label": "ai_video"},
                "assets": {"first_frame": str(src_frame)},
                "description": "A cybernetic cat walks through neon rain",
            },
            {
                "index": 2,
                "start": 4.0,
                "end": 8.0,
                "duration": 4.0,
                "base": {"label": "live_action"},
                "assets": {"first_frame": str(src_frame)},
                "description": "Close-up of neon puddle",
            },
        ],
        "production_plan": {
            "strategy": "frames",
            "units": [
                {"id": "G1", "shots": [1], "strategy": "frames", "duration": 4.0, "calls": 1},
            ],
            "image_tasks": [
                {
                    "shot": 1,
                    "role": "first_frame",
                    "method": "t2i",
                    "prompt": "cybernetic cat in neon rain",
                    "source_frame": str(src_frame),
                    "target": "remake/redrawn/shot001.png",
                },
            ],
        },
        "speech": {
            "lines": [{"start": 0.2, "end": 2.5, "speaker": "S1", "text": "Hello world"}],
        },
    }
    write_json(out / "breakdown.json", breakdown)
    return out


def test_bridge_cli_defaults_to_dry_run_without_external_calls(tmp_path, capsys):
    pkg = _make_package(tmp_path)
    with patch("subprocess.run") as mock_sub, patch("urllib.request.urlopen") as mock_url:
        rc = cli.main(["bridge", str(pkg)])
    assert rc == 0
    mock_sub.assert_not_called()
    mock_url.assert_not_called()
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert out["stage"] == "all"
    assert len(out["images"]["tasks"]) == 1
    assert len(out["video"]["units"]) == 1
    assert out["audio"]["lines"] == 1
    assert out["assemble"]["retime"] == "smooth"


def test_bridge_cli_help_zh_and_en(capsys):
    with pytest.raises(SystemExit) as exc_zh:
        cli.main(["bridge", "--help"])
    assert exc_zh.value.code == 0
    help_zh = capsys.readouterr().out
    assert "--dry-run" in help_zh
    assert "--image-backend" in help_zh
    assert "本地复刻桥接" in help_zh or "预览" in help_zh

    with pytest.raises(SystemExit) as exc_en:
        cli.main(["--lang", "en", "bridge", "--help"])
    assert exc_en.value.code == 0
    help_en = capsys.readouterr().out
    assert "local remake bridge" in help_en or "preview planned work" in help_en
    assert "--clone-source-voices" in help_en


def test_bridge_gateway_t2i_and_env_precedence(tmp_path, monkeypatch, capsys):
    pkg = _make_package(tmp_path)
    monkeypatch.setenv("REA_GEMINI_API_BASE", "https://gem.example.com")
    monkeypatch.setenv("REA_GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("REA_IMAGE_API_BASE", "https://img.example.com/")
    monkeypatch.setenv("REA_IMAGE_API_KEY", "img-key")
    monkeypatch.delenv("REA_IMAGE_MODEL", raising=False)

    calls = []

    class FakeResp:
        def __init__(self, payload: bytes):
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import base64

    png_bytes = b"\x89PNG\r\n\x1a\nfake-png"
    b64 = base64.b64encode(png_bytes).decode("ascii")

    def fake_urlopen(req, timeout=120.0):
        calls.append((req.full_url, dict(req.header_items()), json.loads(req.data.decode("utf-8"))))
        return FakeResp(json.dumps({"data": [{"b64_json": b64}]}).encode("utf-8"))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        rc = cli.main(["bridge", str(pkg), "--stage", "images", "--run"])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is False
    assert len(calls) == 1
    url, headers, body = calls[0]
    assert url == "https://img.example.com/v1/images/generations"
    assert headers["Authorization"] == "Bearer img-key"
    assert body["model"] == "gpt-image-2.6-sunburst"
    assert body["size"] == "1024x1536"
    assert body["n"] == 1
    saved = pkg / "remake" / "redrawn" / "shot001.png"
    assert saved.is_file()
    assert saved.read_bytes() == png_bytes


def test_bridge_gateway_i2i_multipart_url_fallback_and_gemini_env_fallback(tmp_path, monkeypatch, capsys):
    pkg = _make_package(tmp_path)
    bpath = pkg / "breakdown.json"
    b = json.loads(bpath.read_text(encoding="utf-8"))
    b["source"]["aspect_ratio"] = "16:9"
    swap_img = tmp_path / "swap.png"
    swap_img.write_bytes(b"swap-png-bytes")
    b["production_plan"]["own_refs"] = [{"path": str(swap_img)}]
    b["production_plan"]["image_tasks"][0]["method"] = "i2i"
    write_json(bpath, b)

    monkeypatch.delenv("REA_IMAGE_API_BASE", raising=False)
    monkeypatch.delenv("REA_IMAGE_BASE_URL", raising=False)
    monkeypatch.delenv("REA_IMAGE_API_KEY", raising=False)
    monkeypatch.setenv("REA_GEMINI_API_BASE", "https://gem.example.com/v1")
    monkeypatch.setenv("REA_GEMINI_API_KEY", "gem-key")

    calls = []

    class FakeResp:
        def __init__(self, payload: bytes):
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    downloaded_bytes = b"downloaded-png-from-url"

    def fake_urlopen(req, timeout=120.0):
        if isinstance(req, str):
            calls.append(("GET", req, {}, b""))
            return FakeResp(downloaded_bytes)
        calls.append(("POST", req.full_url, dict(req.header_items()), req.data))
        return FakeResp(json.dumps({"data": [{"url": "https://cdn.example.com/out.png"}]}).encode("utf-8"))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        rc = cli.main(["bridge", str(pkg), "--stage", "images", "--run", "--image-model", "custom-img-model"])

    assert rc == 0
    assert len(calls) == 2
    method, url, headers, body = calls[0]
    assert method == "POST"
    assert url == "https://gem.example.com/v1/images/edits"
    assert headers["Authorization"] == "Bearer gem-key"
    assert "multipart/form-data; boundary=" in headers["Content-type"]
    assert b'name="model"\r\n\r\ncustom-img-model' in body
    assert b'name="size"\r\n\r\n1536x1024' in body
    assert b'name="image[]"' in body
    assert b"swap-png-bytes" in body
    assert (pkg / "remake" / "redrawn" / "shot001.png").read_bytes() == downloaded_bytes


def test_bridge_gptimg_and_flow_cli_backends_and_only_force_skip(tmp_path, capsys):
    pkg = _make_package(tmp_path)
    bpath = pkg / "breakdown.json"
    b = json.loads(bpath.read_text(encoding="utf-8"))
    b["production_plan"]["image_tasks"].append({
        "shot": 2,
        "role": "first_frame",
        "method": "i2i",
        "prompt": "neon puddle",
        "source_frame": str(pkg / "shot001_first.jpg"),
        "target": "remake/redrawn/shot002.png",
    })
    write_json(bpath, b)

    # Pre-create shot001 so it is skipped unless --force is given
    existing_1 = pkg / "remake" / "redrawn" / "shot001.png"
    existing_1.parent.mkdir(parents=True, exist_ok=True)
    existing_1.write_bytes(b"already-here")

    seen_cmds = []

    class FakeProc:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(cmd, capture_output=True, text=True, encoding=None, errors=None, check=False):
        seen_cmds.append(cmd)
        out_idx = cmd.index("--output") + 1
        Path(cmd[out_idx]).write_bytes(b"cli-generated")
        return FakeProc(json.dumps({"ok": True, "path": cmd[out_idx]}))

    def fake_which(name):
        return f"C:\\bin\\{name}.cmd" if name in ("gptimg", "flow") else None

    with patch("shutil.which", side_effect=fake_which), patch("subprocess.run", side_effect=fake_run):
        rc1 = cli.main([
            "bridge", str(pkg), "--stage", "images", "--run",
            "--image-backend", "gptimg", "--pool-strategy", "round-robin", "--profile", "acc1",
        ])
    assert rc1 == 0
    res1 = json.loads(capsys.readouterr().out)["results"]["images"]
    assert len(res1["skipped"]) == 1
    assert len(res1["generated"]) == 1
    assert seen_cmds[0][0] == "C:\\bin\\gptimg.cmd"
    assert "--ref" in seen_cmds[0]
    assert "--profile" in seen_cmds[0] and "acc1" in seen_cmds[0]
    assert "--strategy" in seen_cmds[0] and "round-robin" in seen_cmds[0]

    # Now --only 1 --force with flow backend
    seen_cmds.clear()
    with patch("shutil.which", side_effect=fake_which), patch("subprocess.run", side_effect=fake_run):
        rc2 = cli.main([
            "bridge", str(pkg), "--stage", "images", "--run",
            "--image-backend", "flow", "--only", "shot_001", "--force",
        ])
    assert rc2 == 0
    res2 = json.loads(capsys.readouterr().out)["results"]["images"]
    assert len(res2["generated"]) == 1
    assert res2["skipped"] == []
    assert seen_cmds[0][0] == "C:\\bin\\flow.cmd"
    assert existing_1.read_bytes() == b"cli-generated"


def test_bridge_refusal_fails_clearly_with_shot_id_and_no_fallback(tmp_path, monkeypatch, capsys):
    pkg = _make_package(tmp_path)
    monkeypatch.setenv("REA_IMAGE_API_BASE", "https://img.example.com")
    monkeypatch.setenv("REA_IMAGE_API_KEY", "img-key")

    import io
    import urllib.error

    def fake_urlopen(req, timeout=120.0):
        raise urllib.error.HTTPError(
            req.full_url,
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error": {"message": "safety_policy_violation"}}'),
        )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen), patch("subprocess.run") as mock_sub:
        rc = cli.main(["bridge", str(pkg), "--stage", "images", "--run"])

    assert rc == 2
    mock_sub.assert_not_called()
    err = capsys.readouterr().err
    assert "shot_001" in err
    assert "safety_policy_violation" in err


def test_bridge_delegates_video_audio_assemble_stages_lazily(tmp_path):
    pkg = _make_package(tmp_path)
    from types import SimpleNamespace
    from rea_video.bridge import BridgeConfig, run_bridge

    called = []
    fake_video = SimpleNamespace(run_video_stage=lambda cfg, b, plan: called.append("video") or {"stage": "video"})
    fake_audio = SimpleNamespace(
        run_audio_stage=lambda cfg, b, plan: called.append("audio") or {"stage": "audio"},
        run_assemble_stage=lambda cfg, b, plan: called.append("assemble") or {"stage": "assemble"},
    )
    with patch.dict("sys.modules", {"rea_video.bridge_video": fake_video, "rea_video.bridge_audio": fake_audio}):
        run_bridge(BridgeConfig(out_dir=pkg, stage="video", run=True))
        run_bridge(BridgeConfig(out_dir=pkg, stage="audio", run=True))
        run_bridge(BridgeConfig(out_dir=pkg, stage="assemble", run=True))

    assert called == ["video", "audio", "assemble"]


def test_plan_cli_completes_after_cmd_bridge_separation(tmp_path, capsys):
    pkg = tmp_path / "plan_pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    breakdown = {
        "source": {
            "path": "source.mp4",
            "filename": "source.mp4",
            "aspect_ratio": "9:16",
            "duration": 4.0,
            "width": 720,
            "height": 1280,
            "fps": 30.0,
            "has_audio": False,
        },
        "analysis": {"vlm": {"used": False}},
        "shots": [],
    }
    write_json(pkg / "breakdown.json", breakdown)

    rc = cli.main(["plan", str(pkg), "--model", "minimax-h3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "minimax-h3" in out


def test_bridge_dry_run_preview_parity_video_and_audio(tmp_path, capsys):
    pkg = _make_package(tmp_path)
    bpath = pkg / "breakdown.json"
    b = json.loads(bpath.read_text(encoding="utf-8"))
    b["shots"].append(
        {"index": 3, "start": 8.0, "end": 12.0, "duration": 4.0, "base": {"label": "avatar"}}
    )
    b["production_plan"]["units"] = [
        {
            "id": "G1",
            "shots": [1],
            "strategy": "edit",
            "duration": 10.0,
            "calls": 2,
            "refs": [{"tag": "@视频1", "part": 1, "ranges": [[0.0, 4.0]]}, {"tag": "@视频1", "part": 2, "ranges": [[4.0, 8.0]]}],
        }
    ]
    b["production_plan"]["avatar"] = [3]
    write_json(bpath, b)

    # Pre-create G1-1.mp4, custom dubbing_script.json with 2 lines, line_001.wav, and dubbed_mix.wav
    vdir = pkg / "remake" / "bridge" / "video"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "G1-1.mp4").write_bytes(b"g1-1")

    adir = pkg / "remake" / "bridge" / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "line_001.wav").write_bytes(b"wav1")
    (adir / "dubbed_mix.wav").write_bytes(b"mix")
    write_json(
        pkg / "remake" / "dubbing_script.json",
        {
            "version": 1,
            "speakers": {"S1": {"label": "Host", "voice": "Binh", "ref": ""}},
            "lines": [
                {"index": 1, "start": 0.0, "end": 1.5, "speaker": "S1", "text": "Edited line 1"},
                {"index": 2, "start": 2.0, "end": 3.5, "speaker": "S1", "text": "Edited line 2"},
            ],
        },
    )

    with patch("subprocess.run") as mock_sub, patch("urllib.request.urlopen") as mock_url:
        rc = cli.main(["bridge", str(pkg)])
    assert rc == 0
    mock_sub.assert_not_called()
    mock_url.assert_not_called()

    out = json.loads(capsys.readouterr().out)
    v_units = out["video"]["units"]
    assert [u["id"] for u in v_units] == ["G1-1", "G1-2", "avatar_shot003"]
    assert v_units[0]["would_skip"] is True
    assert v_units[1]["would_skip"] is False
    assert v_units[2]["would_skip"] is False
    assert v_units[2]["backend"] == "vids"

    assert out["audio"]["lines"] == 2
    assert out["audio"]["would_skip_lines"] == 1
    assert out["audio"]["mix_would_skip"] is True
    assert out["audio"]["would_skip"] is False


def test_bridge_multi_ref_single_flag_and_unit_only_selector(tmp_path, capsys):
    pkg = _make_package(tmp_path)
    bpath = pkg / "breakdown.json"
    b = json.loads(bpath.read_text(encoding="utf-8"))
    swap = tmp_path / "swap.png"
    swap.write_bytes(b"swap")
    b["production_plan"]["own_refs"] = [{"path": str(swap)}]
    b["production_plan"]["units"] = [{"id": "G1", "shots": [1], "strategy": "text", "duration": 4.0, "calls": 1}]
    b["production_plan"]["image_tasks"] = [
        {
            "shot": 1,
            "role": "first_frame",
            "method": "i2i",
            "prompt": "two refs",
            "source_frame": str(pkg / "shot001_first.jpg"),
            "target": "remake/redrawn/shot001.png",
        },
        {
            "shot": 2,
            "role": "first_frame",
            "method": "t2i",
            "prompt": "other shot",
            "source_frame": str(pkg / "shot001_first.jpg"),
            "target": "remake/redrawn/shot002.png",
        },
    ]
    write_json(bpath, b)

    seen_cmds: list[list[str]] = []

    class FakeProc:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str):
            self.stdout = stdout

    def fake_run(cmd, **_kw):
        seen_cmds.append(cmd)
        out_p = Path(cmd[cmd.index("--output") + 1])
        out_p.write_bytes(b"img")
        return FakeProc(json.dumps({"ok": True, "path": str(out_p)}))

    with (
        patch("shutil.which", return_value="C:\\bin\\flow.cmd"),
        patch("subprocess.run", side_effect=fake_run),
    ):
        rc = cli.main(["bridge", str(pkg), "--stage", "images", "--image-backend", "flow", "--only", "G1", "--run"])

    assert rc == 0
    assert len(seen_cmds) == 1
    cmd = seen_cmds[0]
    assert cmd.count("--ref") == 1
    ref_idx = cmd.index("--ref")
    assert cmd[ref_idx + 1] == str(pkg / "shot001_first.jpg")
    assert cmd[ref_idx + 2] == str(swap)


def test_run_cli_json_flattens_multiline_prompts():
    from rea_video.bridge import _run_cli_json

    seen: list[list[str]] = []

    class FakeProc:
        returncode = 0
        stdout = '{"ok": true}'
        stderr = ""

    def fake_run(cmd, **_kw):
        seen.append(cmd)
        return FakeProc()

    with patch("subprocess.run", side_effect=fake_run):
        res = _run_cli_json(["flow.cmd", "video", "--prompt", "line1\nline2\r\nline3", "--json"])

    assert res == {"ok": True}
    assert seen[0] == ["flow.cmd", "video", "--prompt", "line1 line2 line3", "--json"]


def test_parse_only_unknown_or_empty_unit_never_runs_all():
    from rea_video.bridge import BridgeError, parse_only

    with pytest.raises(BridgeError, match="Unknown generation unit"):
        parse_only("G99", units=[{"id": "G1", "shots": [1]}])
    with pytest.raises(BridgeError, match="Unknown generation unit"):
        parse_only("G1", units=None)
    assert parse_only("G1", units=[{"id": "G1", "shots": []}]) == set()
