from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rea_video.cli import main as cli_main
from rea_video.cli_en import HELP as HELP_EN
from rea_video.generation import spec_of
from rea_video.plan import Planner
from rea_video.remake import assemble


def _make_media(tmp_path: Path, *, source_has_audio: bool) -> tuple[Path, Path, Path]:
    src = tmp_path / "source.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=1"]
    if source_has_audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)]
    subprocess.run(cmd, check=True)

    dub = tmp_path / "dub.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=1", str(dub)],
        check=True,
    )

    bpath = tmp_path / "breakdown.json"
    bpath.write_text(
        json.dumps(
            {
                "source": {
                    "path": str(src),
                    "width": 64,
                    "height": 64,
                    "fps": 10.0,
                    "has_audio": source_has_audio,
                },
                "shots": [
                    {
                        "index": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "base": {"label": "live_action"},
                        "replication": {},
                        "assets": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rpath = tmp_path / "results.json"
    rpath.write_text(json.dumps({"results": {}}), encoding="utf-8")
    return bpath, rpath, dub


def _has_audio_stream(video: Path) -> bool:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "json", str(video)],
        text=True,
    )
    return bool(json.loads(out).get("streams"))


def test_gemini_omni_flash_capabilities_and_prompting() -> None:
    planner = Planner()
    model = planner.models["gemini-omni-flash"]
    spec = spec_of(model)
    assert spec["images"] == 10
    assert spec["ref_total"] == 15.0

    refs_note = planner.prompting["gemini-omni-flash"]["references"]
    assert "10" in refs_note
    assert "15" in refs_note
    assert "最多 6 张" not in refs_note
    assert "3 秒以内" not in refs_note


def test_assemble_raises_on_missing_audio(tmp_path: Path) -> None:
    bpath, rpath, _ = _make_media(tmp_path, source_has_audio=False)
    with pytest.raises(FileNotFoundError):
        assemble(bpath, rpath, out_dir=tmp_path / "out", audio=tmp_path / "missing.wav")


def test_assemble_muxes_audio_override_into_remake_and_compare(tmp_path: Path) -> None:
    bpath, rpath, dub = _make_media(tmp_path, source_has_audio=False)
    out_dir = tmp_path / "out"
    log = json.loads(assemble(bpath, rpath, out_dir=out_dir, audio=dub))

    remake_mp4 = Path(log["remake"])
    compare_mp4 = Path(log["compare"])
    assert remake_mp4.name == "remake.mp4"
    assert remake_mp4.is_file()
    assert compare_mp4.is_file()
    assert _has_audio_stream(remake_mp4)
    assert _has_audio_stream(compare_mp4)


def test_assemble_preserves_default_audio_behavior_when_omitted(tmp_path: Path) -> None:
    s_dir = tmp_path / "s"
    s_dir.mkdir()
    b_silent, r_silent, _ = _make_media(s_dir, source_has_audio=False)
    log_silent = json.loads(assemble(b_silent, r_silent, out_dir=s_dir / "out"))
    assert Path(log_silent["remake"]).name == "remake_silent.mp4"
    assert not _has_audio_stream(Path(log_silent["remake"]))

    sub = tmp_path / "with_audio"
    sub.mkdir()
    b_aud, r_aud, _ = _make_media(sub, source_has_audio=True)
    log_aud = json.loads(assemble(b_aud, r_aud, out_dir=sub / "out"))
    assert Path(log_aud["remake"]).name == "remake.mp4"
    assert _has_audio_stream(Path(log_aud["remake"]))
    assert _has_audio_stream(Path(log_aud["compare"]))


def test_cli_assemble_audio_flag_and_help(tmp_path: Path) -> None:
    assert "assemble.audio" in HELP_EN
    bpath, rpath, dub = _make_media(tmp_path, source_has_audio=False)
    out_dir = tmp_path / "cli_out"
    rc = cli_main(["assemble", str(bpath), str(rpath), "-o", str(out_dir), "--audio", str(dub)])
    assert rc == 0
    assert _has_audio_stream(out_dir / "remake.mp4")
    assert _has_audio_stream(out_dir / "compare.mp4")


