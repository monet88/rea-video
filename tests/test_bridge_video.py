"""Tests for rea-video bridge Video Stage (flow video + vids + remake/mine.json)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rea_video.bridge import BridgeConfig, BridgeError
from rea_video.bridge_video import collect_video_tasks, run_video_stage
from rea_video.util import read_json, write_json


def _make_video_package(
    tmp_path: Path,
    *,
    units: list[dict] | None = None,
    avatar: list[int] | None = None,
    shots: list[dict] | None = None,
    prompts: dict[str, str] | None = None,
) -> Path:
    out = tmp_path / "pkg"
    out.mkdir(parents=True, exist_ok=True)
    src = out / "source.mp4"
    src.write_bytes(b"fake-mp4")
    f1 = out / "shot001_first.jpg"
    f1.write_bytes(b"fake-jpg-1")
    f2 = out / "shot002_first.jpg"
    f2.write_bytes(b"fake-jpg-2")

    default_shots = shots or [
        {"index": 1, "start": 0.0, "end": 4.0, "duration": 4.0, "base": {"label": "ai_video"}, "assets": {"first_frame": str(f1)}},
        {"index": 2, "start": 4.0, "end": 8.0, "duration": 4.0, "base": {"label": "avatar"}, "assets": {"first_frame": str(f2)}, "description": "Avatar host"},
    ]
    breakdown = {
        "source": {"path": str(src), "aspect_ratio": "9:16", "duration": 8.0, "fps": 30.0, "width": 720, "height": 1280},
        "shots": default_shots,
        "production_plan": {
            "strategy": "frames",
            "units": units if units is not None else [],
            "avatar": avatar if avatar is not None else [],
        },
    }
    write_json(out / "breakdown.json", breakdown)
    if prompts:
        write_json(out / "remake" / "prompts.json", prompts)
    return out


def test_collect_video_tasks_includes_units_standalone_avatar_and_only_filter(tmp_path):
    pkg = _make_video_package(
        tmp_path,
        units=[
            {"id": "G1", "shots": [1], "strategy": "frames", "duration": 4.0, "generate_s": 4.0, "calls": 1, "draft": "d1"},
            {"id": "G2", "shots": [3, 4], "strategy": "edit", "duration": 18.0, "generate_s": 18.0, "calls": 2, "draft": "d2",
             "refs": [{"tag": "@视频1", "part": 1, "ranges": [[8.0, 17.0]]}, {"tag": "@视频1", "part": 2, "ranges": [[17.0, 26.0]]}]},
        ],
        avatar=[1, 2],  # shot 1 is already in G1; only shot 2 is standalone avatar
        shots=[
            {"index": 1, "start": 0.0, "end": 4.0, "duration": 4.0, "base": {"label": "avatar"}},
            {"index": 2, "start": 4.0, "end": 8.0, "duration": 4.0, "base": {"label": "avatar"}},
            {"index": 3, "start": 8.0, "end": 17.0, "duration": 9.0, "base": {"label": "ai_video"}},
            {"index": 4, "start": 17.0, "end": 26.0, "duration": 9.0, "base": {"label": "ai_video"}},
        ],
    )
    b = read_json(pkg / "breakdown.json")

    tasks = collect_video_tasks(b, pkg, BridgeConfig(out_dir=pkg))
    assert [t["id"] for t in tasks] == ["G1", "G2-1", "G2-2", "avatar_shot002"]
    assert tasks[0]["backend"] == "vids"  # shot 1 has label=avatar
    assert tasks[1]["backend"] == "flow"
    assert tasks[1]["parts"] == 2
    assert tasks[3]["backend"] == "vids"
    assert tasks[3]["shots"] == [2]

    filtered = collect_video_tasks(b, pkg, BridgeConfig(out_dir=pkg, only="2,4"))
    assert [t["id"] for t in filtered] == ["G2-1", "G2-2", "avatar_shot002"]

    unit_filtered = collect_video_tasks(b, pkg, BridgeConfig(out_dir=pkg, only="G2"))
    assert [t["id"] for t in unit_filtered] == ["G2-1", "G2-2"]


def _opt_values(cmd: list[str], flag: str) -> list[str]:
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == flag and i + 1 < len(cmd)]


def _opt_multi_values(cmd: list[str], flag: str) -> list[str]:
    if flag not in cmd:
        return []
    vals: list[str] = []
    for tok in cmd[cmd.index(flag) + 1 :]:
        if tok.startswith("--"):
            break
        vals.append(tok)
    return vals


def test_run_video_stage_routes_all_flow_strategies_and_redrawn_frames(tmp_path):
    pkg = _make_video_package(
        tmp_path,
        units=[
            {
                "id": "G1",
                "shots": [1, 2],
                "strategy": "edit",
                "duration": 18.0,
                "generate_s": 18.0,
                "calls": 1,
                "draft": "draft G1",
                "refs": [{"tag": "@视频1", "part": 1, "ranges": [[0.0, 8.0]]}],
            },
            {
                "id": "G2",
                "shots": [3],
                "strategy": "keyframes",
                "duration": 5.0,
                "generate_s": 5.0,
                "calls": 1,
                "draft": "draft G2",
                "refs": [{"tag": "@图片1", "keyframe": 3, "t": 8.5}],
            },
            {
                "id": "G3",
                "shots": [4],
                "strategy": "frames",
                "duration": 12.0,
                "generate_s": 12.0,
                "calls": 1,
                "draft": "draft G3",
                "inputs": [{"kind": "last_frame", "need": "yes"}],
            },
            {
                "id": "G4",
                "shots": [5],
                "strategy": "text",
                "duration": 2.0,
                "generate_s": 2.0,
                "calls": 1,
                "draft": "draft G4",
            },
        ],
        shots=[
            {"index": i, "start": float(i - 1), "end": float(i), "duration": 1.0, "base": {"label": "ai_video"}}
            for i in range(1, 6)
        ],
        prompts={"G1": "final prompt G1", "G3": "final prompt G3"},
    )
    redrawn_dir = pkg / "remake" / "redrawn"
    redrawn_dir.mkdir(parents=True, exist_ok=True)
    rd_shot2 = redrawn_dir / "shot002.png"
    rd_shot2.write_bytes(b"redrawn-2")

    ref_clip = pkg / "remake" / "refs" / "G1_p1_v1.mp4"
    ref_clip.parent.mkdir(parents=True, exist_ok=True)
    ref_clip.write_bytes(b"clean-clip")
    kf3 = pkg / "remake" / "refs" / "G2_shot003.jpg"
    kf3.write_bytes(b"kf3")
    f_first = pkg / "remake" / "refs" / "G3_first.jpg"
    f_first.write_bytes(b"first")
    f_last = pkg / "remake" / "refs" / "G3_last.jpg"
    f_last.write_bytes(b"last")

    def fake_ref_clips(_b, _bdir, unit):
        if unit["id"] == "G1":
            return {(1, "@视频1"): str(ref_clip)}
        if unit["id"] == "G2":
            return {(1, "@图片1"): str(kf3)}
        if unit["id"] == "G3":
            return {(1, "first"): str(f_first), (1, "last"): str(f_last)}
        return {}

    cmds: list[list[str]] = []

    def fake_run_cli(cmd, *, shot_id=None):
        cmds.append(cmd)
        out_p = Path(_opt_values(cmd, "--output")[0])
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(b"rendered-mp4")
        return {"ok": True, "path": str(out_p)}

    b = read_json(pkg / "breakdown.json")
    cfg = BridgeConfig(
        out_dir=pkg,
        stage="video",
        run=True,
        resolution="360p",
        pool_strategy="round-robin",
        profile="acct-1",
    )
    with (
        patch("rea_video.integrations.capsule_cinema.ref_clips", side_effect=fake_ref_clips),
        patch("rea_video.bridge_video._resolve_cli", return_value="flow.cmd"),
        patch("rea_video.bridge_video._run_cli_json", side_effect=fake_run_cli),
    ):
        res = run_video_stage(cfg, b, b["production_plan"])

    assert len(res["generated"]) == 4
    assert len(cmds) == 4

    # G1: edit -> --ref-video + auto-included redrawn shot002 + final prompt + duration clamped 3..15
    c1 = cmds[0]
    assert c1[:2] == ["flow.cmd", "video"]
    assert _opt_values(c1, "--prompt") == ["final prompt G1"]
    assert _opt_values(c1, "--ref-video") == [str(ref_clip)]
    assert _opt_values(c1, "--ref") == [str(rd_shot2)]
    assert _opt_values(c1, "--resolution") == ["360p"]
    assert _opt_values(c1, "--duration") == ["15"]
    assert _opt_values(c1, "--strategy") == ["round-robin"]
    assert _opt_values(c1, "--profile") == ["acct-1"]
    assert "--json" in c1

    # G2: keyframes -> --ref + fallback to draft prompt
    c2 = cmds[1]
    assert _opt_values(c2, "--prompt") == ["draft G2"]
    assert _opt_values(c2, "--ref") == [str(kf3)]
    assert "--ref-video" not in c2

    # G3: frames with --last-frame -> duration clamped 3..10, no --ref-video
    c3 = cmds[2]
    assert _opt_values(c3, "--first-frame") == [str(f_first)]
    assert _opt_values(c3, "--last-frame") == [str(f_last)]
    assert _opt_values(c3, "--duration") == ["10"]
    assert "--ref-video" not in c3

    # G4: text -> prompt only, duration clamped >= 3
    c4 = cmds[3]
    assert _opt_values(c4, "--prompt") == ["draft G4"]
    assert _opt_values(c4, "--duration") == ["3"]
    assert "--ref-video" not in c4
    assert "--ref" not in c4
    assert "--first-frame" not in c4


def test_run_video_stage_routes_avatar_and_vids_backend(tmp_path):
    pkg = _make_video_package(
        tmp_path,
        units=[
            {
                "id": "G1",
                "shots": [1],
                "strategy": "edit",
                "duration": 5.0,
                "generate_s": 5.0,
                "calls": 1,
                "draft": "edit unit",
                "refs": [{"tag": "@视频1", "part": 1, "ranges": [[0.0, 5.0]]}],
            },
            {
                "id": "G2",
                "shots": [2],
                "strategy": "keyframes",
                "duration": 4.0,
                "generate_s": 4.0,
                "calls": 1,
                "draft": "keyframes unit",
                "refs": [{"tag": "@图片1", "keyframe": 2, "t": 5.5}],
            },
        ],
        avatar=[3, 4],
        shots=[
            {"index": 1, "start": 0.0, "end": 5.0, "duration": 5.0, "base": {"label": "ai_video"}},
            {"index": 2, "start": 5.0, "end": 9.0, "duration": 4.0, "base": {"label": "ai_video"}},
            {"index": 3, "start": 9.0, "end": 13.0, "duration": 4.0, "base": {"label": "avatar"}, "description": "Host speaks"},
            {"index": 4, "start": 13.0, "end": 17.0, "duration": 4.0, "base": {"label": "avatar"}, "description": "Host outro"},
        ],
    )
    redrawn_dir = pkg / "remake" / "redrawn"
    redrawn_dir.mkdir(parents=True, exist_ok=True)
    rd_avatar3 = redrawn_dir / "shot003.png"
    rd_avatar3.write_bytes(b"avatar3")

    ref_clip = pkg / "remake" / "refs" / "G1_p1_v1.mp4"
    ref_clip.parent.mkdir(parents=True, exist_ok=True)
    ref_clip.write_bytes(b"clip")
    kf2 = pkg / "remake" / "refs" / "G2_shot002.jpg"
    kf2.write_bytes(b"kf2")

    def fake_ref_clips(_b, _bdir, unit):
        if unit["id"] == "G1":
            return {(1, "@视频1"): str(ref_clip)}
        if unit["id"] == "G2":
            return {(1, "@图片1"): str(kf2)}
        return {}

    cmds: list[list[str]] = []

    def fake_run_cli(cmd, *, shot_id=None):
        cmds.append(cmd)
        out_p = Path(_opt_values(cmd, "--output")[0])
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(b"vids-mp4")
        return {"ok": True, "path": str(out_p)}

    b = read_json(pkg / "breakdown.json")

    # 1) video_backend="vids" with no --avatar: G1 -> vids edit, G2 -> vids create --ingredients,
    #    shot 3 -> vids animate --input <redrawn>, shot 4 -> vids create
    cfg_vids = BridgeConfig(out_dir=pkg, stage="video", run=True, video_backend="vids", resolution="360p")
    with (
        patch("rea_video.integrations.capsule_cinema.ref_clips", side_effect=fake_ref_clips),
        patch("rea_video.bridge_video._resolve_cli", return_value="vids.cmd"),
        patch("rea_video.bridge_video._run_cli_json", side_effect=fake_run_cli),
    ):
        res = run_video_stage(cfg_vids, b, b["production_plan"])

    assert len(res["generated"]) == 4
    assert cmds[0][:2] == ["vids.cmd", "edit"]
    assert _opt_values(cmds[0], "--input") == [str(ref_clip)]
    assert _opt_values(cmds[0], "--resolution") == ["720p"]
    assert cmds[1][:2] == ["vids.cmd", "create"]
    assert _opt_values(cmds[1], "--ingredients") == [str(kf2)]
    assert cmds[2][:2] == ["vids.cmd", "animate"]
    assert _opt_values(cmds[2], "--input") == [str(rd_avatar3)]
    assert cmds[3][:2] == ["vids.cmd", "create"]

    # 2) video_backend="flow" + --avatar "Alice" + --force: avatar shots still route to vids create --avatar Alice
    cmds.clear()
    cfg_avatar = BridgeConfig(out_dir=pkg, stage="video", run=True, video_backend="flow", avatar="Alice", only="3", force=True)
    with (
        patch("rea_video.bridge_video._resolve_cli", return_value="vids.cmd"),
        patch("rea_video.bridge_video._run_cli_json", side_effect=fake_run_cli),
    ):
        run_video_stage(cfg_avatar, b, b["production_plan"])

    assert len(cmds) == 1
    assert cmds[0][:2] == ["vids.cmd", "create"]
    assert _opt_values(cmds[0], "--avatar") == ["Alice"]


def test_run_video_stage_mine_json_skip_existing_force_and_incremental_error(tmp_path):
    pkg = _make_video_package(
        tmp_path,
        units=[
            {"id": "G1", "shots": [1], "strategy": "text", "duration": 4.0, "generate_s": 4.0, "calls": 1, "draft": "u1"},
            {
                "id": "G2",
                "shots": [2, 3],
                "strategy": "edit",
                "duration": 10.0,
                "generate_s": 10.0,
                "calls": 2,
                "draft": "u2",
                "refs": [{"tag": "@视频1", "part": 1, "ranges": [[4.0, 9.0]]}, {"tag": "@视频1", "part": 2, "ranges": [[9.0, 14.0]]}],
            },
        ],
        shots=[
            {"index": 1, "start": 0.0, "end": 4.0, "duration": 4.0, "base": {"label": "ai_video"}},
            {"index": 2, "start": 4.0, "end": 9.0, "duration": 5.0, "base": {"label": "ai_video"}},
            {"index": 3, "start": 9.0, "end": 14.0, "duration": 5.0, "base": {"label": "ai_video"}},
        ],
    )
    # Pre-create G1.mp4 on disk without an entry in mine.json
    existing_g1 = pkg / "remake" / "bridge" / "video" / "G1.mp4"
    existing_g1.parent.mkdir(parents=True, exist_ok=True)
    existing_g1.write_bytes(b"existing-g1")

    clip1 = pkg / "remake" / "refs" / "G2_p1.mp4"
    clip2 = pkg / "remake" / "refs" / "G2_p2.mp4"
    clip1.parent.mkdir(parents=True, exist_ok=True)
    clip1.write_bytes(b"c1")
    clip2.write_bytes(b"c2")

    calls: list[str] = []

    def fake_run_cli(cmd, *, shot_id=None):
        calls.append(shot_id or "")
        if shot_id == "G2-2":
            raise BridgeError(f"{shot_id}: policy refusal")
        out_p = Path(_opt_values(cmd, "--output")[0])
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(b"new-mp4")
        return {"ok": True, "path": str(out_p)}

    b = read_json(pkg / "breakdown.json")
    cfg = BridgeConfig(out_dir=pkg, stage="video", run=True)
    with (
        patch("rea_video.integrations.capsule_cinema.ref_clips", return_value={(1, "@视频1"): str(clip1), (2, "@视频1"): str(clip2)}),
        patch("rea_video.bridge_video._resolve_cli", return_value="flow.cmd"),
        patch("rea_video.bridge_video._run_cli_json", side_effect=fake_run_cli),
    ):
        with pytest.raises(BridgeError, match="G2-2"):
            run_video_stage(cfg, b, b["production_plan"])

    # G1 was skipped (not in calls), G2-1 succeeded and was persisted incrementally before G2-2 failed
    assert calls == ["G2-1", "G2-2"]
    mine = read_json(pkg / "remake" / "mine.json")["results"]
    assert "1" in mine
    assert "unit" not in mine["1"]
    assert mine["1"]["video"] == str(existing_g1.resolve())
    assert "G2-1" in mine
    assert mine["G2-1"]["unit"] == "G2"
    assert mine["G2-1"]["part"] == 1
    assert mine["G2-1"]["parts"] == 2
    assert mine["G2-1"]["shots"] == [2, 3]
    assert "G2-2" not in mine

    # With force=True, G1 is also regenerated
    calls.clear()
    with (
        patch("rea_video.integrations.capsule_cinema.ref_clips", return_value={(1, "@视频1"): str(clip1), (2, "@视频1"): str(clip2)}),
        patch("rea_video.bridge_video._resolve_cli", return_value="flow.cmd"),
        patch("rea_video.bridge_video._run_cli_json", side_effect=lambda cmd, shot_id=None: (
            calls.append(shot_id or ""),
            Path(_opt_values(cmd, "--output")[0]).write_bytes(b"regen"),
            {"ok": True, "path": _opt_values(cmd, "--output")[0]},
        )[-1]),
    ):
        res = run_video_stage(BridgeConfig(out_dir=pkg, stage="video", run=True, force=True), b, b["production_plan"])

    assert calls == ["G1", "G2-1", "G2-2"]
    assert len(res["generated"]) == 3


def test_run_video_stage_skipped_frames_bookend_respects_redrawn_subject(tmp_path):
    pkg = _make_video_package(
        tmp_path,
        units=[
            {
                "id": "G1",
                "shots": [1],
                "strategy": "frames",
                "duration": 4.0,
                "generate_s": 4.0,
                "calls": 1,
                "inputs": [{"kind": "last_frame", "need": "yes"}],
            },
        ],
        shots=[
            {"index": 1, "start": 0.0, "end": 4.0, "duration": 4.0, "base": {"label": "ai_video"}},
        ],
    )
    # Pre-create G1.mp4 and redrawn first frame, but NO redrawn last frame
    existing = pkg / "remake" / "bridge" / "video" / "G1.mp4"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"existing")
    rd_dir = pkg / "remake" / "redrawn"
    rd_dir.mkdir(parents=True, exist_ok=True)
    (rd_dir / "shot001.png").write_bytes(b"redrawn-first")

    b = read_json(pkg / "breakdown.json")
    cfg = BridgeConfig(out_dir=pkg, stage="video", run=True)
    res = run_video_stage(cfg, b, b["production_plan"])
    assert res["skipped"] == [str(existing)]
    mine = read_json(pkg / "remake" / "mine.json")["results"]
    assert mine["1"]["bookend"] is False

    # Now add redrawn last frame and clear mine.json -> bookend becomes True
    (pkg / "remake" / "mine.json").unlink()
    (rd_dir / "shot001_last.png").write_bytes(b"redrawn-last")
    run_video_stage(cfg, b, b["production_plan"])
    mine2 = read_json(pkg / "remake" / "mine.json")["results"]
    assert mine2["1"]["bookend"] is True


def test_flow_video_clamps_1080p_and_packs_multi_refs_in_single_flag(tmp_path):
    from rea_video.bridge_video import _build_flow_cmd, _build_vids_cmd

    cfg = BridgeConfig(out_dir=tmp_path, resolution="1080p")
    flow_cmd, _ = _build_flow_cmd(
        "flow.cmd",
        cfg,
        "9:16",
        "prompt",
        "keyframes",
        5.0,
        tmp_path / "g1.mp4",
        ref_clip=None,
        refs=["a.png", "b.png"],
        first_frame=None,
        last_frame=None,
        task_id="G1",
    )
    assert _opt_values(flow_cmd, "--resolution") == ["720p"]
    assert flow_cmd.count("--ref") == 1
    assert _opt_multi_values(flow_cmd, "--ref") == ["a.png", "b.png"]

    vids_cmd, _ = _build_vids_cmd(
        "vids.cmd",
        cfg,
        "9:16",
        "prompt",
        "keyframes",
        5.0,
        tmp_path / "g2.mp4",
        is_avatar=False,
        ref_clip=None,
        refs=["a.png", "b.png"],
        first_frame=None,
        task_id="G2",
    )
    assert vids_cmd.count("--ingredients") == 1
    assert _opt_multi_values(vids_cmd, "--ingredients") == ["a.png", "b.png"]

    vids_edit_cmd, _ = _build_vids_cmd(
        "vids.cmd",
        cfg,
        "9:16",
        "prompt",
        "edit",
        5.0,
        tmp_path / "g1.mp4",
        is_avatar=False,
        ref_clip="clip.mp4",
        refs=["a.png", "b.png", "c.png", "d.png", "e.png"],
        first_frame=None,
        task_id="G1",
    )
    assert _opt_multi_values(vids_edit_cmd, "--ref") == ["a.png", "b.png", "c.png"]



