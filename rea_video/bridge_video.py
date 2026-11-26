"""Video Bridge Stage: renders Generation Units and avatar shots via `flow video` and `vids`, writing `remake/mine.json`."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .bridge import (
    BridgeConfig,
    BridgeError,
    _resolve_cli,
    _run_cli_json,
    _video_backend_for_unit,
    normalize_aspect,
    parse_only,
)
from .generation import (
    load_intent,
    load_prompts,
    redrawn_frame,
    resolve_redrawn_frame,
    unit_frame_aliases,
    unit_images,
    wants_last_frame,
)
from .integrations import capsule_cinema
from .util import read_json, write_json

_FLOW_MAX_REFS = 10
_VIDS_EDIT_MAX_REFS = 7
_VIDS_CREATE_MAX_INGREDIENTS = 3


def _resolve_local(out_dir: Path, raw: str | Path | None) -> str | None:
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = out_dir / p
    return str(p.resolve()) if p.is_file() else str(p)


def _dedupe_paths(paths: list[str | Path | None], *, limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for p in paths:
        if not p:
            continue
        key = str(Path(p).resolve()) if Path(p).exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(str(p))
            if limit is not None and len(out) >= limit:
                break
    return out


def _clamp_duration(raw_s: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(float(raw_s)))))


def _mine_key_and_entry(
    task: dict[str, Any],
    out_path: Path,
    trim_to: float,
    bookend: bool,
) -> tuple[str, dict[str, Any]]:
    shots = task["shots"]
    task_id = task["id"]
    parts = int(task["parts"])
    base: dict[str, Any] = {
        "id": task_id,
        "video": str(out_path.resolve()),
        "shots": shots,
        "trim_to": round(float(trim_to), 3),
        "strategy": task["strategy"],
        "backend": task["backend"],
        "bookend": bookend,
    }
    if len(shots) == 1 and parts == 1:
        return str(shots[0]), base
    return task_id, {
        **base,
        "unit": task["unit_id"],
        "part": int(task["part"]),
        "parts": parts,
    }


def _save_mine(mine_path: Path, mine_data: dict[str, Any], results: dict[str, Any]) -> None:
    write_json(mine_path, {**mine_data, "results": results})


def collect_video_tasks(
    b: dict,
    out_dir: Path,
    cfg: BridgeConfig,
    plan: dict | None = None,
) -> list[dict[str, Any]]:
    """Collect unit/part and standalone avatar video tasks, filtered by `--only`."""
    out_dir = Path(out_dir)
    plan = plan if plan is not None else (b.get("production_plan") or {})
    only_shots = parse_only(cfg.only)
    shots_by_idx = {int(r["index"]): r for r in b.get("shots") or []}
    video_dir = out_dir / "remake" / "bridge" / "video"

    tasks: list[dict[str, Any]] = []
    covered_unit_shots: set[int] = set()

    for unit in plan.get("units") or []:
        u_shots = [int(s) for s in (unit.get("shots") or [])]
        covered_unit_shots.update(u_shots)
        if only_shots is not None and not any(s in only_shots for s in u_shots):
            continue
        refs = unit.get("refs") or []
        parts = sorted({int(r.get("part", 1)) for r in refs if str(r.get("tag", "")).startswith("@视频")})
        if not parts:
            parts = list(range(1, max(1, int(unit.get("calls") or 1)) + 1))
        backend = _video_backend_for_unit(unit, shots_by_idx, cfg)
        is_avatar = any((shots_by_idx.get(s) or {}).get("base", {}).get("label") == "avatar" for s in u_shots)
        for part in parts:
            task_id = unit["id"] if len(parts) == 1 else f"{unit['id']}-{part}"
            tasks.append({
                "id": task_id,
                "unit": unit,
                "unit_id": unit["id"],
                "part": part,
                "parts": len(parts),
                "shots": u_shots,
                "strategy": unit.get("strategy") or "text",
                "backend": backend,
                "is_avatar": is_avatar,
                "output": video_dir / f"{task_id}.mp4",
            })

    for raw_s in plan.get("avatar") or []:
        shot_idx = int(raw_s)
        if shot_idx in covered_unit_shots:
            continue
        if only_shots is not None and shot_idx not in only_shots:
            continue
        task_id = f"avatar_shot{shot_idx:03d}"
        tasks.append({
            "id": task_id,
            "unit": None,
            "unit_id": None,
            "part": 1,
            "parts": 1,
            "shots": [shot_idx],
            "strategy": "avatar",
            "backend": "vids",
            "is_avatar": True,
            "shot_record": shots_by_idx.get(shot_idx) or {},
            "output": video_dir / f"{task_id}.mp4",
        })

    return tasks


def _build_flow_cmd(
    cli_path: str,
    cfg: BridgeConfig,
    aspect: str,
    prompt: str,
    strategy: str,
    dur_s: float,
    out_path: Path,
    *,
    ref_clip: str | None,
    refs: list[str],
    first_frame: str | None,
    last_frame: str | None,
    task_id: str,
) -> tuple[list[str], bool]:
    bookend = False
    has_last = strategy == "frames" and bool(last_frame)
    max_dur = 10 if has_last else 15
    duration = _clamp_duration(dur_s, 3, max_dur)
    cmd = [
        cli_path,
        "video",
        "--prompt",
        prompt,
        "--aspect",
        aspect,
        "--resolution",
        cfg.resolution,
        "--duration",
        str(duration),
        "--output",
        str(out_path),
        "--strategy",
        cfg.pool_strategy,
        "--json",
    ]
    if cfg.profile:
        cmd.extend(["--profile", cfg.profile])

    if strategy in ("edit", "remix"):
        if not ref_clip:
            raise BridgeError(f"{task_id}: {strategy} strategy requires a reference video clip")
        cmd.extend(["--ref-video", ref_clip])
        for rp in refs[:_FLOW_MAX_REFS]:
            cmd.extend(["--ref", rp])
    elif strategy == "keyframes":
        for rp in refs[:_FLOW_MAX_REFS]:
            cmd.extend(["--ref", rp])
    elif strategy == "frames":
        if not first_frame:
            raise BridgeError(f"{task_id}: frames strategy requires a first frame")
        cmd.extend(["--first-frame", first_frame])
        if last_frame:
            cmd.extend(["--last-frame", last_frame])
            bookend = True
    elif strategy == "text":
        pass
    else:
        raise BridgeError(f"{task_id}: unsupported video strategy {strategy!r}")

    return cmd, bookend


def _vids_resolution(res: str | None) -> str:
    return res if res in ("720p", "1080p") else "720p"


def _build_vids_cmd(
    cli_path: str,
    cfg: BridgeConfig,
    aspect: str,
    prompt: str,
    strategy: str,
    dur_s: float,
    out_path: Path,
    *,
    is_avatar: bool,
    ref_clip: str | None,
    refs: list[str],
    first_frame: str | None,
    task_id: str,
) -> tuple[list[str], bool]:
    duration = _clamp_duration(dur_s, 3, 15)
    resolution = _vids_resolution(cfg.resolution)
    tail = [
        "--aspect",
        aspect,
        "--resolution",
        resolution,
        "--duration",
        str(duration),
        "--output",
        str(out_path),
        "--strategy",
        cfg.pool_strategy,
        "--json",
    ]
    if cfg.profile:
        tail.extend(["--profile", cfg.profile])

    if is_avatar or strategy == "avatar":
        if cfg.avatar:
            cmd = [cli_path, "create", "--prompt", prompt, "--avatar", cfg.avatar]
            for rp in refs[:_VIDS_CREATE_MAX_INGREDIENTS]:
                cmd.extend(["--ingredients", rp])
            cmd.extend(tail)
            return cmd, False
        if first_frame:
            return [cli_path, "animate", "--input", first_frame, "--prompt", prompt, *tail], False
        cmd = [cli_path, "create", "--prompt", prompt]
        for rp in refs[:_VIDS_CREATE_MAX_INGREDIENTS]:
            cmd.extend(["--ingredients", rp])
        cmd.extend(tail)
        return cmd, False

    if strategy in ("edit", "remix"):
        if not ref_clip:
            raise BridgeError(f"{task_id}: vids {strategy} requires a reference video clip")
        cmd = [cli_path, "edit", "--input", ref_clip, "--prompt", prompt]
        for rp in refs[:_VIDS_EDIT_MAX_REFS]:
            cmd.extend(["--ref", rp])
        if cfg.avatar:
            cmd.extend(["--avatar", cfg.avatar])
        cmd.extend(tail)
        return cmd, False

    if strategy == "frames":
        if not first_frame:
            raise BridgeError(f"{task_id}: vids frames requires a first frame")
        return [cli_path, "animate", "--input", first_frame, "--prompt", prompt, *tail], False

    if strategy in ("keyframes", "text"):
        cmd = [cli_path, "create", "--prompt", prompt]
        if strategy == "keyframes":
            for rp in refs[:_VIDS_CREATE_MAX_INGREDIENTS]:
                cmd.extend(["--ingredients", rp])
        if cfg.avatar:
            cmd.extend(["--avatar", cfg.avatar])
        cmd.extend(tail)
        return cmd, False

    raise BridgeError(f"{task_id}: unsupported vids strategy {strategy!r}")


def run_video_stage(cfg: BridgeConfig, b: dict, plan: dict | None = None) -> dict[str, Any]:
    """Execute the Video Bridge Stage for planned Generation Units and standalone avatar shots."""
    out_dir = Path(cfg.out_dir)
    plan = plan if plan is not None else (b.get("production_plan") or {})
    tasks = collect_video_tasks(b, out_dir, cfg, plan=plan)
    prompts = load_prompts(out_dir)

    src_aspect = (b.get("source") or {}).get("aspect_ratio")
    flow_aspect = normalize_aspect(src_aspect, ("16:9", "9:16"), default="16:9")
    vids_aspect = normalize_aspect(src_aspect, ("16:9", "9:16", "1:1"), default="1:1")

    mine_path = out_dir / "remake" / "mine.json"
    mine_data: dict[str, Any] = read_json(mine_path) if mine_path.is_file() else {}
    results: dict[str, Any] = dict(mine_data.get("results") or {})

    ref_cache: dict[str, dict[tuple[int, str], str]] = {}
    cli_paths: dict[str, str] = {}
    generated: list[str] = []
    skipped: list[str] = []

    for task in tasks:
        task_id = task["id"]
        out_path: Path = task["output"]
        unit = task["unit"]
        strategy = task["strategy"]
        part = int(task["part"])
        shots = task["shots"]
        dur_s = float((unit.get("call_s") or unit.get("generate_s") or unit.get("duration") or 5.0) if unit else (task.get("shot_record") or {}).get("duration") or 5.0)
        trim_to = float((unit.get("generate_s") or unit.get("duration") or dur_s) if unit else dur_s)

        if out_path.is_file() and not cfg.force:
            bookend = False
            if strategy == "frames" and unit is not None and wants_last_frame(unit):
                intent = (b.get("production_plan") or {}).get("intent") or b.get("intent") or load_intent(out_dir)
                first_rd = resolve_redrawn_frame(out_dir, unit_frame_aliases(unit, part, "first"))
                last_rd = resolve_redrawn_frame(out_dir, unit_frame_aliases(unit, part, "last"))
                subject_redesigned = (
                    first_rd is not None
                    or bool(intent.get("swaps"))
                    or str(intent.get("image_ref") or "auto") not in ("", "auto")
                )
                bookend = not (subject_redesigned and last_rd is None)
            m_key, m_entry = _mine_key_and_entry(task, out_path, trim_to, bookend)
            if m_key not in results:
                results[m_key] = m_entry
                _save_mine(mine_path, mine_data, results)
            skipped.append(str(out_path))
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        files: dict[tuple[int, str], str] = {}
        unit_img_paths: list[str] = []
        if unit is not None:
            uid = unit["id"]
            if uid not in ref_cache:
                ref_cache[uid] = capsule_cinema.ref_clips(b, out_dir, unit)
            files = ref_cache[uid]
            for r in unit_images(unit):
                p = _resolve_local(out_dir, r.get("path")) if r.get("user") else files.get((1, r["tag"]))
                if p:
                    unit_img_paths.append(p)
            prompt = prompts.get(uid) or unit.get("draft") or ""
        else:
            sh = task.get("shot_record") or {}
            shot_idx = shots[0]
            prompt = (
                prompts.get(task_id)
                or prompts.get(f"shot_{shot_idx:03d}")
                or (sh.get("vlm") or {}).get("description")
                or sh.get("description")
                or sh.get("verdict")
                or "Talking head avatar"
            )

        ref_clip = None
        if unit is not None and strategy in ("edit", "remix"):
            ref_vids = [
                files[(part, r["tag"])]
                for r in (unit.get("refs") or [])
                if str(r.get("tag", "")).startswith("@视频")
                and int(r.get("part", 1)) == part
                and (part, r["tag"]) in files
            ]
            ref_clip = ref_vids[0] if ref_vids else None

        extra_redrawn: list[str] = []
        if strategy in ("edit", "remix"):
            for s_idx in shots:
                rd = redrawn_frame(out_dir, f"shot{int(s_idx):03d}")
                if rd is not None:
                    extra_redrawn.append(str(rd.resolve()))

        refs = _dedupe_paths([*unit_img_paths, *extra_redrawn])
        first_frame = files.get((part, "first")) if unit is not None else None
        last_frame = files.get((part, "last")) if unit is not None else None
        if unit is None and shots:
            shot_idx = shots[0]
            rd = redrawn_frame(out_dir, f"shot{shot_idx:03d}")
            if rd is not None:
                first_frame = str(rd.resolve())
            else:
                raw_ff = ((task.get("shot_record") or {}).get("assets") or {}).get("first_frame")
                resolved_ff = _resolve_local(out_dir, raw_ff)
                if resolved_ff and Path(resolved_ff).is_file():
                    first_frame = resolved_ff

        backend = task["backend"]
        if backend == "flow":
            cli_path = cli_paths.get("flow") or cli_paths.setdefault("flow", _resolve_cli("flow"))
            cmd, bookend = _build_flow_cmd(
                cli_path,
                cfg,
                flow_aspect,
                prompt,
                strategy,
                dur_s,
                out_path,
                ref_clip=ref_clip,
                refs=refs,
                first_frame=first_frame,
                last_frame=last_frame,
                task_id=task_id,
            )
        elif backend == "vids":
            cli_path = cli_paths.get("vids") or cli_paths.setdefault("vids", _resolve_cli("vids"))
            cmd, bookend = _build_vids_cmd(
                cli_path,
                cfg,
                vids_aspect,
                prompt,
                strategy,
                dur_s,
                out_path,
                is_avatar=bool(task["is_avatar"]),
                ref_clip=ref_clip,
                refs=refs,
                first_frame=first_frame,
                task_id=task_id,
            )
        else:
            raise BridgeError(f"{task_id}: unsupported video backend {backend!r}")

        _run_cli_json(cmd, shot_id=task_id)
        m_key, m_entry = _mine_key_and_entry(task, out_path, trim_to, bookend)
        results[m_key] = m_entry
        _save_mine(mine_path, mine_data, results)
        generated.append(str(out_path))

    return {
        "stage": "video",
        "generated": generated,
        "skipped": skipped,
        "mine": str(mine_path),
    }
