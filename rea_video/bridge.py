"""Native local remake bridge: dry-run planning, stage orchestration, and Image Stage."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import entities
from .generation import load_intent, redrawn_frame
from .util import read_json

STAGES = ("images", "video", "audio", "assemble")
DEFAULT_IMAGE_MODEL = "gpt-image-2.6-sunburst"


class BridgeError(RuntimeError):
    """Raised when a bridge stage or external backend fails."""


@dataclass
class BridgeConfig:
    out_dir: Path
    stage: str = "all"
    run: bool = False
    only: str | None = None
    force: bool = False
    image_backend: str = "gateway"
    image_model: str | None = None
    video_backend: str = "flow"
    resolution: str = "720p"
    avatar: str | None = None
    pool_strategy: str = "fill-first"
    profile: str | None = None
    tts_voice: str | None = None
    tts_ref: str | None = None
    clone_source_voices: bool = False
    tts_url: str | None = None
    dubbing_script: str | None = None
    keep_original_audio: bool = False
    no_overlays: bool = False
    retime: str = "smooth"


def parse_only(only: str | None) -> set[int] | None:
    """Parse --only shot selector (e.g. '1,3-5,shot_007') into a set of 1-based shot indices, or None for all."""
    if not only or only.strip().lower() == "all":
        return None
    shots: set[int] = set()
    for part in only.split(","):
        tok = part.strip().lower()
        if not tok:
            continue
        m_range = re.fullmatch(r"(?:shot_?)?(\d+)\s*-\s*(?:shot_?)?(\d+)", tok)
        if m_range:
            lo, hi = int(m_range.group(1)), int(m_range.group(2))
            shots.update(range(min(lo, hi), max(lo, hi) + 1))
            continue
        m_one = re.fullmatch(r"(?:shot_?)?(\d+)", tok)
        if m_one:
            shots.add(int(m_one.group(1)))
            continue
        raise BridgeError(f"Invalid --only shot selector: {part!r}")
    return shots or None


def _resolve_cli(name: str) -> str:
    """Resolve a local CLI binary or Windows .cmd shim on PATH."""
    found = shutil.which(name) or shutil.which(f"{name}.cmd")
    if not found:
        raise BridgeError(f"CLI not found on PATH: {name}")
    return found


def _run_cli_json(cmd: list[str], *, shot_id: str | None = None) -> dict[str, Any]:
    """Run a local CLI command with --json and parse its JSON response, raising BridgeError on failure."""
    prefix = f"{shot_id}: " if shot_id else ""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    data: dict[str, Any] | None = None
    if stdout:
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                data = parsed
        except ValueError:
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            data = parsed
                            break
                    except ValueError:
                        continue
    if proc.returncode != 0 or (data is not None and data.get("ok") is False):
        detail = (data or {}).get("error") or stderr or stdout or f"exit code {proc.returncode}"
        raise BridgeError(f"{prefix}{cmd[0]} failed: {detail}")
    if data is None:
        raise BridgeError(f"{prefix}{cmd[0]} returned invalid JSON: {stdout[:200]}")
    return data


def _encode_multipart(fields: dict[str, str], files: list[tuple[str, Path]]) -> tuple[bytes, str]:
    """Encode form fields and file uploads as multipart/form-data bytes and Content-Type header."""
    boundary = f"----ReaVideoBridge{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for k, v in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        chunks.append(f"{v}\r\n".encode("utf-8"))
    for field_name, path in files:
        p = Path(path)
        mime, _ = mimetypes.guess_type(str(p))
        mime = mime or "application/octet-stream"
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{p.name}"\r\n'.encode("utf-8")
        )
        chunks.append(f"Content-Type: {mime}\r\n\r\n".encode("utf-8"))
        chunks.append(p.read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def aspect_to_gateway_size(aspect: str | None) -> str:
    """Map source aspect ratio string (e.g. '9:16', '16:9', '1:1') to gateway image dimensions."""
    raw = str(aspect or "").strip()
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[:xX/]\s*(\d+(?:\.\d+)?)\s*$", raw)
    if m:
        w, h = float(m.group(1)), float(m.group(2))
        if h > w:
            return "1024x1536"
        if w > h:
            return "1536x1024"
        return "1024x1024"
    return "1024x1024"


def normalize_aspect(
    aspect: str | None,
    allowed: tuple[str, ...] = ("16:9", "9:16", "1:1", "4:3", "3:4"),
    default: str = "9:16",
) -> str:
    """Normalize source aspect ratio string to one of `allowed` ratios."""
    raw = str(aspect or "").strip()
    if raw in allowed:
        return raw
    size = aspect_to_gateway_size(raw)
    if size == "1024x1536" and "9:16" in allowed:
        return "9:16"
    if size == "1536x1024" and "16:9" in allowed:
        return "16:9"
    if size == "1024x1024" and "1:1" in allowed:
        return "1:1"
    return default


def urlopen_bytes(req: urllib.request.Request | str, *, timeout: float, label: str) -> bytes:
    """Execute an HTTP request and return raw response bytes, raising BridgeError on failure."""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")[:400]
        except OSError:
            pass
        raise BridgeError(f"{label} (HTTP {exc.code}): {err_body or exc.reason}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BridgeError(f"{label}: {exc}") from None


def resolve_gateway_config(image_model: str | None = None) -> tuple[str, str, str]:
    """Resolve (base_v1_url, api_key, model) from REA_IMAGE_* env vars with REA_GEMINI_* fallback."""
    base = (
        os.getenv("REA_IMAGE_API_BASE")
        or os.getenv("REA_IMAGE_BASE_URL")
        or os.getenv("REA_GEMINI_API_BASE")
        or ""
    ).strip().rstrip("/")
    key = (os.getenv("REA_IMAGE_API_KEY") or os.getenv("REA_GEMINI_API_KEY") or "").strip()
    model = (image_model or os.getenv("REA_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL).strip()
    if not base or not key:
        raise BridgeError(
            "gateway image backend requires REA_IMAGE_API_BASE (or REA_GEMINI_API_BASE) "
            "and REA_IMAGE_API_KEY (or REA_GEMINI_API_KEY)"
        )
    base_v1 = base if base.endswith("/v1") else f"{base}/v1"
    return base_v1, key, model


def _collect_task_refs(out_dir: Path, task: dict, plan: dict) -> list[Path]:
    """Collect existing local reference image paths for an I2I or multi-ref image task."""
    shot_idx = int(task["shot"])
    refs: list[Path] = []
    seen: set[Path] = set()

    def add_if_file(p: str | Path | None) -> None:
        if not p:
            return
        cand = Path(p)
        if not cand.is_absolute():
            cand = out_dir / cand
        if cand.is_file():
            resolved = cand.resolve()
            if resolved not in seen:
                seen.add(resolved)
                refs.append(cand)

    if task.get("role") == "last_frame":
        add_if_file(redrawn_frame(out_dir, f"shot{shot_idx:03d}"))

    img_ref = str(task.get("image_ref") or "auto").strip().lower()
    if img_ref in ("source", "depth", "edge"):
        suffix = "clean" if img_ref == "source" else img_ref
        add_if_file(out_dir / "remake" / "refs" / f"shot{shot_idx:03d}_{suffix}.jpg")

    if img_ref != "none":
        add_if_file(task.get("source_frame"))

    for own in plan.get("own_refs") or []:
        add_if_file(own.get("path"))

    return refs


def _gateway_image_request(
    base_v1: str,
    api_key: str,
    model: str,
    prompt: str,
    size: str,
    ref_paths: list[Path],
    method: str,
    shot_id: str,
) -> bytes:
    """Call OpenAI-compatible /images/generations (T2I) or /images/edits (I2I) and return image bytes."""
    if method == "i2i" and ref_paths:
        url = f"{base_v1}/images/edits"
        field = "image" if len(ref_paths) == 1 else "image[]"
        body, content_type = _encode_multipart(
            {"model": model, "prompt": prompt, "size": size, "n": "1"},
            [(field, p) for p in ref_paths],
        )
    else:
        url = f"{base_v1}/images/generations"
        body = json.dumps({"model": model, "prompt": prompt, "size": size, "n": 1}).encode("utf-8")
        content_type = "application/json"

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
        },
    )
    raw = urlopen_bytes(req, timeout=120.0, label=f"{shot_id}: gateway request failed")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise BridgeError(f"{shot_id}: invalid gateway JSON response: {exc}") from None

    if isinstance(payload, dict) and payload.get("error"):
        raise BridgeError(f"{shot_id}: gateway error: {payload['error']}")

    items = (payload or {}).get("data") if isinstance(payload, dict) else None
    if not items or not isinstance(items, list):
        raise BridgeError(f"{shot_id}: gateway returned no image data")

    first = items[0] if isinstance(items[0], dict) else {}
    if first.get("b64_json"):
        return base64.b64decode(first["b64_json"])
    if first.get("url"):
        return urlopen_bytes(first["url"], timeout=120.0, label=f"{shot_id}: failed to download image URL")
    raise BridgeError(f"{shot_id}: gateway response missing b64_json or url")


def run_images_stage(cfg: BridgeConfig, b: dict, plan: dict) -> dict[str, Any]:
    """Generate Redrawn Frames for planned image_tasks via gateway, gptimg, or flow."""
    out_dir = Path(cfg.out_dir)
    only_shots = parse_only(cfg.only)
    aspect = (b.get("source") or {}).get("aspect_ratio")
    size = aspect_to_gateway_size(aspect)
    cli_aspect = normalize_aspect(aspect, ("16:9", "9:16", "1:1", "4:3", "3:4"), default="1:1")

    gw_cfg: tuple[str, str, str] | None = None
    cli_path: str | None = None

    generated: list[str] = []
    skipped: list[str] = []

    for task in plan.get("image_tasks") or []:
        shot_idx = int(task["shot"])
        if only_shots is not None and shot_idx not in only_shots:
            continue
        shot_id = f"shot_{shot_idx:03d}"
        target = out_dir / task["target"]
        stem = target.stem
        existing = target if target.is_file() else redrawn_frame(out_dir, stem)
        if existing is not None and not cfg.force:
            skipped.append(str(existing))
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        prompt = str(task.get("prompt") or "").strip()
        method = str(task.get("method") or "t2i").strip().lower()
        ref_paths = _collect_task_refs(out_dir, task, plan)

        if cfg.image_backend == "gateway":
            if gw_cfg is None:
                gw_cfg = resolve_gateway_config(cfg.image_model)
            base_v1, api_key, model = gw_cfg
            img_bytes = _gateway_image_request(
                base_v1, api_key, model, prompt, size, ref_paths, method, shot_id
            )
            target.write_bytes(img_bytes)
        elif cfg.image_backend in ("gptimg", "flow"):
            if cli_path is None:
                cli_path = _resolve_cli(cfg.image_backend)
            cmd = [
                cli_path,
                "image",
                "--prompt",
                prompt,
                "--aspect",
                cli_aspect,
                "--output",
                str(target),
                "--strategy",
                cfg.pool_strategy,
                "--json",
            ]
            for rp in ref_paths:
                cmd.extend(["--ref", str(rp)])
            if cfg.profile:
                cmd.extend(["--profile", cfg.profile])
            _run_cli_json(cmd, shot_id=shot_id)
        else:
            raise BridgeError(f"Unsupported image backend: {cfg.image_backend!r}")

        generated.append(str(target))

    return {
        "stage": "images",
        "backend": cfg.image_backend,
        "generated": generated,
        "skipped": skipped,
    }


def _video_backend_for_unit(unit: dict, shots_by_idx: dict[int, dict], cfg: BridgeConfig) -> str:
    if cfg.video_backend == "vids":
        return "vids"
    for s in unit.get("shots") or []:
        rec = shots_by_idx.get(int(s)) or {}
        if (rec.get("base") or {}).get("label") == "avatar":
            return "vids"
    return cfg.video_backend


def plan_dry_run(cfg: BridgeConfig, b: dict, plan: dict) -> dict[str, Any]:
    """Preview planned images, video, audio, and assemble work without calling any external API, CLI, or FFmpeg."""
    from .bridge_video import collect_video_tasks

    out_dir = Path(cfg.out_dir)
    only_shots = parse_only(cfg.only)
    selected = list(STAGES) if cfg.stage == "all" else [cfg.stage]

    img_tasks_preview: list[dict[str, Any]] = []
    for t in plan.get("image_tasks") or []:
        shot_idx = int(t["shot"])
        if only_shots is not None and shot_idx not in only_shots:
            continue
        target = out_dir / t["target"]
        stem = Path(t["target"]).stem
        exists = target.is_file() or redrawn_frame(out_dir, stem) is not None
        img_tasks_preview.append({
            "shot": shot_idx,
            "role": t.get("role", "first_frame"),
            "method": t.get("method", "t2i"),
            "backend": cfg.image_backend,
            "target": str(t["target"]),
            "would_skip": bool(exists and not cfg.force),
        })

    units_preview: list[dict[str, Any]] = [
        {
            "id": t["id"],
            "shots": t["shots"],
            "strategy": t["strategy"],
            "backend": t["backend"],
            "resolution": cfg.resolution,
            "output": str(t["output"]),
            "would_skip": bool(Path(t["output"]).is_file() and not cfg.force),
        }
        for t in collect_video_tasks(b, out_dir, cfg, plan=plan)
    ]

    if cfg.dubbing_script:
        sp = Path(cfg.dubbing_script)
        script_path = sp if sp.is_absolute() else out_dir / sp
    else:
        script_path = out_dir / "remake" / "dubbing_script.json"
    lines = (read_json(script_path).get("lines") or []) if script_path.is_file() else entities.script(b)
    audio_dir = out_dir / "remake" / "bridge" / "audio"
    active_lines = 0
    skip_lines = 0
    for i, ln in enumerate(lines, 1):
        if not str(ln.get("text") or "").strip():
            continue
        active_lines += 1
        idx = int(ln.get("index") or i)
        if (audio_dir / f"line_{idx:03d}.wav").is_file() and not cfg.force:
            skip_lines += 1
    mix_exists = (audio_dir / "dubbed_mix.wav").is_file()
    mix_would_skip = bool(mix_exists and not cfg.force)
    tts_url = cfg.tts_url or os.getenv("REA_TTS_BASE_URL") or "http://127.0.0.1:8000"

    return {
        "dry_run": True,
        "stage": cfg.stage,
        "selected_stages": selected,
        "out_dir": str(out_dir),
        "images": {"backend": cfg.image_backend, "tasks": img_tasks_preview},
        "video": {"backend": cfg.video_backend, "units": units_preview},
        "audio": {
            "tts_url": tts_url,
            "lines": len(lines),
            "would_skip_lines": skip_lines,
            "mix_would_skip": mix_would_skip,
            "would_skip": bool(active_lines > 0 and skip_lines == active_lines and mix_would_skip),
            "voice": cfg.tts_voice,
            "clone_source_voices": cfg.clone_source_voices,
            "keep_original_audio": cfg.keep_original_audio,
        },
        "assemble": {
            "retime": cfg.retime,
            "overlays": not cfg.no_overlays,
            "keep_original_audio": cfg.keep_original_audio,
        },
    }


def run_bridge(cfg: BridgeConfig) -> dict[str, Any]:
    """Execute or dry-run the local remake bridge for the requested stage(s)."""
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    cfg = BridgeConfig(**{**cfg.__dict__, "out_dir": out_dir})
    if cfg.stage != "all" and cfg.stage not in STAGES:
        raise BridgeError(f"Unknown bridge stage: {cfg.stage!r}")
    b = read_json(out_dir / "breakdown.json")
    load_intent(out_dir)
    plan = b.get("production_plan") or {}
    if not cfg.run:
        return plan_dry_run(cfg, b, plan)

    stages = list(STAGES) if cfg.stage == "all" else [cfg.stage]
    results: dict[str, Any] = {}
    for st in stages:
        if st == "images":
            results["images"] = run_images_stage(cfg, b, plan)
        elif st == "video":
            from .bridge_video import run_video_stage

            results["video"] = run_video_stage(cfg, b, plan)
        elif st == "audio":
            from .bridge_audio import run_audio_stage

            results["audio"] = run_audio_stage(cfg, b, plan)
        elif st == "assemble":
            from .bridge_audio import run_assemble_stage

            results["assemble"] = run_assemble_stage(cfg, b, plan)
    return {"dry_run": False, "stage": cfg.stage, "results": results}
