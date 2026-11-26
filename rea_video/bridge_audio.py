"""Audio Bridge Stage (VieNeu-TTS dubbing + stem mix) and Assemble Bridge Stage."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.request
import wave
from pathlib import Path
from typing import Any

from . import entities
from .bridge import BridgeConfig, BridgeError, _encode_multipart, urlopen_bytes
from .util import clamp, read_json, write_json

DEFAULT_TTS_URL = "http://127.0.0.1:8000"
DEFAULT_TTS_MODEL = "vieneu-v3-turbo"


def _resolve_tts_config(tts_url: str | None = None, api_key: str | None = None, model: str | None = None) -> tuple[str, str, str]:
    base = (tts_url or os.getenv("REA_TTS_BASE_URL") or DEFAULT_TTS_URL).strip().rstrip("/")
    base_v1 = base if base.endswith("/v1") else f"{base}/v1"
    key = (api_key if api_key is not None else os.getenv("REA_TTS_API_KEY") or "").strip()
    tts_model = (model or os.getenv("REA_TTS_MODEL") or DEFAULT_TTS_MODEL).strip()
    return base_v1, key, tts_model


def compute_atempo(actual_s: float, target_s: float, lo: float = 0.85, hi: float = 1.25) -> float:
    """Compute gentle FFmpeg atempo factor clamped to [lo, hi] to fit actual_s toward target_s."""
    if actual_s <= 0 or target_s <= 0:
        return 1.0
    return clamp(actual_s / target_s, lo, hi)


def _enrolled_voice_name(speaker: str, ref_path: Path) -> str:
    safe_spk = re.sub(r"[^A-Za-z0-9._-]+", "_", speaker.strip()).strip("_") or "spk"
    digest = hashlib.sha256(f"{speaker}:{ref_path.resolve()}".encode("utf-8")).hexdigest()[:8]
    prefix = f"rea_clone_{safe_spk}"[: 64 - 9]  # reserve _<8 hex>
    return f"{prefix}_{digest}"


def enroll_voice(
    ref_path: Path,
    speaker: str = "speaker",
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Enroll a reference audio clip on VieNeu-TTS via POST /v1/voices under a unique non-built-in name."""
    ref = Path(ref_path)
    if not ref.is_file():
        raise BridgeError(f"TTS reference audio not found: {ref}")
    base_v1, key, _ = _resolve_tts_config(base_url, api_key)
    voice_name = _enrolled_voice_name(speaker, ref)
    body, content_type = _encode_multipart(
        {
            "name": voice_name,
            "denoise": "true",
            "description": f"Cloned voice for {speaker}",
        },
        [("file", ref)],
    )
    headers = {"Content-Type": content_type}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"{base_v1}/voices", data=body, method="POST", headers=headers)
    urlopen_bytes(req, timeout=60.0, label="TTS voice enrollment failed")
    return voice_name


def synthesize_line(
    text: str,
    voice: str,
    out_path: Path,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> Path:
    """Synthesize one line to WAV via POST /v1/audio/speech."""
    base_v1, key, tts_model = _resolve_tts_config(base_url, api_key, model)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": tts_model,
        "input": text,
        "response_format": "wav",
        "voice": voice,
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        f"{base_v1}/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    wav_bytes = urlopen_bytes(req, timeout=120.0, label="TTS synthesis failed")
    out.write_bytes(wav_bytes)
    return out


def _dubbing_script_path(out_dir: Path, cfg: BridgeConfig) -> Path:
    if cfg.dubbing_script:
        p = Path(cfg.dubbing_script)
        return p if p.is_absolute() else out_dir / p
    return out_dir / "remake" / "dubbing_script.json"


def seed_dubbing_script(b: dict, out_dir: Path, cfg: BridgeConfig) -> dict[str, Any]:
    """Build and write an editable Dubbing Script from entities.script(b)."""
    out_dir = Path(out_dir)
    raw_lines = entities.script(b)
    speakers: dict[str, dict[str, str]] = {}
    lines: list[dict[str, Any]] = []

    for idx, ln in enumerate(raw_lines, 1):
        sid = str(ln.get("voice") or ln.get("speaker") or "S1").strip() or "S1"
        label = entities.speaker_label(b, sid) or str(ln.get("speaker") or "").strip() or sid
        if sid not in speakers:
            speakers[sid] = {
                "label": label,
                "voice": str(cfg.tts_voice or ""),
                "ref": str(cfg.tts_ref or ""),
            }
        entry: dict[str, Any] = {
            "index": idx,
            "start": ln.get("start"),
            "end": ln.get("end"),
            "speaker": sid,
            "speaker_label": label,
            "text": str(ln.get("text") or ""),
        }
        if ln.get("delivery"):
            entry["delivery"] = ln["delivery"]
        lines.append(entry)

    payload: dict[str, Any] = {"version": 1, "speakers": speakers, "lines": lines}
    write_json(_dubbing_script_path(out_dir, cfg), payload)
    return payload


def load_or_create_dubbing_script(b: dict, out_dir: Path, cfg: BridgeConfig) -> dict[str, Any]:
    """Load an existing Dubbing Script (preserving user edits) or seed a new one."""
    out_dir = Path(out_dir)
    path = _dubbing_script_path(out_dir, cfg)
    if path.is_file():
        return read_json(path)
    return seed_dubbing_script(b, out_dir, cfg)


def _run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or f"exit code {proc.returncode}"
        raise BridgeError(f"ffmpeg failed: {detail}")


def _probe_audio_duration(path: Path) -> float:
    """Measure WAV duration via audio-stream ffprobe, falling back to stdlib wave."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout:
        try:
            data = json.loads(proc.stdout)
            for container in ((data.get("streams") or [{}])[0], data.get("format") or {}):
                val = container.get("duration") if isinstance(container, dict) else None
                if val is not None and float(val) > 0:
                    return float(val)
        except (ValueError, TypeError, IndexError):
            pass
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            if rate > 0:
                return wf.getnframes() / float(rate)
    except (wave.Error, OSError):
        pass
    return 0.0


def _speaker_turn_start(b: dict, speaker: str) -> float:
    for turn in (b.get("speakers") or {}).get("turns") or []:
        if str(turn.get("speaker") or "") == speaker and isinstance(turn.get("start"), (int, float)):
            return max(0.0, float(turn["start"]))
    return 0.0


def _slice_speaker_sample(b: dict, out_dir: Path, speaker: str, dest: Path, *, force: bool = False) -> Path | None:
    """Slice a 5s WAV reference sample from stems/voice_<spk>.wav or stems/vocals.wav."""
    if dest.is_file() and not force:
        return dest
    stems_dir = out_dir / "stems"
    cand = stems_dir / f"voice_{speaker}.wav"
    if not cand.is_file():
        cand = stems_dir / "vocals.wav"
    if not cand.is_file():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    start_s = _speaker_turn_start(b, speaker)
    _run_ffmpeg(["-ss", f"{start_s:.3f}", "-t", "5", "-i", str(cand), "-ar", "24000", "-ac", "1", str(dest)])
    return dest


def mix_dubbed_audio(aligned_lines: list[dict[str, Any]], out_dir: Path, out_wav: Path) -> Path:
    """Combine aligned line WAVs (adelay) with stems/music.wav and stems/effects.wav via amix."""
    out_dir = Path(out_dir)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    inputs: list[str] = []
    filters: list[str] = []
    mix_labels: list[str] = []

    for idx, item in enumerate(aligned_lines):
        p = Path(item["path"])
        if not p.is_file():
            continue
        in_idx = len(inputs) // 2
        inputs.extend(["-i", str(p)])
        delay_ms = max(0, int(round(float(item.get("start") or 0.0) * 1000)))
        lbl = f"v{idx}"
        filters.append(f"[{in_idx}:a]adelay={delay_ms}|{delay_ms}[{lbl}]")
        mix_labels.append(f"[{lbl}]")

    for stem_name in ("music.wav", "effects.wav"):
        stem = out_dir / "stems" / stem_name
        if stem.is_file():
            in_idx = len(inputs) // 2
            inputs.extend(["-i", str(stem)])
            lbl = stem.stem
            filters.append(f"[{in_idx}:a]anull[{lbl}]")
            mix_labels.append(f"[{lbl}]")

    if not mix_labels:
        raise BridgeError("No aligned voice lines or background stems available to mix")

    n_inputs = len(mix_labels)
    if n_inputs == 1:
        filters.append(f"{mix_labels[0]}alimiter=limit=0.95[aout]")
    else:
        filters.append(
            f"{''.join(mix_labels)}amix=inputs={n_inputs}:duration=longest:normalize=0,alimiter=limit=0.95[aout]"
        )

    _run_ffmpeg([*inputs, "-filter_complex", ";".join(filters), "-map", "[aout]", "-ar", "44100", "-ac", "2", str(out_wav)])
    return out_wav


def run_audio_stage(cfg: BridgeConfig, b: dict, plan: dict | None = None) -> dict[str, Any]:
    """Run the Audio Bridge Stage: Dubbing Script -> voice enrollment -> TTS synthesis + atempo -> stem mix."""
    out_dir = Path(cfg.out_dir)
    audio_dir = out_dir / "remake" / "bridge" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    script_data = load_or_create_dubbing_script(b, out_dir, cfg)
    speakers_cfg = script_data.get("speakers") or {}
    lines = script_data.get("lines") or []
    if not lines:
        return {"stage": "audio", "synthesized": 0, "skipped": 0, "mix": None}

    enrolled_cache: dict[str, str] = {}

    def resolve_voice(sid: str, line_obj: dict[str, Any]) -> str:
        explicit_voice = str(line_obj.get("voice") or "").strip()
        if explicit_voice:
            return explicit_voice
        spk = speakers_cfg.get(sid) or {}
        if str(spk.get("voice") or "").strip():
            return str(spk["voice"]).strip()
        if sid in enrolled_cache:
            return enrolled_cache[sid]

        ref_str = str(line_obj.get("ref") or spk.get("ref") or cfg.tts_ref or "").strip()
        ref_path: Path | None = None
        if ref_str:
            cand = Path(ref_str)
            ref_path = cand if cand.is_absolute() else out_dir / cand
        elif cfg.clone_source_voices:
            safe_spk = re.sub(r"[^A-Za-z0-9._-]+", "_", sid).strip("_") or "spk"
            ref_path = _slice_speaker_sample(
                b, out_dir, sid, audio_dir / "refs" / f"{safe_spk}_5s.wav", force=cfg.force
            )

        if ref_path is not None:
            enrolled = enroll_voice(ref_path, speaker=sid, base_url=cfg.tts_url)
            enrolled_cache[sid] = enrolled
            return enrolled

        return str(cfg.tts_voice or "Binh")

    synthesized = 0
    skipped = 0
    aligned_items: list[dict[str, Any]] = []

    for i, ln in enumerate(lines, 1):
        text = str(ln.get("text") or "").strip()
        if not text:
            continue
        idx = int(ln.get("index") or i)
        sid = str(ln.get("speaker") or "S1").strip() or "S1"
        start = float(ln["start"]) if isinstance(ln.get("start"), (int, float)) else 0.0
        end = float(ln["end"]) if isinstance(ln.get("end"), (int, float)) else start
        aligned_wav = audio_dir / f"line_{idx:03d}.wav"

        if aligned_wav.is_file() and not cfg.force:
            skipped += 1
        else:
            voice = resolve_voice(sid, ln)
            raw_wav = audio_dir / f"line_{idx:03d}_raw.wav"
            synthesize_line(text, voice, raw_wav, base_url=cfg.tts_url)
            actual_s = _probe_audio_duration(raw_wav)
            target_s = end - start if end > start else actual_s
            tempo = compute_atempo(actual_s, target_s)
            if abs(tempo - 1.0) > 0.02:
                _run_ffmpeg(["-i", str(raw_wav), "-filter:a", f"atempo={tempo:.4f}", str(aligned_wav)])
            else:
                shutil.copyfile(raw_wav, aligned_wav)
            synthesized += 1

        aligned_items.append({"path": aligned_wav, "start": start, "end": end})

    mix_wav = audio_dir / "dubbed_mix.wav"
    if aligned_items:
        mix_dubbed_audio(aligned_items, out_dir, mix_wav)
        mix_out: str | None = str(mix_wav)
    else:
        mix_out = None

    return {
        "stage": "audio",
        "synthesized": synthesized,
        "skipped": skipped,
        "enrolled": dict(enrolled_cache),
        "mix": mix_out,
    }


def run_assemble_stage(cfg: BridgeConfig, b: dict, plan: dict | None = None) -> dict[str, Any]:
    """Assemble remake/mine.json with optional dubbed_mix.wav unless keep_original_audio is set."""
    from . import remake

    out_dir = Path(cfg.out_dir)
    rdir = out_dir / "remake"
    rdir.mkdir(parents=True, exist_ok=True)
    mine_path = rdir / "mine.json"
    extra = read_json(mine_path).get("results", {}) if mine_path.is_file() else {}
    code_video = out_dir / "code" / "remake.mp4"
    dubbed_mix = rdir / "bridge" / "audio" / "dubbed_mix.wav"
    audio = dubbed_mix if (dubbed_mix.is_file() and not cfg.keep_original_audio) else None

    log_raw = remake.assemble(
        out_dir / "breakdown.json",
        rdir / "results.json",
        rdir,
        overlays=not cfg.no_overlays,
        code_video=code_video if code_video.is_file() else None,
        extra=extra,
        retime=cfg.retime,
        audio=audio,
    )
    parsed = json.loads(log_raw) if isinstance(log_raw, str) else dict(log_raw)
    return {
        "stage": "assemble",
        "audio": str(audio) if audio is not None else None,
        **parsed,
    }
