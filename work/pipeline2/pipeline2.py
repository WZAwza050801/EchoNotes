"""Run with python -m work.pipeline2.pipeline2 (or this file directly)."""
import argparse
import ast
from datetime import date
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "work.pipeline2"

from .core import (align_windows, cached, digest, load_dotenv, normalize_segments,
                   read_json, write_json, correct_segments)
from .distill import package, repackage
from .media import Bilibili, extract_frames, make_wav, probe
from .models import Chat, load_chat
from .render import compile_pdf, render
from .writing import map_windows, outline, polish, quality_report, verify_formulas


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_name(title):
    value = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", title).strip(" .")[:65]
    return value or "未命名课程"


def existing_browser_cookie():
    """Read the existing pipeline1 literal without importing its side-effectful entrypoint."""
    path = Path(__file__).resolve().parent.parent / "pipeline1" / "pipeline1.py"
    if path.exists():
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HDRS" for t in node.targets)
                    and isinstance(node.value, ast.Dict)):
                for key, value in zip(node.value.keys, node.value.values):
                    if isinstance(key, ast.Constant) and key.value == "Cookie":
                        return ast.literal_eval(value)
    return ""


def run(args, clients=None):
    source_arg = Path(args.source)
    local = source_arg.is_file()
    api = None
    if local:
        source = source_arg.resolve()
        source_id = "local-" + file_hash(source)[:12]
        meta = {"title": args.title or source.stem, "owner": "本地视频", "url": "",
                "source_id": source_id, "date": date.today().isoformat()}
    else:
        api = Bilibili(os.getenv("BILIBILI_COOKIE") or existing_browser_cookie())
        meta = api.metadata(args.source, args.page)
        source_id = meta["bvid"] + f"-P{meta['page']}"
        meta.update(source_id=source_id, date=date.today().isoformat())
        if args.title:
            meta["title"] = args.title
    run_dir = args.work_root.resolve() / source_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / ".running"
    try:
        handle = lock.open("x")
    except FileExistsError:
        raise RuntimeError(f"Run is locked: {lock}. Remove only after confirming the earlier process stopped.") from None
    try:
        with handle:
            handle.write(str(os.getpid()))
        if not local:
            source = run_dir / "source.mp4"
            if not source.exists():
                source = api.download(meta, run_dir)
        source_key = file_hash(source)
        info = probe(source)
        meta["duration"] = info["duration"]
        write_json(run_dir / "meta.json", meta)
        if args.transcript:
            raw = read_json(args.transcript)
            raw["source"] = raw.get("source", "provided_transcript")
        else:
            audio = run_dir / "audio.wav"
            audio_marker = run_dir / "audio-source.json"
            if not audio.exists() or not audio_marker.exists() or read_json(audio_marker) != source_key:
                make_wav(source, audio, info)
                write_json(audio_marker, source_key)
            def transcribe():
                output = run_dir / "asr-output.json"
                command = [args.asr_python, str(Path(__file__).with_name("asr.py")),
                           str(audio), str(output), "--model", args.asr_model, "--language", args.language]
                subprocess.run(command, check=True, timeout=max(1800, info["duration"] * 6),
                               env=os.environ | {"PYTHONUTF8": "1"})
                return read_json(output)
            raw = cached(run_dir / "cache" / "asr.json",
                         [source_key, args.asr_model, args.language, "small-int8-vad-v1"], transcribe)
        # Keep exact input alongside the normalized working copy.
        write_json(run_dir / "transcript.raw.json", raw)
        segments = normalize_segments(raw, info["duration"])
        frames_key = [source_key, args.interval, args.max_frames, args.scene_threshold, "pts-dhash-v1"]
        frame_cache = run_dir / "cache" / "frames.json"
        if frame_cache.exists():
            old = read_json(frame_cache)
            if old.get("key") == digest(frames_key) and any(
                    not (run_dir / f["path"]).exists() or file_hash(run_dir / f["path"]) != f["sha256"]
                    for f in old["data"]):
                frame_cache.unlink()
        def get_frames():
            values = extract_frames(source, run_dir, info, args.interval,
                                    args.max_frames, args.scene_threshold)
            for frame in values:
                frame["sha256"] = file_hash(run_dir / frame["path"])
            return values
        frames = cached(frame_cache, frames_key, get_frames)
        write_json(run_dir / "frames.json", frames)
        if args.prepare_only:
            write_json(run_dir / "transcript.json", {"segments": segments, "source": raw["source"]})
            print(f"[prepared] {run_dir}", flush=True)
            return run_dir
        text, vision = clients or (load_chat("text", args.secrets), load_chat("vision", args.secrets))
        # Persist the throttle clock into the run directory so a restarted
        # process keeps honoring MIN_INTERVAL (file-clock semantics).
        for client in (text, vision):
            if isinstance(client, Chat):
                client.clock_path = run_dir / "cache" / ".request-clock.json"
        print(f"[model] text={text.identity['model']} vision={vision.identity['model']} "
              f"timeout={os.getenv('ECHONOTES_MODEL_TIMEOUT', '180')}s/attempt "
              f"retries={os.getenv('ECHONOTES_MODEL_RETRIES', '3')} "
              f"backoff={os.getenv('ECHONOTES_MODEL_BACKOFF', '10')}s", flush=True)
        fixes = read_json(args.terms).get("fixes", {}) if args.terms else {}
        if args.no_polish:
            segments, warnings = correct_segments(segments, fixes), []
        else:
            segments, warnings = polish(segments, fixes, text, run_dir)
        write_json(run_dir / "transcript.json", {"segments": segments, "source": raw["source"]})
        windows = align_windows(segments, frames, info["duration"], args.window_seconds, args.max_images)
        write_json(run_dir / "alignment.json", windows)
        blocks = map_windows(windows, vision, text, run_dir)
        write_json(run_dir / "blocks.json", blocks)
        structure = outline(blocks, text, run_dir)
        checks = verify_formulas(blocks, frames, vision, run_dir, enabled=not args.no_verify)
        report = quality_report(blocks, checks, segments, frames, warnings)
        report["sampling"] = read_json(run_dir / "sampling.json")
        lecture = {"schema_version": 1, "meta": meta, "blocks": blocks, "outline": structure,
                   "frames": frames, "segments": segments, "quality": report,
                   "models": {"text": text.identity, "vision": vision.identity}}
        write_json(run_dir / "lecture.json", lecture)
        render(lecture, run_dir)
        report["compilation"] = {"status": "skipped"} if args.no_compile else compile_pdf(run_dir)
        write_json(run_dir / "quality.json", report)
        write_json(run_dir / "lecture.json", lecture)
        # 成品夹：每门课一个文件夹（BV号-P页-课程名），只留 PDF/tex/lecture.json/
        # 去重帧/README。默认自动清理运行目录（成品夹已含全部所需产物）；
        # 失败重试时缓存始终保留，只有成功收尾才清。
        deliverable = package(lecture, run_dir, args.output_root)
        if not (Path(deliverable) / "lecture.pdf").exists():
            raise RuntimeError("Deliverable PDF missing; run directory kept for inspection")
        if args.keep_cache:
            print(f"[done] {deliverable} (run cache kept: {run_dir})", flush=True)
        else:
            shutil.rmtree(run_dir)
            print(f"[done] {deliverable} (run cache cleaned)", flush=True)
        return deliverable
    finally:
        lock.unlink(missing_ok=True)


def parser():
    root = argparse.ArgumentParser(description="EchoNotes 课程视频 → 有证据的 LaTeX 讲义")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check local dependencies without reading secrets")
    p = commands.add_parser("run")
    p.add_argument("source", help="BV ID, full Bilibili URL, or local video file")
    p.add_argument("--page", type=int)
    p.add_argument("--title")
    p.add_argument("--transcript", type=Path, help="Existing timestamped transcript JSON; bypass ASR")
    p.add_argument("--terms", type=Path, help='Course-only {"fixes":{"old":"new"}} glossary')
    p.add_argument("--secrets", type=Path, default=os.getenv("ECHONOTES_SECRETS_FILE"))
    p.add_argument("--asr-python", default=os.getenv("ECHONOTES_ASR_PYTHON", sys.executable))
    p.add_argument("--asr-model", default=os.getenv("ECHONOTES_ASR_MODEL", "small"))
    p.add_argument("--language", default="zh")
    p.add_argument("--work-root", type=Path, default=Path("work/pipeline2/runs"))
    p.add_argument("--output-root", type=Path, default=Path("output/课程讲义"))
    p.add_argument("--interval", type=float, default=30)
    p.add_argument("--max-frames", type=int, default=240)
    p.add_argument("--scene-threshold", type=float, default=.12)
    p.add_argument("--window-seconds", type=float, default=600)
    p.add_argument("--max-images", type=int, default=8)
    p.add_argument("--prepare-only", action="store_true", help="Stop after transcript and frames; no API calls")
    p.add_argument("--no-polish", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--keep-cache", action="store_true",
                   help="Keep the run directory after a successful finish (default: auto-clean)")
    d = commands.add_parser("distill", help="Repackage an existing run into the clean per-course folder")
    d.add_argument("lecture", help="Path to lecture.json (run directory or old archive)")
    d.add_argument("--output-root", type=Path, default=Path("output/课程讲义"))
    return root


def main():
    load_dotenv()
    args = parser().parse_args()
    if args.command == "doctor":
        print(json.dumps({"executables": {x: bool(shutil.which(x)) for x in ("ffmpeg", "ffprobe", "xelatex")},
                          "python_modules": {x: importlib.util.find_spec(x) is not None
                                             for x in ("PIL", "faster_whisper")},
                          "asr_python_override": bool(os.getenv("ECHONOTES_ASR_PYTHON"))}, indent=2))
        return
    if args.command == "distill":
        print(f"[done] {repackage(args.lecture, args.output_root)}")
        return
    if (args.interval <= 0 or args.max_frames < 1 or args.window_seconds <= 0 or args.max_images < 1
            or not 0 <= args.scene_threshold <= 1):
        raise ValueError("Invalid sampling/window options")
    run(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"[error] {error}", file=sys.stderr)
        sys.exit(1)
