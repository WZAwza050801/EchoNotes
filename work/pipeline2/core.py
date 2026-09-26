"""Pure evidence contracts, sampling and content-addressed stage cache."""
import hashlib
import json
import math
import os
from pathlib import Path


def load_dotenv(path=None):
    """Load KEY=VALUE lines from .env into os.environ (existing values win).

    Lets users configure keys by filling .env as the README promises, without
    exporting anything. Never overrides variables already set in the shell.
    """
    path = Path(path) if path else Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cached(path, inputs, produce):
    path = Path(path)
    key = digest(inputs)
    if path.exists():
        record = read_json(path)
        if record.get("key") == key:
            return record["data"]
    data = produce()
    write_json(path, {"key": key, "data": data})
    return data


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def choose_times(duration, scenes, interval=30, budget=240):
    if not finite_number(duration) or duration <= 0 or interval <= 0 or budget < 1:
        raise ValueError("duration / interval / frame budget must be positive")
    # Preserve the uniform backbone. Fail explicitly rather than silently thinning.
    count = math.ceil(duration / interval)
    if count > budget:
        raise ValueError(f"Uniform sampling needs {count} frames; increase --max-frames or --interval")
    selected = [round(i * interval, 6) for i in range(count)]
    candidates = {float(t) for t in scenes if finite_number(t) and 0 <= t < duration}
    while candidates and len(selected) < budget:
        point = max(sorted(candidates), key=lambda t: min(abs(t - x) for x in selected))
        candidates.remove(point)
        if min(abs(point - x) for x in selected) >= 0.25:
            selected.append(point)
    return sorted(selected)


def normalize_segments(data, duration):
    result = []
    for i, segment in enumerate(data["segments"]):
        start, end = segment["start"], segment["end"]
        if (not finite_number(start) or not finite_number(end)
                or not 0 <= start < duration or not start < end <= duration + 1):
            raise ValueError(f"Invalid transcript time at segment {i}")
        if not isinstance(segment["text"], str):
            raise ValueError("Transcript text must be a string")
        result.append({**segment, "id": f"s{i:06d}", "start": start,
                       "end": min(end, duration)})
    return sorted(result, key=lambda s: s["start"])


def correct_segments(segments, fixes):
    result = []
    for segment in segments:
        text = segment["text"]
        for old, new in fixes.items():
            if not old or not isinstance(new, str):
                raise ValueError("Term corrections must map non-empty strings to strings")
            text = text.replace(old, new)
        result.append({**segment, "text": text})
    return result


def align_windows(segments, frames, duration, window_seconds=600, max_images=8):
    if window_seconds <= 0 or max_images < 1:
        raise ValueError("Window duration and image limit must be positive")
    result = []
    for n in range(math.ceil(duration / window_seconds)):
        start, end = n * window_seconds, min((n + 1) * window_seconds, duration)
        selected = sorted([f for f in frames if start <= f["actual_t"] < end],
                          key=lambda f: f["actual_t"])
        # Split at midpoints between image groups, so speech remains continuous.
        boundaries = [start]
        boundaries += [(selected[i - 1]["actual_t"] + selected[i]["actual_t"]) / 2
                       for i in range(max_images, len(selected), max_images)]
        boundaries.append(end)
        for left, right in zip(boundaries, boundaries[1:]):
            result.append({
                "id": f"w{len(result):04d}", "start": left, "end": right,
                "segments": [s for s in segments if s["start"] < right and s["end"] > left],
                "frames": [f for f in selected if left <= f["actual_t"] < right],
            })
    return result


KINDS = {"definition", "theorem", "proof", "derivation", "example", "explanation"}


def string_list(value):
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def normalize_uncertainties(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        raise ValueError("Uncertainties must be null, a string, or a list")
    result = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                result.append(item)
        elif isinstance(item, dict):
            text = next((item.get(key) for key in ("note", "reason", "text", "description")
                         if isinstance(item.get(key), str) and item[key].strip()), None)
            if text is None:
                raise ValueError("Uncertainty object has no textual note")
            result.append(text)
        else:
            raise ValueError("Uncertainty list contains a non-text value")
    return result


def normalize_map(data, window=None):
    allowed_frames = {f["id"] for f in window["frames"]} if window else None
    if isinstance(data, dict) and isinstance(data.get("blocks"), list):
        for block in data["blocks"]:
            if isinstance(block, dict):
                block["uncertainties"] = normalize_uncertainties(block.get("uncertainties"))
                kind = block.get("kind")
                if isinstance(kind, str) and kind not in KINDS:
                    block["uncertainties"].append(
                        f"模型原始知识块类型为“{kind}”，已按 explanation 排版。")
                    block["kind"] = "explanation"
                for name in ("formulas", "symbols"):
                    value = block.get(name)
                    if value is None:
                        block[name] = []
                    elif isinstance(value, dict):
                        block[name] = [value]
                # Conservative repair for a missing/mistyped formula.uncertain:
                # mark it for human review instead of rejecting the whole window.
                for index, formula in enumerate(block["formulas"], 1):
                    if isinstance(formula, dict) and not isinstance(formula.get("uncertain"), bool):
                        formula["uncertain"] = True
                        block["uncertainties"].append(
                            f"第 {index} 条公式的 uncertain 标记缺失或类型错误，"
                            "已保守按“待核验”处理，请人工复核该公式。")
                if allowed_frames is not None and isinstance(block.get("frame_ids"), list):
                    for formula in block["formulas"]:
                        source = formula.get("frame_id") if isinstance(formula, dict) else None
                        if source is not None and source in allowed_frames and source not in block["frame_ids"]:
                            block["frame_ids"].append(source)
    return data


def validate_map(data, window):
    if not isinstance(data, dict) or not isinstance(data.get("title"), str):
        raise ValueError("Map requires a title")
    if not isinstance(data.get("blocks"), list) or not data["blocks"]:
        raise ValueError("Map must contain at least one evidence-backed block")
    allowed_segments = {s["id"] for s in window["segments"]}
    allowed_frames = {f["id"] for f in window["frames"]}
    for block in data["blocks"]:
        if block.get("kind") not in KINDS:
            raise ValueError("Unknown block kind")
        for name in ("title", "text"):
            if not isinstance(block.get(name), str) or not block[name].strip():
                raise ValueError(f"Missing block {name}")
        for name, allowed in (("segment_ids", allowed_segments), ("frame_ids", allowed_frames)):
            if not string_list(block.get(name)) or not set(block[name]).issubset(allowed):
                raise ValueError(f"Invented or invalid {name}")
        if not block["segment_ids"] and not block["frame_ids"]:
            raise ValueError("Block has no evidence")
        if not string_list(block.get("uncertainties")):
            raise ValueError("Uncertainties must be a list of strings")
        if not isinstance(block.get("formulas"), list) or not isinstance(block.get("symbols"), list):
            raise ValueError("Formulas and symbols must be lists")
        for formula in block["formulas"]:
            if not isinstance(formula.get("latex"), str) or not formula["latex"].strip():
                raise ValueError("Empty formula")
            if formula.get("frame_id") is not None and formula["frame_id"] not in block["frame_ids"]:
                raise ValueError("Formula source must belong to the block")
            if not isinstance(formula.get("uncertain"), bool):
                raise ValueError("Formula uncertainty is required")
        for symbol in block["symbols"]:
            if not all(isinstance(symbol.get(k), str) for k in ("symbol", "meaning")):
                raise ValueError("Invalid symbol definition")
    return data


def validate_outline(data, blocks):
    sections = data.get("sections") if isinstance(data, dict) else None
    if not isinstance(sections, list) or not sections:
        raise ValueError("Reduce requires sections")
    ids = []
    for section in sections:
        if not isinstance(section.get("title"), str) or not string_list(section.get("block_ids")):
            raise ValueError("Invalid outline section")
        if not section["block_ids"]:
            raise ValueError("Empty outline section")
        ids.extend(section["block_ids"])
    expected = [b["id"] for b in blocks]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Reduce must retain every block exactly once")
    return data
