"""Offline, script-only promo finder for a timestamped screenplay ZIP.

Usage: python promo_finder.py "path/to/scripts.zip" --out raw_outputs
Only Python's standard library is required. No source scripts are extracted.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path


TIME = r"(\d{2}:\d{2}\.\d{2})"
EVENT_RE = re.compile(
    rf"^\*\*(.*?):\*\*\r?\nStart:\s*\({TIME}\)\s*\|\s*End:\s*\({TIME}\)\r?\n([^\r\n]*)",
    re.MULTILINE,
)
SCENE_RE = re.compile(r"^SCENE\s+\d+\s*$", re.MULTILINE | re.IGNORECASE)
SCENE_END_RE = re.compile(rf"^Scene end:\s*{TIME}", re.MULTILINE | re.IGNORECASE)
RAW_TIME_RE = re.compile(r"^Start:\s*\(\d{2}:\d{2}\.\d{2}\)\s*\|\s*End:", re.MULTILINE)

SHOCK = re.compile(
    r"\b(?:slap|strike|struck|bleed|blood|explod|shatter|scream|gun|knife|blade|"
    r"fireball|flame|flaming|arrest|hanged|hanging|gallows|poison|witch|dead|death|dying|"
    r"betray|betrayal|secret|thug|chained|chains|threat|drag|seize|beg|"
    r"punish|destroy|burn|confess|recogniz|reveal|kneel|collapse|affair|"
    r"crawl|humiliat|imposter)\w*\b",
    re.IGNORECASE,
)
CONFLICT = re.compile(
    r"\b(?:no|never|not|cannot|can't|won't|must|how dare|who are you|"
    r"what are you|why|leave|stop|choose|tell me|give me|take her|"
    r"you're|you are|i am|i'm|he is|she is|they are|before|unless|or else)\b",
    re.IGNORECASE,
)
OPEN_LOOP = re.compile(
    r"\b(?:who|what|why|how|where|will|would|could|can|do you|did you|"
    r"tell me|speak|choose|if|unless|before|or else|about to|soon|"
    r"stop|wait|take her|arrest|hang|execute|kill|death|dying|"
    r"coming|arriv|approach|burst|break|enter|"
    r"point|aim|discover|recogniz|reveal|secret)\w*\b",
    re.IGNORECASE,
)
RESOLUTION = re.compile(
    r"\b(?:finally safe|all is well|we begin again|i accept|thank you|"
    r"we understand each other|no longer afraid|everything is over|"
    r"saved|free at last|triumphant|peaceful|fading to black)\b",
    re.IGNORECASE,
)
IMMINENT = re.compile(
    r"\b(?:about to|raises?|reaches?|bursts?|explodes?|strikes?|swings?|"
    r"rushes?|breaks?|shatters?|grabs?|drags?|enters?|emerges?|steps? through|"
    r"stops? mere|points?|aims?|lunges?|approaches?|arrives?|knocks?|"
    r"seizes?|grasps?|fireball|gallows|branding rod)\b",
    re.IGNORECASE,
)
FUTURE_STAKE = re.compile(
    r"\b(?:will|would|could|going to|tomorrow|tonight|before|unless|if|"
    r"or else|hang|gallows|execute|trial|arrest|take her|rightful|coming home|not lawful)\b",
    re.IGNORECASE,
)
PENDING_DECISION = re.compile(
    r"\b(?:accept|choose|decide|fail|refuse|answer me|speak to me|"
    r"marry me|not yet|no one will)\b",
    re.IGNORECASE,
)
UNKNOWN_ARRIVAL = re.compile(
    r"\b(?:silhouett\w*|armed figures|unknown figure|unexpected visitor|"
    r"someone (?:arrives|enters|approaches)|stranger (?:arrives|enters|approaches))\b",
    re.IGNORECASE,
)
RELATION = re.compile(
    r"\b(?:my|your|his|her)\s+(?:wife|husband|mother|father|sister|brother|"
    r"bride|groom|son|daughter|family|home|book|ring|life)\b",
    re.IGNORECASE,
)
PRONOUN_OPEN = re.compile(r"^(?:\W*[.]{0,3}\s*)?(?:he|she|it|they|this|that|him|her|them|there)\b", re.IGNORECASE)


def to_seconds(value: str) -> float:
    minute, seconds = value.split(":")
    return round(int(minute) * 60 + float(seconds), 2)


def to_stamp(value: float) -> str:
    minute, seconds = divmod(round(value * 100), 6000)
    return f"{minute:02d}:{seconds // 100:02d}.{seconds % 100:02d}"


@dataclass(frozen=True)
class Event:
    id: str
    episode: int
    scene: int
    start: float
    end: float
    kind: str
    speaker: str
    text: str


@dataclass
class Episode:
    number: int
    filename: str
    duration: float
    scene_count: int
    logline: str
    events: list[Event]


def parse_zip(path: Path) -> list[Episode]:
    episodes: list[Episode] = []
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Bad ZIP CRC: {bad_member}")
        filenames = sorted((n for n in archive.namelist() if n.lower().endswith(".txt")))
        if not filenames:
            raise ValueError("ZIP contains no .txt screenplay episodes")
        for filename in filenames:
            match = re.search(r"episode[-_ ](\d+)", filename, re.IGNORECASE)
            if not match:
                raise ValueError(f"Cannot identify episode from {filename}")
            number = int(match.group(1))
            source = archive.read(filename).decode("utf-8-sig")
            logline_match = re.search(r"^EPISODE LOGLINE:\s*(.*)$", source, re.MULTILINE | re.IGNORECASE)
            logline = logline_match.group(1) if logline_match else ""
            scene_marks = list(SCENE_RE.finditer(source))
            if not scene_marks:
                raise ValueError(f"Episode {number} has no scenes")
            events: list[Event] = []
            declared_end = 0.0
            for scene_index, mark in enumerate(scene_marks, 1):
                section_end = scene_marks[scene_index].start() if scene_index < len(scene_marks) else len(source)
                section = source[mark.start():section_end]
                end_match = SCENE_END_RE.search(section)
                if not end_match:
                    raise ValueError(f"Episode {number} scene {scene_index} lacks an end time")
                declared_end = max(declared_end, to_seconds(end_match.group(1)))
                timed_lines = len(RAW_TIME_RE.findall(section))
                event_matches = list(EVENT_RE.finditer(section))
                if timed_lines != len(event_matches):
                    raise ValueError(
                        f"Episode {number} scene {scene_index}: {timed_lines} time lines but "
                        f"{len(event_matches)} parsed events"
                    )
                for event_match in event_matches:
                    label = event_match.group(1).strip()
                    start = to_seconds(event_match.group(2))
                    end = to_seconds(event_match.group(3))
                    if start < 0 or end < start:
                        raise ValueError(f"Episode {number} has invalid event time {start}-{end}")
                    kind = label if label in {"CAMERA", "ACTION", "SOUND FX"} else "DIALOGUE"
                    speaker = "" if kind != "DIALOGUE" else re.sub(r"\s*\(.*", "", label).strip()
                    events.append(
                        Event(
                            f"E{number:03d}-{len(events)+1:03d}", number, scene_index,
                            start, end, kind, speaker, event_match.group(4).strip(),
                        )
                    )
            if any(event.end > declared_end + 0.01 for event in events):
                raise ValueError(f"Episode {number} event extends beyond episode end")
            episodes.append(Episode(number, filename, declared_end, len(scene_marks), logline, events))
    numbers = [episode.number for episode in episodes]
    if len(numbers) != len(set(numbers)):
        raise ValueError("Duplicate episode numbers")
    return episodes


def evidence_score(event: Event) -> float:
    text = event.text
    if event.kind == "SOUND FX":
        return 0.25
    if event.kind == "CAMERA":
        base = 0.35
    elif event.kind == "ACTION":
        base = 1.2
    else:
        base = 1.35
    base += min(2.1, 0.85 * len(SHOCK.findall(text)))
    base += min(1.2, 0.55 * len(CONFLICT.findall(text)))
    if "?" in text:
        base += 0.25
    if "!" in text:
        base += 0.25
    if text.lstrip('"').startswith(("...", "…")):
        base -= 1.6
    if PRONOUN_OPEN.search(text.strip('"')):
        base -= 0.7
    if event.kind == "CAMERA":
        base = min(base, 3.3)
    return min(5.0, max(0.0, base))


def cliff_score(event: Event) -> float:
    text = event.text
    if event.kind == "SOUND FX":
        return 0.0
    score = 0.5 if event.kind == "CAMERA" else 0.9
    if "?" in text:
        score += 2.0
    score += min(1.0, 0.5 * len(OPEN_LOOP.findall(text)))
    if FUTURE_STAKE.search(text):
        score += 0.9
    if PENDING_DECISION.search(text):
        score += 1.8
    if event.kind in {"ACTION", "CAMERA"} and UNKNOWN_ARRIVAL.search(text):
        score += 1.4
    if IMMINENT.search(text):
        score += 1.6 if event.kind in {"ACTION", "CAMERA"} else 0.6
    if SHOCK.search(text):
        score += 0.5
    if RESOLUTION.search(text):
        score -= 1.8
    if re.search(r"\b(?:fade|fades|walk away|embrace|smiles? warmly|peaceful)\b", text, re.IGNORECASE):
        score -= 1.0
    if event.kind == "CAMERA":
        score = min(score, 2.5)
    return max(0.0, min(5.0, score))


def intersect(event: Event, start: float, end: float) -> bool:
    return event.start < end and event.end > start


def window_events(episode: Episode, start: float, end: float) -> list[Event]:
    return [event for event in episode.events if intersect(event, start, end)]


def dialogue_cut(episode: Episode, point: float) -> bool:
    return any(
        event.kind == "DIALOGUE" and event.start + 0.05 < point < event.end - 0.05
        for event in episode.events
    )


def score_window(episode: Episode, start: float, end: float) -> dict | None:
    if not (20 <= end - start <= 60) or dialogue_cut(episode, start) or dialogue_cut(episode, end):
        return None
    events = window_events(episode, start, end)
    if not events:
        return None
    opening = [
        e for e in events
        if e.start >= start - 0.05 and e.start < start + 3 and e.kind != "SOUND FX"
    ]
    ending = [e for e in events if e.end >= end - 2.5 and e.start <= end and e.kind != "SOUND FX"]
    if not opening or not ending:
        return None
    hook_event = max(opening, key=evidence_score)
    cliff_event = max(ending, key=cliff_score)
    hook = evidence_score(hook_event)
    cliff = cliff_score(cliff_event)
    if hook < 2.1 or cliff < 2.6:
        return None
    spoken = [e for e in events if e.kind == "DIALOGUE"]
    speakers = set(e.speaker for e in spoken)
    early_text = " ".join(e.text for e in events if e.start < start + 10 and e.kind != "CAMERA")
    clip_text = " ".join(e.text for e in events if e.kind != "CAMERA")
    context = 2.0
    context += 0.7 if len(speakers) >= 2 else 0
    context += 0.5 if RELATION.search(clip_text) else 0
    context += 0.4 if len(spoken) >= 4 else 0
    context += 0.4 if len(early_text) >= 60 else 0
    context -= 0.75 if PRONOUN_OPEN.search(hook_event.text.strip('"')) else 0
    context -= 0.65 if len(set(e.scene for e in events)) > 1 else 0
    context = max(0, min(5, context))
    if cliff_event.end < end - 1.0:
        cliff -= 0.4  # the dramatic beat is followed by too much tail
    if abs(episode.duration - end) <= 1.0:
        cliff += 0.3  # episode endings are priors, not automatic winners
    duration = end - start
    duration_bonus = max(0, 1.0 - abs(duration - 38) / 22)
    total = round(3.3 * hook + 3.5 * cliff + 1.5 * context + duration_bonus, 3)
    return {
        "episode": episode.number,
        "segments": [{"episode": episode.number, "start": to_stamp(start), "end": to_stamp(end)}],
        "duration_seconds": round(duration, 2),
        "hook": hook_event.text,
        "hook_time": to_stamp(hook_event.start),
        "hook_event_id": hook_event.id,
        "cliffhanger": cliff_event.text,
        "cliffhanger_time": to_stamp(cliff_event.start),
        "cliffhanger_event_id": cliff_event.id,
        "scores": {"hook": round(hook, 2), "cliffhanger": round(cliff, 2), "context_proxy": round(context, 2), "total": total},
        "why_it_works": "Immediate " + ("visual action" if hook_event.kind != "DIALOGUE" else "confrontation")
        + " leads to an unresolved " + ("question" if "?" in cliff_event.text else "turn") + ".",
        "review_note": "Script-only suggestion; check footage, context, and the exact cut before posting.",
    }


def candidates_for_episode(episode: Episode) -> list[dict]:
    interesting = [e for e in episode.events if e.kind in {"ACTION", "DIALOGUE"}]
    starts = {0.0} | {e.start for e in interesting}
    ends = {episode.duration} | {e.end for e in interesting}
    camera_boundaries = {e.start for e in episode.events if e.kind == "CAMERA"}
    for boundary in camera_boundaries:
        if any(0 <= e.start - boundary <= 2.5 for e in interesting):
            starts.add(boundary)
    camera_ends = {e.end for e in episode.events if e.kind == "CAMERA"}
    for boundary in camera_ends:
        if any(0 <= boundary - e.end <= 2.0 for e in interesting):
            ends.add(boundary)
    results = []
    for start in sorted(starts):
        for end in sorted(ends):
            if end - start < 20:
                continue
            if end - start > 60:
                break
            candidate = score_window(episode, start, end)
            if candidate:
                results.append(candidate)
    return sorted(results, key=lambda x: x["scores"]["total"], reverse=True)


def overlap_fraction(a: dict, b: dict) -> float:
    if a["episode"] != b["episode"]:
        return 0.0
    sa, ea = (to_seconds(a["segments"][0][k]) for k in ("start", "end"))
    sb, eb = (to_seconds(b["segments"][0][k]) for k in ("start", "end"))
    overlap = max(0.0, min(ea, eb) - max(sa, sb))
    return overlap / max(0.01, min(ea - sa, eb - sb))


def diversify(candidates: list[dict], limit: int, per_episode: int = 2) -> list[dict]:
    selected: list[dict] = []
    counts: dict[int, int] = {}
    for candidate in candidates:
        episode = candidate["episode"]
        if counts.get(episode, 0) >= per_episode:
            continue
        if any(overlap_fraction(candidate, prior) >= 0.5 for prior in selected):
            continue
        selected.append(dict(candidate))
        counts[episode] = counts.get(episode, 0) + 1
        if len(selected) >= limit:
            break
    for rank, candidate in enumerate(selected, 1):
        candidate["rank"] = rank
    return selected


def run(input_path: Path, output_dir: Path, top: int) -> None:
    episodes = parse_zip(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_candidates = []
    episode_best = []
    for episode in episodes:
        candidates = candidates_for_episode(episode)
        all_candidates.extend(candidates)
        episode_best.extend(diversify(candidates, 1, 1))
    all_candidates.sort(key=lambda x: x["scores"]["total"], reverse=True)
    ranked = diversify(all_candidates, top)
    for candidate in episode_best:
        candidate.pop("rank", None)
    audit = {
        "input_zip": input_path.name,
        "episodes": len(episodes),
        "scenes": sum(e.scene_count for e in episodes),
        "timestamped_events": sum(len(e.events) for e in episodes),
        "dialogue_events": sum(ev.kind == "DIALOGUE" for e in episodes for ev in e.events),
        "action_events": sum(ev.kind == "ACTION" for e in episodes for ev in e.events),
        "camera_events": sum(ev.kind == "CAMERA" for e in episodes for ev in e.events),
        "sound_events": sum(ev.kind == "SOUND FX" for e in episodes for ev in e.events),
        "total_runtime_seconds": round(sum(e.duration for e in episodes), 2),
        "episode_duration_median_seconds": round(statistics.median(e.duration for e in episodes), 2),
        "valid_scored_windows": len(all_candidates),
        "episode_best_count": len(episode_best),
        "episodes_without_scored_window": sorted(
            set(e.number for e in episodes) - set(c["episode"] for c in episode_best)
        ),
        "ranked_count": len(ranked),
        "mode": "offline heuristic baseline; not an audience performance predictor",
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "all_candidates.json").write_text(json.dumps(all_candidates, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "episode_best.json").write_text(json.dumps(episode_best, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "ranked_promos.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "ranked_promos.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "episode", "start", "end", "duration_seconds", "hook", "cliffhanger", "why_it_works", "score"])
        for candidate in ranked:
            segment = candidate["segments"][0]
            writer.writerow([
                candidate["rank"], candidate["episode"], segment["start"], segment["end"],
                candidate["duration_seconds"], candidate["hook"], candidate["cliffhanger"],
                candidate["why_it_works"], candidate["scores"]["total"],
            ])
    print(json.dumps(audit, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--out", type=Path, default=Path("raw_outputs"))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    run(args.zip_path, args.out, args.top)


if __name__ == "__main__":
    main()
