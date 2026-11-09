"""Find cross-episode, two-segment promo proposals from timed screenplay events.

The first segment supplies an immediate hook and situation; the later segment
escalates the same topic and cuts before a consequence. This is a script-only
search queue: every join needs a cold-viewer and footage check.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from promo_finder import Episode, Event, UNKNOWN_ARRIVAL, dialogue_cut, to_seconds, to_stamp
from promo_pipeline import (
    PENDING, PENDING_ACTION, REFERENCE_OPEN, RESOLUTION, STAKE,
    cliff_strength, complete_events, event_words, hook_strength,
)


TOPIC_PATTERNS = {
    "poisoning": re.compile(r"\b(?:poison|arsenic|antidote|toxin|sick|dying)\w*\b", re.I),
    "healing": re.compile(r"\b(?:heal|herb|medicine|remedy|treat|doctor|fever)\w*\b", re.I),
    "identity": re.compile(r"\b(?:duke|heir|rightful|crest|ring|son|daughter|identity|disguise)\w*\b", re.I),
    "marriage": re.compile(r"\b(?:wedd|marry|marriage|husband|wife|bride|groom|priest|license)\w*\b", re.I),
    "trial": re.compile(r"\b(?:witch|trial|witness|magistrate|testimony|court|verdict|judge)\w*\b", re.I),
    "evidence": re.compile(r"\b(?:book|ledger|letter|evidence|proof|document|watch|signet)\w*\b", re.I),
    "captivity": re.compile(r"\b(?:arrest|prison|cell|chain|guard|escape|captive|dungeon)\w*\b", re.I),
    "violence": re.compile(r"\b(?:kill|murder|hang|gallows|execute|blood|weapon|gun|knife|fire)\w*\b", re.I),
}
STRONG_TOPICS = {"poisoning", "healing", "identity", "marriage", "trial"}
TOPIC_PRIORITY = ("poisoning", "identity", "trial", "marriage", "healing", "evidence", "captivity", "violence")
GENERIC_ROLE = {"lady", "lord", "the", "a", "his", "her", "former", "groom", "mother", "father", "guard", "maid", "constable", "priest", "duchess", "duke", "soldier", "man", "woman", "child", "voice", "blond", "elder", "young", "old", "tall", "high", "scarred", "new", "armed", "court", "royal", "nobleman", "noblewoman"}
THREAT = re.compile(r"\b(?:will (?:die|hang|suffer|lose|be arrested|be dead)|unless|fail(?:,| or)|refuse|let (?:her|him|them) suffer|hangs? from|be executed|or (?:your|his|her) .*? (?:die|hang))\b", re.I)
VISUAL_ARRIVAL = re.compile(r"\b(?:silhouett\w* (?:armed )?figures?|armed figures? (?:emerge|appear|arrive)|unknown (?:figures?|person|intruder)|masked (?:figures?|men)|(?:doors?|gates?) (?:burst|explode|shatter) (?:open|inward))\b", re.I)
DRAMATIC_ACTION = re.compile(r"\b(?:slaps?|strikes?|stabs?|shoots?|arrests?|seizes?|drags?|attacks?|explod\w*|shatter\w*|burns?|bleed\w*|blood|gun|knife|blade|fireball|chok\w*|hits?)\b", re.I)
CALM_ACTION = re.compile(r"\b(?:quietly|studies|smiles?|laugh\w*|kneels?|gazes?|glances?|looks?|turns?|watches?|firelight)\b", re.I)
CLOSED_DECLARATION = re.compile(r"\b(?:nothing breaks that|you are safe now|it is over|we have won|we are free)\b", re.I)
GENERIC_JOIN_STOP = {"said", "says", "asks", "looks", "turns", "stands", "walks", "room", "door", "hand", "eyes", "voice", "again", "here", "there", "now", "then", "just", "can", "would", "could", "should", "will", "before", "after", "across", "behind", "toward", "still", "only", "them", "their", "some", "that", "this", "from", "into", "with"}


def opening_strength(event: Event) -> float:
    strength = hook_strength(event)
    if event.kind in {"ACTION", "CAMERA"} and CALM_ACTION.search(event.text) and not DRAMATIC_ACTION.search(event.text):
        strength = min(strength, 1.5)
    return strength


def ending_strength(event: Event) -> float:
    strength = cliff_strength(event)
    if THREAT.search(event.text):
        strength = max(strength, 2.8)
    if event.kind in {"ACTION", "CAMERA"} and (VISUAL_ARRIVAL.search(event.text) or UNKNOWN_ARRIVAL.search(event.text)):
        strength = max(strength, 3.1)
    return strength


def names(events: list[Event]) -> set[str]:
    result: set[str] = set()
    for event in events:
        if event.kind != "DIALOGUE":
            continue
        for word in re.findall(r"[A-Za-z]{3,}", event.speaker):
            word = word.lower()
            if word not in GENERIC_ROLE:
                result.add(word)
    return result


def topics(events: list[Event]) -> set[str]:
    text = " ".join(event.text for event in events if event.kind != "SOUND FX")
    return {name for name, pattern in TOPIC_PATTERNS.items() if pattern.search(text)}


def source_segment(episode: Episode, start: float, end: float, role: str) -> dict | None:
    if not 8 <= end - start <= 30 or dialogue_cut(episode, start) or dialogue_cut(episode, end):
        return None
    events = complete_events(episode, start, end)
    substantive = [e for e in events if e.kind in {"DIALOGUE", "ACTION"}]
    if len(substantive) < 2:
        return None
    chars = names(events)
    tags = topics(events)
    if role == "opening":
        early = [e for e in events if e.kind != "SOUND FX" and e.end <= start + 3.01]
        uncertain = False
        if not early:
            early = [e for e in events if e.kind != "SOUND FX" and e.start < start + 3.01]
            uncertain = True
        if not early:
            return None
        cue = max(early, key=opening_strength)
        strength = opening_strength(cue)
        if strength < 1.8:
            return None
        # The short opening must offer more than an isolated exclamation.
        context = min(2.0, 0.3 * len(substantive) + 0.5 * bool(STAKE.search(" ".join(e.text for e in events))) + 0.4 * bool(chars))
        score = 2.3 * strength + context + 0.3 * len(tags) - (1.7 if uncertain else 0.0)
    else:
        final = [e for e in events if e.kind != "SOUND FX" and e.end >= end - 2.01]
        if not final:
            return None
        cue = max(final, key=ending_strength)
        strength = ending_strength(cue)
        if strength < 2.1 or RESOLUTION.search(cue.text) or CLOSED_DECLARATION.search(cue.text):
            return None
        if cue.end < max(e.end for e in substantive) - 0.05:
            return None
        open_loop = "?" in cue.text or bool(PENDING_ACTION.search(cue.text)) or bool(THREAT.search(cue.text)) or bool(VISUAL_ARRIVAL.search(cue.text)) or bool(UNKNOWN_ARRIVAL.search(cue.text))
        if not open_loop:
            return None
        first_substantive = substantive[0]
        abrupt = bool(REFERENCE_OPEN.search(first_substantive.text))
        context = min(2.0, 0.25 * len(substantive) + 0.5 * bool(STAKE.search(" ".join(e.text for e in events))) + 0.4 * bool(chars))
        score = 2.3 * strength + context + 0.3 * len(tags) - (0.9 if abrupt else 0.0)
        uncertain = False
    return {
        "episode": episode.number,
        "start": start,
        "end": end,
        "cue": cue,
        "score": round(score, 3),
        "timing_uncertain": uncertain,
        "characters": chars,
        "topics": tags,
        "events": events,
    }


def micro_options(episode: Episode, rows: list[dict], role: str, limit: int = 26) -> list[dict]:
    # Use different anchor moments, not just 18 slightly shifted copies of one.
    anchors: list[float] = []
    options: dict[tuple[float, float], dict] = {}
    row_anchors = [to_seconds(row["segments"][0]["start" if role == "opening" else "end"]) for row in rows[:120]]
    if role == "opening":
        source_anchors = [e.start for e in sorted(episode.events, key=opening_strength, reverse=True) if e.kind in {"DIALOGUE", "ACTION"}][:24]
        source_anchors.append(0.0)
    else:
        source_anchors = [e.end for e in sorted(episode.events, key=ending_strength, reverse=True) if e.kind in {"DIALOGUE", "ACTION", "CAMERA"}][:24]
        source_anchors.append(episode.duration)
    for anchor in source_anchors + row_anchors:
        if any(abs(anchor - old) < 3.5 for old in anchors):
            continue
        anchors.append(anchor)
        boundaries = [e.end if role == "opening" else e.start for e in episode.events if e.kind in {"DIALOGUE", "ACTION", "CAMERA"}]
        for target in (13.0, 20.0, 27.0):
            eligible = [b for b in boundaries if 8 <= (b - anchor if role == "opening" else anchor - b) <= 30]
            if not eligible:
                continue
            boundary = min(eligible, key=lambda b: abs((b - anchor if role == "opening" else anchor - b) - target))
            start, end = (anchor, boundary) if role == "opening" else (boundary, anchor)
            option = source_segment(episode, start, end, role)
            if option:
                key = (start, end)
                if key not in options or option["score"] > options[key]["score"]:
                    options[key] = option
        if len(anchors) >= 45:
            break
    return sorted(options.values(), key=lambda x: x["score"], reverse=True)[:limit]


def join_score(opening: dict, ending: dict, allow_generic: bool = False) -> tuple[float, set[str], set[str], set[str]] | None:
    duration = opening["end"] - opening["start"] + ending["end"] - ending["start"]
    if not 20 <= duration <= 60:
        return None
    shared_topics = opening["topics"] & ending["topics"]
    shared_names = opening["characters"] & ending["characters"]
    if not shared_names:
        return None
    gap = ending["episode"] - opening["episode"]
    if gap != 1:
        return None
    # Reject nearly repeated dialogue: a promo should escalate, not replay.
    opener_words = set().union(*(event_words(e) for e in opening["events"] if e.kind in {"DIALOGUE", "ACTION"}))
    closer_words = set().union(*(event_words(e) for e in ending["events"] if e.kind in {"DIALOGUE", "ACTION"}))
    shared_words = (opener_words & closer_words) - GENERIC_JOIN_STOP - shared_names
    strong_match = bool(shared_topics & STRONG_TOPICS)
    if not strong_match and (not allow_generic or len(shared_words) < 2):
        return None
    overlap = len(opener_words & closer_words) / max(1, len(opener_words | closer_words))
    if overlap >= 0.48:
        return None
    continuity = 1.5 * len(shared_topics & STRONG_TOPICS) + 0.7 * min(2, len(shared_names))
    continuity += 1.0
    continuity += 0.3 * min(3, len(shared_topics))
    if not strong_match:
        continuity += 0.35 * min(3, len(shared_words)) - 1.5
    score = opening["score"] + ending["score"] + continuity - 2.0 * overlap
    return round(score, 3), shared_topics, shared_names, shared_words


def record(opening: dict, ending: dict, score: float, shared_topics: set[str], shared_names: set[str], shared_words: set[str]) -> dict:
    segments = [
        {"episode": opening["episode"], "start": to_stamp(opening["start"]), "end": to_stamp(opening["end"])},
        {"episode": ending["episode"], "start": to_stamp(ending["start"]), "end": to_stamp(ending["end"])},
    ]
    strong_match = bool(shared_topics & STRONG_TOPICS)
    topic = next((name for name in TOPIC_PRIORITY if name in shared_topics & STRONG_TOPICS), "recurring story terms")
    character = sorted(shared_names)[0]
    hook_excerpt = opening["cue"].text.strip().replace("\n", " ")[:92]
    cliff_excerpt = ending["cue"].text.strip().replace("\n", " ")[:92]
    return {
        "segments": segments,
        "duration_seconds": round((opening["end"] - opening["start"]) + (ending["end"] - ending["start"]), 2),
        "hook": opening["cue"].text,
        "hook_event_id": opening["cue"].id,
        "hook_episode": opening["episode"],
        "hook_time": to_stamp(opening["cue"].start),
        "cliffhanger": ending["cue"].text,
        "cliffhanger_event_id": ending["cue"].id,
        "cliffhanger_episode": ending["episode"],
        "cliffhanger_time": to_stamp(ending["cue"].start),
        "why_it_works": f"The opening cues '{hook_excerpt}' and the ending cues '{cliff_excerpt}'; {character.title()} and {('the ' + topic + ' topic') if strong_match else 'repeated story terms'} link the segments provisionally.",
        "post_first_reason": f"Ranked by immediate opening cue and pending ending cue (automatic score {score}); verify the causal join before posting.",
        "join_rationale": (f"Shared character: {character.title()}; shared story topic: {topic}; chronological gap: 1 episode." if strong_match else f"Fallback: shared speaker {character.title()} and repeated terms {', '.join(sorted(shared_words)[:4])}; story join unverified."),
        "join_basis": "named_speaker_and_topic" if strong_match else "generic_word_fallback",
        "scores": {"opening": opening["score"], "ending": ending["score"], "continuity_and_total": score},
        "hook_timing_uncertain": opening["timing_uncertain"],
        "review_note": "Script-only stitched proposal. Check each source cut and the cross-episode join in footage and with a cold viewer.",
    }


def select(rows: list[dict], top: int) -> list[dict]:
    selected = []
    pair_counts: dict[tuple[int, int], int] = {}
    opening_counts: dict[int, int] = {}
    ending_counts: dict[int, int] = {}
    for row in sorted(rows, key=lambda r: r["scores"]["continuity_and_total"], reverse=True):
        if row["hook_timing_uncertain"]:
            continue
        pair = tuple(segment["episode"] for segment in row["segments"])
        if pair_counts.get(pair, 0) >= 1:
            continue
        if opening_counts.get(pair[0], 0) >= 1 or ending_counts.get(pair[1], 0) >= 1:
            continue
        # Avoid showing the same source interval repeatedly with minor shifts.
        redundant = False
        for prior in selected:
            if pair != tuple(segment["episode"] for segment in prior["segments"]):
                continue
            overlaps = []
            for a, b in zip(row["segments"], prior["segments"]):
                sa, ea, sb, eb = to_seconds(a["start"]), to_seconds(a["end"]), to_seconds(b["start"]), to_seconds(b["end"])
                overlaps.append(max(0.0, min(ea, eb) - max(sa, sb)) / max(0.01, min(ea - sa, eb - sb)))
            if min(overlaps) >= 0.65:
                redundant = True
                break
        if redundant:
            continue
        selected.append(dict(row))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        opening_counts[pair[0]] = opening_counts.get(pair[0], 0) + 1
        ending_counts[pair[1]] = ending_counts.get(pair[1], 0) + 1
        if len(selected) >= top:
            break
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
    return selected


def write_outputs(output_dir: Path, name: str, rows: list[dict]) -> None:
    (output_dir / f"{name}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "segments_in_order", "duration_seconds", "hook", "cliffhanger", "why_it_works", "post_first_reason", "score"])
        for row in rows:
            path = " + ".join(f"Ep {s['episode']} {s['start']}-{s['end']}" for s in row["segments"])
            writer.writerow([row.get("rank", ""), path, row["duration_seconds"], row["hook"], row["cliffhanger"], row["why_it_works"], row["post_first_reason"], row["scores"]["continuity_and_total"]])


def build_composites(episodes: list[Episode], output_dir: Path, top: int = 20) -> dict:
    source = json.loads((output_dir / "narrative_all_candidates.json").read_text(encoding="utf-8"))
    by_episode: dict[int, list[dict]] = {}
    for row in source:
        by_episode.setdefault(row["episode"], []).append(row)
    openers = {}
    closers = {}
    for episode in episodes:
        rows = sorted(by_episode.get(episode.number, []), key=lambda x: x["scores"]["total"], reverse=True)
        openers[episode.number] = micro_options(episode, rows, "opening")
        closers[episode.number] = micro_options(episode, rows, "ending")
    def propose(allow_generic: bool) -> dict[tuple, dict]:
        proposals: dict[tuple, dict] = {}
        for episode in episodes:
            later_number = episode.number + 1
            if later_number not in closers:
                continue
            for opening in openers[episode.number]:
                for ending in closers[later_number]:
                    joined = join_score(opening, ending, allow_generic)
                    if joined is None:
                        continue
                    score, shared_topics, shared_names, shared_words = joined
                    row = record(opening, ending, score, shared_topics, shared_names, shared_words)
                    key = tuple((s["episode"], s["start"], s["end"]) for s in row["segments"])
                    proposals[key] = row
        return proposals

    proposals = propose(allow_generic=False)
    fallback_used = not proposals
    if fallback_used:
        proposals = propose(allow_generic=True)
    all_rows = sorted(proposals.values(), key=lambda r: r["scores"]["continuity_and_total"], reverse=True)
    ranked = select(all_rows, top)
    write_outputs(output_dir, "cross_episode_all_candidates", all_rows)
    write_outputs(output_dir, "cross_episode_ranked", ranked)
    audit = {
        "method": "two chronological, source-timed segments from different episodes; lexical topic/character continuity; script-only review queue",
        "candidate_count": len(all_rows),
        "ranked_count": len(ranked),
        "generic_fallback_used": fallback_used,
        "abstention_reason": "No adjacent episode pair passed named-speaker and topic/word-continuity gates." if not ranked else "",
        "episodes_in_ranked": sorted({s["episode"] for row in ranked for s in row["segments"]}),
        "episode_pairs_in_ranked": [list(dict.fromkeys(s["episode"] for s in row["segments"])) for row in ranked],
        "limitations": "No video join or true cold-viewer comprehension is checked automatically. Names/topics are proxies.",
    }
    (output_dir / "cross_episode_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit
