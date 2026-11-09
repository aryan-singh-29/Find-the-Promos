"""Offline, script-only promo experiment.

Runs two explicitly limited methods on the same timestamped screenplay pack:
  * hive_lite: a script-only HIVE-inspired surrogate (highlight -> boundaries)
  * narrative: adds turning-point discovery, clip-contained evidence, and scored
    hook/context/unresolved-consequence proxies for human review.

Neither method reproduces HIVE's multimodal LLM or predicts ad performance.
Usage: python promo_pipeline.py scripts.zip --out comparison
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from promo_finder import (
    Episode,
    Event,
    RESOLUTION,
    candidates_for_episode,
    cliff_score,
    dialogue_cut,
    evidence_score,
    parse_zip,
    score_window,
    to_seconds,
    to_stamp,
)


TURN = re.compile(
    r"\b(?:but|instead|actually|truth|real|true|lie|lied|father|mother|sister|"
    r"husband|wife|marry|married|duke|heir|poison|witch|betray|traitor|"
    r"forgive|confess|secret|prove|witness|arrest|hang|execute|kill|"
    r"choose|unless|if you|not your|i am|i'm|you are|you're|never)\b",
    re.I,
)
REVEAL = re.compile(
    r"\b(?:poison|witch|traitor|betray|murder|kill|dead|duke|heir|"
    r"secret|imposter|arrest|hang|gallows|pregnan|stole|thief|"
    r"real (?:father|mother|husband|wife)|not your (?:father|mother|husband|wife))\w*\b",
    re.I,
)
PENDING = re.compile(
    r"\b(?:will|would|could|unless|if you|if she|if he|before|"
    r"or else|choose|decide|about to|going to|answer me|tell me)\b",
    re.I,
)
PENDING_ACTION = re.compile(
    r"\b(?:aims? (?:the )?(?:gun|pistol|rifle|knife|sword)|"
    r"raises? (?:the |a )?(?:gun|pistol|rifle|knife|sword|blade)|"
    r"reaches? for (?:the |a )?(?:gun|knife|blade)|"
    r"about to (?:strike|shoot|stab|kill)|lunges? (?:at|toward)|"
    r"bursts? (?:through|in)|guards? (?:rush|advance))\b",
    re.I,
)
CAUSE = re.compile(
    r"\b(?:because|so that|if|unless|before|or else|for|to save|"
    r"in order|which means|then|therefore|only if|as long as)\b",
    re.I,
)
STAKE = re.compile(
    r"\b(?:die|dead|death|kill|hang|gallows|execute|arrest|prison|"
    r"lose|lost|destroy|burn|poison|marry|wedding|husband|wife|"
    r"daughter|son|father|mother|sister|brother|home|inherit|"
    r"banish|exile|trial|traitor|witch)\b",
    re.I,
)
HIGH_ACTION = re.compile(
    r"\b(?:slap|strik|hits?|stab|shoot|fires?|grabs?|drags?|"
    r"collaps|falls?|bursts?|shatter|breaks?|kneels?|aims?|"
    r"seiz|poison|flam|burn|chok|push|pull|lung|scream)\w*\b",
    re.I,
)
POINTLESS = re.compile(
    r"\b(?:smiles?|looks?|gazes?|glances?|turns?|stands?|"
    r"walks?|pauses?|nods?|sits?|breathes?|stares?)\b",
    re.I,
)
REFERENCE_OPEN = re.compile(
    r"^(?:[\s\"'….-]*)(?:then|so|and|but|he|she|they|it|this|"
    r"that|him|her|them|there|his|hers|theirs)\b",
    re.I,
)
WORD = re.compile(r"[a-z]{3,}", re.I)
STOP = {"the", "and", "you", "your", "her", "him", "his", "she", "that", "this", "with", "for", "are", "was", "were", "have", "from", "they", "them", "not", "but", "will", "what", "who", "how", "why"}


def complete_events(episode: Episode, start: float, end: float) -> list[Event]:
    """Only evidence wholly within the interval may justify a selection."""
    return [e for e in episode.events if e.start >= start - 0.01 and e.end <= end + 0.01]


def event_words(event: Event) -> set[str]:
    return {w.lower() for w in WORD.findall(event.text) if w.lower() not in STOP}


def hook_strength(event: Event) -> float:
    text = event.text
    if event.kind == "SOUND FX":
        return 0.0
    strength = 0.4 if event.kind == "CAMERA" else 1.0
    if event.kind == "ACTION":
        strength += 1.4 if HIGH_ACTION.search(text) else 0.1
    if event.kind == "DIALOGUE":
        strength += 0.45 if ("?" in text or "!" in text) else 0.0
        strength += 0.9 if STAKE.search(text) else 0.0
        strength += 1.1 if REVEAL.search(text) else 0.0
        strength += 0.6 if PENDING.search(text) else 0.0
    if event.kind == "CAMERA" and HIGH_ACTION.search(text):
        strength += 0.8
    if REFERENCE_OPEN.search(text):
        strength -= 1.2
    if event.kind in {"ACTION", "CAMERA"} and POINTLESS.search(text) and not HIGH_ACTION.search(text):
        strength -= 0.7
    if len(text.strip()) < 12 and not HIGH_ACTION.search(text):
        strength -= 0.5
    words = WORD.findall(text)
    if event.kind == "DIALOGUE" and len(words) <= 5 and not REVEAL.search(text):
        strength = min(strength, 1.5)
    if event.kind == "DIALOGUE" and len(words) <= 2:
        strength = min(strength, 1.2)
    return max(0.0, min(5.0, strength))


def cliff_strength(event: Event) -> float:
    text = event.text
    if event.kind == "SOUND FX":
        return 0.0
    strength = 0.2 if event.kind == "CAMERA" else 0.6
    strength += 1.2 if "?" in text else 0.0
    strength += 1.0 if STAKE.search(text) else 0.0
    strength += 1.0 if PENDING.search(text) else 0.0
    strength += 0.8 if REVEAL.search(text) and "?" in text else 0.0
    strength += 1.2 if event.kind == "ACTION" and PENDING_ACTION.search(text) else 0.0
    strength -= 0.8 if event.kind == "CAMERA" and not HIGH_ACTION.search(text) else 0.0
    strength -= 0.6 if POINTLESS.search(text) and not (STAKE.search(text) or HIGH_ACTION.search(text)) else 0.0
    return max(0.0, min(5.0, strength))


def narrative_score(episode: Episode, start: float, end: float) -> dict | None:
    duration = end - start
    if not 20 <= duration <= 60 or dialogue_cut(episode, start) or dialogue_cut(episode, end):
        return None
    events = complete_events(episode, start, end)
    first = [e for e in events if e.start < start + 3.01 and e.end <= start + 3.01]
    last = [e for e in events if e.end >= end - 2.01 and e.end <= end + 0.01]
    first = [e for e in first if e.kind != "SOUND FX"]
    last = [e for e in last if e.kind != "SOUND FX"]
    timing_uncertain = False
    if not first:
        first = [e for e in events if e.start < start + 3.01 and e.kind != "SOUND FX"]
        timing_uncertain = True
    if not first or not last:
        return None
    hook = max(first, key=hook_strength)
    cliff = max(last, key=cliff_strength)
    hs = hook_strength(hook)
    cs = cliff_strength(cliff)
    if hs < 0.7 or cs < 0.8:
        return None
    spoken = [e for e in events if e.kind == "DIALOGUE"]
    actions = [e for e in events if e.kind == "ACTION"]
    early = [e for e in events if e.start < start + 12]
    speakers = {e.speaker for e in spoken}
    early_text = " ".join(e.text for e in early if e.kind != "CAMERA")
    clip_text = " ".join(e.text for e in spoken + actions)
    context = 0.5
    context += 0.7 if len(speakers) >= 2 else 0.0
    context += 0.6 if STAKE.search(early_text) else 0.0
    context += 0.45 if TURN.search(early_text) else 0.0
    context += 0.35 if CAUSE.search(clip_text) else 0.0
    context += 0.3 if len(spoken) >= 3 else 0.0
    context += 0.25 if len(actions) >= 1 else 0.0
    context -= 0.6 if REFERENCE_OPEN.search(hook.text) else 0.0
    context -= 0.45 if len({e.scene for e in events}) > 1 else 0.0
    context = max(0.0, min(3.0, context))
    # Repetition is not an escalation: compare final beat against earlier beats.
    final_words = event_words(cliff)
    earlier = [e for e in events if e.end < cliff.start - 3 and e.kind in {"DIALOGUE", "ACTION"}]
    similarities = [len(final_words & event_words(e)) / max(1, len(final_words | event_words(e))) for e in earlier]
    repeat = max(similarities, default=0.0)
    novelty = max(0.0, 1.0 - 1.8 * repeat)
    open_loop = cs >= 2.15 and (
        "?" in cliff.text or bool(PENDING.search(cliff.text)) or bool(PENDING_ACTION.search(cliff.text))
    ) and not bool(RESOLUTION.search(cliff.text))
    causal_setup = bool(STAKE.search(clip_text) and (CAUSE.search(clip_text) or len(spoken) >= 3))
    # These are conservative proxies, never a substitute for a cold viewer.
    gate_proxy = {
        "source_and_duration": True,
        "hook_first_3_seconds": hs >= 2.45 and not timing_uncertain,
        "standalone_context": context >= 1.7,
        "unresolved_cut": open_loop and novelty >= 0.35,
    }
    # Distinct story beats are favored, but the score is only a sorting aid.
    turning = min(2.0, 0.4 * sum(bool(REVEAL.search(e.text)) for e in events if e.kind == "DIALOGUE"))
    escalation = 0.4 if causal_setup else 0.0
    length = max(0.0, 1.0 - abs(duration - 38) / 22)
    failed_proxy_count = sum(not passed for passed in gate_proxy.values())
    score = round(3.2 * hs + 2.9 * cs + 1.8 * context + 1.2 * novelty + turning + escalation + 0.4 * length - 3.5 * failed_proxy_count, 3)
    return {
        "episode": episode.number,
        "segments": [{"episode": episode.number, "start": to_stamp(start), "end": to_stamp(end)}],
        "duration_seconds": round(duration, 2),
        "hook": hook.text,
        "hook_time": to_stamp(hook.start),
        "hook_event_id": hook.id,
        "cliffhanger": cliff.text,
        "cliffhanger_time": to_stamp(cliff.start),
        "cliffhanger_event_id": cliff.id,
        "why_it_works": f"Opens on {hook.text[:90]!r} and cuts at {cliff.text[:90]!r}; an editor must confirm the open loop.",
        "scores": {"hook_proxy": round(hs, 2), "cliff_proxy": round(cs, 2), "context_proxy": round(context, 2), "novelty_proxy": round(novelty, 2), "turning_proxy": round(turning, 2), "total": score},
        "gate_proxies": gate_proxy,
        "hook_timing_uncertain": timing_uncertain,
        "review_note": "Script-only proxy gates; a cold viewer and source footage must confirm the hook, standalone meaning, and cut.",
    }


def narrative_candidates(episode: Episode) -> tuple[list[dict], int]:
    """Find timed candidate openings and endings, then score legal pairs."""
    starts = [
        (hook_strength(e), e.start)
        for e in episode.events
        if e.kind in {"DIALOGUE", "ACTION", "CAMERA"}
        and hook_strength(e) >= 0.8
    ]
    ends = [
        (cliff_strength(e), e.end)
        for e in episode.events
        if e.kind in {"DIALOGUE", "ACTION", "CAMERA"} and cliff_strength(e) >= 0.8
    ]
    # Include episode edges, but do not make them automatic winners.
    start_scores: dict[float, float] = {}
    end_scores: dict[float, float] = {}
    for strength, time in starts:
        start_scores[time] = max(strength, start_scores.get(time, 0.0))
    for strength, time in ends:
        end_scores[time] = max(strength, end_scores.get(time, 0.0))
    starts = sorted(((score, time) for time, score in start_scores.items()), key=lambda pair: (-pair[0], pair[1]))[:44]
    ends = sorted(((score, time) for time, score in end_scores.items()), key=lambda pair: (-pair[0], pair[1]))[:44]
    if not any(time == 0.0 for _, time in starts):
        starts.append((0.0, 0.0))
    if not any(time == episode.duration for _, time in ends):
        ends.append((0.0, episode.duration))
    found: dict[tuple[float, float], dict] = {}
    legal_pairs = 0
    for _, start in starts:
        for _, end in ends:
            if not 20 <= end - start <= 60:
                continue
            legal_pairs += 1
            candidate = narrative_score(episode, start, end)
            if candidate:
                found[(start, end)] = candidate
    return sorted(found.values(), key=lambda row: row["scores"]["total"], reverse=True), legal_pairs


def hive_lite_candidates(episode: Episode) -> list[dict]:
    """A documented text-only adaptation, not the multimodal HIVE model.

    Stage 1: rank dramatic highlight events within the episode.
    Stage 2: use the existing lexical opening/ending search around those events.
    Stage 3: use continuous prefix/suffix trimming only; interior pruning is omitted.
    """
    salient = [e for e in episode.events if e.kind in {"ACTION", "DIALOGUE"}]
    highlights = sorted(salient, key=evidence_score, reverse=True)[:12]
    windows = candidates_for_episode(episode)
    output = []
    for row in windows:
        start = to_seconds(row["segments"][0]["start"])
        end = to_seconds(row["segments"][0]["end"])
        included = [e for e in highlights if e.start >= start - 0.01 and e.end <= end + 0.01]
        if not included:
            continue
        best = max(included, key=evidence_score)
        row = dict(row)
        row["method"] = "hive_lite_script_surrogate"
        row["highlight_event_id"] = best.id
        row["scores"] = dict(row["scores"])
        row["scores"]["total"] = round(row["scores"]["total"] + 0.6 * evidence_score(best), 3)
        output.append(row)
    return sorted(output, key=lambda row: row["scores"]["total"], reverse=True)


def story_tokens(row: dict) -> set[str]:
    return {w.lower() for w in WORD.findall(row["hook"] + " " + row["cliffhanger"]) if w.lower() not in STOP}


def select_slate(rows: list[dict], limit: int, cross_episode_dedup: bool) -> list[dict]:
    selected: list[dict] = []
    episode_counts: dict[int, int] = {}
    for row in sorted(rows, key=lambda r: r["scores"]["total"], reverse=True):
        episode = row["episode"]
        if episode_counts.get(episode, 0) >= 2:
            continue
        s, e = (to_seconds(row["segments"][0][key]) for key in ("start", "end"))
        duplicate = False
        for prior in selected:
            ps, pe = (to_seconds(prior["segments"][0][key]) for key in ("start", "end"))
            overlap = max(0, min(e, pe) - max(s, ps)) / max(0.01, min(e - s, pe - ps)) if episode == prior["episode"] else 0.0
            if overlap >= 0.5:
                duplicate = True
                break
            if cross_episode_dedup and episode != prior["episode"]:
                a, b = story_tokens(row), story_tokens(prior)
                if len(a & b) >= 4 and len(a & b) / max(1, len(a | b)) >= 0.65:
                    duplicate = True
                    break
        if duplicate:
            continue
        selected.append(dict(row))
        episode_counts[episode] = episode_counts.get(episode, 0) + 1
        if len(selected) >= limit:
            break
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
    return selected


def output_rows(directory: Path, name: str, rows: list[dict]) -> None:
    (directory / f"{name}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "episode", "start", "end", "duration_seconds", "hook", "cliffhanger", "score", "why_it_works"])
        for row in rows:
            segment = row["segments"][0]
            writer.writerow([row.get("rank", ""), row["episode"], segment["start"], segment["end"], row["duration_seconds"], row["hook"], row["cliffhanger"], row["scores"]["total"], row["why_it_works"]])


def run(input_zip: Path, output_dir: Path, top: int) -> None:
    episodes = parse_zip(input_zip)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_all: list[dict] = []
    narrative_all: list[dict] = []
    baseline_best: list[dict] = []
    narrative_best: list[dict] = []
    coverage = []
    for episode in episodes:
        baseline = hive_lite_candidates(episode)
        narrative, legal_pairs = narrative_candidates(episode)
        # Reconsider the baseline's own shortlist with the narrative rubric.
        # This makes it possible to improve boundaries without losing recall
        # solely because a different first-stage retriever missed the beat.
        seen = {(row["segments"][0]["start"], row["segments"][0]["end"]) for row in narrative}
        for base_row in baseline[:50]:
            segment = base_row["segments"][0]
            key = (segment["start"], segment["end"])
            if key in seen:
                continue
            reranked = narrative_score(episode, to_seconds(key[0]), to_seconds(key[1]))
            if reranked:
                narrative.append(reranked)
                seen.add(key)
        narrative.sort(key=lambda row: row["scores"]["total"], reverse=True)
        baseline_all.extend(baseline)
        narrative_all.extend(narrative)
        if baseline:
            baseline_best.append(baseline[0])
        if narrative:
            narrative_best.append(narrative[0])
        coverage.append({"episode": episode.number, "baseline_candidates": len(baseline), "narrative_candidates": len(narrative), "legal_narrative_pairs": legal_pairs})
        print(f"episode {episode.number:02d}: baseline {len(baseline)}, narrative {len(narrative)}", flush=True)
    # Identical diversification is applied to both methods in the comparison.
    baseline_slate = select_slate(baseline_all, top, cross_episode_dedup=True)
    narrative_slate = select_slate(narrative_all, top, cross_episode_dedup=True)
    narrative_proxy_pass_slate = select_slate(
        [row for row in narrative_all if all(row["gate_proxies"].values())],
        top,
        cross_episode_dedup=True,
    )
    # Conditional ranking ablation: same intervals, different scoring. This
    # isolates ranking from candidate generation, but only on shared survivors.
    shared_baseline: list[dict] = []
    shared_narrative: list[dict] = []
    all_intervals = {
        (row["episode"], row["segments"][0]["start"], row["segments"][0]["end"])
        for row in baseline_all + narrative_all
    }
    by_episode = {episode.number: episode for episode in episodes}
    for episode_number, start_stamp, end_stamp in sorted(all_intervals):
        episode = by_episode[episode_number]
        start, end = to_seconds(start_stamp), to_seconds(end_stamp)
        base = score_window(episode, start, end)
        new = narrative_score(episode, start, end)
        if base and new:
            shared_baseline.append(base)
            shared_narrative.append(new)
    shared_baseline_slate = select_slate(shared_baseline, top, cross_episode_dedup=True)
    shared_narrative_slate = select_slate(shared_narrative, top, cross_episode_dedup=True)
    output_rows(output_dir, "baseline_ranked", baseline_slate)
    output_rows(output_dir, "narrative_ranked", narrative_slate)
    output_rows(output_dir, "narrative_proxy_pass_ranked", narrative_proxy_pass_slate)
    output_rows(output_dir, "shared_pool_baseline_ranked", shared_baseline_slate)
    output_rows(output_dir, "shared_pool_narrative_ranked", shared_narrative_slate)
    output_rows(output_dir, "baseline_episode_best", baseline_best)
    output_rows(output_dir, "narrative_episode_best", narrative_best)
    # Keep unedited candidate pools, including rejected alternatives available to the editor.
    output_rows(output_dir, "baseline_all_candidates", baseline_all)
    output_rows(output_dir, "narrative_all_candidates", narrative_all)
    from composite_promos import build_composites
    cross_episode_audit = build_composites(episodes, output_dir, top)
    report = {
        "input_zip": input_zip.name,
        "input_sha256": hashlib.sha256(input_zip.read_bytes()).hexdigest(),
        "source_format": "episode-NNN UTF-8 .txt with SCENE, Scene end, and timed event lines (parser v1)",
        "episodes": len(episodes),
        "scenes": sum(ep.scene_count for ep in episodes),
        "timestamped_events": sum(len(ep.events) for ep in episodes),
        "baseline_method": "HIVE-inspired script-only surrogate; lexical highlights -> opening/ending; no multimodal LLM or interior pruning",
        "narrative_method": "adds turning-point search plus source-timed hook, causal-context and unresolved-cut scoring proxies; uncertain timing is flagged",
        "evaluation_status": "automated outputs only; independent blind review required before any quality claim",
        "baseline_candidate_count": len(baseline_all),
        "narrative_candidate_count": len(narrative_all),
        "shared_pool_candidate_count": len(shared_baseline),
        "baseline_covered_episodes": len(baseline_best),
        "narrative_covered_episodes": len(narrative_best),
        "narrative_proxy_pass_ranked_count": len(narrative_proxy_pass_slate),
        "cross_episode_candidate_count": cross_episode_audit["candidate_count"],
        "cross_episode_ranked_count": cross_episode_audit["ranked_count"],
        "cross_episode_generic_fallback_used": cross_episode_audit["generic_fallback_used"],
        "coverage": coverage,
    }
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "coverage"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--out", type=Path, default=Path("comparison"))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    run(args.zip_path, args.out, args.top)


if __name__ == "__main__":
    main()
