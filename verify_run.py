"""Check source-grounded invariants of a full promo experiment run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from promo_finder import dialogue_cut, parse_zip, to_seconds


FILES = [
    "baseline_ranked.json", "narrative_ranked.json", "narrative_proxy_pass_ranked.json",
    "baseline_episode_best.json", "narrative_episode_best.json",
    "shared_pool_baseline_ranked.json", "shared_pool_narrative_ranked.json",
    "baseline_all_candidates.json", "narrative_all_candidates.json",
]


def verify(zip_path: Path, output_dir: Path) -> dict:
    episodes = {ep.number: ep for ep in parse_zip(zip_path)}
    checked = 0
    failures = []
    for filename in FILES:
        rows = json.loads((output_dir / filename).read_text(encoding="utf-8"))
        for index, row in enumerate(rows, 1):
            segment = row["segments"][0]
            episode = episodes.get(row["episode"])
            if episode is None:
                failures.append(f"{filename} row {index}: unknown episode")
                continue
            start, end = to_seconds(segment["start"]), to_seconds(segment["end"])
            if not 0 <= start < end <= episode.duration + 0.01 or not 20 <= end - start <= 60:
                failures.append(f"{filename} row {index}: invalid interval")
            if dialogue_cut(episode, start) or dialogue_cut(episode, end):
                failures.append(f"{filename} row {index}: mid-dialogue cut")
            by_id = {event.id: event for event in episode.events}
            for field in ("hook_event_id", "cliffhanger_event_id"):
                event = by_id.get(row.get(field))
                if event is None:
                    failures.append(f"{filename} row {index}: missing {field}")
                    continue
                if event.text != row["hook" if field == "hook_event_id" else "cliffhanger"]:
                    failures.append(f"{filename} row {index}: quoted event mismatch")
                if filename.startswith("narrative") or filename == "shared_pool_narrative_ranked.json":
                    if event.start < start - 0.01 or event.end > end + 0.01:
                        failures.append(f"{filename} row {index}: evidence crosses interval")
                    if field == "hook_event_id" and event.end > start + 3.01 and not row.get("hook_timing_uncertain"):
                        failures.append(f"{filename} row {index}: hook evidence after first three seconds")
            checked += 1
    for filename in ("cross_episode_ranked.json", "cross_episode_all_candidates.json"):
        rows = json.loads((output_dir / filename).read_text(encoding="utf-8"))
        for index, row in enumerate(rows, 1):
            segments = row.get("segments", [])
            if len(segments) != 2 or segments[0]["episode"] >= segments[1]["episode"]:
                failures.append(f"{filename} row {index}: segments are not an ordered cross-episode pair")
                continue
            total = 0.0
            for segment in segments:
                episode = episodes.get(segment["episode"])
                if episode is None:
                    failures.append(f"{filename} row {index}: unknown segment episode")
                    continue
                start, end = to_seconds(segment["start"]), to_seconds(segment["end"])
                total += end - start
                if not 0 <= start < end <= episode.duration + 0.01 or dialogue_cut(episode, start) or dialogue_cut(episode, end):
                    failures.append(f"{filename} row {index}: invalid source segment or mid-dialogue cut")
            if not 20 <= total <= 60 or abs(total - row["duration_seconds"]) > 0.02:
                failures.append(f"{filename} row {index}: combined duration mismatch")
            for field, segment in (("hook", segments[0]), ("cliffhanger", segments[-1])):
                episode = episodes[segment["episode"]]
                event_id = row[field + "_event_id"]
                event = next((e for e in episode.events if e.id == event_id), None)
                start, end = to_seconds(segment["start"]), to_seconds(segment["end"])
                if event is None or event.text != row[field] or event.start < start - 0.01 or event.end > end + 0.01:
                    failures.append(f"{filename} row {index}: {field} evidence not wholly inside its source segment")
                elif field == "hook" and not row.get("hook_timing_uncertain") and event.end > start + 3.01:
                    failures.append(f"{filename} row {index}: hook evidence exceeds first three assembled seconds")
            if not row.get("why_it_works") or not row.get("post_first_reason"):
                failures.append(f"{filename} row {index}: missing explanation")
            checked += 1
    result = {"checked_rows": checked, "failures": failures, "status": "pass" if not failures else "fail"}
    (output_dir / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = verify(args.zip_path, args.output_dir)
    print(json.dumps({"checked_rows": result["checked_rows"], "failure_count": len(result["failures"]), "status": result["status"]}, indent=2))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
