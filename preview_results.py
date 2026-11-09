"""Generate an offline HTML editor view from a screenplay run.

Usage: python preview_results.py scripts.zip comparison --html comparison/promo_review.html
Only source events wholly inside a proposed interval are quoted. No network calls.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

from promo_finder import parse_zip, to_seconds, to_stamp
from promo_pipeline import complete_events


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def event_line(event, offset: float, assembled_shift: float = 0.0) -> str:
    label = event.speaker if event.kind == "DIALOGUE" else event.kind
    return (
        f"<li><span class='time'>+{esc(to_stamp(assembled_shift+event.start-offset))}"
        f" to +{esc(to_stamp(assembled_shift+event.end-offset))}</span> "
        f"<strong>{esc(label)}</strong>: {esc(event.text)}</li>"
    )


def safe_cue(row: dict, episode, event_id: str, start: float, end: float) -> tuple[str, str]:
    event = next((e for e in episode.events if e.id == row.get(event_id)), None)
    if event is None:
        return "No matching source event", "unverified"
    if event.start < start - 0.01 or event.end > end + 0.01:
        return "Source event crosses the proposed clip boundary; inspect footage", "uncertain"
    if event_id == "hook_event_id" and event.end > start + 3.01:
        return event.text, "first-three-second timing uncertain"
    return event.text, "timed inside clip"


def card(row: dict, episode, method: str) -> str:
    segment = row["segments"][0]
    start, end = to_seconds(segment["start"]), to_seconds(segment["end"])
    events = sorted(complete_events(episode, start, end), key=lambda e: (e.start, e.end, e.id))
    first = [e for e in events if e.end <= start + 3.01]
    crossing = sum(e.start < end and e.end > start and e not in events for e in episode.events)
    hook, hook_note = safe_cue(row, episode, "hook_event_id", start, end)
    cliff, cliff_note = safe_cue(row, episode, "cliffhanger_event_id", start, end)
    gates = row.get("gate_proxies")
    if gates:
        failed = [name.replace("_", " ") for name, passed in gates.items() if not passed]
        status = "All automated proxies pass; footage review still required" if not failed else "Review flags: " + ", ".join(failed)
    else:
        status = "Lexical baseline; no narrative proxy checks"
    first_html = "".join(event_line(e, start) for e in first) or "<li>No whole timed event proves the first-three-second content. Check footage.</li>"
    script_html = "".join(event_line(e, start) for e in events)
    return f"""
    <article class="card" data-method="{esc(method)}" data-episodes="{esc(row['episode'])}">
      <div class="card-head"><span class="rank">#{esc(row.get('rank', '?'))}</span>
      <h2>Episode {esc(row['episode'])}</h2><span class="method-badge">{esc('Narrative' if method == 'narrative' else 'Lexical baseline')}</span><span class="interval">{esc(segment['start'])} to {esc(segment['end'])} ({esc(row['duration_seconds'])}s)</span></div>
      <p class="status">{esc(status)}</p>
      <div class="two"><div><h3>Opening cue</h3><p>{esc(hook)}</p><small>{esc(hook_note)}</small></div>
      <div><h3>Ending cue</h3><p>{esc(cliff)}</p><small>{esc(cliff_note)}</small></div></div>
      <p class="rationale"><strong>System rationale:</strong> {esc(row['why_it_works'])}</p>
      <p class="meta">Sorting score {esc(row['scores']['total'])}; {crossing} boundary-crossing source descriptions withheld.</p>
      <details><summary>Inspect timed script for this clip</summary>
        <h3>First three seconds - whole timed events only</h3><ol>{first_html}</ol>
        <h3>Full interval - whole timed events only</h3><ol>{script_html}</ol>
      </details>
    </article>"""


def composite_card(row: dict, episodes: dict[int, object], method: str = "composite") -> str:
    segments = row["segments"]
    first_segment = segments[0]
    last_segment = segments[-1]
    first_episode = episodes[first_segment["episode"]]
    last_episode = episodes[last_segment["episode"]]
    first_start, first_end = to_seconds(first_segment["start"]), to_seconds(first_segment["end"])
    last_start, last_end = to_seconds(last_segment["start"]), to_seconds(last_segment["end"])
    hook, hook_note = safe_cue(row, first_episode, "hook_event_id", first_start, first_end)
    cliff, cliff_note = safe_cue(row, last_episode, "cliffhanger_event_id", last_start, last_end)
    first_events = complete_events(first_episode, first_start, first_end)
    first_three = [e for e in first_events if e.end <= first_start + 3.01]
    first_html = "".join(event_line(e, first_start) for e in first_three) or "<li>No whole timed event proves the first-three-second content. Check footage.</li>"
    segment_lines = []
    script_sections = []
    cumulative = 0.0
    crossing_total = 0
    for index, segment in enumerate(segments, 1):
        episode = episodes[segment["episode"]]
        start, end = to_seconds(segment["start"]), to_seconds(segment["end"])
        events = sorted(complete_events(episode, start, end), key=lambda e: (e.start, e.end, e.id))
        crossing = sum(e.start < end and e.end > start and e not in events for e in episode.events)
        crossing_total += crossing
        segment_lines.append(
            f"<li><strong>Segment {index}:</strong> Episode {esc(segment['episode'])}, "
            f"{esc(segment['start'])} to {esc(segment['end'])} "
            f"({end-start:.1f}s; assembled +{esc(to_stamp(cumulative))} to +{esc(to_stamp(cumulative+end-start))})</li>"
        )
        script_sections.append(
            f"<h3>Segment {index} - episode {esc(segment['episode'])} "
            f"({esc(segment['start'])} to {esc(segment['end'])})</h3>"
            f"<p class='meta'>{crossing} boundary-crossing descriptions withheld.</p>"
            f"<ol>{''.join(event_line(e, start, cumulative) for e in events)}</ol>"
        )
        cumulative += end - start
        if index < len(segments):
            script_sections.append("<p class='join-marker'>JOIN: source episode changes here</p>")
    reviewed = method == "reviewed"
    provider = method == "provider"
    review_status = row.get("status", "automatic proposal").replace("_", " ")
    if reviewed:
        review_detail = f"<p class='status'><strong>{esc(review_status.title())}.</strong> {esc(row['editor_check'])}</p>"
    elif provider:
        score = row["provider_review"]
        review_detail = f"<p class='status'><strong>{esc(row['provider_name'])} provider, {esc(row['provider_model'])}.</strong> Hook {score['hook']}/5; context {score['context']}/5; JOIN {score['join']}/5; cliff {score['cliffhanger']}/5. {esc(score['failure'])}</p>"
    else:
        review_detail = "<p class='status'>Automatic chronological join; editor must verify story continuity and footage.</p>"
    title = "Reviewed cross-episode promo" if reviewed else ("Provider-ranked cross-episode proposal" if provider else "Automatic cross-episode proposal")
    rationale = row["provider_review"]["reason"] if provider else row["why_it_works"]
    rationale_label = "Provider reason" if provider else "Why it works"
    rank_reason = (f"Provider weighted score {row['provider_review']['weighted_total']}; original automatic rank {row.get('automatic_rank') or 'unranked'}." if provider else row["post_first_reason"])
    return f"""
    <article class="card composite" data-method="{esc(method)}" data-episodes="{esc(','.join(str(s['episode']) for s in segments))}">
      <div class="card-head"><span class="rank">#{esc(row.get('rank', '?'))}</span>
      <h2>{esc(title)}</h2><span class="method-badge">2 source segments</span>
      <span class="interval">{esc(row['duration_seconds'])}s total</span></div>
      <ol class="segments">{''.join(segment_lines)}</ol>
      {review_detail}
      <div class="two"><div><h3>Hook in first segment</h3><p>{esc(hook)}</p><small>{esc(hook_note)}</small></div>
      <div><h3>Cliffhanger in final segment</h3><p>{esc(cliff)}</p><small>{esc(cliff_note)}</small></div></div>
      <p class="rationale"><strong>{esc(rationale_label)}:</strong> {esc(rationale)}</p>
      <p class="rationale"><strong>Ranking reason:</strong> {esc(rank_reason)}</p>
      <p class="meta">Join evidence: {esc(row['join_rationale'])} Automatic score {esc(row['scores']['continuity_and_total'])}; {crossing_total} boundary-crossing descriptions withheld.</p>
      <details><summary>Inspect assembled script and join</summary>
        <h3>First three assembled seconds - whole timed events only</h3><ol>{first_html}</ol>
        {''.join(script_sections)}
      </details>
    </article>"""


def candidate_fingerprint(row: dict) -> tuple:
    return (tuple((s["episode"], s["start"], s["end"]) for s in row["segments"]), row["hook_event_id"], row["hook"], row["cliffhanger_event_id"], row["cliffhanger"])


def generate(zip_path: Path, output_dir: Path, html_path: Path, reviewed_path: Path | None = None, provider_path: Path | None = None) -> None:
    episodes = {episode.number: episode for episode in parse_zip(zip_path)}
    report = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    if report.get("input_sha256") and report["input_sha256"] != hashlib.sha256(zip_path.read_bytes()).hexdigest():
        raise ValueError("The screenplay ZIP does not match this run_report.json")
    baseline = json.loads((output_dir / "baseline_ranked.json").read_text(encoding="utf-8"))
    narrative = json.loads((output_dir / "narrative_ranked.json").read_text(encoding="utf-8"))
    cross_path = output_dir / "cross_episode_ranked.json"
    composite = json.loads(cross_path.read_text(encoding="utf-8")) if cross_path.exists() else []
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8")) if reviewed_path else []
    provider_rows = json.loads(provider_path.read_text(encoding="utf-8")) if provider_path else []
    if reviewed or provider_rows:
        candidate_pool = json.loads((output_dir / "cross_episode_all_candidates.json").read_text(encoding="utf-8"))
        fingerprints = {candidate_fingerprint(row) for row in candidate_pool}
        for label, rows in (("editorial", reviewed), ("provider", provider_rows)):
            if any(candidate_fingerprint(row) not in fingerprints for row in rows):
                raise ValueError(f"{label} overlay does not match exact source segments and cues in this run")
    cards = "\n".join(composite_card(row, episodes, "reviewed") for row in reviewed)
    cards += "\n" + "\n".join(composite_card(row, episodes, "provider") for row in provider_rows)
    cards += "\n" + "\n".join(composite_card(row, episodes) for row in composite)
    cards += "\n" + "\n".join(card(row, episodes[row["episode"]], method) for method, rows in (("narrative", narrative), ("baseline", baseline)) for row in rows)
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Promo review</title><style>
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#1b2b3d;background:#f4f7f8}*{box-sizing:border-box}
body{margin:0}header{background:#15283f;color:#fff;padding:1.4rem max(1rem,calc((100vw - 1100px)/2))}h1{margin:0 0 .25rem;font-size:1.75rem}header p{margin:.25rem 0;color:#d9e9ee}
main{max-width:1100px;margin:auto;padding:1rem}.notice{background:#fff6de;border-left:4px solid #c78b10;padding:.8rem 1rem;margin:.3rem 0 1rem}
.controls{display:flex;gap:.7rem;flex-wrap:wrap;align-items:end;margin-bottom:1rem}.controls label{display:grid;gap:.2rem;font-weight:600}.controls select,.controls input{font:inherit;padding:.5rem;border:1px solid #aebbc5;border-radius:.35rem;background:#fff}.controls input{min-width:16rem}
.card{background:#fff;border:1px solid #d6e0e6;border-radius:.65rem;padding:1rem;margin:0 0 1rem;box-shadow:0 2px 6px #14304d0d}.card-head{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}.card-head h2{font-size:1.15rem;margin:0}.rank{background:#087b83;color:#fff;border-radius:1rem;padding:.15rem .6rem;font-weight:700}.method-badge{background:#e3edf1;color:#315466;border-radius:1rem;padding:.18rem .55rem;font-size:.75rem;font-weight:700}.interval{margin-left:auto;color:#536678;font-variant-numeric:tabular-nums}.status{background:#edf5f4;padding:.5rem .7rem;border-radius:.35rem;font-size:.9rem}.segments{background:#f3f8f9;border-radius:.4rem;padding:.55rem .8rem .55rem 2rem}.segments li{margin:.2rem 0}.composite{border-left:4px solid #087b83}.two{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.two>div{border-left:3px solid #8fbfc0;padding-left:.7rem}.two h3,details h3{font-size:.9rem;margin:.2rem 0}.two p{margin:.2rem 0}.two small,.meta{color:#637280}.rationale{font-size:.9rem}.meta{font-size:.82rem}.join-marker{background:#15283f;color:#fff;text-align:center;padding:.45rem;font-weight:700;letter-spacing:.03em}details{border-top:1px solid #dde5e9;padding-top:.6rem}summary{cursor:pointer;color:#075e65;font-weight:700}ol{padding-left:1.4rem}li{margin:.3rem 0;line-height:1.35}.time{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#486575;font-size:.85em}footer{max-width:1100px;margin:1rem auto;padding:1rem;color:#637280;font-size:.85rem}
@media(max-width:700px){.two{grid-template-columns:1fr}.interval{margin-left:0}header{padding:1rem}}
</style></head><body><header><h1>Promo review queue</h1><p>Offline, source-grounded screenplay run</p></header><main>
<p class="notice"><strong>Editor check required.</strong> These are script suggestions, not video cuts or performance predictions. Open the actual footage to verify the first three seconds, standalone meaning, and final unresolved consequence. Descriptions crossing a cut are withheld below.</p>
<p><strong>Run summary:</strong> __EPISODES__ episodes parsed; __CROSSCOUNT__ automatic cross-episode proposals assembled from two ordered segments; __REVIEWCOUNT__ editorial shortlist entries; __PROVIDERCOUNT__ provider-ranked proposals. The single-episode narrative method proposes clips in __NARRCOVER__ episodes, versus __BASECOVER__ for the baseline. Scores only sort candidates.</p>
<div class="controls"><label>View<select id="method">__REVIEWOPTION____PROVIDEROPTION__<option value="composite">Automatic cross-episode proposals</option><option value="narrative">Single-episode narrative</option><option value="baseline">Single-episode lexical baseline</option><option value="all">All</option></select></label><label>Episode or cue search<input id="query" type="search" placeholder="e.g. 23 or poison"></label><span id="count"></span></div>
__CARDS__</main><footer>Generated from the input ZIP and unedited JSON outputs. No network assets or calls.</footer>
<script>const m=document.getElementById('method'),q=document.getElementById('query'),cards=[...document.querySelectorAll('.card')],count=document.getElementById('count');function refresh(){let n=0,term=q.value.trim().toLowerCase(),episode=/^\\d+$/.test(term);for(const card of cards){const match=!term||(episode?card.dataset.episodes.split(',').includes(String(Number(term))):card.textContent.toLowerCase().includes(term));const show=(m.value==='all'||card.dataset.method===m.value)&&match;card.hidden=!show;if(show)n++}count.textContent=n+' clips shown'}m.addEventListener('change',refresh);q.addEventListener('input',refresh);refresh();</script></body></html>"""
    document = document.replace("__EPISODES__", str(report["episodes"]))
    document = document.replace("__BASECOVER__", str(report["baseline_covered_episodes"]))
    document = document.replace("__NARRCOVER__", str(report["narrative_covered_episodes"]))
    document = document.replace("__CROSSCOUNT__", str(len(composite)))
    document = document.replace("__REVIEWCOUNT__", str(len(reviewed)))
    document = document.replace("__PROVIDERCOUNT__", str(len(provider_rows)))
    document = document.replace("__REVIEWOPTION__", '<option value="reviewed">Reviewed posting order</option>' if reviewed else "")
    document = document.replace("__PROVIDEROPTION__", '<option value="provider">Provider reranking (script proxy)</option>' if provider_rows else "")
    document = document.replace("__CARDS__", cards)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(document, encoding="utf-8")
    print(f"Wrote {html_path} ({len(reviewed)} editorial, {len(provider_rows)} provider, {len(composite)} cross-episode, {len(narrative)} narrative, {len(baseline)} baseline cards)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--html", type=Path, default=Path("comparison/promo_review.html"))
    parser.add_argument("--reviewed", type=Path, help="Optional editorial shortlist JSON")
    parser.add_argument("--provider-reviewed", type=Path, help="Optional provider-ranked JSON from review_cross_episode.py")
    args = parser.parse_args()
    generate(args.zip_path, args.output_dir, args.html, args.reviewed, args.provider_reviewed)


if __name__ == "__main__":
    main()
