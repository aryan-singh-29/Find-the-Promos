"""One-command, fresh-folder rerun on a new same-format screenplay ZIP.

Generates raw candidates, validates them, applies an optional swappable provider
to cross-episode promos, and builds an offline viewer. It never applies
the supplied show's editorial annotations to a new input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from preview_results import generate
from promo_finder import parse_zip
from promo_pipeline import run as run_pipeline
from review_cross_episode import ChatProvider, RulesProvider, review as review_provider
from verify_run import verify


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_output(input_zip: Path, digest: str, requested: Path | None) -> Path:
    if requested is not None:
        if requested.exists():
            raise FileExistsError(f"Output already exists: {requested}. Choose a fresh --out folder so runs cannot mix.")
        return requested
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", input_zip.stem).strip("-")[:40] or "scripts"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path("runs") / f"{stem}-{digest[:8]}-{stamp}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = Path(str(base) + f"-{suffix}")
        suffix += 1
    return candidate


def run(input_zip: Path, output_dir: Path | None, top: int, provider_name: str, endpoint: str | None, model: str | None, api_key_env: str, limit: int, review_pool: str = "ranked") -> dict:
    if not input_zip.is_file():
        raise FileNotFoundError(f"Cannot find input ZIP: {input_zip}")
    if top < 1 or limit < 1:
        raise ValueError("--top and --limit must be positive")
    if review_pool not in {"ranked", "all"}:
        raise ValueError("--review-pool must be ranked or all")
    if provider_name == "chat" and (not endpoint or not model):
        raise ValueError("chat provider requires --endpoint and --model")
    provider = RulesProvider() if provider_name == "rules" else ChatProvider(endpoint, model, api_key_env)
    # Fail malformed/empty inputs before making an output folder.
    episodes = parse_zip(input_zip)
    digest = source_hash(input_zip)
    target = choose_output(input_zip, digest, output_dir)
    target.mkdir(parents=True, exist_ok=False)
    run_pipeline(input_zip, target, top)
    validation = verify(input_zip, target)
    if validation["failures"]:
        raise ValueError(f"Mechanical validation failed with {len(validation['failures'])} errors; see {target / 'validation.json'}")
    viewer = target / "promo_review.html"
    run_report = json.loads((target / "run_report.json").read_text(encoding="utf-8"))
    manifest = {
        "input_name": input_zip.name,
        "input_sha256": digest,
        "source_format": run_report["source_format"],
        "episodes": len(episodes),
        "provider": provider.name,
        "model": provider.model,
        "provider_candidate_pool": review_pool,
        "candidate_count": run_report["cross_episode_candidate_count"],
        "automatic_ranked_count": run_report["cross_episode_ranked_count"],
        "provider_reviewed_count": 0,
        "provider_status": "pending",
        "validation_checked_rows": validation["checked_rows"],
        "validation_failures": len(validation["failures"]),
        "abstained": run_report["cross_episode_ranked_count"] == 0,
        "editorial_annotations_applied": False,
        "viewer": viewer.name,
        "note": "The rules provider is a deterministic proxy. Chat opinions and script-only cuts need footage and editor review.",
    }
    # A usable automatic viewer and provenance survive an optional API failure.
    (target / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    generate(input_zip, target, viewer)
    try:
        candidate_file = "cross_episode_ranked.json" if review_pool == "ranked" else "cross_episode_all_candidates.json"
        provider_report = review_provider(input_zip, target / candidate_file, target, provider, limit)
    except Exception as error:
        manifest["provider_status"] = "failed"
        manifest["provider_error"] = f"{type(error).__name__}: {error}"
        (target / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise RuntimeError(f"Provider review failed; automatic viewer remains at {viewer}. {error}") from error
    manifest["provider_reviewed_count"] = provider_report["reviewed_count"]
    manifest["provider_status"] = "complete"
    generate(input_zip, target, viewer, provider_path=target / "cross_episode_provider_reviewed.json")
    (target / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(target), **manifest}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--out", type=Path, help="Fresh output folder; default runs/<name>-<hash>-<UTC time>")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--provider", choices=("rules", "chat"), default="rules")
    parser.add_argument("--endpoint", help="Full OpenAI-compatible /chat/completions URL for chat provider")
    parser.add_argument("--model", help="Model identifier for chat provider")
    parser.add_argument("--api-key-env", default="PROMO_API_KEY", help="Environment variable holding provider key")
    parser.add_argument("--limit", type=int, default=20, help="Maximum automatic cross-episode promos sent to provider")
    parser.add_argument("--review-pool", choices=("ranked", "all"), default="ranked", help="Use diversified shortlist or top raw candidates for provider review")
    args = parser.parse_args()
    try:
        run(args.zip_path, args.out, args.top, args.provider, args.endpoint, args.model, args.api_key_env, args.limit, args.review_pool)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
