"""Public Streamlit interface for the reusable screenplay promo finder."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from run_interview import run as run_finder


APP_DIR = Path(__file__).resolve().parent
DEMO_ZIP = APP_DIR / "sample_input.zip"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED = 100 * 1024 * 1024
MAX_MEMBER_UNCOMPRESSED = 30 * 1024 * 1024


def clear_previous_run() -> None:
    """A changed input or limit should not leave old results on screen."""
    st.session_state.pop("promo_result", None)


def check_zip(data: bytes) -> None:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("ZIP exceeds the 20 MB upload limit.")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > 100:
            raise ValueError("ZIP has more than 100 files.")
        if not any(member.filename.lower().endswith(".txt") for member in members):
            raise ValueError("ZIP needs episode-NNN.txt screenplay files.")
        if any(member.file_size > MAX_MEMBER_UNCOMPRESSED for member in members):
            raise ValueError("An episode exceeds the 30 MB uncompressed limit.")
        if sum(member.file_size for member in members) > MAX_TOTAL_UNCOMPRESSED:
            raise ValueError("ZIP exceeds the 100 MB uncompressed limit.")


def result_bundle(output_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.glob("*")):
            if path.is_file():
                archive.write(path, path.name)
    return buffer.getvalue()


def execute(source: bytes, top: int) -> dict:
    check_zip(source)
    with tempfile.TemporaryDirectory(prefix="promo_finder_") as scratch:
        root = Path(scratch)
        input_path = root / "input.zip"
        output_dir = root / "result"
        input_path.write_bytes(source)
        with contextlib.redirect_stdout(io.StringIO()):
            run_finder(input_path, output_dir, top, "rules", None, None, "PROMO_API_KEY", top)
        ranked = json.loads((output_dir / "cross_episode_ranked.json").read_text(encoding="utf-8"))
        reviewed = json.loads((output_dir / "cross_episode_provider_reviewed.json").read_text(encoding="utf-8"))
        report = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
        validation = json.loads((output_dir / "validation.json").read_text(encoding="utf-8"))
        return {
            "ranked": ranked,
            "reviewed": reviewed,
            "report": report,
            "validation": validation,
            "ranked_json": (output_dir / "cross_episode_ranked.json").read_bytes(),
            "ranked_csv": (output_dir / "cross_episode_ranked.csv").read_bytes(),
            "bundle": result_bundle(output_dir),
        }


st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")
st.title("🎬 Script to promos")
st.caption("Source-timed, ranked cross-episode promo proposals for short-form drama")
st.write(
    "Upload a ZIP of timestamped episode screenplays, or try the fully invented two-episode demo. "
    "The app finds 20–60 second promos, each made of two ordered clips from adjacent episodes."
)

with st.sidebar:
    st.header("Run settings")
    source_mode = st.radio("Input", ["Invented demo", "Upload my ZIP"], index=0, on_change=clear_previous_run)
    upload = None
    if source_mode == "Upload my ZIP":
        upload = st.file_uploader("Screenplay ZIP", type=["zip"])
        st.caption("Public hosting processes uploads on the app server. Use the local CLI for confidential scripts.")
    top = st.slider(
        "Maximum promos to return", min_value=1, max_value=20, value=10,
        help="Upper limit, not a target. The tool may find fewer, and keeps at most one promo per adjacent episode pair.",
        on_change=clear_previous_run,
    )
    if source_mode == "Invented demo":
        st.caption("This two-episode demo has only one adjacent episode pair, so it can return at most one ranked promo.")
    run_clicked = st.button("Find promos", type="primary", use_container_width=True)
    st.divider()
    st.caption("Input: episode-NNN.txt files with SCENE markers and timed dialogue/action lines. See README for the exact format.")
    st.caption("Default reviewer: deterministic rules. An OpenAI-compatible model can be substituted through the CLI.")

if run_clicked:
    st.session_state.pop("promo_result", None)
    if source_mode == "Upload my ZIP" and upload is None:
        st.error("Choose a ZIP file first.")
    else:
        source = DEMO_ZIP.read_bytes() if source_mode == "Invented demo" else upload.getvalue()
        try:
            with st.spinner("Scanning timed beats, assembling clips, and checking cuts…"):
                st.session_state["promo_result"] = execute(source, top)
            st.success("Run complete. Results are ready below.")
        except (ValueError, zipfile.BadZipFile, RuntimeError, KeyError, OSError) as error:
            st.error(f"Could not process this ZIP: {error}")

result = st.session_state.get("promo_result")
if result is None:
    st.info("Click **Find promos** to run with the current input and maximum.")
    st.stop()

report = result["report"]
validation = result["validation"]
ranked = result["ranked"]
metric_cols = st.columns(4)
metric_cols[0].metric("Episodes", report["episodes"])
metric_cols[1].metric("Stitched candidates", report["cross_episode_candidate_count"])
metric_cols[2].metric("Ranked promos", len(ranked))
metric_cols[3].metric("Cut checks failed", len(validation["failures"]))
if len(ranked) < top:
    st.info(
        f"Found {len(ranked)} ranked promo{'s' if len(ranked) != 1 else ''} with a maximum of {top}. "
        "The maximum is a cap; cuts still need to pass the source and continuity checks, "
        "and ranking keeps at most one promo per adjacent episode pair."
    )

st.subheader("What to post first")
st.caption("Ranked by timed hook, unresolved ending, and lexical continuity. Every join needs a cold-viewer and footage review.")
if not ranked:
    st.warning("No cross-episode pair passed the continuity and duration gates. The tool abstained on this input.")
else:
    for row in ranked:
        with st.container(border=True):
            st.markdown(f"### #{row['rank']} · {row['duration_seconds']:.1f} seconds")
            for index, segment in enumerate(row["segments"], 1):
                st.write(f"**Segment {index}:** Episode {segment['episode']} · {segment['start']} → {segment['end']}")
            st.write("**Hook:**", row["hook"])
            st.write("**Cliffhanger:**", row["cliffhanger"])
            st.write("**Why it works:**", row["why_it_works"])
            st.write("**Why post it at this rank:**", row["post_first_reason"])
            if row.get("review_note"):
                st.caption(row["review_note"])

st.subheader("Export this run")
download_cols = st.columns(3)
download_cols[0].download_button("Ranked JSON", result["ranked_json"], "cross_episode_ranked.json", "application/json", use_container_width=True)
download_cols[1].download_button("Ranked CSV", result["ranked_csv"], "cross_episode_ranked.csv", "text/csv", use_container_width=True)
download_cols[2].download_button("All raw outputs ZIP", result["bundle"], "promo_run_outputs.zip", "application/zip", use_container_width=True)

with st.expander("Checks, reviewer, and limits"):
    st.write(f"Mechanical checks: {validation['checked_rows']} rows, {len(validation['failures'])} failures.")
    st.write(f"The rules provider reviewed {len(result['reviewed'])} shortlisted promos as a deterministic proxy; its opinions are saved in the raw outputs.")
    st.write("This is a screenplay-only proposal queue. It cannot establish whether the cut works visually or whether a new viewer understands the join. The score is not an audience performance prediction.")
    if report.get("cross_episode_generic_fallback_used"):
        st.write("This input used the generic word-overlap fallback because no stronger topic match was found.")
