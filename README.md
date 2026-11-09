# Script to promos

A rerunnable Streamlit prototype that finds and ranks **two-segment, cross-episode promo proposals** from timestamped screenplays. Every proposal lists the episode and source start/end of each segment in order, a hook, a cliffhanger, and why the proposal might work. The app also exports unedited output files from the run.

The included demo is **invented**. No private show scripts, derived outputs, editor annotations, or examples are in this repository.

## Run locally

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Choose **Invented demo** and click **Find promos**, or upload a ZIP containing `episode-001.txt`, `episode-002.txt`, etc. The app runs in a temporary folder, then shows ranked results and lets you download the JSON, CSV, and complete raw output bundle. The ZIP must contain at least one UTF-8 `.txt` episode with a numbered filename; cross-episode proposals require at least two adjacent numbered episodes. Uploads are limited to 20 MB compressed and 100 MB uncompressed.

Each episode uses screenplay lines such as:

```text
EPISODE LOGLINE: A navigator finds a false signal.
SCENE 1
**MIRA:**
Start: (00:00.00) | End: (00:02.00)
"The signal will kill the crew!"
**ACTION:**
Start: (00:02.00) | End: (00:04.00)
Mira hits the beacon console.
Scene end: 00:04.00
```

The exact accepted format is demonstrated by `sample_input.zip`. An uploaded ZIP can contain a full show; the parser does not use any show-specific editorial notes.

## Run from the command line / swap reviewer

The same pipeline also has a one-command CLI:

```bash
python run_interview.py sample_input.zip --out runs/demo --top 10
```

The default `rules` reviewer is free and deterministic. To use a stronger model, provide an OpenAI-compatible chat-completions endpoint and model identifier:

```bash
export PROMO_API_KEY=your_key
python run_interview.py sample_input.zip --out runs/model-demo --top 10 --provider chat --endpoint https://YOUR_PROVIDER/v1/chat/completions --model YOUR_MODEL
```

On PowerShell, set the key with `$env:PROMO_API_KEY = 'your_key'`. Use a fresh `--out` folder for each run. The model reviews the **same assembled, source-grounded two-segment packets** and cannot change the source cut points. No paid API is required for the app or the CLI default. The public web UI runs the rules reviewer only; its server does not hold a shared model API key.

## Outputs and interpretation

`cross_episode_ranked.json` and `.csv` are the main deliverables. `cross_episode_all_candidates.json` preserves alternatives. `run_report.json`, `validation.json`, `run_manifest.json`, and `cross_episode_audit.json` record counts, provenance, checks, and abstention. `cross_episode_provider_reviewed.json` records the separate reviewer scores. `promo_review.html` is an offline editor view. The export bundle includes these and the baseline/narrative comparison files.

Ranking is a **script-only editing queue**, not a prediction of viewer response. It checks timestamp and dialogue boundaries mechanically, then scores hook, cliffhanger, and lexical continuity. Cross-episode joins can be confusing even when names and topics match. An editor must check the footage and ask a cold viewer whether the joined story makes sense. If no pair passes the gates, the tool returns no promos instead of inventing one.

For confidential scripts, run the CLI locally. Uploading to a public Streamlit deployment sends the ZIP to that app's server.

## Deploy

Push this directory alone to a **new public GitHub repository**. In [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), select that repository, choose `app.py` as the entrypoint, and deploy. Do not push a parent workspace or any private input ZIP, generated output, research notes, or Git history from work on private scripts.
