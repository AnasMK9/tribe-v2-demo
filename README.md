---
title: TRIBE v2 Neuromarketing
emoji: 🧠
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
hardware: t4-small
---

# TRIBE v2 — Neuromarketing Creative Analyzer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnasMK9/tribe-v2-neuromarketing/blob/main/colab_demo.ipynb) [![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/anasmk9/tribe-v2-neuromarketing)

Upload a 15–60s video ad. Get a predicted neural response on Meta's TRIBE v2 brain encoder, broken down by region, over time, with plain-English interpretation for marketers.

## What this does

1. Runs real TRIBE v2 inference on your video (video + audio + optional text)
2. Aggregates per-vertex fMRI predictions into 7 neuromarketing ROI clusters (Visual Attention, Face & Social, Motion & Dynamics, Language & Meaning, Emotion & Reward, Memory Encoding, Object & Scene)
3. Scores 4 marketing dimensions: Attention, Engagement, Comprehension, Memorability
4. Surfaces **spikes**: moments in your video where a given region lit up significantly above baseline (z-score > 2). Click a spike to see the exact frame and what drove it.
5. Renders a brain-surface heatmap on the fsaverage5 cortical mesh.

## Setup (local, Apple Silicon or Linux+CUDA)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then paste your HF_TOKEN (needed for gated LLaMA 3.2-3B)
python app.py
```

First run downloads ~5 GB of model weights and takes a while.

## Hardware expectations

| Tier | Per 30s clip | Cost |
|---|---|---|
| HF Space T4 small (this Space) | ~2–5 min | ~$0.60/hr while running; pause between sessions |
| Local Apple Silicon (MPS if supported, else CPU) | 5–20 min | $0 |
| HF Space free CPU | 15–30 min | $0 (not recommended for video) |

**Pause this Space when not testing** to avoid idle GPU billing.

## License & ethics

TRIBE v2 is **CC-BY-NC-4.0**. This Space is for non-commercial research and internal testing only. Do not build a paid product on top of it.

Encoding ≠ behavior. TRIBE v2 predicts population-average fMRI response, not conversion, not purchase intent, not your specific target audience. Use these scores as directional proxies, not ground truth. See the Methodology tab in the app.

## Credits

- Meta AI Research — [TRIBE v2 model](https://huggingface.co/facebook/tribev2)
- Reference Space: [vishnuverse-in/ad-brain-scorer](https://huggingface.co/spaces/vishnuverse-in/ad-brain-scorer) (CPU deployment pattern)
