# Nesy-Gen Demo

This folder contains a lightweight Streamlit demo for the paper `Towards Explainable Graph-Verified Neuro-symbolic Chest X-Ray Report Generation`.

## What the demo is

- A presentation-friendly, GitHub-friendly interface for the paper's method
- An interactive verifier for sample chest X-ray cases
- A compact dashboard for the paper's reported ablation results
- A figure browser for the manuscript visuals

## What the demo is not

- It is not the full clinical training pipeline
- It does not require the original model checkpoints
- It is not intended for diagnostic use

## Run locally

```bash
pip install -r requirements.txt
streamlit run demo/paper_demo_app.py
```

## Conference use

- Best entry page: `Overview`
- Best live walkthrough page: `Interactive verifier`
- Best fallback page for fast Q&A: `Paper figures`
- Best page for performance questions: `Results dashboard`

## Why Streamlit

- Fast to launch for reviewers and collaborators
- Easy to host on GitHub/Streamlit Community Cloud
- Good fit for a method demo where explanation matters as much as prediction
