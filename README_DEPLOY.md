# Deployment Guide — AI Smart City Dashboard

This repository contains a Gradio-based dashboard in `dashboard/app.py` and a top-level `app.py` that exposes the Gradio `demo` interface. Use the instructions below to publish to GitHub and Hugging Face Spaces.

## 1) Prepare repository

- Make sure `requirements.txt` contains all dependencies (it now includes `gradio`).
- Confirm `app.py` (top-level) exists — Spaces expects `app.py` at repo root.

## 2) Publish to GitHub

1. Initialize git (if not already):

```bash
git init
git add .
git commit -m "Add Gradio dashboard and deployment files"
```

2. Create a GitHub repo and push (replace `<origin-url>`):

```bash
git remote add origin <origin-url>
git branch -M main
git push -u origin main
```

## 3) Deploy to Hugging Face Spaces

You can create a Space and push this repository. Two approaches:

A) Manual (simple):

1. Create a new Space on https://huggingface.co/spaces using the Web UI. Choose `Gradio` as the SDK.
2. Clone the new Space repo URL (it will look like `https://huggingface.co/spaces/<username>/<space-name>`).
3. Push the repo contents to that Space:

```bash
git clone https://huggingface.co/spaces/<username>/<space-name>
cd <space-name>
# copy files here or set the remote and push
git remote add upstream https://huggingface.co/spaces/<username>/<space-name>
git push upstream main
```

B) Using `huggingface-cli` (recommended for automation):

1. Install `huggingface_hub` and login:

```bash
pip install huggingface_hub
huggingface-cli login
```

2. Create the Space and push (replace names):

```bash
huggingface-cli repo create <username>/<space-name> --type space
git remote add hf https://huggingface.co/spaces/<username>/<space-name>
# push files
git push hf main
```

Note: Spaces will automatically install `requirements.txt` and run `app.py`.

## 4) CI / Automation

I added a simple GitHub Actions CI workflow that runs on pushes to `main` to check imports and linting; you can extend it to automatically mirror the repo to Hugging Face using a secret `HF_TOKEN`.

---

If you want, I can:
- Push the current repository to GitHub for you (if you provide the remote or grant access),
- Create the Hugging Face Space automatically using your `HF_TOKEN` (set as a secret in GitHub Actions), or
- Add an automated GitHub Action that mirrors commits to a Hugging Face Space (requires `HF_TOKEN`).

Tell me which of the above you'd like me to do next.