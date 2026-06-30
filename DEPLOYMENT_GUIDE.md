# Deployment Guide — Urdu Fake News Detector

## What you need before starting
- GitHub account (you have this)
- Your trained `classifier.pkl` (8.2 MB)
- Your Groq API key

---

## Step 1 — Copy your trained model into this deploy folder

Copy your file from:
```
C:\Users\Laptop\Desktop\New folder (2)\models\classifier.pkl
```
into:
```
deploy/app/models/classifier.pkl
```

---

## Step 2 — Create a GitHub repository

1. Go to github.com → click **New repository**
2. Name it: `urdu-fake-news-detector`
3. Set to **Public** (required for free Streamlit Cloud deployment)
4. Do NOT initialize with README (we already have our files)
5. Click **Create repository**

---

## Step 3 — Push this folder to GitHub

Open a terminal inside this `deploy` folder and run:

```bash
git init
git add .
git commit -m "Initial deployment of Urdu Fake News Detector"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/urdu-fake-news-detector.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

**Important:** the `.gitignore` file ensures your `secrets.toml` (with your real API key) is NEVER uploaded to GitHub. Only `app.py`, `hybrid_engine.py`, `requirements.txt`, and your model file get pushed.

---

## Step 4 — Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io**
2. Sign in with your GitHub account
3. Click **Create app** → **From existing repo**
4. Select your repository: `urdu-fake-news-detector`
5. Set:
   - **Branch:** main
   - **Main file path:** `app/app.py`
6. Click **Advanced settings** before deploying:
   - Under **Secrets**, paste:
     ```toml
     GROQ_API_KEY = "your_actual_groq_key_here"
     ```
7. Click **Deploy**

---

## Step 5 — Wait and test

Deployment takes 2-5 minutes the first time (installing dependencies). Once live, you'll get a public URL like:
```
https://urdu-fake-news-detector.streamlit.app
```

Test it with the same articles we used before to confirm the deployed version behaves identically to your local version.

---

## Troubleshooting

**"Model not found" error:**
Check that `app/models/classifier.pkl` was actually pushed to GitHub (GitHub web UI → browse your repo → confirm the file is there and not 0 bytes — sometimes large files fail silently if Git LFS wasn't needed but something else went wrong).

**"GROQ_API_KEY not found" error:**
Go to your deployed app → Settings (3-dot menu) → Secrets → confirm the key was saved correctly with no extra quotes or spaces.

**App works locally but crashes on deploy:**
Usually a missing package — check the deployed app's logs (bottom right "Manage app" → logs) for the exact import error, then add the missing package to `requirements.txt`.
