# Deploy guide

## 1. Push to GitHub (from Termux)
    pkg install git
    git config --global user.name "Your Name"
    git config --global user.email "you@example.com"
    unzip prismora-master.zip && cd prismora
    git init && git add . && git commit -m "Prismora initial commit"
    git branch -M main
    # create an EMPTY repo named prismora on github.com first, then:
    git remote add origin https://github.com/Nikhil-creat/prismora.git
    git push -u origin main
GitHub asks for a username and a **personal access token** (Settings > Developer settings > Tokens), not your password. Never commit `.env`.

## 2. Frontend (free, static)
Publish the `frontend/` folder. Pick any one:
- **GitHub Pages:** repo Settings > Pages > Source: GitHub Actions. The included workflow deploys on every push. URL: https://nikhil-creat.github.io/prismora/
- **Vercel:** Add New Project > import repo. `vercel.json` already sets the output folder. No build command.
- **Netlify:** Add new site > import repo. `netlify.toml` sets publish to `frontend`.
- **Cloudflare Pages:** Workers & Pages > Create > Pages > connect repo. Build command empty, output directory `frontend`.

## 3. Backend (Render, free Docker web service)
1. Render > New > Blueprint > pick the repo (it reads `render.yaml`).
2. Set env vars: `API_KEYS` (e.g. `long-random-string:admin`), `GROQ_API_KEY`, `GEMINI_API_KEY`, and `CORS_ORIGINS` = your frontend URL.
3. Free instances sleep when idle and have limited memory. Leave `ENABLE_MODELS=0`; the CLIP/RAG endpoints need more RAM than the free tier, so they may fail there. Run the full stack locally with Docker for those.
4. Put the Render URL in `frontend/config.js` as `window.PRISMORA_API`, commit, push.

## 4. Free API keys
Groq: console.groq.com > API Keys. Gemini: aistudio.google.com/apikey. Model names change over time; override with `GROQ_MODEL` / `GEMINI_MODEL`.

## Limits to know
Anyone can read keys pasted into a public site's JS, so the site asks each visitor for their own key and never ships yours. Provenance manifests are HMAC-signed, not real C2PA.
