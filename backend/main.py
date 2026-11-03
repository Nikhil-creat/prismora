"""Prismora API: upload, diagnostics, symbolic rendering, agent swarm (SSE), RAG, assistant."""
import io, os, time, json, asyncio
import numpy as np, cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent_swarm import run_swarm
from rag_engine import index_image, search

ALLOWED = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
from security import guard as _g
from main_helpers import load
app = FastAPI(title="Prismora API")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED, allow_methods=["GET", "POST"], allow_headers=["*"])

def guard(key):
    _g(key, "viewer", 0, "core")

def diagnostics(a: np.ndarray) -> dict:
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    noise = float(np.std(g.astype(float) - cv2.GaussianBlur(g, (5, 5), 0)))
    return {"width": a.shape[1], "height": a.shape[0], "blur_score": blur, "is_blurry": blur < 100,
            "noise_sigma": noise, "histogram": np.histogram(g, 16, (0, 255))[0].tolist()}

def matrix(a: np.ndarray, mode: str, cols: int) -> str:
    h = max(1, int(cols * a.shape[0] / a.shape[1] * 0.5))
    g = cv2.resize(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (cols, h), interpolation=cv2.INTER_AREA) / 255
    ramp = {"ascii": list("@%#*+=-:. "), "emoji": ["🌑","🌒","🌓","🌔","🌕","⭐","✨","🤍"],
            "latex": ["\\sum","\\int","\\nabla","x^2","=","-","."," "]}[mode]
    idx = np.minimum(len(ramp) - 1, (g * len(ramp)).astype(int))
    sep = " " if mode == "latex" else ""
    return "\n".join(sep.join(ramp[i] for i in row) for row in idx)

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    guard(x_api_key); a = load(await file.read()); return diagnostics(a)

@app.post("/api/symbolic")
async def symbolic(mode: str = "emoji", cols: int = 80, file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    guard(x_api_key)
    if mode not in ("emoji", "ascii", "latex"): raise HTTPException(400, "mode must be emoji, ascii or latex")
    return {"mode": mode, "text": matrix(load(await file.read()), mode, max(20, min(cols, 200)))}

@app.post("/api/index")
async def index(file: UploadFile = File(...), label: str = "", x_api_key: str | None = Header(None)):
    guard(x_api_key); return index_image(load(await file.read()), label)

@app.get("/api/search")
def similar(q: str, x_api_key: str | None = Header(None)):
    guard(x_api_key); return search(q)

@app.post("/api/agents/stream")
async def agents(file: UploadFile = File(...), goal: str = "improve quality", x_api_key: str | None = Header(None)):
    guard(x_api_key); a = load(await file.read())
    async def gen():
        for ev in run_swarm(a, goal, diagnostics):
            yield f"data: {json.dumps(ev)}\n\n"; await asyncio.sleep(0.05)
    return StreamingResponse(gen(), media_type="text/event-stream")

class Chat(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(body: Chat, x_api_key: str | None = Header(None)):
    guard(x_api_key)
    import httpx
    gk, mk = os.getenv("GROQ_API_KEY"), os.getenv("GEMINI_API_KEY")
    async with httpx.AsyncClient(timeout=30) as c:
        if gk:
            r = await c.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {gk}"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": body.message}]})
            return {"provider": "groq", "reply": r.json()["choices"][0]["message"]["content"]}
        if mk:
            r = await c.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={mk}",
                json={"contents": [{"role": "user", "parts": [{"text": body.message}]}]})
            return {"provider": "gemini", "reply": r.json()["candidates"][0]["content"]["parts"][0]["text"]}
    raise HTTPException(503, "Set GROQ_API_KEY or GEMINI_API_KEY")

import extra
app.include_router(extra.router)
