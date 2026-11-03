import io, os, re, json, base64, httpx
from fastapi import APIRouter, UploadFile, File, Header, HTTPException, Request
from pydantic import BaseModel
from security import guard, usage, audit, KEYS
import models, provenance
from main_helpers import load

router = APIRouter(prefix="/api")
async def img(f, key, need, ep):
    data = await f.read(); a = load(data); guard(key, need, a.shape[0] * a.shape[1], ep); return a, data

@router.post("/forensics")
async def forensics(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    a, _ = await img(file, x_api_key, "viewer", "forensics"); return models.forensics(a)
@router.post("/segment")
async def segment(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    a, _ = await img(file, x_api_key, "editor", "segment"); r = models.segment(a); models.free_vram(); return r
@router.post("/ocr")
async def ocr(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    a, _ = await img(file, x_api_key, "viewer", "ocr"); return models.ocr(a)
@router.post("/upscale")
async def upscale(factor: int = 2, file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    a, d = await img(file, x_api_key, "editor", "upscale"); r = models.upscale(a, max(1, min(factor, 4))); r["provenance"] = provenance.manifest(base64.b64decode(r["png_b64"]), actions=["upscale"]); models.free_vram(); return r
@router.post("/depth")
async def depth(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    a, _ = await img(file, x_api_key, "editor", "depth"); return models.depth_contours(a)
@router.post("/provenance/verify")
async def verify(manifest: dict, x_api_key: str | None = Header(None)):
    guard(x_api_key, "viewer", 0, "verify"); return {"valid": provenance.verify(dict(manifest))}
@router.get("/usage")
def my_usage(x_api_key: str | None = Header(None)):
    guard(x_api_key, "viewer", 0, "usage"); return usage.get(x_api_key, {})
@router.get("/audit")
def get_audit(x_api_key: str | None = Header(None)):
    guard(x_api_key, "admin", 0, "audit"); return audit[-100:]

DOCS = {"security": "Uploads capped at 50MB, magic-byte checked, EXIF stripped. Roles: viewer, editor, admin. Rate limit via Redis token bucket.",
 "agents": "LangGraph swarm: Planner, Vision Analyst, Transformer, Critic with up to 3 self-healing retries; SSE stream at /api/agents/stream.",
 "rag": "CLIP embeddings in Qdrant; /api/index adds images, /api/search queries by text.",
 "deploy": "Frontend is static (Vercel/Netlify/Cloudflare Pages/GitHub Pages). Backend is Docker (Render). Set VITE-free window.PRISMORA_API in config.js.",
 "models": "Optional: SAM 2, Grounding DINO, EasyOCR, Real-ESRGAN, Depth Anything V2, NSFW classifier. Fallbacks run without them."}
class Ask(BaseModel): message: str; history: list = []
@router.post("/assistant")
async def assistant(b: Ask, x_api_key: str | None = Header(None)):
    """Agentic RAG: retrieve doc snippets + caller usage, then ask Groq (preferred) or Gemini."""
    guard(x_api_key, "viewer", 0, "assistant")
    words = set(re.findall(r"\w+", b.message.lower())); ctx = [v for k, v in DOCS.items() if k in words or words & set(re.findall(r"\w+", v.lower()))][:3]
    sys = "You are Prismora AI. Use only this context and say so if unsure:\n" + "\n".join(ctx) + f"\nCaller usage: {usage.get(x_api_key, {})}"
    msgs = [{"role": "system", "content": sys}, *b.history[-8:], {"role": "user", "content": b.message}]
    gk, mk = os.getenv("GROQ_API_KEY"), os.getenv("GEMINI_API_KEY")
    async with httpx.AsyncClient(timeout=30) as c:
        if gk:
            r = await c.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {gk}"}, json={"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), "messages": msgs})
            if r.is_success: return {"provider": "groq", "reply": r.json()["choices"][0]["message"]["content"], "sources": ctx}
        if mk:
            r = await c.post(f"https://generativelanguage.googleapis.com/v1beta/models/{os.getenv('GEMINI_MODEL','gemini-2.0-flash')}:generateContent?key={mk}", json={"system_instruction": {"parts": [{"text": sys}]}, "contents": [{"role": "user", "parts": [{"text": b.message}]}]})
            if r.is_success: return {"provider": "gemini", "reply": r.json()["candidates"][0]["content"]["parts"][0]["text"], "sources": ctx}
    raise HTTPException(503, "No working GROQ_API_KEY or GEMINI_API_KEY")
