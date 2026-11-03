"""Auth (API key -> role), RBAC, token-bucket rate limit (Redis or memory), metering, audit log."""
import os, time, json
from fastapi import HTTPException
ROLES = {"viewer": 1, "editor": 2, "admin": 3}
KEYS = {}
for item in filter(None, os.getenv("API_KEYS", "dev-key:admin").split(",")):
    k, _, r = item.partition(":"); KEYS[k] = r if r in ROLES else "viewer"
RATE, BURST = float(os.getenv("RATE_PER_SEC", "1")), int(os.getenv("BURST", "10"))
_r = None
try:
    import redis; _r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"), socket_timeout=1); _r.ping()
except Exception: _r = None
_mem, usage, audit = {}, {}, []

def _take(key):
    now = time.time()
    if _r:
        t, tok = [float(x) for x in (_r.hmget(f"rl:{key}", "t", "tok") if _r.exists(f"rl:{key}") else (now, BURST))]
    else: t, tok = _mem.get(key, (now, BURST))
    tok = min(BURST, tok + (now - t) * RATE)
    if tok < 1: raise HTTPException(429, "Rate limit reached. Retry in a few seconds.")
    if _r: _r.hset(f"rl:{key}", mapping={"t": now, "tok": tok - 1}); _r.expire(f"rl:{key}", 3600)
    else: _mem[key] = (now, tok - 1)

def guard(key, need="viewer", pixels=0, endpoint=""):
    if key not in KEYS: raise HTTPException(401, "Invalid or missing X-API-Key")
    if ROLES[KEYS[key]] < ROLES[need]: raise HTTPException(403, f"Requires {need} role")
    _take(key)
    u = usage.setdefault(key, {"requests": 0, "pixels": 0}); u["requests"] += 1; u["pixels"] += pixels
    audit.append({"t": time.time(), "key": key[:4] + "***", "role": KEYS[key], "endpoint": endpoint}); del audit[:-500]
    return KEYS[key]
