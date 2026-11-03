"""C2PA-style provenance: an HMAC-signed manifest. This is NOT a real C2PA claim; use c2pa-python with a CA-issued cert for production."""
import os, hmac, hashlib, json, time
SECRET = os.getenv("PROVENANCE_SECRET", "change-me").encode()
def manifest(data: bytes, tool="Prismora", actions=()):
    m = {"generator": tool, "sha256": hashlib.sha256(data).hexdigest(), "created": int(time.time()), "actions": list(actions), "ai_generated": True}
    m["signature"] = hmac.new(SECRET, json.dumps(m, sort_keys=True).encode(), hashlib.sha256).hexdigest(); return m
def verify(m: dict):
    sig = m.pop("signature", ""); ok = hmac.compare_digest(sig, hmac.new(SECRET, json.dumps(m, sort_keys=True).encode(), hashlib.sha256).hexdigest()); return ok
