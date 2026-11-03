"""Visual RAG: CLIP embeddings stored in Qdrant, with an in-memory fallback if Qdrant is unreachable."""
import os, uuid, numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

COL = "visionforge"; _model = None; _mem = []
def _clip():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("clip-ViT-B-32")
    return _model
def _client():
    try:
        c = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"), timeout=3)
        if COL not in [x.name for x in c.get_collections().collections]:
            c.create_collection(COL, vectors_config=VectorParams(size=512, distance=Distance.COSINE))
        return c
    except Exception: return None

def index_image(arr, label):
    v = _clip().encode(Image.fromarray(arr)).tolist(); c = _client(); pid = str(uuid.uuid4())
    if c: c.upsert(COL, [PointStruct(id=pid, vector=v, payload={"label": label})])
    else: _mem.append((pid, np.array(v), label))
    return {"id": pid, "label": label, "store": "qdrant" if c else "memory"}

def search(text, k=5):
    v = _clip().encode(text); c = _client()
    if c: return [{"label": h.payload["label"], "score": h.score} for h in c.search(COL, query_vector=v.tolist(), limit=k)]
    sims = sorted(((float(np.dot(e, v) / (np.linalg.norm(e) * np.linalg.norm(v))), l) for _, e, l in _mem), reverse=True)[:k]
    return [{"label": l, "score": s} for s, l in sims]
