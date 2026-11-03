import io, numpy as np
from PIL import Image
from fastapi import HTTPException
MAX_BYTES = 50 * 1024 * 1024
MAGIC = {b"\x89PNG\r\n\x1a\n": "png", b"\xff\xd8\xff": "jpeg", b"RIFF": "webp"}
def load(data: bytes) -> np.ndarray:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File is over 50 MB")
    if not any(data.startswith(m) for m in MAGIC) or (data[:4] == b"RIFF" and data[8:12] != b"WEBP"):
        raise HTTPException(415, "Unsupported file type (PNG, JPEG, WebP only)")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    clean = Image.new("RGB", img.size); clean.putdata(list(img.getdata()))  # drops EXIF
    return np.array(clean)

