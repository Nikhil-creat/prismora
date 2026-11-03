import os, base64, numpy as np, cv2
from celery import Celery
import models
app = Celery("prismora", broker=os.getenv("REDIS_URL", "redis://redis:6379/0"), backend=os.getenv("REDIS_URL", "redis://redis:6379/0"))
@app.task
def upscale_task(png_b64: str, factor: int = 2):
    a = cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(png_b64), np.uint8), 1), cv2.COLOR_BGR2RGB); r = models.upscale(a, factor); models.free_vram(); return r
