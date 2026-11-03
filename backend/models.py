"""Optional heavy models. Each loads lazily when ENABLE_MODELS=1 and the package exists; otherwise a classical fallback runs and the response says so."""
import os, io, base64, numpy as np, cv2
ON = os.getenv("ENABLE_MODELS", "0") == "1"
def b64(a, ext=".png"): return base64.b64encode(cv2.imencode(ext, cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if a.ndim == 3 else a)[1]).decode()

def free_vram():
    try:
        import torch, gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception: pass

def segment(a):
    """GrabCut fallback; SAM 2 / Grounding DINO plug in here when installed."""
    h, w = a.shape[:2]; mask = np.zeros((h, w), np.uint8); rect = (w // 10, h // 10, w * 8 // 10, h * 8 // 10)
    cv2.grabCut(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == 1) | (mask == 3), 255, 0).astype("uint8")
    return {"engine": "grabcut-fallback", "coverage": float(m.mean() / 255), "mask_png_b64": b64(m)}

def ocr(a):
    try:
        import easyocr
        if not ON: raise ImportError
        r = easyocr.Reader(["en"], gpu=False).readtext(a); free_vram()
        return {"engine": "easyocr", "items": [{"text": t, "conf": float(c), "box": [list(map(int, p)) for p in b]} for b, t, c in r]}
    except Exception: return {"engine": "unavailable", "items": [], "note": "Install easyocr and set ENABLE_MODELS=1"}

def upscale(a, f=2):
    out = cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_LANCZOS4)
    out = cv2.addWeighted(out, 1.4, cv2.GaussianBlur(out, (0, 0), 1.2), -0.4, 0)  # Real-ESRGAN plugs in here
    return {"engine": "lanczos+unsharp-fallback", "size": [out.shape[1], out.shape[0]], "png_b64": b64(out)}

def depth_contours(a, levels=12):
    """Depth Anything V2 when enabled; else luminance-as-depth approximation (labelled)."""
    engine = "luminance-approximation"; d = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (0, 0), 3)
    if ON:
        try:
            from transformers import pipeline; from PIL import Image
            p = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
            d = np.array(p(Image.fromarray(a))["depth"].convert("L").resize((a.shape[1], a.shape[0]))); engine = "depth-anything-v2"; free_vram()
        except Exception: pass
    q = (d.astype(float) / 256 * levels).astype(np.uint8); edges = cv2.Canny(q * (255 // levels), 10, 30)
    return {"engine": engine, "contours_png_b64": b64(255 - edges)}

def forensics(a):
    """Error-level analysis heuristic. A manipulation hint, NOT a deepfake verdict."""
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(a).save(buf, "JPEG", quality=90); re = np.array(Image.open(buf).convert("RGB"))
    ela = np.abs(a.astype(float) - re.astype(float)).mean(axis=2); score = float(min(1, ela.std() / 6))
    res = {"manipulation_hint": round(score, 3), "method": "ELA heuristic", "nsfw": None}
    if ON:
        try:
            from transformers import pipeline; res["nsfw"] = pipeline("image-classification", model="Falconsai/nsfw_image_detection")(Image.fromarray(a)); free_vram()
        except Exception: pass
    return res
