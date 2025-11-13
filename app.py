import base64
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

import numpy as np
from PIL import Image
import torch

from floodrisk import UNet, load_population

app = FastAPI(title='FloodRisk AI API')

# lazy global model (loaded on first request)
_model = None

def get_model(weights_path: str = 'models/unet.pt', base: int = 32):
    global _model
    if _model is None:
        m = UNet(in_channels=3, base=base)
        try:
            m.load_state_dict(torch.load(weights_path, map_location='cpu'))
        except Exception:
            # if no weights, start with random (demo only)
            pass
        m.eval()
        _model = m
    return _model

def encode_png(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')

@app.post('/segment')
async def segment(image: UploadFile = File(...), threshold: float = Form(0.5), size: int = Form(256)):
    content = await image.read()
    img = Image.open(BytesIO(content)).convert('RGB').resize((size, size))
    x = torch.from_numpy(np.array(img).transpose(2,0,1)).float()/255.0
    x = x.unsqueeze(0)
    model = get_model()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0,0].numpy()
    mask = (probs>threshold).astype(np.uint8)*255
    # overlay
    img_np = np.array(img)
    overlay = img_np.copy()
    overlay[mask>0] = (overlay[mask>0]*0.3 + np.array([0,0,255])*0.7).astype(np.uint8)
    return JSONResponse({
        'mask_png_base64': encode_png(mask),
        'overlay_png_base64': encode_png(overlay)
    })

@app.post('/risk')
async def risk(image: UploadFile = File(...), population: Optional[UploadFile] = File(None), threshold: float = Form(0.5), size: int = Form(256)):
    content = await image.read()
    img = Image.open(BytesIO(content)).convert('RGB').resize((size, size))
    x = torch.from_numpy(np.array(img).transpose(2,0,1)).float()/255.0
    x = x.unsqueeze(0)
    model = get_model()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0,0].numpy()
    mask = (probs>threshold).astype(np.float32)

    # load population grid
    pop_path = None
    if population is not None:
        tmp = await population.read()
        # save temp file to disk for loader simplicity
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='_'+population.filename) as tf:
            tf.write(tmp)
            pop_path = tf.name
    pop = load_population(pop_path, mask.shape)
    if pop.shape != mask.shape:
        from skimage.transform import resize
        pop = resize(pop, mask.shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    risk_score = float((mask * pop).sum())

    # heatmap
    import matplotlib.pyplot as plt
    import io
    plt.figure(figsize=(6,5))
    plt.imshow(pop, cmap='YlOrRd', alpha=0.6)
    plt.imshow(mask, cmap='Blues', alpha=0.4)
    plt.title(f'Flood × Population exposure (score={risk_score:.3f})')
    plt.colorbar(label='Population density (normalized)')
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', dpi=150)
    plt.close()
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

    return JSONResponse({'risk_score': risk_score, 'risk_heatmap_png_base64': heatmap_b64})