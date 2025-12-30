import io, torch, uvicorn

from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image

from model_clip import CLIPModel
from h3_classification import H3Classifier

COUNTRIES = [
    'AL', 'AD', 'AR', 'AU', 'AT', 'BD', 'BE', 'BT', 'BO', 'BW', 'BR', 'BG', 'KH', 'CA', 'CL', 'CO',
    'HR', 'CZ', 'DK', 'DO', 'EC', 'EE', 'SZ', 'FI', 'FR', 'DE', 'GH', 'GR', 'GL', 'GT', 'HU', 'IS',
    'IN', 'ID', 'IE', 'IL', 'IT', 'JP', 'JO', 'KE', 'KG', 'LV', 'LB', 'LS', 'LI', 'LT', 'LU', 'MY',
    'MX', 'MN', 'ME', 'NA', 'NL', 'NZ', 'NG', 'MK', 'NO', 'OM', 'PS', 'PA', 'PE', 'PH', 'PL', 'PT',
    'QA', 'RO', 'RU', 'RW', 'SM', 'ST', 'SN', 'RS', 'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'LK', 'SE',
    'CH', 'TW', 'TH', 'TR', 'TN', 'UA', 'UG', 'AE', 'GB', 'US', 'UY', 'VN',
]


class GeoResponse(BaseModel):
    lat: float = Field(...)
    lon: float = Field(...)
    confidence: float = Field(...)


def load_model_and_classes(ckpt_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        "device": device,
        "cache_dir": "cache",
        "eval_interval": 500,
        "countries": COUNTRIES,
        "num_countries": len(COUNTRIES),
        "steps": 7500,
        "max_lr_head": 1e-4,
        "batch_size": 512,
        "accum_steps": 1,
        "classes": 861,
        "tau_km": 75,
        "model": "ViT-L/14@336px",
        'backbone_unfrozen': True,
        'h3_resolution': 2,
        'h3_mappings': 'h3_utils/h3_to_class_res2_min200_ring20.json',
        'h3_counts': 'h3_utils/h3_counts_res2.json',
    }

    classifier = H3Classifier(config)
    device = config["device"]

    print(f"Using device: {device}")
    print("Loading model architecture...")
    model = CLIPModel(config).to(device)

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raw_state = checkpoint["state_dict"]
    else:
        raw_state = checkpoint

    fixed_state = {}
    for k, v in raw_state.items():
        new_k = k
        # 2. Clean standard prefixes
        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod."):]
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        
        if hasattr(model, 'model') and not new_k.startswith('model.'):
             pass 
        elif not hasattr(model, 'model') and new_k.startswith('model.'):
             new_k = new_k[len("model."):]

        fixed_state[new_k] = v
    try:
        model.load_state_dict(fixed_state, strict=True)
        print("Success: All weights (including ViT) loaded strictly.")
    except RuntimeError as e:
        print(f"Strict loading failed. Missing/Unexpected keys:\n{e}")
        res = model.load_state_dict(fixed_state, strict=False)
        print("Loaded with strict=False. Missing keys (weights not updated):", res.missing_keys)

    model.eval()

    print("Loading class centers with get_clusters...")
    class_centers = classifier.CLASS_CENTERS_XYZ

    if class_centers.shape[1] < 2:
        raise ValueError(
            f"Expected cluster_centers to have at least 2 columns (lat, lon), "
            f"got shape: {class_centers.shape}"
        )

    print("Model and classes loaded.")
    return model, class_centers, config


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing model...")
    model, class_centers, config = load_model_and_classes('models/neuroguessr-861-large-acw-streetview-h3-unfrozen-2-best.pth')
    model.eval()

    app.state.model = model
    app.state.class_centers = class_centers
    app.state.config = config
    print("Ready for production.")
    yield


app = FastAPI(title="GeoAPI", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "geo ok"}


@app.post("/geolocate", response_model=GeoResponse)
async def geolocate(request: Request, file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="`file` must not be empty.")
    
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))

    model = request.app.state.model
    class_centers = request.app.state.class_centers
    config = request.app.state.config

    try:
        with torch.inference_mode():
            lat, lon, confidence = model.guess(class_centers, config, img)
        return GeoResponse(lat=lat, lon=lon, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geolocation failed: {e}")
    

if __name__ == "__main__":
    print("Starting Geo API server...")
    uvicorn.run(app, host="0.0.0.0", port=1616)
