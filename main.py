from typing import Optional
from pathlib import Path
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from screenshot import collect_image, url_path
from pharec import Pharec

origins = [
    "http://localhost:4000",
    "http://localhost:8000",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api/v1")

image_size = (256, 512) # height, width
model_path = "models/2021-12-11_18;40;33.772059_wpd2_valacc0.9462_e8_b16.tf"

pharec = Pharec(model_path, image_size)

"""
    One get("/") endpoint to serve frontend
    One post("/check_url") endpoint to screenshot and run model
"""

class check_url_req(BaseModel):
    url: str
    description: Optional[str] = None

@app.get("/")
def read_root():
    return {"Hello": "World"}

@router.post("/check_url")
def read_item(url_req: check_url_req):
    url = url_req.url
    image_path = f"collected_images/img_{url_path(url)}.png" 
    if not Path(image_path).exists():
        collect_image(url)
        print("Screenshot taken... : path", image_path)

    image = pharec.load_image(image_path)
    print("Image loaded... : image shape", image.shape)
    pred_domain = pharec.predict_domain(image)
    print("Predicted domain:", pred_domain)

    return {"url": url_req.url, "predicted_domain": pred_domain}

app.include_router(router)
