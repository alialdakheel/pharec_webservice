from typing import Optional
from pathlib import Path
from fastapi import FastAPI, APIRouter, Request
from pydantic import BaseModel, constr
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from screenshot import collect_image, url_path
from pharec import Pharec

app = FastAPI()

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200/minute"],
)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

router = APIRouter(prefix="/api/v1")

image_size = (256, 512) # height, width
model_path = "models/2021-12-11_18;40;33.772059_wpd2_valacc0.9462_e8_b16.tf"

pharec = Pharec(model_path, image_size)

"""
    One get("/") endpoint to serve frontend
    One post("/check_url") endpoint to screenshot and run model
"""

class check_url_req(BaseModel):
    url: constr(min_length=8, max_length=26)
    description: Optional[str] = None

@router.post("/check_url")
@limiter.limit("100/minute")
def read_item(request: Request, url_req: check_url_req):
    url = url_req.url
    image_path = f"collected_images/img_{url_path(url)}.png" 
    if not Path(image_path).exists():
        collect_image(url)
        print("Screenshot taken... : path", image_path)

    image = pharec.load_image(image_path)
    print("Image loaded... : image shape", image.shape)
    pred_domain = pharec.predict_domain(image)
    print("Predicted domain:", pred_domain)

    return {
        "url": url_req.url,
        "predicted_domain": pred_domain,
        "image_path": image_path
    }

app.include_router(router)
app.mount('/collected_images', StaticFiles(directory='collected_images'))
app.mount('/', StaticFiles(directory='pharec_frontend/dist', html=True))
