from typing import Optional

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from screenshot import collect_image

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


@router.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None, a: Optional[int] = None):
    return {"item_id": item_id, "q": q, "a": a}

@router.post("/check_url")
def read_item(url_req: check_url_req):
    return {"url": url_req.url, "pass": 1}

app.include_router(router)
