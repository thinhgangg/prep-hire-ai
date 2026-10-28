
from dotenv import load_dotenv
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.api.v1.api import api_router
from app.core.config import settings

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

DEMO_DIR = Path(__file__).resolve().parent / "static" / "interview_demo"
app.mount("/demo-assets", StaticFiles(directory=DEMO_DIR), name="demo-assets")


@app.get("/demo", include_in_schema=False)
def interview_demo() -> FileResponse:
    return FileResponse(DEMO_DIR / "index.html")
