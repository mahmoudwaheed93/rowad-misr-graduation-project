from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from app.core.config import settings
from app.api.routes import router

BASE_DIR = Path(__file__).resolve().parent.parent
UI_INDEX = BASE_DIR / "ui" / "index.html"

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Egyptian Real Estate RAG & Agentic AI Backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def serve_portal():
    if UI_INDEX.exists():
        return FileResponse(UI_INDEX)
    return HTMLResponse(
        f"<h3>ui/index.html not found at {UI_INDEX}</h3><p>Go to <a href='/docs'>/docs</a></p>",
        status_code=404,
    )


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": settings.PROJECT_NAME}