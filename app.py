import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import trafilatura
from lingodotdev import LingoDotDevEngine

app = FastAPI()


# --- /load-article ---

class ArticleRequest(BaseModel):
    url: str

@app.post("/load-article")
async def load_article(req: ArticleRequest):
    html = trafilatura.fetch_url(req.url)
    text = trafilatura.extract(html)
    if not text:
        return {"error": "Could not extract article"}

    paragraphs_en = [p.strip() for p in text.split("\n") if p.strip()]

    source = {str(i): p for i, p in enumerate(paragraphs_en)}
    async with LingoDotDevEngine({"api_key": os.getenv("LINGODOTDEV_API_KEY")}) as engine:
        translated = await engine.localize_object(
            source, {"source_locale": "en", "target_locale": "zh-Hans"}
        )
    paragraphs_zh = [translated.get(str(i), p) for i, p in enumerate(paragraphs_en)]

    return {"en": paragraphs_en, "zh": paragraphs_zh}


# --- serve frontend ---

@app.get("/")
async def index():
    return FileResponse("index.html")
