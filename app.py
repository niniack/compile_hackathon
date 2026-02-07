import os, re
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

def split_sentences(text):
    """Split Chinese text into sentences on common punctuation."""
    parts = re.split(r'(?<=[。！？；])', text)
    return [s.strip() for s in parts if s.strip()]

@app.post("/load-article")
def load_article(req: ArticleRequest):
    html = trafilatura.fetch_url(req.url)
    text = trafilatura.extract(html)
    if not text:
        return {"error": "Could not extract article"}

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    # Each paragraph becomes {text, sentences[]}
    result = []
    for p in paragraphs:
        sents = split_sentences(p)
        result.append({"text": p, "sentences": sents})
    return {"paragraphs": result}


# --- /translate ---

class TranslateRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate(req: TranslateRequest):
    api_key = os.getenv("LINGODOTDEV_API_KEY")
    if not api_key:
        return {"error": "LINGODOTDEV_API_KEY not set"}

    async with LingoDotDevEngine({"api_key": api_key}) as engine:
        result = await engine.localize_text(
            req.text, {"source_locale": "zh-Hans", "target_locale": "en"}
        )
    return {"translation": result}


# --- serve frontend ---

@app.get("/")
async def index():
    return FileResponse("index.html")
