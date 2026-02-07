import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import trafilatura

app = FastAPI()

class ArticleRequest(BaseModel):
    url: str

@app.post("/load-article")
def load_article(req: ArticleRequest):
    html = trafilatura.fetch_url(req.url)
    text = trafilatura.extract(html)
    if not text:
        return {"error": "Could not extract article"}

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return {"paragraphs": paragraphs}


# --- serve frontend ---

@app.get("/")
async def index():
    return FileResponse("index.html")
