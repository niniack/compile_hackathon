import os, re
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import trafilatura
from lingodotdev import LingoDotDevEngine

app = FastAPI()


# --- Sentence tracker (server-side state) ---

class SentenceTracker:
    def __init__(self):
        self.paragraphs = []   # [{text, sentences}]
        self.sentences = []    # flat list of sentence strings
        self.states = []       # 'pending' | 'done' per sentence
        self.translations = {} # idx -> user's accepted translation text
        self.lingo = {}         # idx -> cached lingo.dev translation (one-shot)
        self.focus = 0

    def load(self, paragraphs):
        self.paragraphs = paragraphs
        self.sentences = [s for p in paragraphs for s in p["sentences"]]
        self.states = ["pending"] * len(self.sentences)
        self.translations = {}
        self.lingo = {}
        self.focus = 0

    def set_focus(self, idx):
        if 0 <= idx < len(self.sentences):
            self.focus = idx

    def accept(self, idx):
        if 0 <= idx < len(self.sentences):
            self.states[idx] = "done"
            # auto-advance to next pending
            for i in range(idx + 1, len(self.sentences)):
                if self.states[i] == "pending":
                    self.focus = i
                    return
            self.focus = -1  # all done

    def to_dict(self):
        return {
            "paragraphs": self.paragraphs,
            "sentences": self.sentences,
            "states": self.states,
            "translations": self.translations,
            "lingo": self.lingo,
            "focus": self.focus,
            "done": sum(1 for s in self.states if s == "done"),
            "total": len(self.sentences),
        }

tracker = SentenceTracker()


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
    result = []
    for p in paragraphs:
        sents = split_sentences(p)
        result.append({"text": p, "sentences": sents})

    tracker.load(result)
    return tracker.to_dict()


# --- /state, /focus, /accept ---

@app.get("/state")
def get_state():
    return tracker.to_dict()

class FocusRequest(BaseModel):
    idx: int

@app.post("/focus")
def set_focus(req: FocusRequest):
    tracker.set_focus(req.idx)
    return tracker.to_dict()

class AcceptRequest(BaseModel):
    idx: int
    text: str = ""

@app.post("/accept")
def accept(req: AcceptRequest):
    tracker.translations[req.idx] = req.text
    tracker.accept(req.idx)
    return tracker.to_dict()


# --- /lingo (one-shot cached translation) ---

class LingoRequest(BaseModel):
    idx: int

@app.post("/lingo")
async def lingo_translate(req: LingoRequest):
    idx = req.idx
    # Already cached? Return immediately, no API call
    if idx in tracker.lingo:
        return tracker.to_dict()

    if idx < 0 or idx >= len(tracker.sentences):
        return {"error": "Invalid sentence index"}

    api_key = os.getenv("LINGODOTDEV_API_KEY")
    if not api_key:
        return {"error": "LINGODOTDEV_API_KEY not set"}

    async with LingoDotDevEngine({"api_key": api_key}) as engine:
        result = await engine.localize_text(
            tracker.sentences[idx],
            {"source_locale": "zh-Hans", "target_locale": "en"}
        )
    tracker.lingo[idx] = result
    return tracker.to_dict()


# --- serve frontend ---

@app.get("/")
async def index():
    return FileResponse("index.html")
