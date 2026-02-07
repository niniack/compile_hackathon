import os, re
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import trafilatura
from lingodotdev import LingoDotDevEngine
from huggingface_hub import InferenceClient

app = FastAPI()

# Free HF Inference API client (needs HF_TOKEN in .env)
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"), provider="novita")


# --- Sentence tracker (server-side state) ---

class SentenceTracker:
    def __init__(self):
        self.paragraphs = []   # [{text, sentences}]
        self.sentences = []    # flat list of sentence strings
        self.states = []       # 'pending' | 'done' per sentence
        self.translations = {} # idx -> user's accepted translation text
        self.lingo = {}         # idx -> cached lingo.dev translation (one-shot)
        self.scores = {}        # idx -> latest judge score (1-10)
        self.focus = 0

    def load(self, paragraphs):
        self.paragraphs = paragraphs
        self.sentences = [s for p in paragraphs for s in p["sentences"]]
        self.states = ["pending"] * len(self.sentences)
        self.translations = {}
        self.lingo = {}
        self.scores = {}
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
            "scores": self.scores,
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


# --- /judge (evaluate translation) ---

class JudgeRequest(BaseModel):
    idx: int
    user_translation: str

@app.post("/judge")
async def judge_translation(req: JudgeRequest):
    idx = req.idx

    if idx < 0 or idx >= len(tracker.sentences):
        raise HTTPException(status_code=400, detail="Invalid sentence index")

    if idx not in tracker.lingo:
        raise HTTPException(status_code=400, detail="No Lingo.dev translation available for this sentence")

    lingo_translation = tracker.lingo[idx]
    user_translation = req.user_translation

    prompt = (
        "You are a translation quality judge. Compare a user's translation against a ground-truth reference.\n"
        "Rate the user translation from 1-10 and give one-sentence feedback on accuracy and fluency. "
        "Don't reveal the true translation, they might try again.\n\n"
        f"Ground Truth: {lingo_translation}\n"
        f"User Translation: {user_translation}\n\n"
        "Judgment:"
    )

    try:
        response = hf_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="meta-llama/Llama-3.1-8B-Instruct",
            max_tokens=200,
        )
        judgment = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM evaluation failed: {str(e)}")

    # Extract numeric score from judgment text
    score = None
    m = re.search(r'(\d+)\s*/\s*10', judgment)
    if m:
        score = int(m.group(1))
    else:
        m = re.search(r'\b(10|[1-9])\b', judgment)
        if m:
            score = int(m.group(1))

    if score is not None:
        tracker.scores[idx] = score

    return {"judgment": judgment, "score": score}


# --- serve frontend ---

@app.get("/")
async def index():
    return FileResponse("index.html")
