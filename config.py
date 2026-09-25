import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is missing from .env")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is missing from .env")


# =========================
# Wiki configuration
# =========================

WIKI_INDEX_PATH = "wiki_index/index.faiss"
WIKI_METADATA_PATH = "wiki_index/metadata.json"


# =========================
# AI configuration
# =========================

MODEL = "openai/gpt-oss-20b"

MAX_RESPONSE_WORDS = 200


# =========================
# RAG configuration
# =========================

INITIAL_CHUNKS = 80

MIN_ARTICLES = 3
MAX_ARTICLES = 5

MAX_CHUNKS_PER_ARTICLE = 2

# Maximum relative similarity drop allowed
# when deciding whether to add another article.
ARTICLE_SCORE_DROP = 0.05