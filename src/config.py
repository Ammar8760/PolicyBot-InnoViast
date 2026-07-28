"""
Central configuration.

Poore project ki har "tunable" value yahan ek jagah hai. Isse do faide hain:
  1. Experiment karna asaan — chunk size badalna ho to sirf yahan badlo
  2. Evaluation reproducible rehti hai — settings document ho jati hain
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# .env file se environment variables load karo (agar mojood ho)
load_dotenv()


# ---------------------------------------------------------------- paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
UPLOADS_DIR = PROJECT_ROOT / "knowledge_base" / "_uploads"


# ------------------------------------------------------------ api / model --
def get_secret(name: str, default: str = "") -> str:
    """
    Secret dhoondo. Do jagah dekhte hain kyunki:
      - Local development mein secrets .env file mein hote hain
      - Streamlit Cloud par "Secrets" panel mein hote hain

    Secret kabhi bhi code mein hardcode nahi karna.
    """
    value = os.getenv(name, "").strip()
    if value:
        return value

    # Streamlit secrets sirf tab available hote hain jab app chal rahi ho
    try:
        import streamlit as st

        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


def get_api_key() -> str:
    """Gemini API key (embeddings + generation dono ke liye)."""
    return get_secret("GOOGLE_API_KEY")


CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Gemini ka embedding model 3072 dimensions deta hai. Hum 768 par truncate
# karte hain — quality ka farq bohot kam hai lekin storage/speed behtar.
EMBEDDING_DIM = 768


# -------------------------------------------------------------- chunking ---
# CHUNK_SIZE = ek tukde mein max kitne characters
#   - Bohot chota (200) => context toot jata hai, jawab adhoora
#   - Bohot bara (3000) => ek chunk mein 5 topics, retrieval ghalat
CHUNK_SIZE = 900

# OVERLAP = do consecutive chunks kitna share karte hain
#   - Isse boundary par toota hua sentence dono chunks mein reh jata hai
CHUNK_OVERLAP = 150


# -------------------------------------------------------- vector backend ---
# Do backends support hote hain — interface bilkul same hai, isliye baqi
# poore code ko farq nahi parta ke neeche kya chal raha hai.
#
#   "chroma"   -> local disk par (chroma_db/ folder)
#                 Tez, offline, koi signup nahi. LOCAL DEVELOPMENT ke liye.
#
#   "pinecone" -> cloud par (managed serverless vector DB)
#                 Persistent rehta hai. DEPLOYMENT ke liye — kyunki
#                 Streamlit Cloud ka filesystem ephemeral hai aur restart
#                 par chroma_db/ mit jata hai.
#
# .env mein ye line daal kar switch karein:  VECTOR_BACKEND=pinecone
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma").strip().lower()

COLLECTION_NAME = "policy_kb"

# --- Pinecone-specific (sirf VECTOR_BACKEND=pinecone par istemal hote hain)
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "policybot").strip()
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "policy-kb").strip()

# Free (Starter) tier par serverless indexes sirf aws / us-east-1 mein
# banti hain. Paid plan par ye badla ja sakta hai.
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws").strip()
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1").strip()


def get_pinecone_key() -> str:
    return get_secret("PINECONE_API_KEY")

# Har sawal par kitne chunks laane hain
TOP_K = 4

# Fallback ka faisla isi number se hota hai.
# Cosine similarity 0.0 (bilkul mukhtalif) se 1.0 (bilkul same) tak hoti hai.
# Agar best chunk ka score is se kam hai => hum maante hain jawab KB mein nahi
# hai, aur LLM ko call kiye baghair fallback message dikha dete hain.
#
# ---------------------------------------------------------------------------
# CALIBRATION (`python -m tools.calibrate`, 37 chunks, gemini-embedding-001):
#
#   IN-SCOPE      (13 sawal)  min 0.637   max 0.758   avg 0.692
#   OUT-OF-SCOPE  ( 4 sawal)  min 0.536   max 0.671   avg 0.592
#
# 0.60 chuna gaya kyunki:
#   - Har in-scope sawal pass hota hai (sab se kamzor 0.637 > 0.60)
#   - 4 mein se 3 out-of-scope sawal yahin ruk jate hain
#
# Ranges thori OVERLAP karti hain: "Nexora ka CEO kaun hai?" 0.671 leta hai
# kyunki "Nexora Technologies" har document ke header mein hai — halanki CEO
# ka zikr kahin nahi. Yani koi bhi threshold akela poora kaam nahi kar sakti.
#
# Isi liye teen layers hain: is case ko Layer 2 (system prompt) pakarta hai,
# aur model sahi tor par mana kar deta hai. Threshold ek SASTA PEHLA FILTER
# hai — mukammal hal nahi. (Verify: python -m tools.ask "Nexora ka CEO kaun hai?")
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.60


# ------------------------------------------------------------- generation --
# Temperature kam rakhi hai (0.1) — hum creativity nahi chahte, hum chahte
# hain ke model bilkul context ke mutabiq jawab de.
TEMPERATURE = 0.1

# Kushada rakha hai. Gemini 2.5 mein thinking tokens bhi isi budget se katte
# hain (hum generator.py mein thinking band kar dete hain, lekin phir bhi
# margin rakhna behtar hai — kam budget ka natija khaali jawab hota hai).
MAX_OUTPUT_TOKENS = 2048

APP_TITLE = "PolicyBot"
APP_SUBTITLE = "Nexora Technologies — HR & Policy Assistant"
