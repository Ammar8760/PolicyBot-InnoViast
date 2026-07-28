"""
Generation — context + sawal se grounded jawab banana.

Assignment ka Quality Bar:
    "Responses must be grounded in the knowledge base and should not
     invent unsupported facts."

Grounding ki teen layers hain is project mein:
  Layer 1 (retriever.py) : score kam ho to LLM ko call hi nahi karte
  Layer 2 (yahan)        : system prompt mein sakht rules
  Layer 3 (app.py)       : sources UI par dikhate hain, user khud verify kare

Sirf prompt par bharosa karna kaafi nahi hota — isliye teeno layers.
"""

from __future__ import annotations

from dataclasses import dataclass

from google.genai import types

from . import config, embedder
from .retriever import RetrievalResult, build_context

# ------------------------------------------------------------ system prompt --
SYSTEM_PROMPT = """You are PolicyBot, an HR and policy assistant for Nexora Technologies.

You answer ONLY from the CONTEXT provided in the user message. The context
contains excerpts from the company's official policy documents.

RULES — follow these strictly:

1. GROUNDING: Every factual claim in your answer must come from the context.
   Never use outside knowledge. Never guess. Never fill gaps with what is
   "usually" true at other companies.

2. INSUFFICIENT CONTEXT: If the context does not contain enough information
   to answer, reply with exactly:
   "I could not find this in the knowledge base."
   Then name the topics the context DOES cover, so the user can rephrase.

3. PARTIAL ANSWERS: If the context answers part of the question, answer that
   part, then state clearly which part is not covered by the documents.

4. CITATIONS: After each fact, cite the source file in square brackets,
   e.g. [leave_policy.md]. If a fact combines two sources, cite both.

5. NUMBERS: Quote figures, dates, and limits exactly as written in the
   context. Do not round, convert, or recalculate them.

6. STYLE: Be concise and direct. Use short paragraphs or bullets. Do not
   start with filler like "Based on the provided context...". Just answer.

7. LANGUAGE: Reply in the same language the user asked in. If the user
   writes in Roman Urdu, reply in Roman Urdu — but keep policy terms,
   figures, and file names exactly as they appear in the context.
"""

USER_TEMPLATE = """CONTEXT:
{context}

---

QUESTION: {question}"""


@dataclass
class Answer:
    """Ek mukammal jawab — text + sources + kya ye fallback tha."""

    text: str
    result: RetrievalResult
    is_fallback: bool

    @property
    def sources(self):
        return self.result.hits


# ---------------------------------------------------------------- fallback --
def fallback_message(result: RetrievalResult, available_topics: list[str]) -> str:
    """
    Fallback sirf "I don't know" nahi hona chahiye. Assignment kehta hai
    fallback ko user ko GUIDE karna chahiye. Isliye hum batate hain:
      - kya nahi mila
      - kitna qareeb tha (transparency)
      - KB mein kya kya hai
      - aage kya karein
    """
    topics = "\n".join(f"- {t}" for t in available_topics) or "- (knowledge base khaali hai)"

    closest = ""
    if result.all_hits:
        best = result.all_hits[0]
        closest = (
            f"\nSabse qareeb match `{best.source}` mein tha, lekin uska relevance "
            f"score **{best.score:.2f}** hai — hamari threshold **{result.threshold:.2f}** "
            f"se kam. Is liye maine andaza lagane ke bajaye rukna behtar samjha.\n"
        )

    return (
        "**Is sawal ka jawab knowledge base mein nahi mila.**\n"
        f"{closest}\n"
        "**Knowledge base mein ye documents mojood hain:**\n"
        f"{topics}\n\n"
        "**Aap ye kar sakte hain:**\n"
        "- Sawal ko policy ke alfaaz mein dobara likhein "
        "(jaise *\"leave\"*, *\"reimbursement\"*, *\"remote work\"*)\n"
        "- Sidebar se mutalliqa document upload karein\n"
        "- Sidebar mein relevance threshold thori kam karein aur dobara poochein"
    )


# -------------------------------------------------------------- generation --
def generate(result: RetrievalResult, available_topics: list[str]) -> Answer:
    """Retrieval ke natije se final jawab banao."""

    # Layer 1 — kaafi relevant context nahi mila? LLM ko call hi mat karo.
    if not result.has_context:
        return Answer(
            text=fallback_message(result, available_topics),
            result=result,
            is_fallback=True,
        )

    client = embedder.get_client()  # wahi client reuse kar rahe hain
    context = build_context(result.hits)
    prompt = USER_TEMPLATE.format(context=context, question=result.question)

    base_kwargs = dict(
        system_instruction=SYSTEM_PROMPT,
        temperature=config.TEMPERATURE,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
    )

    # Gemini 2.5 models mein "thinking" default on hoti hai. Woh reasoning
    # tokens bhi max_output_tokens ke budget se hi katte hain — agar budget
    # thinking mein khatam ho jaye to response.text KHAALI aa jata hai.
    #
    # Hamare use-case mein deep reasoning ki zaroorat nahi: context pehle se
    # diya hua hai, model ko sirf usko parh kar summarise karna hai. Isliye
    # thinking band kar dete hain — jawab tez bhi aata hai aur sasta bhi.
    try:
        response = client.models.generate_content(
            model=config.CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                **base_kwargs,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    except Exception:
        # Purane models (jaise gemini-2.0-flash) thinking_config support
        # nahi karte — un par bina us parameter ke chalao.
        response = client.models.generate_content(
            model=config.CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(**base_kwargs),
        )

    text = (response.text or "").strip()
    if not text:
        text = (
            "Model se koi jawab nahi aaya. Baraye meherbani sawal dobara poochein, "
            "ya thora mukhtalif alfaaz mein likhein."
        )

    # Agar model ne khud kaha ke nahi mila, to usse bhi fallback maano —
    # taake UI usko warning ke tor par dikhaye, normal answer ke tor par nahi.
    model_declined = "could not find this in the knowledge base" in text.lower()

    return Answer(text=text, result=result, is_fallback=model_declined)
