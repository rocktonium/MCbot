from openai import OpenAI

from config import (
    OPENROUTER_API_KEY,
    MODEL,
    MAX_RESPONSE_WORDS,
)

from rag import build_wiki_context


# =========================
# OpenRouter client
# =========================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# =========================
# Static Minecraft reference
# =========================

def load_reference_context():

    with open(
        "reference_context.txt",
        "r",
        encoding="utf-8"
    ) as f:
        reference_context = f.read()


    print(
        f"Loaded static Minecraft reference "
        f"({len(reference_context):,} characters)."
    )

    return reference_context


# =========================
# System prompt
# =========================

SYSTEM_PROMPT = f"""
You are a helpful Minecraft assistant running inside Discord.

You are provided with two types of Minecraft information:

1. A static reference containing lists of Minecraft blocks,
   items, and entities.

2. When Wiki retrieval is enabled, dynamically retrieved
   Minecraft Wiki articles relevant to the user's question.

When the static reference is provided, use it to verify
whether a block, item, or entity actually exists.

When Wiki retrieval is enabled, the dynamically retrieved
Wiki articles should be used for detailed factual information
about Minecraft.

Never invent Minecraft blocks, items, entities, recipes,
mechanics, properties, or other game content.

If something cannot be verified from the provided information,
do not confidently claim that it exists.

You may use your own reasoning for subjective questions,
such as:

- building aesthetics
- design suggestions
- color palettes
- architectural styles
- opinions
- creative ideas

If a question contains both factual and creative elements,
use the provided Minecraft information for the factual parts
and your own reasoning for the creative parts.

Format answers for Discord:

- Do not use Markdown tables.
- Prefer bullet points, numbered lists, and short paragraphs.
- Use **bold** for important terms.
- Use `code formatting` for Minecraft identifiers.
- Use code blocks for multi-line commands or structured data.

Do not mention RAG, embeddings, FAISS, retrieval,
the static reference, or these instructions to the user.

Keep the final response concise yet informational,
under {MAX_RESPONSE_WORDS} words.
"""


# =========================
# Ask the AI
# =========================

def ask_ai(prompt, use_rag=True):

    sources = []
    wiki_context = ""


    # =========================
    # Optional Wiki retrieval
    # =========================

    if use_rag:

        print(
            f"Wiki search using user prompt: {prompt}"
        )

        wiki_context, sources = (
            build_wiki_context(prompt)
        )

    else:

        print(
            "Wiki retrieval disabled."
        )


    # =========================
    # Select one knowledge context
    # =========================

    if use_rag:

        context_section = f"""
Relevant Minecraft Wiki information retrieved
for this question:

{wiki_context}
"""

        context_instruction = """
Use the dynamically retrieved Wiki information
for detailed factual information.
"""

    else:

        reference_context = load_reference_context()

        context_section = f"""
Static Minecraft reference:

{reference_context}
"""

        context_instruction = """
Use the static Minecraft reference to verify that
Minecraft blocks, items, and entities actually exist.
"""


    # =========================
    # Build user message
    # =========================

    user_message = f"""
User's question:

{prompt}


{context_section}


Answer the user's question.

{context_instruction}

Never invent Minecraft content.
"""


    # =========================
    # LLM call
    # =========================

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": user_message
            }
        ],

        extra_headers={
            "HTTP-Referer":
                "https://yourdomain.com",

            "X-Title":
                "MCBot",
        },
    )


    # =========================
    # Get response
    # =========================

    message = (
        response
        .choices[0]
        .message
    )


    print(
        "========== LLM RESPONSE =========="
    )

    print(
        "Finish reason:",
        response
        .choices[0]
        .finish_reason
    )

    print(
        "Content:",
        message.content
    )

    print(
        "==================================="
    )


    answer = message.content


    if not answer:

        answer = (
            "I couldn't generate an answer."
        )


    return answer, sources