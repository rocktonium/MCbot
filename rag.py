import json

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from config import (
    WIKI_INDEX_PATH,
    WIKI_METADATA_PATH,
    INITIAL_CHUNKS,
    MIN_ARTICLES,
    MAX_ARTICLES,
    MAX_CHUNKS_PER_ARTICLE,
    ARTICLE_SCORE_DROP,
)


# =========================
# Load embedding model
# =========================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================
# Load FAISS index
# =========================

print("Loading Minecraft Wiki index...")

wiki_index = faiss.read_index(
    WIKI_INDEX_PATH
)


# =========================
# Load metadata
# =========================

print("Loading Wiki metadata...")

with open(
    WIKI_METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    wiki_metadata = json.load(f)


print(
    f"Loaded {len(wiki_metadata)} Wiki chunks."
)


# =========================
# Permanent reference pages
# =========================

# These pages are ALWAYS included in the
# LLM context.
#
# They are NOT retrieved through similarity.
# They do NOT count toward the 3–8 dynamic
# articles.

ALWAYS_INCLUDED_PAGES = {
    "Block",
    "Item",
    "Entity",
}


def load_reference_pages():
    """
    Find every chunk belonging to the permanent
    Blocks, Items, and Entities pages.

    This happens once when the bot starts.
    """

    pages = {
        "Block": [],
        "Item": [],
        "Entity": [],
    }

    for chunk in wiki_metadata:

        title = chunk.get("title")

        if title in ALWAYS_INCLUDED_PAGES:

            pages[title].append(
                chunk
            )

    return pages


reference_pages = load_reference_pages()


print(
    "Loaded permanent Wiki reference pages:"
)

for title in [
    "Block",
    "Item",
    "Entity",
]:

    print(
        f"  {title}: "
        f"{len(reference_pages[title])} chunks"
    )


# =========================
# Build permanent context
# =========================

def build_reference_context():
    """
    Build the context for the permanent
    Blocks, Items, and Entities pages.

    This context is independent of the
    user's query.
    """

    context_parts = []

    for title in [
        "Block",
        "Item",
        "Entity",
    ]:

        chunks = reference_pages.get(
            title,
            []
        )

        if not chunks:

            print(
                f"Warning: permanent Wiki page "
                f"'{title}' was not found."
            )

            continue


        # Preserve the order in which the
        # chunks appear in metadata.json.
        #
        # This assumes metadata.json was
        # generated in page/chunk order,
        # which is normally what we want.
        page_context = (
            f"### Minecraft Wiki: {title}\n"
        )


        for chunk in chunks:

            section = chunk.get(
                "section",
                "Article"
            )

            page_context += (
                f"\n#### {section}\n"
            )

            page_context += (
                chunk.get("text", "")
                + "\n"
            )


        context_parts.append(
            page_context
        )


    return "\n\n".join(
        context_parts
    )


# Build this ONCE instead of rebuilding it
# for every user request.

PERMANENT_CONTEXT = (
    build_reference_context()
)


print(
    "Permanent Wiki context loaded."
)


# =========================
# Search Wiki
# =========================

def search_wiki(
    query,
    top_k=INITIAL_CHUNKS
):
    """
    Search the Minecraft Wiki FAISS index
    using the user's exact query.

    The LLM never generates this query.
    """

    # =========================
    # Create query embedding
    # =========================

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # =========================
    # FAISS search
    # =========================

    scores, indices = wiki_index.search(
        query_embedding,
        top_k
    )


    # =========================
    # Build results
    # =========================

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        # FAISS can return -1 if there
        # aren't enough results.
        if index < 0:
            continue


        result = wiki_metadata[index].copy()

        result["score"] = float(
            score
        )

        results.append(
            result
        )


    return results


# =========================
# Build Wiki context
# =========================

def build_wiki_context(query):
    """
    Build the complete Wiki context for an
    LLM request.

    Context consists of:

        1. Permanent Blocks page
        2. Permanent Items page
        3. Permanent Entities page
        4. 3–8 dynamically retrieved articles
    """

    # =========================
    # Semantic retrieval
    # =========================

    print(
        f"Searching Wiki with: {query}"
    )

    results = search_wiki(
        query,
        top_k=INITIAL_CHUNKS
    )


    # =========================
    # Group chunks by article
    # =========================

    articles = {}


    for result in results:

        title = result["title"]


        # The permanent pages are already
        # included separately, so don't let
        # them participate in dynamic retrieval.
        if title in ALWAYS_INCLUDED_PAGES:
            continue


        if title not in articles:

            articles[title] = []


        articles[title].append(
            result
        )


    # =========================
    # Rank articles
    # =========================

    article_scores = []


    for title, chunks in articles.items():

        if not chunks:
            continue


        # The strongest chunk determines
        # the article's relevance.
        best_score = max(
            chunk["score"]
            for chunk in chunks
        )


        article_scores.append({

            "title": title,

            "chunks": chunks,

            "score": best_score

        })


    # Highest similarity first.

    article_scores.sort(
        key=lambda article: article["score"],
        reverse=True
    )


    # =========================
    # Select minimum articles
    # =========================

    selected = article_scores[
        :min(
            MIN_ARTICLES,
            len(article_scores)
        )
    ]


    # =========================
    # Dynamically add articles
    # =========================

    for i in range(
        len(selected),
        min(
            MAX_ARTICLES,
            len(article_scores)
        )
    ):

        previous_score = (
            article_scores[i - 1]["score"]
        )

        current_score = (
            article_scores[i]["score"]
        )


        # Calculate relative drop.

        score_drop = (
            previous_score
            - current_score
        ) / max(
            abs(previous_score),
            0.0001
        )


        # If the next article is still
        # reasonably close in relevance,
        # include it.

        if score_drop <= ARTICLE_SCORE_DROP:

            selected.append(
                article_scores[i]
            )

        else:

            # Once relevance drops
            # significantly, stop.
            break


    # =========================
    # Build dynamic context
    # =========================

    dynamic_context_parts = []

    sources = []


    for article in selected:

        title = article["title"]

        chunks = article["chunks"]

        best_score = article["score"]


        # Sort chunks by similarity.

        chunks = sorted(
            chunks,
            key=lambda chunk: chunk["score"],
            reverse=True
        )


        # Only include the best chunks
        # from each dynamically retrieved
        # article.

        chunks = chunks[
            :MAX_CHUNKS_PER_ARTICLE
        ]


        article_context = (
            f"### Minecraft Wiki: {title}\n"
        )


        for chunk in chunks:

            section = chunk.get(
                "section",
                "Article"
            )


            article_context += (
                f"\n#### {section}\n"
            )


            article_context += (
                chunk.get("text", "")
                + "\n"
            )


        dynamic_context_parts.append(
            article_context
        )


        # Store source information for
        # Discord's retrieval display.

        sources.append({

            "title": title,

            "url": chunks[0].get(
                "url",
                ""
            ),

            "score": best_score

        })


    # =========================
    # Combine contexts
    # =========================

    dynamic_context = (
        "\n\n".join(
            dynamic_context_parts
        )
    )


    # Permanent pages always come first.

    if PERMANENT_CONTEXT:

        if dynamic_context:

            context = (
                PERMANENT_CONTEXT
                + "\n\n"
                + dynamic_context
            )

        else:

            context = (
                PERMANENT_CONTEXT
            )

    else:

        context = dynamic_context


    # =========================
    # Debugging
    # =========================

    print(
        "Wiki retrieval complete:"
    )

    print(
        f"  Permanent pages: "
        f"{len(ALWAYS_INCLUDED_PAGES)}"
    )

    print(
        f"  Dynamic articles: "
        f"{len(selected)}"
    )


    for source in sources:

        print(
            f"  {source['title']}: "
            f"{source['score']:.4f}"
        )


    # =========================
    # Return
    # =========================

    return context, sources