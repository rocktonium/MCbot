import os
import re
import json
import time
import requests
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


WIKI_API = "https://minecraft.wiki/api.php"

INDEX_DIR = "wiki_index"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.json")
PROGRESS_FILE = os.path.join(INDEX_DIR, "progress.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Article filtering
MIN_CHARACTERS = 150
MIN_TEXT_RATIO = 0.10

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

# Save progress every N articles
SAVE_EVERY = 500

os.makedirs(INDEX_DIR, exist_ok=True)

session = requests.Session()

session.headers.update({
    "User-Agent": "MinecraftDiscordBot/1.0"
})


def get_all_pages():

    print("Getting list of Minecraft Wiki articles...")

    pages = []
    apcontinue = None

    while True:

        params = {
            "action": "query",
            "list": "allpages",
            "apnamespace": 0,
            "apfilterredir": "nonredirects",
            "aplimit": "max",
            "format": "json",
        }

        if apcontinue:
            params["apcontinue"] = apcontinue

        response = session.get(
            WIKI_API,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        for page in data["query"]["allpages"]:

            pages.append({
                "pageid": page["pageid"],
                "title": page["title"]
            })

        print(
            f"Found {len(pages)} articles..."
        )

        if "continue" not in data:
            break

        apcontinue = data["continue"]["apcontinue"]

        time.sleep(0.1)

    return pages


def get_page_text(pageid):

    params = {
        "action": "parse",
        "pageid": pageid,
        "prop": "wikitext",
        "format": "json",
    }

    response = session.get(
        WIKI_API,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    try:
        return data["parse"]["wikitext"]["*"]
    except KeyError:
        return ""


def clean_wikitext(text):

    # Remove comments
    text = re.sub(
        r"<!--.*?-->",
        "",
        text,
        flags=re.DOTALL
    )

    # Remove templates
    previous = None

    while previous != text:

        previous = text

        text = re.sub(
            r"\{\{[^{}]*\}\}",
            "",
            text
        )

    # Remove references
    text = re.sub(
        r"<ref.*?</ref>",
        "",
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r"<ref[^>]*/>",
        "",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    # [[Page|display text]]
    text = re.sub(
        r"\[\[[^|\]]+\|([^\]]+)\]\]",
        r"\1",
        text
    )

    # [[Page]]
    text = re.sub(
        r"\[\[([^\]]+)\]\]",
        r"\1",
        text
    )

    # External links
    text = re.sub(
        r"\[https?://[^\s\]]+\s*([^\]]*)\]",
        r"\1",
        text
    )

    # Categories
    text = re.sub(
        r"\[\[Category:[^\]]+\]\]",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Headings
    text = re.sub(
        r"={2,6}\s*(.*?)\s*={2,6}",
        r"\1",
        text
    )

    # Formatting
    text = text.replace("'''", "")
    text = text.replace("''", "")

    # Remove common image/file syntax that survived
    text = re.sub(
        r"\[\[(File|Image):.*?\]\]",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Normalize whitespace
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


def should_keep_article(raw_text, cleaned_text):

    if len(cleaned_text) < MIN_CHARACTERS:
        return False

    if len(raw_text) == 0:
        return False

    text_ratio = (
        len(cleaned_text)
        / len(raw_text)
    )

    if text_ratio < MIN_TEXT_RATIO:
        return False

    return True


def chunk_text(text):
    """
    Split primarily by wiki section headings.

    Normal sections stay together.
    Very large sections are split into smaller chunks.
    """

    # Match wiki headings:
    #
    # == Section ==
    # === Subsection ===
    # ==== Sub-subsection ====
    #
    heading_pattern = re.compile(
        r"^(={2,6})\s*(.*?)\s*\1\s*$",
        re.MULTILINE
    )

    matches = list(
        heading_pattern.finditer(text)
    )

    sections = []

    # Text before the first heading
    if matches:

        intro = text[
            :matches[0].start()
        ].strip()

        if intro:
            sections.append({
                "section": "Introduction",
                "text": intro
            })

    else:

        # Article has no headings
        return [{
            "section": "Article",
            "text": text
        }]

    for i, match in enumerate(matches):

        heading_level = len(
            match.group(1)
        )

        heading = match.group(2).strip()

        start = match.end()

        if i + 1 < len(matches):

            end = matches[i + 1].start()

        else:

            end = len(text)

        section_text = text[
            start:end
        ].strip()

        if not section_text:
            continue

        sections.append({
            "section": heading,
            "level": heading_level,
            "text": section_text
        })

    chunks = []

    MAX_WORDS = 500
    OVERLAP = 75

    for section in sections:

        section_name = section["section"]
        section_text = section["text"]

        words = section_text.split()

        # Small/normal section:
        if len(words) <= MAX_WORDS:

            chunks.append({
                "section": section_name,
                "text": section_text
            })

            continue

        # Large section:
        start = 0
        chunk_number = 0

        while start < len(words):

            end = min(
                start + MAX_WORDS,
                len(words)
            )

            chunk = " ".join(
                words[start:end]
            )

            if len(chunk.strip()) > 100:

                chunks.append({
                    "section": (
                        f"{section_name} "
                        f"(part {chunk_number + 1})"
                    ),
                    "text": chunk
                })

            if end == len(words):
                break

            start = end - OVERLAP

            chunk_number += 1

    return chunks


def load_progress():

    if not os.path.exists(PROGRESS_FILE):
        return {
            "next_page": 0
        }

    with open(
        PROGRESS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_progress(next_page):

    with open(
        PROGRESS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "next_page": next_page
            },
            f
        )


def load_metadata():

    if not os.path.exists(METADATA_FILE):
        return []

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_metadata(metadata):

    temp_file = METADATA_FILE + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False
        )

    os.replace(
        temp_file,
        METADATA_FILE
    )


def main():

    pages = get_all_pages()

    print()
    print(
        f"Total wiki pages found: "
        f"{len(pages)}"
    )

    progress = load_progress()

    start_index = progress["next_page"]

    metadata = load_metadata()

    print(
        f"Resuming from article "
        f"{start_index + 1}"
    )

    print(
        f"Existing chunks: "
        f"{len(metadata)}"
    )

    print()

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # Create or load FAISS index
    if os.path.exists(INDEX_FILE):

        index = faiss.read_index(
            INDEX_FILE
        )

        print(
            "Loaded existing FAISS index."
        )

    else:

        index = None

    batch_texts = []
    batch_metadata = []

    processed_since_save = 0

    for i in range(
        start_index,
        len(pages)
    ):

        page = pages[i]

        title = page["title"]
        pageid = page["pageid"]

        print(
            f"[{i + 1}/{len(pages)}] "
            f"{title}"
        )

        try:

            raw_text = get_page_text(
                pageid
            )

            cleaned_text = clean_wikitext(
                raw_text
            )

            if not should_keep_article(
                raw_text,
                cleaned_text
            ):

                print(
                    "  -> skipped"
                )

                save_progress(i + 1)

                continue

            chunks = chunk_text(
                cleaned_text
            )

            for chunk_number, chunk in enumerate(chunks):
                section = chunk["section"]
                chunk_text_value = chunk["text"]

                # Put article + section into the embedding text.
                # This helps the embedding model understand
                # what the chunk is actually about.
                embedding_text = (
                    f"Article: {title}\n"
                    f"Section: {section}\n\n"
                    f"{chunk_text_value}"
                )

                batch_texts.append(embedding_text)

                batch_metadata.append({
                    "title": title,
                    "pageid": pageid,
                    "section": section,
                    "chunk": chunk_number,
                    "url": (
                        "https://minecraft.wiki/w/"
                        + title.replace(" ", "_")
                    ),
                    "text": chunk_text_value
                })

        except Exception as e:

            print(
                f"  -> ERROR: {e}"
            )

        processed_since_save += 1

        # Save periodically
        if (
            processed_since_save >= SAVE_EVERY
            or i == len(pages) - 1
        ):

            if batch_texts:

                print()
                print(
                    "Embedding batch..."
                )

                embeddings = model.encode(
                    batch_texts,
                    batch_size=64,
                    show_progress_bar=True,
                    normalize_embeddings=True
                )

                embeddings = np.asarray(
                    embeddings,
                    dtype="float32"
                )

                if index is None:

                    index = faiss.IndexFlatIP(
                        embeddings.shape[1]
                    )

                index.add(
                    embeddings
                )

                metadata.extend(
                    batch_metadata
                )

                batch_texts.clear()
                batch_metadata.clear()

            print(
                f"Saving progress at "
                f"{i + 1}/{len(pages)}..."
            )

            faiss.write_index(
                index,
                INDEX_FILE
            )

            save_metadata(
                metadata
            )

            save_progress(
                i + 1
            )

            processed_since_save = 0

            print(
                f"Saved {len(metadata)} "
                f"total chunks."
            )

            print()

        time.sleep(0.1)

    print()
    print("==========================")
    print("INDEXING COMPLETE")
    print("==========================")
    print(
        f"Wiki pages: {len(pages)}"
    )
    print(
        f"Indexed chunks: "
        f"{len(metadata)}"
    )
    print(
        f"Index: {INDEX_FILE}"
    )
    print(
        f"Metadata: {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()