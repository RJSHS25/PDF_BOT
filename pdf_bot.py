import os
import re
import json
import hashlib

import fitz  # PyMuPDF
import streamlit as st
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BOT_NAME = "GurucoolBOT"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")
CACHE_FILE = os.path.join(BASE_DIR, "processed_data.json")

MIN_PARA_LEN = 40          # ignore tiny fragments (page numbers, headers)
TOP_CHUNKS = 8              # how many chunks to consider before sentence-ranking
TOP_MATCHES = 3             # how many distinct matches to show the user
MAX_ANSWER_SENTENCES = 2    # keep each match's snippet short
DOCX_SECTION_CHUNK_CHARS = 500  # roughly how much text goes in one docx chunk
CACHE_VERSION = 3           # bump whenever the chunk/metadata schema changes

SUPPORTED_EXTENSIONS = (".pdf", ".docx")


# ------------------------------------------------------------------
# 1. Document discovery + fingerprinting (so we know when to re-process)
# ------------------------------------------------------------------
def list_documents(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(SUPPORTED_EXTENSIONS))


def folder_fingerprint(folder):
    """A cheap hash of filenames + sizes + mtimes, so we can detect changes
    without re-reading every file's contents."""
    items = []
    for f in list_documents(folder):
        path = os.path.join(folder, f)
        stat = os.stat(path)
        items.append(f"{f}:{stat.st_size}:{int(stat.st_mtime)}")
    raw = "|".join(items)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# 2a. PDF extraction — paragraphs tagged with "Page N"
# ------------------------------------------------------------------
AUTHOR_PATTERNS = [
    re.compile(r"\bby\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3})"),
    re.compile(r"\bauthor[:\s]+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3})", re.IGNORECASE),
    re.compile(r"\bwritten\s+by\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3})", re.IGNORECASE),
]


def guess_author_from_text(pages_text):
    for text in pages_text:
        for pattern in AUTHOR_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
    return None


def read_pdf(path, filename):
    """Returns (chunks, page_texts). Each chunk location is 'Page N'."""
    chunks = []
    page_texts = {}
    doc = fitz.open(path)
    for page_no, page in enumerate(doc, start=1):
        text = page.get_text()
        location = f"Page {page_no}"
        page_texts[location] = text.strip()
        paragraphs = [p.strip() for p in text.split("\n\n")]
        for para in paragraphs:
            para = re.sub(r"\s+", " ", para).strip()
            if len(para) > MIN_PARA_LEN:
                chunks.append({"source": filename, "location": location, "text": para})
    doc.close()
    return chunks, page_texts


def process_pdf(path, filename):
    doc = fitz.open(path)
    page_count = doc.page_count
    first_page_text = doc[0].get_text() if page_count else ""

    title = filename
    for line in first_page_text.split("\n"):
        line = line.strip()
        if len(line) > 4:
            title = line
            break

    meta_author = (doc.metadata or {}).get("author", "").strip()
    if meta_author:
        author = meta_author
    else:
        sample_pages = [doc[i].get_text() for i in range(min(2, page_count))]
        author = guess_author_from_text(sample_pages)
    doc.close()

    chunks, page_texts = read_pdf(path, filename)
    file_meta = {
        "filename": filename,
        "file_type": "pdf",
        "guessed_title": title,
        "author": author,
        "extent_label": f"{page_count} pages"
    }
    return chunks, page_texts, file_meta


# ------------------------------------------------------------------
# 2b. Word (.docx) extraction — paragraphs tagged with "Section: <heading>"
# ------------------------------------------------------------------
def read_docx(path, filename):
    """Returns (chunks, page_texts). Word has no fixed page numbers, so we
    cite by section (nearest preceding heading) instead — it's a stable
    reference the document itself defines."""
    document = Document(path)
    chunks = []
    sections = {}  # section title -> list of paragraph strings (for full-section reading)
    current_section = "Introduction"
    sections[current_section] = []
    buffer = []

    def flush_buffer():
        if buffer:
            text = " ".join(buffer).strip()
            if len(text) > MIN_PARA_LEN:
                chunks.append({
                    "source": filename,
                    "location": f"Section: {current_section}",
                    "text": text
                })
            buffer.clear()

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = para.style.name if para.style else ""
        if style_name.startswith("Heading") or style_name == "Title":
            flush_buffer()
            current_section = text
            sections.setdefault(current_section, [])
            continue
        sections[current_section].append(text)
        buffer.append(text)
        if sum(len(b) for b in buffer) > DOCX_SECTION_CHUNK_CHARS:
            flush_buffer()
    flush_buffer()

    page_texts = {
        f"Section: {title}": "\n\n".join(paras)
        for title, paras in sections.items() if paras
    }
    return chunks, page_texts


def process_docx(path, filename):
    document = Document(path)
    core = document.core_properties

    title = (core.title or "").strip()
    if not title:
        for para in document.paragraphs:
            if para.text.strip():
                title = para.text.strip()
                break
    if not title:
        title = filename

    author = (core.author or "").strip() or None

    chunks, page_texts = read_docx(path, filename)
    file_meta = {
        "filename": filename,
        "file_type": "docx",
        "guessed_title": title,
        "author": author,
        "extent_label": f"{len(page_texts)} section(s)"
    }
    return chunks, page_texts, file_meta


# ------------------------------------------------------------------
# 2c. Unified processing across all supported file types
# ------------------------------------------------------------------
def process_all_documents(folder):
    all_chunks = []
    all_page_texts = {}
    metadata = {"files": []}

    for f in list_documents(folder):
        path = os.path.join(folder, f)
        ext = os.path.splitext(f)[1].lower()

        if ext == ".pdf":
            chunks, page_texts, file_meta = process_pdf(path, f)
        elif ext == ".docx":
            chunks, page_texts, file_meta = process_docx(path, f)
        else:
            continue  # unsupported, shouldn't happen given list_documents filter

        metadata["files"].append(file_meta)
        all_chunks.extend(chunks)
        all_page_texts[f] = page_texts

    return all_chunks, metadata, all_page_texts


# ------------------------------------------------------------------
# 3. Cache load/save
# ------------------------------------------------------------------
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def save_cache(fingerprint, chunks, metadata, page_texts):
    with open(CACHE_FILE, "w", encoding="utf-8") as fh:
        json.dump({
            "fingerprint": fingerprint,
            "version": CACHE_VERSION,
            "chunks": chunks,
            "metadata": metadata,
            "page_texts": page_texts
        }, fh)


def get_chunks_and_metadata():
    """Returns (chunks, metadata, page_texts), using the on-disk cache
    whenever the documents/ folder hasn't changed since last run."""
    fingerprint = folder_fingerprint(DOCS_DIR)
    cached = load_cache()
    if cached and cached.get("fingerprint") == fingerprint and cached.get("version") == CACHE_VERSION:
        return cached["chunks"], cached["metadata"], cached.get("page_texts", {})

    chunks, metadata, page_texts = process_all_documents(DOCS_DIR)
    if chunks:
        save_cache(fingerprint, chunks, metadata, page_texts)
    return chunks, metadata, page_texts


# ------------------------------------------------------------------
# 4. Retrieval: chunk-level, then sentence-level for concise snippets
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_vectorizer(chunks_tuple):
    """chunks_tuple is a tuple of texts (hashable) so st.cache_resource
    can key on it correctly."""
    texts = list(chunks_tuple)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors


def display_name(filename, metadata):
    """Prefer the guessed title (e.g. 'Originals') over the raw filename
    for citations."""
    for f in metadata["files"]:
        if f["filename"] == filename:
            title = f.get("guessed_title")
            return title if title else filename
    return filename


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


DEFINITION_TERM_PATTERN = re.compile(
    r"^(?:what\s+(?:is|are)|define|what's|meaning\s+of)\s+(.+)", re.IGNORECASE
)


def extract_definition_term(question):
    match = DEFINITION_TERM_PATTERN.match(question.strip().rstrip("?. "))
    if match:
        return match.group(1).strip()
    return None


def concise_snippet(chunk_text, question, term):
    """Pick the 1-2 best sentences within a chunk for a short answer,
    boosting sentences that look like a definition when relevant."""
    sentences = split_sentences(chunk_text)
    if not sentences:
        return chunk_text[:400]

    sent_vectorizer = TfidfVectorizer(stop_words="english")
    try:
        sent_vectors = sent_vectorizer.fit_transform(sentences)
        q_vec = sent_vectorizer.transform([question])
        sent_scores = cosine_similarity(q_vec, sent_vectors)[0]

        if term:
            def_pattern = re.compile(
                rf"\b{re.escape(term.lower())}\b\s+(is|are|refers to|means)\b",
                re.IGNORECASE
            )
            for idx, sent in enumerate(sentences):
                if def_pattern.search(sent):
                    sent_scores[idx] += 1.0

        top_idx = sorted(sent_scores.argsort()[::-1][:MAX_ANSWER_SENTENCES])
        return " ".join(sentences[i] for i in top_idx)
    except ValueError:
        return " ".join(sentences[:MAX_ANSWER_SENTENCES])


def get_top_matches(chunks, question, top_n=TOP_MATCHES):
    """Returns up to top_n distinct matches, each with a short snippet plus
    the full chunk text (for the 'read more' expander)."""
    texts = tuple(c["text"] for c in chunks)
    vectorizer, vectors = build_vectorizer(texts)
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, vectors)[0]
    ranked = scores.argsort()[::-1]

    term = extract_definition_term(question)
    matches = []
    seen_locations = set()

    for i in ranked:
        if scores[i] <= 0 or len(matches) >= top_n:
            break
        chunk = chunks[i]
        key = (chunk["source"], chunk["location"])
        if key in seen_locations:
            continue  # skip duplicate paragraphs from the same page/section
        seen_locations.add(key)

        matches.append({
            "source": chunk["source"],
            "location": chunk["location"],
            "snippet": concise_snippet(chunk["text"], question, term),
            "full_text": chunk["text"],
            "score": round(float(scores[i]), 2)
        })

    return matches


# ------------------------------------------------------------------
# 5. Generic / small-talk handling (no retrieval needed)
# ------------------------------------------------------------------
GREETINGS = {"hi", "hello", "hey", "hii", "hiya", "good morning", "good afternoon", "good evening"}
THANKS = {"thanks", "thank you", "thx", "ty"}
BYE = {"bye", "goodbye", "see you", "exit", "quit"}


def handle_generic(question, metadata):
    q = question.strip().lower().rstrip("!?. ")

    if re.search(r"\byour name\b", q) or q in {"who are you", "what are you"}:
        return f"I'm **{BOT_NAME}** — your local document Q&A assistant. Ask me anything about the files loaded from the documents/ folder!"

    if q in GREETINGS or any(q.startswith(g) for g in GREETINGS):
        return f"Hi! I'm {BOT_NAME}. Ask me anything about the document(s) loaded from the documents/ folder."

    if any(t in q for t in THANKS):
        return "You're welcome! Anything else you'd like to ask?"

    if q in BYE:
        return "Goodbye! Come back anytime you have more questions."

    if "how are you" in q:
        return f"Doing well, thanks for asking! I'm {BOT_NAME} — what can I help you find in the document?"

    if re.search(r"\bauthor\b", q) or re.search(r"who\s+(wrote|is the author)", q):
        if not metadata["files"]:
            return "I don't have any documents loaded yet — add one to the documents/ folder."
        lines = []
        for f in metadata["files"]:
            author = f.get("author") or "not stated in the document"
            lines.append(f"- **{f['guessed_title']}** — author: {author}")
        return "\n".join(lines)

    if re.search(r"\b(title|name of the (book|document|pdf|file))\b", q):
        if metadata["files"]:
            lines = [f"- **{f['guessed_title']}** ({f['filename']})" for f in metadata["files"]]
            return "Here's what I have loaded:\n" + "\n".join(lines)
        return "I don't have any documents loaded yet — add one to the documents/ folder."

    if re.search(r"how many (pages|documents|files|pdfs|sections)", q):
        lines = [f"- **{f['guessed_title']}**: {f['extent_label']}" for f in metadata["files"]]
        return f"I have {len(metadata['files'])} document(s) loaded:\n" + "\n".join(lines)

    if re.search(r"what (documents|files|pdfs) (do you have|are loaded|can you read)", q):
        if metadata["files"]:
            names = ", ".join(f["filename"] for f in metadata["files"])
            return f"Loaded documents: {names}"
        return "No documents are loaded yet."

    return None  # not a generic question -> fall through to retrieval


# ------------------------------------------------------------------
# 6. Streamlit chat UI
# ------------------------------------------------------------------
st.set_page_config(page_title=BOT_NAME, page_icon="🎓")
st.title(f"🎓 {BOT_NAME}")

os.makedirs(DOCS_DIR, exist_ok=True)
chunks, metadata, page_texts = get_chunks_and_metadata()

if not chunks:
    st.warning(f"No PDF or Word files found in `{DOCS_DIR}`. Add files there and refresh the page.")
    st.stop()

with st.sidebar:
    st.subheader("Loaded documents")
    for f in metadata["files"]:
        st.write(f"**{f['guessed_title']}** ({f['file_type'].upper()}, {f['extent_label']})")
    if st.button("🔄 Re-scan documents/ folder"):
        st.cache_resource.clear()
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "type": "text",
         "content": f"Hi! I'm {BOT_NAME}. Ask me anything about the document(s) loaded from the documents/ folder."}
    ]


def render_matches(matches, metadata_, page_texts_, msg_idx):
    """Renders each match with its snippet + citation + an expander to read
    the full page (PDF) or full section (Word)."""
    for rank, m in enumerate(matches, start=1):
        title = display_name(m["source"], metadata_)
        st.markdown(f"**{rank}.** {m['snippet']}")
        st.caption(f"Source: {title}, {m['location']} (relevance score {m['score']})")

        full_text = page_texts_.get(m["source"], {}).get(m["location"], m["full_text"])
        label = "page" if m["location"].startswith("Page") else "section"
        with st.expander(f"📖 Read full {label} — {title}, {m['location']}"):
            st.write(full_text)


# Replay chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["type"] == "text":
            st.markdown(msg["content"])
        elif msg["type"] == "matches":
            render_matches(msg["matches"], metadata, page_texts, idx)

# New input
question = st.chat_input("Ask a question...")
if question:
    st.session_state.messages.append({"role": "user", "type": "text", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    generic_reply = handle_generic(question, metadata)

    with st.chat_message("assistant"):
        if generic_reply is not None:
            st.markdown(generic_reply)
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": generic_reply})
        else:
            matches = get_top_matches(chunks, question)
            if not matches:
                reply = "Sorry, I couldn't find anything relevant to that in the document(s)."
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": reply})
            else:
                new_idx = len(st.session_state.messages)
                render_matches(matches, metadata, page_texts, new_idx)
                st.session_state.messages.append({"role": "assistant", "type": "matches", "matches": matches})
