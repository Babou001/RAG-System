import paths
import os
import shutil
import pickle
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from time import time
import sys
import chromadb
from langchain_chroma import Chroma
from chromadb.config import Settings
from uuid import uuid4
import pymupdf


def load_doc(filepath):
    """Safely open a PDF and return the document object (or None)."""
    try:
        return pymupdf.open(filepath)
    except Exception as e:
        print(f"Failed to load document: {e}")
        return None


def is_skippable_page(page_txt: str) -> bool:
    """Return *True* if the page looks like Table‑of‑Contents or Update‑History.

    Heuristic:
    • keyword hit ("table of contents" or "document update history")
    • OR ≥ 3 dotted‑leader lines that make up ≥ 30 % of the visible lines.
    """
    # A. direct keyword hit
    if re.search(r"\b(table\s+of\s+contents)\b",
                 page_txt, flags=re.I):
        return True

    # B. dotted‑leader lines like "2.3  Installing …… 17"
    dot_line_pat = re.compile(r'\.{2,}\s*\d+\s*$')
    lines = [ln for ln in page_txt.splitlines() if ln.strip()]
    dot_lines = [ln for ln in lines if dot_line_pat.search(ln)]
    return len(dot_lines) >= 3 and len(dot_lines) / max(len(lines), 1) >= 0.30


def strip_sections(raw_txt: str) -> str:
    """Fallback regex cleaner (kept for edge‑cases)."""
    # DOCUMENT UPDATE HISTORY … CLASSIFICATION
    raw_txt = re.sub(
        r'DOCUMENT UPDATE HISTORY.*?CLASSIFICATION',
        '',
        raw_txt,
        flags=re.S | re.I
    )

    # Table of contents … 1   Introduction
    raw_txt = re.sub(
        r'Table of contents.*?\n1\s+Introduction',
        '1   Introduction',
        raw_txt,
        flags=re.S | re.I
    )

    # collapse 3+ consecutive blank lines
    return re.sub(r'\n{3,}', '\n\n', raw_txt)


# ────────────────────────────────────────────────────────────
# Core extractor (minimal changes)
# ────────────────────────────────────────────────────────────

def extract_text_with_layout(
    filepath,
    images_dir="images",
    space_multiplier=0.5,
):
    """Extract text with rudimentary layout, skipping TOC/History pages."""
    doc = load_doc(filepath)
    if doc is None:
        return ""

    meta = doc.metadata

    #print(f"\n\nDocument metadata : \n{doc.metadata}\n\n")

    os.makedirs(images_dir, exist_ok=True)
    full_lines = []

    for pno in range(doc.page_count):
        page = doc.load_page(pno)

        #   decide early whether we keep this page
        plain_txt = page.get_text("text")
        if is_skippable_page(plain_txt):
            print(f"Skipping page {pno + 1} (TOC / history)")
            continue  # jump to next page

        # page delimiter (only for retained pages)
        full_lines.append(f"--- Page {pno + 1} ---")

        page_dict = page.get_text("dict")

        # text blocks, top‑to‑bottom
        blocks = sorted(page_dict.get("blocks", []), key=lambda b: b['bbox'][1])
        for block in blocks:
            # process lines within block
            lines = sorted(block.get("lines", []), key=lambda l: l['bbox'][1])
            for line in lines:
                spans = sorted(line.get("spans", []), key=lambda s: s['bbox'][0])
                line_text = ""
                cursor_x = None
                for span in spans:
                    x0, _, x1, _ = span['bbox']
                    txt = span.get('text', '')
                    if cursor_x is not None:
                        gap = x0 - cursor_x
                        n_spaces = max(1, int(gap // (span['size'] * space_multiplier)))
                        line_text += ' ' * n_spaces
                    line_text += txt
                    cursor_x = x1
                full_lines.append(line_text.rstrip())
            # blank line after block
            full_lines.append("")

    doc.close()

    # join lines & collapse 3+ newlines to 2
    text = "\n".join(full_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return strip_sections(text) , meta


# to get the text from file
def get_text(path):
    """To get text from PDF file using markitdown package"""
    try:
        return extract_text_with_layout(path)
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return ""


def get_all_files(directory=paths.data_path, skip_existing=True):
    """Returns only new files that have not been processed before."""
    all_files = []
    try:
        processed_files = set(os.listdir(paths.preprocessed_data)) if skip_existing else set()
        for root, _, files in os.walk(directory):
            for file in files:
                if file not in processed_files:
                    all_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"Error accessing directory {directory}: {e}")
    return all_files


def get_chunks(chunk_size=20000, overlap=2000):
    """To get chunks from text"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = []
    files = get_all_files()
    
    for fpath in files:
        text, meta = get_text(fpath)
        text = text.lower()
        if not text:
            continue
        # Merge absolute path with PDF metadata
        meta_clean = {k: v for k, v in meta.items() if v}  # drop Nones
        meta_clean["source"] = os.path.abspath(fpath)
        docs.append(Document(page_content=text, metadata=meta_clean))
    
    all_splits = text_splitter.split_documents(docs)
    
    directory_path = paths.chunks_dir_path
    file_path = f"{directory_path}/all_splits.pkl"
    try:
        os.makedirs(directory_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(all_splits, f)
    except Exception as e:
        print(f"Error saving chunks to {file_path}: {e}")
    
    return all_splits


def get_vectorizer(
    save_path: str = paths.preprocessed_data,
    collection_name: str = "rag_docs",
    host: str = "localhost",
    port: int = 8010,
):
    """
    Connects to (or starts) the Chroma collection that holds your vectors.
    If the collection is empty it populates it from the PDF corpus.
    Returns a LangChain-wrapped Chroma vector store so the rest of the
    codebase continues to work unchanged.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=paths.bert_model_path)

    # 1) Connect to the running Chroma server
    client = chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(anonymized_telemetry=False),
    )

    # 2) Re-use or create the collection (server-side metadata knows if it exists)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},          # optional tuning
    )

    # 3) Wrap it in the LangChain adapter
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

    # 4) First-run bootstrap: load documents only if the DB is still empty
    if collection.count() == 0:
        print("Creating Chroma collection and embedding all documents …")
        all_splits = get_chunks()
        ids = [str(uuid4()) for _ in range(len(all_splits))]
        vectorstore.add_documents(all_splits, ids=ids)
        print(f"{len(all_splits)} chunks inserted in collection ‹{collection_name}›")

    else:
        print(f"Using existing Chroma collection ‹{collection_name}› "
              f"({collection.count()} chunks)")

    return vectorstore



def add_documents(
    upload_directory: str = paths.upload_dir_path,
    collection_name: str = "rag_docs",
    host: str = "localhost",
    port: int = 8010,
):
    """
    Embeds only the *new* PDFs present in `upload_directory`
    and appends them to the shared Chroma collection.
    """
    try:
        existing_files = set(os.listdir(paths.data_path)) \
                         if os.path.exists(paths.data_path) else set()
        new_files = [f for f in os.listdir(upload_directory) if f not in existing_files]

        if not new_files:
            print("No new documents to process.")
            return

        print(f"Processing {len(new_files)} new documents …")

        # ------------------------------------------------------------------ #
        # 1) Read & split the new PDFs
        # ------------------------------------------------------------------ #
        splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=2000)
        new_docs = []
        for fname in new_files:
            raw = get_text(os.path.join(upload_directory, fname))
            if raw:
                new_docs.append(
                    Document(page_content=raw, metadata={"source": fname})
                )

        if not new_docs:
            print("No readable text found in the uploaded files.")
            return

        new_splits = splitter.split_documents(new_docs)

        # ------------------------------------------------------------------ #
        # 2) Connect to Chroma and append embeddings
        # ------------------------------------------------------------------ #
        embedding_model = HuggingFaceEmbeddings(model_name=paths.bert_model_path)
        client = chromadb.HttpClient(host=host, port=port,
                                     settings=Settings(anonymized_telemetry=False))
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )

        ids = [str(uuid4()) for _ in range(len(new_splits))]
        vectorstore.add_documents(new_splits, ids=ids)
        print(f"Added {len(new_splits)} chunks from {len(new_docs)} file(s) to Chroma.")

        # ------------------------------------------------------------------ #
        # 3) Move the PDFs into your long-term data folder
        # ------------------------------------------------------------------ #
        for fname in new_files:
            try:
                shutil.move(
                    os.path.join(upload_directory, fname),
                    os.path.join(paths.data_path, fname)
                )
            except Exception as e:
                print(f"Error moving file {fname} → {paths.data_path}: {e}")

    except Exception as e:
        print(f"Error adding documents: {e}")


execution_type = sys.argv[1]

if execution_type == "preprocess" :
    start = time() 
    get_vectorizer()
    end = time()
    duration = end -start 
    m, s = divmod(duration, 60)
    print(f"Time for preprocessing : {int(m)}:{int(s):02d}")


elif execution_type == "add_doc" : 
    start = time() 
    add_documents()
    end = time()
    duration = end -start 
    m, s = divmod(duration, 60)
    print(f"Time for adding documents : {int(m)}:{int(s):02d}")
    


else : 
    print("Parameter would be 'preprocess' or 'add_doc'" \
    "\n\nExample : for data preprocessing -------->  python preprocess.py preprocess")