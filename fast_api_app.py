from fastapi import FastAPI, UploadFile, File
import retriever as rt
import generator
import paths
import redis_db
import os
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
import json
import preprocess
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio, shutil
from uuid import uuid4
from typing import Literal
from pydantic import BaseModel


app = FastAPI()

vectorstore_lock = asyncio.Lock()          # protects concurrent writes


# Global chat queue, worker tasks, and a lock to serialize model calls
chat_queue = asyncio.Queue()
worker_tasks = []
model_lock = asyncio.Lock()  # Ensures only one model call is active at a time

# Load models and initialize chains/clients
embedding_model = HuggingFaceEmbeddings(model_name=paths.bert_model_path)
vectorstore = rt.load_vectorstore(embedding_model)
generator_model = generator.get_model()
generator_chain = generator.get_chain_generator(generator_model)
retriever_chain, retriever, collection = rt.get_retriever(vectorstore, generator_chain)
redis_client = redis_db.create_redis_client()


@app.on_event("startup")
async def startup_event():
    global worker_tasks
    num_workers = 1  # Adjust the number of workers as needed
    for _ in range(num_workers):
        task = asyncio.create_task(chat_worker())
        worker_tasks.append(task)


@app.on_event("shutdown")
async def shutdown_event():
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)


async def chat_worker():
    """Worker that continuously processes chat requests from the queue."""
    while True:
        session_id, user_input, future = await chat_queue.get()
        try:
            chat_hist = generator.get_chat_hist_instance(session_id)
            # Serialize access to the generator chain to avoid connection issues
            async with model_lock:
                chat_hist , duration = await generator.generate_chat_st(user_input, retriever_chain, chat_hist)
            result = ( chat_hist.messages[-1].content , duration )
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            chat_queue.task_done()


@app.get("/")
async def home():
    return {"message": "FastAPI is running!"}


class RetrievePayload(BaseModel):
    query:  str
    mode:  Literal["vector", "words"] = "vector"

@app.post("/retrieve")
async def retrieve_documents(payload: RetrievePayload):
    paths, metas = await rt.get_best_files(
        query     = payload.query,
        retriever = retriever,
        mode=payload.mode,
        
    )
    return {"documents": paths, "metadatas": metas}


@app.post("/chat")
async def chat(user_input: str, session_id: str):
    # Create a future to hold the result of this chat request
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    # Put the task into the queue
    await chat_queue.put((session_id, user_input, future))
    # Wait for the worker to process the task and return the result
    response , duration = await future
    return {"response": response , "duration" : duration}

# =====================================================================

@app.get("/chat/history")
async def chat_history(session_id: str):
    """
    Récupère l’historique complet (role, content, duration) tel qu’enregistré dans Redis.
    """
    # on récupère la clé dans generator
    chat_hist = generator.get_chat_hist_instance(session_id)
    key = chat_hist.key  # ex. "chat_history:<session_id>"
    # lrange renvoie les JSON strings {"role","content","duration"?}
    raw = redis_client.lrange(key, 0, -1)
    # reconvertit en liste de dicts
    history = [json.loads(item) for item in raw]
    # on filtre juste les system, si souhaité
    history = [m for m in history if m.get("role") != "system"]
    return {"history": history}




@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    # Save into the temporary uploads directory
    tmp_path = os.path.join(paths.upload_dir_path, file.filename)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # Extract plain text plus PDF metadata using our new extractor
    text, meta = preprocess.get_text(tmp_path)
    if not text:
        return {"error": "Unable to extract text from PDF."}

    # Merge metadata and split into chunks
    meta_clean = {k: v for k, v in (meta or {}).items() if v}
    meta_clean["source"] = os.path.join(paths.data_path, file.filename)

    splitter = RecursiveCharacterTextSplitter(chunk_size=20_000, chunk_overlap=2_000)
    docs = splitter.split_documents(
        [Document(page_content=text, metadata=meta_clean)]
    )

    # Embed and append to the **Chroma** collection (thread‑safe)
    async with vectorstore_lock:
        ids = [str(uuid4()) for _ in range(len(docs))]
        vectorstore.add_documents(docs, ids=ids)
        # For Chroma (HTTP client) data are immediately persisted server‑side

    # Move the PDF from /uploads to the canonical data/ corpus folder
    try:
        shutil.move(tmp_path, os.path.join(paths.data_path, file.filename))
    except Exception as e:
        # If move fails, don’t block the API call; just log.
        print(f"Warning: could not move file {file.filename}: {e}")

    return {"message": "File uploaded and indexed.", "filename": file.filename}

