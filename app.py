from fastapi import FastAPI, UploadFile, File
import retriever as rt
import generator
import paths
import redis_db
import os
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

# Global chat queue and worker tasks list
chat_queue = asyncio.Queue()
worker_tasks = []

# Shared components for retrieval (kept global for /retrieve endpoint)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = rt.load_vectorstore(embedding_model)
# Using one chain instance for retrieval-related tasks;
# This is independent of chat generation.
retriever_chain, retriever = rt.get_retriever(vectorstore, generator.get_chain_generator(generator.get_model()))
redis_client = redis_db.create_redis_client()


@app.on_event("startup")
async def startup_event():
    global worker_tasks
    num_workers = 2  # Create two workers; adjust as needed.
    for i in range(num_workers):
        # Each worker gets its own model instance and chain.
        model_instance = generator.get_model()
        chain_instance = generator.get_chain_generator(model_instance)
        chain_instance, _ = rt.get_retriever(vectorstore, chain_instance)
        task = asyncio.create_task(chat_worker(chain_instance))
        worker_tasks.append(task)


@app.on_event("shutdown")
async def shutdown_event():
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)


async def chat_worker(chain_instance):
    """
    Worker that continuously processes chat requests from the queue using its own model instance.
    """
    while True:
        session_id, user_input, future = await chat_queue.get()
        try:
            chat_hist = generator.get_chat_hist_instance(session_id)
            # Use the worker's own chain instance to generate a response.
            chat_hist = await generator.generate_chat_st(user_input, chain_instance, chat_hist)
            result = chat_hist.messages[-1].content
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            chat_queue.task_done()


@app.get("/")
async def home():
    return {"message": "FastAPI is running!"}


@app.post("/retrieve")
async def retrieve_documents(query: str):
    results = await asyncio.ensure_future(rt.get_best_files(query, retriever))
    results = set(results)
    return {"documents": results}


@app.post("/chat")
async def chat(user_input: str, session_id: str):
    """
    Enqueue a chat request and await the response from one of the available workers.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await chat_queue.put((session_id, user_input, future))
    response = await future
    return {"response": response}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(paths.upload_dir_path, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return {"message": "File uploaded successfully!", "filename": file.filename}
