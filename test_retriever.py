import chromadb 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import paths
from chromadb.config import Settings


embedding_model = HuggingFaceEmbeddings(model_name=paths.bert_model_path)

# 1) Connect to the running Chroma server
client = chromadb.HttpClient(
    host="localhost",
    port=8010,
    settings=Settings(anonymized_telemetry=False),
)

# 2) Re-use or create the collection (server-side metadata knows if it exists)
collection = client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"},          # optional tuning
)


res = collection.get(
    where={"source": {"$eq": "C:\\Users\\elhadsey\\OneDrive - myidemia\\Documents\\IA\\projects\\version 8\\data\\[HSM] HowTo install HSM to-do-list - v1.0.EN.pdf"}},
    include=["metadatas", ]     # ids alone if you prefer
)

print(res)  
print(f"Nombre  de chuncks : {collection.count()}\n")

# client.delete_collection(name="rag_docs")
#collection.delete(ids="565d059d-0d22-41f7-88e1-03a7e9fb5735")

print(f"Nombre  de chuncks : {collection.count()}")
