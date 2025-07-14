import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import paths
from uuid import uuid4
from chromadb.utils import embedding_functions
import os

# Pour éviter les warnings de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=default, 1=info, 2=warning, 3=error

# Connexion au client Chroma
client = chromadb.HttpClient(host="localhost", port=8010)

# Définition de l'embedder LangChain
embedder = HuggingFaceEmbeddings(model_name=paths.bert_model_path)

# Embedding function compatible avec Chroma (via chromadb directement)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=paths.bert_model_path
)

# Récupération ou création de la collection
collection = client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"}
    
)

# Création du vectordb (retriever)
vectordb = Chroma(
    embedding_function=embedder,
    client=client,
    collection_name="rag_docs"
)

#print(collection.count())

# Effectuer une recherche de similarité textuelle
query = "central architecture"
docs_and_scores = vectordb.similarity_search_with_relevance_scores(query, k=88)

# Trier les documents selon leur score décroissant
sorted_docs = sorted(docs_and_scores, key=lambda x: x[1])

# Afficher les documents triés
for doc, score in sorted_docs:
    print(f"================================== Score: {score:.4f} =========================================")
    print(f"Content: {doc.metadata}")
    print("---")
