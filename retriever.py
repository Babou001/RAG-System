import os
import paths
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, List, Tuple
import asyncio
import re

from langchain_chroma import Chroma
import chromadb
from chromadb.utils import embedding_functions

from langchain.vectorstores.base import VectorStoreRetriever



NB_DOCS = 2
OPTIMIZED_INDEX_PATH = paths.faiss_index_path



class VectorStoreRetrieverChromaWorkAround(VectorStoreRetriever):
    actual_k: int = NB_DOCS

    def invoke(self, query: str):
        docs = super().invoke(query)
        # Trie les documents par score décroissant (par exemple en supposant que le score est dans metadata['score'])
        sorted_docs = sorted(docs, key=lambda doc: doc.metadata.get("score", 0), reverse=True)
        return sorted_docs[: self.actual_k]

    async def ainvoke(self, query: str):
        docs = await super().ainvoke(query)
        sorted_docs = sorted(docs, key=lambda doc: doc.metadata.get("score", 0), reverse=True)
        return sorted_docs[: self.actual_k]



 
# Vector store helpers

def load_vectorstore(embedding_model, collection_name: str = "rag_docs"):
    
    client = chromadb.HttpClient(host="localhost", port=8010)
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )



# Retriever & chain

def get_retriever(vectorstore, generator_chain, k: int = NB_DOCS):

    def parse_retriever_input(params: Dict):
        return params["messages"][-1].content
    

    retriever = VectorStoreRetrieverChromaWorkAround(
        vectorstore=vectorstore,
        search_kwargs={"k": 20},
        search_type="similarity",
        actual_k=k,  # final number of docs returned
    )

    collection = vectorstore._collection  # underlying Chroma collection

    retrieval_chain = RunnablePassthrough.assign(
        context=parse_retriever_input | retriever
    ).assign(answer=generator_chain)

    return retrieval_chain, retriever, collection


# Misc util --------------------------------------------------------------------

def verification(filepath: str) -> str:
    """Return an absolute path relative to the data folder if necessary."""
    if os.path.isabs(filepath):
        return filepath
    return os.path.join(paths.base_dir, "data", filepath)


# Async helper to fetch best files --------------------------------------------
async def get_best_files(
    query: str,
    retriever,
    mode: str = "vector",  # "vector" | "words"
    k: int = 5,
) -> Tuple[List[str], List[Dict]]:
    """Return (list_of_file_paths, list_of_metadatas) for the *k* best docs."""

    docs = await retriever.ainvoke(query)  # async embed‑&‑search

    if mode == "vector":
        doc_paths, metas = [], []
        for d in docs[:k]:
            doc_paths.append(d.metadata.get("source"))
            metas.append(d.metadata)

    else:  # -------- literal “Words” mode --------
        query_lower = query.lower()

        collection = retriever.vectorstore._collection
        collection._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=paths.bert_model_path
        )

        res = await asyncio.to_thread(
            collection.query,
            query_texts=[query_lower],
            n_results=k,
            where_document={"$contains": query_lower},
        )
        doc_paths = [m.get("source") for m in res["metadatas"][0]]
        metas = res["metadatas"][0]

    return doc_paths, metas