import os
import paths
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough
from typing import Dict
import asyncio
from langchain_chroma import Chroma
import chromadb

# Download necessary NLTK data
#nltk.download("punkt")

NB_DOCS = 2
OPTIMIZED_INDEX_PATH = paths.faiss_index_path



#…
def load_vectorstore(embedding_model, collection_name="rag_docs"):
    client = chromadb.HttpClient(host="localhost", port=8010)
    return Chroma(client=client,
                  collection_name=collection_name,
                  embedding_function=embedding_model)




def get_retriever(vectorstore, generator_chain, k: int = NB_DOCS) :
    """
    Initialize retriever from chroma's vectorstore.
    """

    def parse_retriever_input(params: Dict):
        return params["messages"][-1].content    


    retriever = vectorstore.as_retriever(search_kwargs={"k": k},
                                                     search_type="mmr")
    

    collection = vectorstore._collection  

    

    retrieval_chain = RunnablePassthrough.assign(
        context=parse_retriever_input | retriever
    ).assign(answer=generator_chain)

    return retrieval_chain, collection


def verification(filepath: str) -> str:
    if os.path.isabs(filepath):
        # already an absolute path on any OS
        return filepath
    else:
        return os.path.join(paths.base_dir, "data", filepath)


async def get_best_files(query: str, retriever, k: int = 5) -> list:
    docs = await asyncio.ensure_future(retriever.ainvoke(query))
    q_lower = query.lower()
    tokens = q_lower.split()

    # 1) On cherche d’abord l’expression exacte "central architecture"
    exact = [d for d in docs if q_lower in d.page_content.lower()]

    if exact:
        selected = exact[:k]
    else:
        # 2) Sinon, on ne garde que les docs qui contiennent TOUS les tokens
        both = [
            d for d in docs
            if all(tok in d.page_content.lower() for tok in tokens)
        ]
        selected = both[:k] if both else docs[:k]

    best_files = [
        verification(d.metadata.get('source', 'Unknown source'))
        for d in selected
    ]
    return best_files