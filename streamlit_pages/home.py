import streamlit as st
import paths



st.logo(image=paths.image_logo, size="large")
# Main title and overview
st.title("IDEMIA RAG 🎯")
st.subheader("Home Page Overview", divider="violet")

# Introduction section with colored highlights
st.markdown('''
Welcome to the **IDEMIA RAG** application!  
This platform leverages advanced AI techniques to enable:

- :violet[Document Mining AI] using **BERT** + **BM25**
- :blue[Add Documents] upload and ingestion pipeline
- :green[Chat AI] powered by Meta's **LLaMA** model for RAG
- :orange[Dashboard] to visualize metrics and usage
''')



# Navigation links to app sections
st.subheader("Navigate to Features", divider="violet")
nav_col1, nav_col2 = st.columns(2)
with nav_col1:
    st.page_link(
        "streamlit_pages/document_mining.py",
        label="📑 Document Mining",
        icon="🔍"
    )
    st.page_link(
        "streamlit_pages/chatbot.py",
        label="💬 Chat AI",
        icon="🤖"
    )
with nav_col2:
    st.page_link(
        "streamlit_pages/documents.py",
        label="📂 Add Documents",
        icon="➕"
    )
    st.page_link(
        "streamlit_pages/dashboard.py",
        label="📊 Dashboard",
        icon="📈"
    )


st.subheader(" How It Works", divider="violet")

# Footer explanation
st.markdown('''
- :violet[**Document Mining**]: We combine transformer-based embeddings (BERT) with classic BM25 ranking to ensure both semantic and lexical search (**Similarity Search**).  
- :violet[**Add Documents**]: Upload your own files (PDF) and our `add_documents` function will handle ingestion and indexing for RAG later.  
- :violet[**Chat AI**]: Ask questions directly via a Retrieval-Augmented Generation (RAG) chatbot powered by Meta's open-source LLaMA model, pulling in the most relevant document snippets behind the scenes.  
- :violet[**Dashboard**]: Monitor usage, performance metrics, and get insights into your document corpus and query statistics.
''')



