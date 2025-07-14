# document_mining.py  ── version stable ─────────────────────
import os, requests, streamlit as st
from streamlit_pdf_viewer import pdf_viewer   # ou votre composant maison

FASTAPI_URL = "http://127.0.0.1:8000"

# ---------- 1. session_state safe defaults -----------------
ss = st.session_state
ss.setdefault("pdf_refs", [])
ss.setdefault("metadatas", [])
ss.setdefault("selected_pdf", None)
ss.setdefault("selected_meta", {})

# ---------- 2. Barre latérale : options -------------------
st.sidebar.header("Options")
show_meta = st.sidebar.checkbox("Afficher les métadonnées", value=False)


# ---------- 2b.  Mode de recherche ------------------------
mode = st.sidebar.radio(
    "Mécanisme de recherche",
    ["Vector", "Words"],
    index=0,
    horizontal=True,
).lower()   

# ---------- 3. Zone de requête ------------------------------
st.page_link("streamlit_pages/home.py", label="Home", icon="🏠")
query = st.text_input("Entrez votre requête")

if st.button("Chercher") and query.strip():
    payload = {
        "query":  query,
        "mode" : mode
        
    }
    r = requests.post(f"{FASTAPI_URL}/retrieve", json=payload).json()

    ss.pdf_refs  = r.get("documents", [])
    ss.metadatas = r.get("metadatas", [])
    ss.selected_pdf  = None
    ss.selected_meta = {}
    st.rerun()                     # relance pour afficher les résultats

# ---------- 4. Affichage des résultats ----------------------
st.subheader("Documents trouvés")

if not ss.pdf_refs:
    st.info("Aucun document pour ces filtres.")
else:
    for i, path in enumerate(ss.pdf_refs):
        name = os.path.basename(path) or f"doc_{i}"
        if st.button(name, key=f"btn_{i}"):
            ss.selected_pdf  = path
            ss.selected_meta = ss.metadatas[i] if i < len(ss.metadatas) else {}
            st.rerun()

# ---------- 5. Visionneuse PDF + métadonnées ----------------
if ss.selected_pdf:
    st.markdown("### Visionneuse PDF")
    with open(ss.selected_pdf, "rb") as f:
        pdf_viewer(input=f.read(), width=700, height=900)

    if show_meta and ss.selected_meta:
        st.markdown("#### Métadonnées")
        st.json(ss.selected_meta)
