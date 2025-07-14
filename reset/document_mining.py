import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import requests
from streamlit import session_state as ss
import os
import paths

# Declare session state variable for storing PDF references
if 'pdf_refs' not in ss:
    ss.pdf_refs = []
if 'selected_pdf' not in ss:
    ss.selected_pdf = None  # Track the currently selected PDF

st.info('Check the explanations to see how to view documents', icon="ℹ️")


st.page_link("streamlit_pages/home.py", label="Home", icon="🏠")

st.header("Search for Documents :mag:")
with st.expander("See explanation"):
    st.write('''
        To display the document, click on the document and then on the 'Open Document Viewer' button.
    ''')
    video_file = open(paths.exp_video, "rb")
    video_bytes = video_file.read()

    st.video(video_bytes)

query = st.text_input("Enter your query")

FASTAPI_URL = "http://127.0.0.1:8000"  # Update this when deploying

if st.button("Search"):
    response = requests.post(f"{FASTAPI_URL}/retrieve", params={"query": query}).json()
    st.write("## Results")
    ss.pdf_refs = response["documents"]  # Store document paths in session state
    ss.selected_pdf = None  # Reset selected PDF

st.write("## Documents")
for doc_path in ss.pdf_refs:
    file_name = os.path.basename(doc_path)
    if st.button(f"View {file_name}"):
        ss.selected_pdf = doc_path
        st.rerun()

if ss.selected_pdf:
    with st.popover("Open Document Viewer", use_container_width=True):
        st.write("## Document Viewer")
        with open(ss.selected_pdf, "rb") as file:
            binary_data = file.read()
            pdf_viewer(input=binary_data, width=700)