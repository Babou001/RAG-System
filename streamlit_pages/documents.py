import streamlit as st
import os

st.page_link("streamlit_pages/home.py", label="Home", icon="🏠")
st.header("Dataset PDF Documents :file_folder:")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file is not None:
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully!")
