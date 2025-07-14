import streamlit as st
import paths



# Main pages
home_page            = st.Page("streamlit_pages/home.py",            title="Home",           icon=":material/home:",       default=True)
documents_page       = st.Page("streamlit_pages/documents.py",       title="Documents",      icon=":material/folder:")
document_mining_page = st.Page("streamlit_pages/document_mining.py", title="Document Mining",icon=":material/search:")
chatbot_page         = st.Page("streamlit_pages/chatbot.py",         title="Chatbot",        icon=":material/smart_toy:")
dashbord_page        = st.Page("streamlit_pages/dashboard.py",        title="Dashboard",      icon=":material/bar_chart:")



    
# Navigation 
pg = st.navigation({
    'Tools'  : [home_page, documents_page, document_mining_page, chatbot_page, dashbord_page]
},
position="hidden" 
    )

st.set_page_config(page_title="IDEMIA RAG Project", page_icon=":robot_face:")

st.logo(image=paths.image_logo, size="large")

# Exécution de la page sélectionnée
pg.run()
