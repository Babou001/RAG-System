# chatbot.py
import streamlit as st
import requests
import uuid
import time
from datetime import datetime
import redis_db
from streamlit_cookies_manager import EncryptedCookieManager


# 
FASTAPI_URL = "http://127.0.0.1:8000"
headers = {'User-Agent': 'Mozilla/5.0'}

# Client Redis global
redis_client = redis_db.create_redis_client()

# Initialize cookie manager for persistent session_id
cookies = EncryptedCookieManager(
    prefix="llama_chat",                        # app prefix
    password=st.secrets["COOKIE_PASSWORD"],     
)
if not cookies.ready():
    # wait until cookies are loaded/synced
    st.stop()

def save_feedback(msg_index):
    
    fb = st.session_state[f"feedback_{msg_index}"]
    st.session_state.history[msg_index]["feedback"] = fb

    
    date_str = datetime.utcnow().date().isoformat()
    key = f"feedback:{'positive' if fb else 'negative'}:{date_str}"
    redis_client.incr(key)

def stream_data_with_time(text, duration_str):
    """Streaming mot-à-mot + affichage de la durée à la fin."""
    for word in (text + f" \n\n*⏱️ {duration_str}*").split(" "):
        yield word + " "
        time.sleep(0.05)

# ========================================================================
# Manage persistent session_id via cookie
session_id = cookies.get("session_id")
if not session_id:
    session_id = str(uuid.uuid4())
    cookies["session_id"] = session_id
    cookies.save()                        # persist immediately
st.session_state.session_id = session_id


if "history" not in st.session_state:
    try:
        resp = requests.get(
            f"{FASTAPI_URL}/chat/history",
            params={"session_id": session_id},
            headers=headers,
            timeout=5
        )
        data = resp.json()
        st.session_state.history = data.get("history", [])
    except Exception as e:
        # Si échec (API down, pas d'historique, etc.), on repart vierge
        st.warning("⚠️Impossible de charger l’historique des messages, nouvelle session.")
        st.session_state.history = []




# =========================================================================


# to manage the sidebar toggle
if "disable" not in st.session_state : 
    st.session_state.disable = False


# toggle
st.page_link("streamlit_pages/home.py", label="Home", icon="🏠", disabled=st.session_state.disable)


st.header("Chat with Llama Model 🤖")

# Show chat history before new user inputs
for idx, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # on récupère directement la durée (float en secondes) formatée
            sec = msg.get("duration", 0.0)
            m, s = divmod(sec, 60)
            duration_str = f"{int(m)}:{int(s):02d}"
            st.write(msg["content"] + f"  \n\n*⏱️ {duration_str}*")
        else:
            st.write(msg["content"])


        # show duration if it is AI response
        if msg["role"] == "assistant":
            default = msg.get("feedback", None)
            st.session_state[f"feedback_{idx}"] = default
            st.feedback(
                options="thumbs",
                key=f"feedback_{idx}",
                disabled=default is not None,
                on_change=save_feedback,
                args=[idx],
            )

if "disable_button" not in st.session_state : 
    st.session_state.disable_button = False

def disable():
    st.session_state.disable_button = True
    st.session_state.disable = True
    

# New user input
user_input = st.chat_input("Enter your message:", disabled=st.session_state.disable_button , on_submit=disable)

if user_input :
    # display user input
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    # save (stats) for daily users
    date_str = datetime.utcnow().date().isoformat()
    redis_client.sadd(f"users:{date_str}", session_id)

    # call API +  time
    with st.spinner("Génération en cours…", show_time=True):
        resp = requests.post(
            f"{FASTAPI_URL}/chat",
            params={"user_input": user_input, "session_id": session_id},
            headers=headers
        ).json()

    # calculation and duration storing
    elapsed = resp.get("duration",0.0)
    m, s = divmod(elapsed, 60)
    formatted = f"{int(m)}:{int(s):02d}"

    # add AI response on chat history
    ai_text = resp["response"]
    st.session_state.history.append({
        "role": "assistant",
        "content": ai_text,
        "duration": elapsed
    })

   

    # display ai response and feedbacks widget
    with st.chat_message("assistant"):
        st.write_stream(stream_data_with_time(ai_text, formatted))
        idx = len(st.session_state.history) - 1
        st.session_state[f"feedback_{idx}"] = None
        st.feedback(
            options="thumbs",
            key=f"feedback_{idx}",
            on_change=save_feedback,
            args=[idx],
        )
    
    st.session_state.disable_button = False
    st.session_state.disable = False
    st.rerun()