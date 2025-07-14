# dashbord.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import redis_db


# --- AUTHENTIFICATION ---
# On utilise st.session_state pour ne pas redemander le mot de passe à chaque interaction
if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    pwd = st.text_input(
        "🔒 Please enter the administrator password", 
        type="password"
    )
    st.page_link("streamlit_pages/home.py", label="", icon="🔙")
    if pwd:
        if pwd == st.secrets["dashboard_password"]:
            st.session_state.authed = True
            st.rerun()  # relance l’app pour afficher le dashboard
        else:
            st.error("Mot de passe incorrect")
    # Tant que st.session_state.authed est False, on stoppe l’exécution
    st.stop()


# Redis client
r = redis_db.create_redis_client()

# 7 last days
today = datetime.utcnow().date()
days = [today - timedelta(days=i) for i in range(6, -1, -1)]
dates = [d.isoformat() for d in days]

# Retrieve data from redis
user_counts = [r.scard(f"users:{date}") for date in dates]
pos = [int(r.get(f"feedback:positive:{date}") or 0) for date in dates]
neg = [int(r.get(f"feedback:negative:{date}") or 0) for date in dates]
responses = [int(r.get(f"responses:{date}") or 0) for date in dates]



# the mean of duration
avg_response_times = []
for date in dates:
    # list string converted to float
    lst = r.lrange(f"response_times:{date}", 0, -1)
    floats = [float(v) for v in lst] if lst else []
    avg = sum(floats) / len(floats) if floats else 0
    avg_response_times.append(round(avg, 2))

# DataFrame for duration
df_rt = pd.DataFrame({"Temps de réponse moyen (s)": avg_response_times}, index=dates)


# Calculation of all feedbacks and percentage
total_fb = [pos[i] + neg[i] for i in range(len(dates))]
rates = [(total_fb[i] / responses[i]) if responses[i] > 0 else 0 for i in range(len(dates))]

# Dataframe for daily users
df_users = pd.DataFrame({
    "Date": dates,
    "Utilisateurs": user_counts
}).set_index("Date")

# DataFrame for daily feedbacks
df_fb = pd.DataFrame({
    "Positifs": pos,
    "Négatifs": neg
}, index=dates)

# DataFrame for  feedback percentage
df_rate = pd.DataFrame({
    "Taux de feedback (%)": [round(rate * 100, 2) for rate in rates]
}, index=dates)

st.page_link("streamlit_pages/home.py", label="Home", icon="🏠")

# Streamlit UI
st.title("📊 Dashboard RAG")

st.subheader("Users per day")
st.bar_chart(df_users)

st.subheader("Positives vs négatives feedbacks (per day)")
st.bar_chart(df_fb)

st.subheader("Feedbacks rate (feedbacks / réponses IA)")

# KPI for most recent percentage of feedbacks
dernier_taux = df_rate["Taux de feedback (%)"].iloc[-1]
st.metric("Current feedbacks rate (%)", f"{dernier_taux}%")

# Graph for the feedbacks of the week
st.area_chart(df_rate)

# Graph for the duration 
st.subheader("⏱️ average response time per day in seconds")
st.line_chart(df_rt)
