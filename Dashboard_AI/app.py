import streamlit as st
from PIL import Image
from prompt import generate_reply

st.set_page_config(page_title="Social Dashboard", layout="wide")

st.title("🤖 Social Media AI Dashboard")
st.markdown("Incoming messages are shown below the icons. Replies are generated using Groq's LLM.")

#  messages 
messenger_msg = "If I make it, it will be history"
facebook_msg = "🚨 𝐁𝐑𝐄𝐀𝐊𝐈𝐍𝐆: Kylian Mbappé has been discharged from the hospital and has returned to the training camp. Mbappé will continue receiving specific medical treatment and will gradually return to team activity."
youtube_msg = "I wanna be one of the ones bouncing off the ship 😂"

col1, col2, col3 = st.columns(3)

# COLUMN 1: Messenger
with col1:
    st.image("Dashboard_AI/icons/ig.jpg", width=100)
    st.info(f"📩 Incoming: {messenger_msg}")
    messenger_reply = generate_reply(messenger_msg)
    st.text_area("Instagram Reply", value=messenger_reply, height=120)

# COLUMN 2: Facebook
with col2:
    st.image("Dashboard_AI/icons/fb.png", width=100)
    st.warning(f"📩 Incoming: {facebook_msg}")
    facebook_reply = generate_reply(facebook_msg)
    st.text_area("Facebook Reply", value=facebook_reply, height=120)

# COLUMN 3: YouTube
with col3:
    st.image("Dashboard_AI/icons/yt.jpg", width=100)
    st.success(f"📩 Incoming: {youtube_msg}")
    youtube_reply = generate_reply(youtube_msg)
    st.text_area("YouTube Reply", value=youtube_reply, height=120)
