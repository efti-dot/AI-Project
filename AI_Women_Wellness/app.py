import streamlit as st
from config import OpenAIConfig

# API key
api_key = "api_key"
openai_config = OpenAIConfig(api_key=api_key)

st.title("AI Women Wellness Coach!")
st.write("Talk to your wellness companion. I'm here to help you with your wellness journey.")

with st.sidebar:
    st.header("Navigation")
    st.write("Share your experience:")

    user_mood = st.selectbox("Mood", ["Happy", "Sad", "Anxious", "Stressed", "Neutral"], index=0)
    symptoms = st.multiselect("Select symptoms:", ["Headache", "Fatigue", "Nausea", "Stress", "Back Pain", "Bloating", "Cravings", "Low Mood", 
                                                   "Hot Flashes", "Poor Sleep", "None", "Other"])
    energy_level = st.slider("How energetic do you feel today?", 0, 10, 5)
    submit_btn = st.button("Submit")

    st.checkbox("Enable Notifications", value=True)
    st.selectbox("Language", ["English"], index=0)

if "messages_ai_coach" not in st.session_state:
    st.session_state.messages_ai_coach = []

# --- Ask for user input if not provided ---
if not st.session_state.messages_ai_coach:
    # Friendly prompt to gather user input
    warm_prompt = """
    Hi there! I’m your wellness coach. Before we get started, I would love to know a bit about how you’re feeling today.
    
    Please share: Your current mood, symptoms and energy level.
    
    Once you click "Submit," I’ll give you personalized suggestions and tips based on your input!
    """

    st.session_state.messages_ai_coach.append({"role": "assistant", "content": warm_prompt})
    with st.chat_message("assistant"):
        st.write(warm_prompt)

# --- Show message history ---
for message in st.session_state.messages_ai_coach:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Generate suggestions after user clicks submit ---
if submit_btn:
    # Construct the prompt using the selected mood, symptoms, and energy level
    prompt = f"""
    The user is currently feeling: {user_mood}.
    Symptoms reported: {', '.join(symptoms) if symptoms else 'None'}.
    Energy level: {energy_level}/10.

    Based on this, please provide a friendly, supportive wellness suggestion and personalized nutrition or lifestyle tip.
    """
    
    # Get response from the AI based on the prompt
    ai_response = openai_config.get_response(prompt)

    # Store assistant's response in session state
    st.session_state.messages_ai_coach.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.write(ai_response)

user_input = st.chat_input("Ask your coach...", key="user_input")

if user_input:
    st.session_state.messages_ai_coach.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    prompt = f"""
    The user is currently feeling: {user_mood}.
    Symptoms reported: {', '.join(symptoms) if symptoms else 'None'}.
    Energy level: {energy_level}/10.
    User said: {user_input}

    Based on this, provide a thoughtful response and potentially adjust wellness suggestions or tips.
    """

    # Get response from the AI based on user input
    response = openai_config.get_response(prompt)
    
    st.session_state.messages_ai_coach.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)










def lifestyle_tips():
    st.subheader("Lifestyle Tips")
    st.write("Get personalized tips for a healthier lifestyle.")
    
    if "messages_lifestyle_tips" not in st.session_state:
        st.session_state.messages_lifestyle_tips = []

    for message in st.session_state.messages_lifestyle_tips:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask your coach...", key="user_input")

    if user_input:
        st.session_state.messages_lifestyle_tips.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Add the dynamic prompt based on user's choice
        if user_mood == "Talk with Coach":
            response = openai_config.get_response(f"Wellness coach: {user_input}")
        elif user_mood == "Lifestyle Tips":
            response = openai_config.get_response(f"Provide wellness lifestyle tips based on: {user_input}")
        elif user_mood == "Mood Tracker":
            response = openai_config.get_response(f"How can I improve my mood based on: {user_input}")
        else:
            response = openai_config.get_response(user_input)

        st.session_state.messages_lifestyle_tips.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.write(response)




