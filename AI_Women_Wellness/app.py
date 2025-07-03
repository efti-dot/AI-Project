import streamlit as st


st.title("AI Women Wellness Coach!")
st.write("Talk to your wellness companion. I'm here to help you with your wellness journey.")


with st.sidebar:
    st.header("Navigation")
    st.write("Customize your experience:")

    
    user_choice = st.selectbox("Options", ["Talk with Coach", "Lifestyle Tips", "Mood Tracker", "Suggestion"], index=0)

    
    cycle_phase = st.selectbox("Phase", ["Menstrual", "Follicular", "Ovulation", "Luteal", "Perimenopause", "Menopause"], index=0)

    st.checkbox("Enable Notifications", value=True)
    st.selectbox("Language", ["English"], index=0)



def talk_with_coach():
    st.subheader("Talk with Your Coach")
    st.write("Ask your wellness coach anything about your health, fitness, or wellness journey.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    
    user_input = st.chat_input("Ask your coach...", key="user_input")

    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        
        response = f"Thank you for your question! Let's explore that topic together."
        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.write(response)


def lifestyle_tips():
    st.subheader("Lifestyle Tips")
    st.write("Get personalized tips for a healthier lifestyle.")
    tips = [
        "Drink plenty of water daily.",
        "Incorporate more fruits and vegetables into your diet.",
        "Exercise regularly, even if it's just a short walk.",
        "Practice mindfulness and meditation.",
        "Get enough sleep each night."
    ]
    for tip in tips:
        st.write(f"- {tip}")


def mood_tracker():
    st.subheader("Mood Tracker")
    st.write("Track your mood and well-being.")
    mood = st.selectbox("How are you feeling today?", ["Happy", "Sad", "Anxious", "Excited", "Calm"])

    if mood:
        st.session_state.messages.append({"role": "user", "content": f"Mood: {mood}"})
        response = f"Thank you for sharing your mood: {mood}. It's important to acknowledge how you feel."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)


def suggestion():
    st.subheader("Suggestions")
    st.write("Get suggestions for improving your wellness.")
    suggestions = [
        "Try a new hobby or activity.",
        "Spend time in nature.",
        "Connect with friends or family.",
        "Set small, achievable goals.",
        "Practice gratitude daily."
    ]
    for suggestion in suggestions:
        st.write(f"- {suggestion}")


if user_choice == "Talk with Coach":
    talk_with_coach()
elif user_choice == "Lifestyle Tips":
    if cycle_phase == "Menstrual":
        st.write("Tip: During your Menstrual phase, prioritize rest and relaxation. Focus on calming activities such as meditation and light walking.")
    elif cycle_phase == "Follicular":
        st.write("Tip: During the Follicular phase, it's a great time to plan new projects, focus on nutrition, and engage in more vigorous activities like strength training.")
    elif cycle_phase == "Ovulation":
        st.write("Tip: During the Ovulatory phase, try social activities, communication, and networking. Your energy and confidence will be high!")
    elif cycle_phase == "Luteal":
        st.write("Tip: In the Luteal phase, focus on self-care, emotional balance, and stress reduction. Try gentle exercises like yoga and focus on healthy eating.")
    elif cycle_phase == "Perimenopause":
        st.write("Tip: During Perimenopause, maintain a balanced diet and focus on gentle exercises. Managing stress and sleep is also important.")
    elif cycle_phase == "Menopause":
        st.write("Tip: During Menopause, prioritize your bone health and stay active with regular, low-impact exercises.")

elif user_choice == "Mood Tracker":
    mood_tracker()

elif user_choice == "Suggestion":
    st.write(f"Here are suggestions for your {cycle_phase} phase. These suggestions are tailored to your hormonal changes and can help improve your wellness.")
    if cycle_phase == "Menstrual":
        st.write("Suggestion: Get plenty of sleep, eat foods rich in iron, and reduce stress.")
    elif cycle_phase == "Follicular":
        st.write("Suggestion: Engage in high-energy activities and focus on balanced nutrition, especially protein-rich foods.")
    elif cycle_phase == "Ovulation":
        st.write("Suggestion: Consider taking advantage of your high energy levels with more social interactions, creative projects, and cardio workouts.")
    elif cycle_phase == "Luteal":
        st.write("Suggestion: Practice mindfulness, ensure you're getting enough rest, and avoid overexerting yourself.")
    elif cycle_phase == "Perimenopause":
        st.write("Suggestion: Incorporate strength training and focus on a healthy diet to support hormonal balance.")
    elif cycle_phase == "Menopause":
        st.write("Suggestion: Focus on maintaining bone health and managing stress through relaxation techniques.")

