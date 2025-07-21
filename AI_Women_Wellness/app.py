import streamlit as st
from config import OpenAIConfig
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.runtime import Runtime

# API key
api_key = "api_key"
openai_config = OpenAIConfig(api_key=api_key)

if "page" not in st.session_state:
    st.session_state.page = "Welcome"

if "mood" not in st.session_state:
    st.session_state.mood = "Welcome"

if st.session_state.get("go_to_coach", False):
    st.session_state.mood = "Talk with Coach"


def sidebar_nav():
    st.header("Navigation")
    st.write("Share your experience:")
    st.selectbox(
        "Function",
        ["Welcome", "Basic info", "Symptom", "Lifestyle", "Goal", "Daily remainder", "Confirmation", "Talk with Coach", "Tracker"],
        key="mood"
    )
    st.checkbox("Enable Notifications", value=True)
    st.selectbox("Language", ["English"], index=0)



def welcome_page():
    st.subheader("Welcome to FENYX")
    st.write("Let's get to know your body better so we can support you every step of the way.")
    start_btn = st.button("Get started!")
    if start_btn:
        st.session_state.mood = "Basic info"
        st.rerun()

def basic_info_page():
    st.subheader("Basic Info")
    st.write("Please tell us about your cycle.")
    period = st.selectbox("Do you have a regular cycle?", ["Yes", "No"], index=0)
    phase = st.selectbox("Are you in perimenopause, menopause, or post-menopause?", ["Yes", "No"], index=0)
    hormones = st.selectbox("Are you currently on hormonal birth control or HRT?", ["Yes", "No"], index=0)
    ctn_btn = st.button("Continue")
    if ctn_btn:
        st.session_state.mood = "Symptom"
        st.rerun()

def symptom_page():
    st.subheader("Symptom Tracker")
    st.write("What are your top 3 concerns right now? (Max 3)")
    symptoms = st.multiselect("Select symptoms:", ["Headache", "Fatigue", "Nausea", "Stress", "Back Pain", "Bloating", "Cravings", "Low Mood", 
                                                   "Hot Flashes", "Poor Sleep", "None", "Other"])
    ctn_btn = st.button("Continue")
    if ctn_btn:
        st.session_state.mood = "Lifestyle"
        st.rerun()

def lifestyle_page():
    st.subheader("Lifestyle and Nutrition")
    st.write("Tell us about your lifestyle and nutrition habits.")
    dietary_style = st.selectbox("Dietary Style", ["Omnivore", "Vegetarian", "Vegan", "Pescatarian", "Keto", "Other"], index=0)
    activity_level = st.selectbox("Activity Level", ["Low", "Moderate", "High"], index=0)
    stress_level = st.slider("Stress Level", 0, 10, 5)
    ctn_btn = st.button("Continue")
    if ctn_btn:
        st.session_state.mood = "Goal"
        st.rerun()

def goal_page():
    st.subheader("Goal Setting")
    st.write("What would you like FENYX to help you with?")
    goals = st.multiselect("Select your goals:", ["Balance hormones", "Track my cycle", "Improve mood", "Boost energy", "Reduce cravings", 
                                                  "Support weight changes", "Feel more in control", "Learn what’s happening in my body"])
    ctn_btn = st.button("Continue")
    if ctn_btn:
        st.session_state.mood = "Daily remainder"
        st.rerun()

def daily_remainder_page():
    st.subheader("Set Daily Reminder")
    st.write("Would you like a daily reminder to check in with your body?")
    daily_reminder = st.checkbox("Yes, remind me daily", value=True)
    ctn_btn = st.button("Continue")
    if ctn_btn:
        st.session_state.mood = "Confirmation"
        st.rerun()

def confirmation_page():
    st.subheader("You’re In FENYX!")
    st.write("Your personalized dashboard is ready. Let’s rise together.")
    dashboard_btn = st.button("Go to Dashboard")
    if dashboard_btn:
        st.session_state.mood = "Talk with Coach"
        st.rerun()
    


def talk_with_coach_page():
    st.subheader("Talk with Your Wellness Coach")
    st.write("Share your feelings, symptoms, and energy levels to get personalized wellness suggestions.")


    if "messages_ai_coach" not in st.session_state:
        st.session_state.messages_ai_coach = []
        

    #redirect from Tracker
    if st.session_state.get("go_to_coach"):
        st.session_state.go_to_coach = False

        
        mood = st.session_state.get("user_mood", "Neutral")
        symptoms = st.session_state.get("symptoms", [])
        energy = st.session_state.get("energy_level", 5)

        prompt = f"""
        The user is currently feeling: {mood}.
        Symptoms reported: {', '.join(symptoms) if symptoms else 'None'}.
        Energy level: {energy}/10.

        Based on this, provide a warm, supportive wellness suggestion and a personalized lifestyle or nutrition tip.
        """

        ai_response = openai_config.get_response(prompt)

        
        st.session_state.messages_ai_coach.append({"role": "assistant", "content": ai_response})
        #st.session_state.go_to_coach = True 

    if not st.session_state.messages_ai_coach:
        warm_prompt = """
        Hi there! I’m your wellness coach. 💬  
        I’d love to help you feel your best.  
        Could you please tell me how you're feeling today — your **mood**, any **symptoms**, and your **energy level**?

        Or you can use the **Wellness Tracker** on the left to submit your inputs!
        """
        st.session_state.messages_ai_coach.append({"role": "assistant", "content": warm_prompt})


    for message in st.session_state.messages_ai_coach:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask your coach anything...", key="user_input")
    
    
    if user_input:
        st.session_state.messages_ai_coach.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        response = openai_config.get_response(user_input)
        st.session_state.messages_ai_coach.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)



def rerun():
    ctx = get_script_run_ctx()
    Runtime.instance().request_rerun(ctx.session_id)



def tracker_page():
    st.subheader("Wellness Tracker")
    st.write("Track your wellness journey, symptoms, and mood over time.")

    user_mood = st.selectbox("Mood", ["Happy", "Sad", "Anxious", "Stressed", "Neutral"], index=0)
    symptoms = st.multiselect("Select symptoms:", ["Headache", "Fatigue", "Nausea", "Stress", "Back Pain", "Bloating", "Cravings", "Low Mood", 
                                                   "Hot Flashes", "Poor Sleep", "None", "Other"])
    energy_level = st.slider("How energetic do you feel today?", 0, 10, 5)
    submit_btn = st.button("Submit")
    
    if submit_btn:
        st.session_state.user_mood = user_mood
        st.session_state.symptoms = symptoms
        st.session_state.energy_level = energy_level
        st.session_state.go_to_coach = True
        st.rerun()


# --- PAGE ROUTING ---
if st.session_state.mood == "Welcome":
    welcome_page()
elif st.session_state.mood == "Basic info":
    basic_info_page()
elif st.session_state.mood == "Symptom":
    symptom_page()
elif st.session_state.mood == "Lifestyle":
    lifestyle_page()
elif st.session_state.mood == "Goal":
    goal_page()
elif st.session_state.mood == "Daily remainder":
    daily_remainder_page()
elif st.session_state.mood == "Confirmation":
    confirmation_page()
elif st.session_state.mood == "Talk with Coach":
    talk_with_coach_page()
elif st.session_state.mood == "Tracker":
    tracker_page()

# --- MOVE SIDEBAR TO THE END ---
with st.sidebar:
    sidebar_nav()

