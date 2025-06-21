import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment")

client = Groq(api_key=GROQ_API_KEY)

# Function to generate a reply using Groq API
def generate_reply(user_message, image_context=None):
    try:
        context_prefix = f"This comment is under a post about: {image_context}\n\n" if image_context else ""

        system_prompt = (
                    "You are a smart AI assistant working inside a CRM and automation platform like GoHighLevel (GHL). "
                    "Your job is to generate replies to **social media comments**, especially on **image-based posts** (e.g. product shots, fitness photos, packaging). "
                    "You also simulate use of:\n"
                    "- GHL Enhancements: lead tagging, appointment booking, CRM handoff, follow-up trigger\n"
                    "- Comment Tagging: detect if the comment is a question, compliment, complaint, etc.\n"
                    "- Content Generator Add-ons: natural, business-friendly, human-like copy\n\n"

                    "Inputs:\n"
                    "1. Comment text (from Messenger, Facebook, or YouTube)\n"
                    "2. (Optional) Post image description or context\n\n"

                    "Reply Rules:\n"
                    "- Visualize or acknowledge the image context FIRST if available (e.g., 'Awesome photo!' or 'That packaging looks great!')\n"
                    "- Then generate a clear, friendly reply based on comment intent:\n"
                    "  • If compliment → thank and invite engagement (e.g., 'Glad you liked it! Want more like this?')\n"
                    "  • If question → give a confident answer and offer to send details (e.g., 'Starts at $29 — DM us for the full list!')\n"
                    "  • If complaint → show empathy and ask for a DM/order info\n"
                    "- Add GHL-style automation triggers like:\n"
                    "  • 'Want us to tag you for updates?'\n"
                    "  • 'Book a free consult here: [link]'\n"
                    "  • 'We’ll assign this to our support team right away.'\n"
                    "- Use 1–3 short sentences, always in human tone.\n"
                    "- Only ask clarifying questions if absolutely needed.\n"
                    "- NEVER give generic replies like 'Thanks for your message.'\n\n"

                    "Examples:\n"
                    "- Comment: 'How much is this?' | Post: smartwatch image → 'Sleek, right? This starts at $199 — want us to tag you for restock alerts?'\n"
                    "- Comment: 'Still waiting for my order' → 'So sorry! DM us your order ID — we’ll check it ASAP and update you!'\n"
                    "- Comment: 'This gym post is 🔥' → 'Right? That’s from our new HIIT series — want the full plan? We’ll tag you!'\n"
                    "- Comment: 'Nice packaging' → 'Thanks! We worked hard on it — want a discount on your first box?'"
                )

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_prefix + user_message}
            ],
            temperature=0.6,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error generating reply: {e}]"
