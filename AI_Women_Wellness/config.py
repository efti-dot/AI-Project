# config.py
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch

class LocalAIConfig:
    def __init__(self, model_name: str = "gpt2-medium"):
        """
        Initializes the local Hugging Face transformer model for text generation.
        """
        print("🔁 Loading local model...")
        self.device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device=self.device
        )

    def get_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 150):
        """
        Generates a text response using GPT-2 locally.
        """
        try:
            system_prompt = (
                "You are a calm, empathetic, and friendly wellness coach. "
                "Your responses should be thoughtful and supportive. "
                "Your goal is to help users feel empowered and informed about their health and well-being.\n"
            )
            final_prompt = system_prompt + prompt

            results = self.generator(final_prompt, max_length=max_tokens, temperature=temperature, num_return_sequences=1)
            return results[0]['generated_text'][len(final_prompt):].strip()  # remove the prompt from output
        except Exception as e:
            print(f"Error generating response locally: {e}")
            return "Sorry, I couldn't process your request at the moment. Please try again later."


    '''def get_prompt(self, cycle_phase: str) -> str:
        """
        Returns a pre-defined prompt based on the user's hormonal cycle phase.
        
        :param cycle_phase: The phase of the user's hormonal cycle.
        :return: A tailored prompt for the AI based on the cycle phase.
        """
        prompts = {
            "Menstrual": """
            🌸 You're in the Menstrual phase, and it's time to slow down and prioritize **self-care**. 🌸
            - **Sleep**: Ensure you get 7-9 hours of deep, restful sleep. Your body is working hard to recover, so give it the rest it needs.
            - **Stress**: Embrace calmness—practice deep breathing, meditation, or take time for quiet reflection. It's your time to recharge.
            - **Nutrition**: Replenish your energy with **iron-rich foods** like leafy greens, lentils, and beets. Stay hydrated and reduce caffeine intake.
            - **Exercise**: Gentle walks or light stretching can relieve cramps and support your recovery. Don’t push yourself too hard—be kind to your body.
            This is your time for **self-compassion**, relaxation, and healing. You're doing **amazing**! 🌼 Take it easy, and honor your body’s needs.
            """,
            "Follicular": """
            🌸 You're in the Follicular phase, the time of renewal and energy! 🌸
            - **Sleep**: Quality sleep is key for mental clarity. Aim for 7-8 hours to recharge your energy.
            - **Stress**: This is a great time for mindfulness, yoga, or journaling to keep your mind calm and focused.
            - **Nutrition**: Boost your energy with foods like leafy greens, flaxseeds, and avocados to support brain health.
            - **Exercise**: It's the perfect time to take on new projects and challenge yourself with strength training or cardio.
            You’ve got the energy to take on anything! Stay active, stay healthy, and enjoy this phase to the fullest. 💪
            """,
            "Ovulation": """
            🌸 You're in the Ovulation phase, and your energy is peaking! 🌸
            - **Sleep**: Continue prioritizing 7-8 hours of sleep to keep your body energized throughout the day.
            - **Stress**: With your energy levels high, stay active with light cardio, and make time for socializing.
            - **Nutrition**: Enjoy high-protein foods, like lean meats and legumes, and focus on fiber-rich veggies.
            - **Exercise**: Take advantage of your vitality with HIIT workouts, or go for a run to get that natural high.
            You're in your power phase—social, energetic, and ready to go! Shine bright, and keep that energy flowing. ✨
            """,
            "Luteal": """
            🌸 You're in the Luteal phase, a time for nurturing your body and mind. 🌸
            - **Sleep**: Ensure 8-9 hours of sleep to fight fatigue and combat mood swings.
            - **Stress**: Engage in relaxing activities like yoga, meditation, or a warm bath to reduce stress.
            - **Nutrition**: Magnesium-rich foods like dark chocolate, avocado, and sweet potatoes will help alleviate PMS symptoms.
            - **Exercise**: Opt for light exercises like walking or gentle yoga to balance energy levels and relax your body.
            This is a time for rest and self-care. Take things slow, and know that you’re doing your best! 🌷
            """,
            "Perimenopause": """
            🌸 You're in Perimenopause, and it’s all about balance and self-care. 🌸
            - **Sleep**: Prioritize quality sleep (7-8 hours), and avoid caffeine late in the day to prevent insomnia.
            - **Stress**: Keep your stress under control with activities like mindfulness or journaling.
            - **Nutrition**: Support your hormonal health with chia seeds, berries, and healthy fats from avocados.
            - **Exercise**: Light exercise, like walking or swimming, is great for maintaining joint health and boosting energy.
            This is your time to embrace change. Be kind to yourself and practice mindful living. You’ve got this! 🌿
            """,
            "Menopause": """
            🌸 You're in Menopause, and it’s time to take care of your body and mind as you transition. 🌸
            - **Sleep**: Aim for 7-8 hours of restful sleep, and avoid large meals before bedtime to help with insomnia.
            - **Stress**: Reduce stress with activities like yoga, tai chi, or deep breathing exercises.
            - **Nutrition**: Focus on bone health with protein-rich foods, leafy greens, and anti-inflammatory spices like turmeric.
            - **Exercise**: Low-impact exercises, such as walking or cycling, are great for maintaining cardiovascular health.
            You’re navigating through a big life change, but remember to take it one step at a time. You’re stronger than you think! 💪🌸
            """
        }
        return prompts.get(cycle_phase, "Please specify a valid phase like menstrual, follicular, ovulation, luteal, perimenopause or menopause.")'''
