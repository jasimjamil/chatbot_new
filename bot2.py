import speech_recognition as sr
from gtts import gTTS
import os
from transformers import pipeline
from googletrans import Translator
from playsound import playsound
import streamlit as st


def speech_to_text():
    
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Speak now:")
            audio_data = recognizer.listen(source)
            text = recognizer.recognize_google(audio_data)
            st.success(f"User Input (Speech-to-Text): {text}")
            return text
    except OSError as e:
        st.error("No default microphone device found, please use text input.")
        return st.text_input("Enter your message:")
    except sr.UnknownValueError:
        st.error("Sorry, I did not understand that.")
        return ""
    except sr.RequestError:
        st.error("Request Error, please check your internet connection.")
        return ""

# Function for Text-to-Speech
def text_to_speech(text, language_code='en'):
    tts = gTTS(text=text, lang=language_code, slow=False)
    filename = "response.mp3"
    tts.save(filename)
    st.audio(filename)  
    os.remove(filename)  


def translate_text(text, target_language='en'):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    st.success(f"Translated Text ({target_language}): {translated.text}")
    return translated.text


@st.cache_resource
def load_chatbot():
    st.info("Loading Chatbot Model... Please wait.")
    return pipeline("text-generation", model="microsoft/DialoGPT-medium", max_length=100, pad_token_id=50256)

chatbot = load_chatbot()

def voice_chatbot():
    st.title("🎙️ Multi-Language Voice-to-Voice Chatbot")
    st.write("This chatbot allows you to communicate using your voice, supporting multiple languages!")

    # Input for Target Language
    target_language = st.text_input("Enter your target language code (e.g., 'en' for English, 'fr' for French):", "en")

    if st.button("Start Conversation"):
        st.warning("Speak 'exit', 'quit', or 'stop' to end the conversation.")
        
        while True:
            
            user_input = speech_to_text()
            if not user_input or user_input.lower() in ["exit", "quit", "stop"]:
                st.warning("Exiting Chatbot. Goodbye!")
                break

            translated_input = translate_text(user_input, target_language='en')

            # Step 3: Generate chatbot response
            response = chatbot(translated_input, do_sample=True, max_length=100, pad_token_id=50256)
            chatbot_reply = response[0]['generated_text']
            st.info(f"Chatbot Reply (in English): {chatbot_reply}")

            # Step 4: Translate chatbot reply back to target language
            translated_reply = translate_text(chatbot_reply, target_language=target_language)

            # Step 5: Convert chatbot reply to speech
            text_to_speech(translated_reply, language_code=target_language)

# Run the Streamlit app
if __name__ == "__main__":
    voice_chatbot()
