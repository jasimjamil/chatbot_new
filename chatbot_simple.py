import os
import requests
from flask import Flask, request, jsonify, render_template
import streamlit as st
from transformers import pipeline, MarianMTModel, MarianTokenizer
import subprocess  # Import subprocess for launching Flask in a separate process

# Initialize Flask app
app = Flask(__name__)

# Setup Text Generation with Hugging Face
text_generator = pipeline("text-generation", model="gpt2")

# Setup Translation with Hugging Face (using MarianMT)
translation_model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"  # English to Romance languages
translation_model = MarianMTModel.from_pretrained(translation_model_name)
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # A simple HTML form for interaction

# Text generation endpoint
@app.route('/generate-text', methods=['POST'])
def generate_text():
    prompt = request.json.get('prompt', '')
    try:
        response = text_generator(prompt, max_length=200, num_return_sequences=1)
        return jsonify({"text": response[0]['generated_text']})
    except Exception as e:
        return jsonify({"error": str(e)})

# Translation endpoint
@app.route('/translate', methods=['POST'])
def translate_text():
    text = request.json.get('text', '')
    target_language = request.json.get('target_language', 'es')  # Default to Spanish
    try:
        # Use MarianMT model for translation
        tokenized_text = translation_tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
        translated_tokens = translation_model.generate(**tokenized_text)
        translated_text = translation_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        return jsonify({"error": str(e)})

# Word counter endpoint
@app.route('/word-count', methods=['POST'])
def word_count():
    text = request.json.get('text', '')
    word_count = len(text.split())
    return jsonify({"word_count": word_count})

# Streamlit frontend
def run_streamlit_app():
    # Set the Flask API base URL
    API_URL = "http://127.0.0.1:5001"  # Update port to 5001

    # Streamlit app title
    st.title("AI Text Generator and Translator")

    # Text Generation Section
    st.header("Text Generation")
    text_prompt = st.text_input("Enter a prompt for text generation:")
    if st.button("Generate Text"):
        if text_prompt:
            try:
                response = requests.post(f"{API_URL}/generate-text", json={"prompt": text_prompt})
                generated_text = response.json().get("text", "Error: Unable to generate text")
                st.text_area("Generated Text", generated_text, height=200)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a prompt!")

    # Translation Section
    st.header("Text Translation")
    text_to_translate = st.text_area("Enter text to translate:")
    target_language = st.selectbox("Select target language:", ["es", "fr", "de", "it"])  # Spanish, French, German, Italian
    if st.button("Translate Text"):
        if text_to_translate:
            try:
                response = requests.post(
                    f"{API_URL}/translate", json={"text": text_to_translate, "target_language": target_language}
                )
                translated_text = response.json().get("translated_text", "Error: Unable to translate text")
                st.text_area("Translated Text", translated_text, height=200)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter text to translate!")

    # Word Counter Section
    st.header("Word Counter")
    text_for_word_count = st.text_area("Enter text to count words:")
    if st.button("Count Words"):
        if text_for_word_count:
            try:
                response = requests.post(f"{API_URL}/word-count", json={"text": text_for_word_count})
                word_count = response.json().get("word_count", "Error: Unable to count words")
                st.success(f"Word Count: {word_count}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter text!")

# Run Flask and Streamlit together
def run_flask_app():
    subprocess.Popen(['python', 'flask_app.py'])  # Replace 'flask_app.py' with your Flask app script's name

if __name__ == '__main__':
    run_flask_app()  # Run Flask app
    run_streamlit_app()  # Run Streamlit app
