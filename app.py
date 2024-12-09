from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline, Speech2TextProcessor, Speech2TextForConditionalGeneration
import torch
import librosa
import soundfile as sf

app = FastAPI()

# Load Hugging Face pipelines
image_generator = pipeline("image-generation", model="CompVis/stable-diffusion-v1-4")
text_translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
speech_to_text_processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
speech_to_text_model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")

@app.post("/generate-image/")
async def generate_image(prompt: str):
    images = image_generator(prompt, num_images=1)
    return JSONResponse(content={"images": images[0]["image"]})

@app.post("/translate-text/")
async def translate_text(text: str):
    translation = text_translator(text)
    return {"translated_text": translation[0]["translation_text"]}

@app.post("/speech-to-text/")
async def speech_to_text(file: UploadFile = File(...)):
    audio, sr = librosa.load(file.file, sr=16000)
    sf.write("temp.wav", audio, sr)
    speech, _ = librosa.load("temp.wav", sr=16000)
    input_values = speech_to_text_processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    logits = speech_to_text_model.generate(input_values)
    transcription = speech_to_text_processor.batch_decode(logits)[0]
    return {"transcription": transcription}

@app.get("/")
def home():
    return {"message": "Welcome to the modern chatbot API"}
