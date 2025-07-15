# app.py

import streamlit as st
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from TTS.api import TTS
import tempfile
import os
import torchaudio

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_llm_model():
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return tokenizer, model, generator

@st.cache_resource
def load_tts_model():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

whisper_model = load_whisper_model()
tokenizer, model, generator = load_llm_model()
tts_model = load_tts_model()

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def generate_response(input_text):
    prompt = f"You are a friendly local language tutor.\nUser: {input_text}\nTutor:"
    output = generator(prompt, max_new_tokens=150, do_sample=True, pad_token_id=generator.tokenizer.eos_token_id)
    response = output[0]["generated_text"]
    prompt_end_index = response.find("Tutor:")
    if prompt_end_index != -1:
        return response[prompt_end_index + len("Tutor:"):].strip()
    return response.strip()

def synthesize_speech(text, output_path):
    cleaned_text = "".join([c for c in text if c.isalnum() or c.isspace() or c in ".,!?'\""])
    tts_model.tts_to_file(text=cleaned_text, file_path=output_path)
    return output_path

st.title("Language Tutor App")
st.write("Upload an audio file and get a language tutor's response!")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    audio_bytes = uploaded_file.getvalue()
    st.audio(audio_bytes, format=uploaded_file.type)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(audio_bytes)
        temp_file_path = tmp_file.name

    with st.spinner("Transcribing audio..."):
        input_text = transcribe_audio(temp_file_path)
    st.text_area("Transcription:", input_text, height=150)

    with st.spinner("Generating response..."):
        model_response = generate_response(input_text)
    st.text_area("Tutor's Response:", model_response, height=150)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_res_tmp:
        audio_response_path = synthesize_speech(model_response, audio_res_tmp.name)

    st.audio(audio_response_path)

    # Clean up temporary files
    os.remove(temp_file_path)
    os.remove(audio_response_path)