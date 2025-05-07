import base64
from PIL import Image
import numpy as np
import io, torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info



def preprocess_image(image: Image.Image) -> str:
    """Convert PIL image to base64 string for Ollama API."""
    if image is None:
        return None
    # Convert image to JPEG and encode as base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def preprocess_audio(audio: tuple) -> str:
    """Convert audio numpy array to base64 WAV for Ollama API."""
    if audio is None:
        return None
    # Audio is a tuple (sample_rate, numpy_array) from Gradio
    sample_rate, audio_data = audio
    # Save audio to WAV buffer
    buffered = io.BytesIO()
    sf.write(buffered, audio_data, sample_rate, format="WAV")
    buffered.seek(0)
    return base64.b64encode(buffered.read()).decode("utf-8")

def call_multimodal_model(text: str, image: dict = None, audio: dict = None) -> str:
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path="Qwen/Qwen2.5-Omni-7B",
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text",
                 "text": "You are Agent Qwen, a virtual Geek Squad Agent who can understand text, audio, and video and"
                         "perceiving auditory and visual inputs, as well as generating text and speech."
                         "Your goal is to learn about the customers issue with their device. Then generate a report"
                         "based on what the customer says so that another agent can work on the computer and fix the"
                         "issue."
                         "Strictly adhere to these guidelines and make sure to summarize what the customer has said the"
                         " issue is. We dont want every detail in the report on the details that will help the other agent"
                         "find the solution the fix the device."}
            ],
        },
        {
            "role": "user",
            "content": [
            ],
        },
    ]

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
                       use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference: Generation of the output text and audio
    text_ids, audio = model.generate(**inputs, use_audio_in_video=True, speaker = 'Ethan')

    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(text)
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )