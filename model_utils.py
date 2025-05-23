import torch
import tempfile
import soundfile as sf
from transformers import pipeline
from transformers.models.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from dotenv import load_dotenv

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

# Initialize Qwen model and processor
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype=torch_dtype,
    device_map="auto",
    enable_audio_output=True
)
processor = Qwen2_5OmniProcessor.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    max_pixels=100000
)

# Initialize Whisper pipeline
whisper = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    device_map="auto",
    torch_dtype=torch_dtype,
    generate_kwargs={"language": "en"}
)

# System prompt for Qwen
SYSTEM_PROMPT = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": (
                "You are Agent Qwen, a virtual human Geek Squad Agent developed by the Qwen Team, Alibaba Group,"
                "who can understand text, audio, and video and perceive auditory and visual inputs, as well as generate text and speech. "
                "If the customer doesnt state their name at the beginning of the interaction you must ask them and call them by their name."
                "Your goal is to learn about the customer's issue with their device. Then generate a report "
                "based on what the customer says so that another agent can work on the computer and fix the issue. "
                "Strictly adhere to these guidelines and make sure to summarize what the customer has said the issue is. "
                "We don't want every detail in the report, only the details that will help the other agent find the solution to fix the device."
            ),
        }
    ]
}

def transcribe_audio(audio_path):
    """Transcribe audio file using Whisper pipeline."""
    if not audio_path:
        return None
    try:
        result = whisper(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        return "Error transcribing audio."

def save_audio_to_temp_file(audio_data):
    """Save audio data to a temporary WAV file."""
    if audio_data is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(
                tmp_file.name,
                audio_data.reshape(-1).detach().cpu().numpy(),
                samplerate=24000,
            )
            return tmp_file.name
    except Exception as e:
        print(f"Failed to save audio: {e}")
        return None

def decode_model_response(text_ids):
    """Decode model-generated text IDs to a clean response."""
    try:
        full_response = processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        parts = full_response.split("assistant", full_response.count("assistant"))
        return parts[-1].strip() or parts[-2].strip() if len(parts) > 1 else full_response.strip()
    except Exception as e:
        return f"Error processing response: {str(e)}"

def prepare_user_message(text, image, audio, video, transcribed_text):
    """Format user message for chat history display."""
    if audio and transcribed_text:
        return transcribed_text
    if image:
        return "Image uploaded!"
    if image and text or transcribed_text:
        return text
    if video:
        return "Video uploaded!"
    if video and text or transcribed_text:
        return text
    return text or "Multimodal input"

def process_input(image, audio, video, text, chat_history):
    """Process multimodal input and generate a response."""
    # Initialize conversation
    conversation = [SYSTEM_PROMPT] + [
        item for item in (chat_history or [])
        if isinstance(item, dict) and "role" in item and "content" in item
    ]

    # Transcribe audio if present
    transcribed_text = transcribe_audio(audio)

    # Prepare user input
    user_input = {
        "text": text or None,
        "image": image or None,
        "audio": audio or None,
        "video": video or None
    }
    conversation.append({"role": "user", "content": user_input_to_content(user_input)})

    # Prepare model inputs
    text_for_model = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(
        text=text_for_model,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True
    ).to(model.device).to(model.dtype)

    # Generate response
    try:
        with torch.no_grad():
            text_ids, audio_data = model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=True,
                speaker="Ethan",
            )
    except Exception as e:
        print(f"Audio generation failed: {e}")
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=False,
            return_audio=False,
        )
        audio_data = None

    # Process outputs
    audio_path = save_audio_to_temp_file(audio_data)
    assistant_response = decode_model_response(text_ids)
    user_message = prepare_user_message(text, image, audio, video, transcribed_text)

    # Update chat history
    chat_history = chat_history or []
    chat_history.extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response}
    ])

    return chat_history, assistant_response, audio_path

def user_input_to_content(user_input):
    """Convert user input dictionary to content list for conversation."""
    if isinstance(user_input, str):
        return user_input
    content = []
    for key, value in user_input.items():
        if value and key in ("text", "image", "audio", "video"):
            content.append({"type": key, key: value})
    return content or user_input

def process_multimodal_textbox(multimodal_input):
    """Extract multimodal inputs from Gradio MultimodalTextbox."""
    if not isinstance(multimodal_input, dict):
        return None, None, None, ""

    text = multimodal_input.get("text", "")
    image, audio, video = None, None, None

    for file in multimodal_input.get("files", []):
        file_path = file if isinstance(file, str) else file.get("path", "")
        mime_type = file.get("mime_type", "") if isinstance(file, dict) else ""

        if (mime_type.startswith("image") or
            file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))):
            image = file_path
        elif (mime_type.startswith("audio") or
              file_path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))):
            audio = file_path
        elif (mime_type.startswith("video") or
              file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))):
            video = file_path

    return image, audio, video, text