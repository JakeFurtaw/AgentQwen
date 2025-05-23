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
    device_map="auto",
    max_pixels=100000
)

# Initialize Whisper pipeline
whisper = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    device_map="auto",
    torch_dtype=torch_dtype,
    generate_kwargs={"language": "en"}  # Force English transcription
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

def process_input(image, audio, video, text, chat_history):
    # Initialize conversation for this session
    conversation = [SYSTEM_PROMPT]

    # Add previous chat history
    if isinstance(chat_history, list):
        for item in chat_history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                conversation.append(item)
    else:
        chat_history = []

    # Transcribe audio using Whisper if present
    transcribed_text = None
    if audio is not None:
        try:
            # Transcribe audio file
            result = whisper(audio)
            transcribed_text = result["text"]
            print(f"Transcribed text: {transcribed_text}")
        except Exception as e:
            print(f"Whisper transcription failed: {e}")
            transcribed_text = "Error transcribing audio."

    # Combine multimodal inputs, exclude audio for Qwen processing
    user_input = {
        "text": text if text is not None or "" else None,
        "image": image if image is not None else None,
        "audio": audio if audio is not None else None,
        "video": video if video is not None else None
    }

    # Add current user input
    user_content = user_input_to_content(user_input)
    conversation.append({"role": "user", "content": user_content})

    # Prepare for inference
    text_for_model = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

    inputs = processor(
        text=text_for_model,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True
    )

    inputs = inputs.to(model.device).to(model.dtype)

    # Generate response
    try:
        with torch.no_grad():
            text_ids, audio = model.generate(
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
        audio = None

    # Save audio to temporary file if available
    audio_path = None
    if audio is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(
                    tmp_file.name,
                    audio.reshape(-1).detach().cpu().numpy(),
                    samplerate=24000,
                )
                audio_path = tmp_file.name
        except Exception as e:
            print(f"Failed to save audio: {e}")
            audio_path = None

    # Clean up text response
    word = "assistant"
    try:
        full_response = processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        if full_response and word in full_response:
            num_occurrences = full_response.count(word)
            if num_occurrences > 0:
                parts = full_response.split(word, num_occurrences)
                assistant_response = parts[-1].strip() if parts[-1].strip() else parts[-2].strip() if len(
                    parts) > 1 else full_response.strip()
            else:
                assistant_response = full_response.strip()
        else:
            assistant_response = "No response generated."
    except Exception as e:
        assistant_response = f"Error processing response: {str(e)}"

    # Format user message for chat history display
    user_message = (text if text is not None or "" else "")
    if image is not None:
        user_message = (user_message or "Image uploaded!")
    if audio is not None:
        user_message = (transcribed_text if transcribed_text is not None or "" else "Audio uploaded!")
    if video is not None:
        user_message = (user_message or "Video uploaded!")

    if not user_message.strip():
        user_message = "Multimodal input"

    # Update chat history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": assistant_response})

    return chat_history, assistant_response, audio_path

def user_input_to_content(user_input):
    if isinstance(user_input, str):
        return user_input
    elif isinstance(user_input, dict):
        content = []
        if "text" in user_input and user_input["text"]:
            content.append({"type": "text", "text": user_input["text"]})
        if "image" in user_input and user_input["image"]:
            content.append({"type": "image", "image": user_input["image"]})
        if "audio" in user_input and user_input["audio"]:
            content.append({"type": "audio", "audio": user_input["audio"]})
        if "video" in user_input and user_input["video"]:
            content.append({"type": "video", "video": user_input["video"]})
        return content
    return user_input

def process_multimodal_textbox(multimodal_input):
    image, audio, video, text = None, None, None, ""

    if not isinstance(multimodal_input, dict):
        return image, audio, video, text

    text = multimodal_input.get("text", "")
    for file in multimodal_input.get("files", []):
        if isinstance(file, str):
            if file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                audio = file
            elif file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                image = file
            elif file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video = file
        elif isinstance(file, dict):
            file_path = file.get("path", "")
            mime_type = file.get("mime_type", "")
            if mime_type.startswith("image") or file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                image = file_path
            elif mime_type.startswith("audio") or file_path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                audio = file_path
            elif mime_type.startswith("video") or file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video = file_path

    return image, audio, video, text