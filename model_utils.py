import torch, tempfile
import soundfile as sf
from transformers.models.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype=torch_dtype,
    device_map="auto",
    enable_audio_output=True,
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

# System prompt
SYSTEM_PROMPT = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": (
                "You are Agent Qwen, a virtual Geek Squad Agent who can understand text, audio, and video and "
                "perceive auditory and visual inputs, as well as generate text and speech. "
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

    # Combine multimodal inputs
    user_input = {
        "text": text,
        "image": image if image is not None else None,
        "audio": audio if audio is not None else None,
        "video": video if video is not None else None
    }

    # Add current user input
    user_content = user_input_to_content(user_input)
    conversation.append({"role": "user", "content": user_content})

    # Prepare for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

    inputs = processor(
        text=text,
        audios=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # Generate response
    try:
        text_ids, audio = model.generate(
            **inputs,
            use_audio_in_video=True,
            return_audio=True
        )
    except Exception as e:
        print(f"Audio generation failed: {e}")
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=True,
            return_audio=False
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

    # Decode text response
    text_response = processor.batch_decode(
        text_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Clean up text response
    text_response = text_response.strip()

    # Format user message for chat history display
    user_message_for_display = str(text) if text is not None else ""
    if image is not None:
        user_message_for_display = (user_message_for_display or "Image uploaded") + " [Image]"
    if audio is not None:
        user_message_for_display = (user_message_for_display or "Audio uploaded") + " [Audio]"
    if video is not None:
        user_message_for_display = (user_message_for_display or "Video uploaded") + " [Video]"

    # If empty, provide a default message
    if not user_message_for_display.strip():
        user_message_for_display = "Multimodal input"

    # Update chat history with messages format
    chat_history.append({"role": "user", "content": user_message_for_display})
    chat_history.append({"role": "assistant", "content": text_response})

    return chat_history, text_response, audio_path


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
    """
    Process input from Gradio's MultimodalTextbox, which returns a dict with 'text' and 'files'.
    Returns: image, audio, video, text
    """
    text = multimodal_input.get("text", "")
    image, audio, video = None, None, None

    for file in multimodal_input.get("files", []):
        file_path = file.get("path", "")
        mime_type = file.get("mime_type", "")
        if mime_type.startswith("image"):
            image = file_path
        elif mime_type.startswith("audio"):
            audio = file_path
        elif mime_type.startswith("video"):
            video = file_path

    return image, audio, video, text