import gradio as gr
from model_utils import process_input, process_multimodal_textbox, clear_audio_files

custom_css = """
/* General container styling */
.gradio-container {
    background-color: #000000; /* Geek Squad Black */
    color: #FFFFFF; /* White text */
    font-family: 'Arial', sans-serif;
}

/* Chatbot styling */
.gr-chatbot {
    background-color: #1C2526; /* Dark gray for contrast Provincia di Savona*/
    border: 2px solid #F60; /* Geek Squad Orange */
    border-radius: 10px;
    padding: 10px;
    color: #FFFFFF;
}

/* Chatbot message bubbles */
.gr-chatbot .message {
    background-color: #2E2E2E; /* Slightly lighter dark for messages */
    border-radius: 8px;
    padding: 8px;
    margin: 5px 0;
}

/* Input box styling */
.multimodal-textbox {
    background-color: #000000; /* White background for input */
    color: #000000; /* Black text */
    border: 1px solid #F60; /* Orange border */
    border-radius: 5px;
    padding: 10px;
}

/* Placeholder text */
.multimodal-textbox::placeholder {
    color: #666666; /* Gray placeholder */
}

/* Button styling */
button {
    background-color: #F60; /* Geek Squad Orange */
    color: #FFFFFF; /* White text */
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-weight: bold;
    transition: background-color 0.3s;
}

button:hover {
    background-color: #CC5500; /* Darker orange on hover */
}

/* Clear button specific styling */
#clear-btn {
    background-color: #666666; /* Gray for clear button */
}

#clear-btn:hover {
    background-color: #555555; /* Darker gray on hover */
}

/* Markdown header styling */
h1 {
    color: #F60; /* Orange headers */
    text-align: center;
    margin-bottom: 20px;
    font-size: 3rem;
}
p {
    color: #F60; /* Orange headers */
    text-align: center;
    margin-bottom: 20px;
    font-size: 1rem;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css) as demo:
    with gr.Column(variant="compact"):
        gr.Markdown("# Agent Qwen")
        gr.Markdown(
            "You can interact with Agent Qwen using text, images, audio, or video files. Provide as much "
            "detail about your device issue as possible. Note: Audio responses may not always be available."
        )
        chatbot = gr.Chatbot(label="Agent Qwen", height="70vh", autoscroll=True, type="messages")
        with gr.Row():
            user_input = gr.MultimodalTextbox(
                show_label=False,
                placeholder="Enter your message here...",
                sources=["upload", "microphone"]
            )
        audio_output = gr.Audio(label="Agent Qwen's Response", autoplay=True)
        clear_btn = gr.Button("Clear")


    # Event handlers
    def handle_user_input(multimodal_input, chat_history):
        try:
            # Process MultimodalTextbox input
            image, audio, video, text = process_multimodal_textbox(multimodal_input)

            # Call backend to process input
            new_chat_history, text_response, audio_path = process_input(
                image, audio, video, text, chat_history
            )

            # Return updated components
            return new_chat_history, {"text": "", "files": []}, audio_path
        except Exception as e:
            # Display error in chatbot with correct message format
            error_message = f"Error processing input: {str(e)}"
            if not isinstance(chat_history, list):
                chat_history = []
            new_chat_history = chat_history + [{"role": "assistant", "content": error_message}]
            return new_chat_history, {"text": "", "files": []}, None


    user_input.submit(
        fn=handle_user_input,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input, audio_output],
        queue=True
    )


    def clear_chat():
        clear_audio_files()
        return [], {"text": "", "files": []}, None


    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot, user_input, audio_output],
        queue=False
    )

demo.launch(inbrowser=True)