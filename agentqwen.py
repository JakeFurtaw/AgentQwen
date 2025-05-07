import gradio as gr
from model_utils import preprocess_image, preprocess_audio, call_multimodal_model

def process_inputs(user_input, chat_history):
    """Process user inputs and call the multimodal model."""
    text = user_input.get("text", "")
    image = user_input.get("image", None)
    audio = user_input.get("audio", None)

    # Preprocess image and audio if provided
    processed_image = preprocess_image(image) if image else None
    processed_audio = preprocess_audio(audio) if audio else None

    # Call the multimodal model
    response = call_multimodal_model(text, processed_image, processed_audio)

    # Update chat history
    chat_history.append((text, response))
    return chat_history, None, None

custom_css = """
/* General container styling */
.gradio-container {
    background-color: #000000; /* Geek Squad Black */
    color: #FFFFFF; /* White text */
    font-family: 'Arial', sans-serif;
}

/* Chatbot styling */
.gr-chatbot {
    background-color: #1C2526; /* Dark gray for contrast */
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
        gr.Markdown("You can interact with Agent Qwen using text, images, and audio files. Make sure to provide as much "
                    "detail about your issue.")
        chatbot = gr.Chatbot(label="Agent Qwen", height="80vh", autoscroll=True, type='messages')
        with gr.Row():
            user_input = gr.MultimodalTextbox(show_label= False, placeholder="Enter your message here...",
                                              sources=['upload', 'microphone'])


    # Clear button
    clear_btn = gr.Button("Clear")

    # State to maintain chat history
    chat_history = gr.State([])

    # Event handlers
    user_input.submit(
        fn=process_inputs,
        inputs=[user_input, chat_history],
        outputs=[chatbot, user_input]
    )

    clear_btn.click(
        fn=lambda: ([], None, []),
        inputs=None,
        outputs=[chatbot, user_input, chat_history]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True)