import json

import gradio as gr
import requests


def query_ollama(prompt):
    try:
        # Check cache first
        cached_response = None
        if cached_response is not None:
            print("Using cached response")
            yield cached_response
            return

        # Initialize response text
        response_text = ""
        buffer = ""
        chunk_size = 0

        # Make streaming request to Ollama
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'my-phi3-finetuned',
                'prompt': prompt,
                'stream': True,
                'context_size': 2048,
                'num_predict': 1000,
            },
            stream=True
        )

        # Process the stream
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if 'response' in json_response:
                    chunk = json_response['response']
                    buffer += chunk
                    chunk_size += len(chunk)

                    # Only yield when buffer reaches certain size or on completion
                    if chunk_size >= 20 or json_response.get('done', False):
                        response_text += buffer
                        yield response_text
                        buffer = ""
                        chunk_size = 0

                if json_response.get('done', False):
                    # Yield any remaining buffer
                    if buffer:
                        response_text += buffer
                        yield response_text
                    # Save the complete response to cache
                    break

    except Exception as e:
        yield f"Error: {str(e)}"


def create_interface():
    iface = gr.Interface(
        fn=query_ollama,
        inputs=gr.Textbox(
            label="Prompt",
            placeholder="Ask your question here...",
            lines=3
        ),
        outputs=gr.Textbox(
            label="Response",
            lines=5,
        ),
        title="Bose AI",
        description="Chat about Bose products",

    )
    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.queue()
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        root_path="",
        show_error=True
    )