import json

import gradio as gr
import requests


def query_ollama(prompt, history):
    try:

        # Initialize response text
        response_text = ""
        buffer = ""
        chunk_size = 0

        # Make streaming request to Ollama
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'bose-slm',
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


if __name__ == "__main__":
    iface = gr.ChatInterface(
        fn=query_ollama,
    )

    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )