
---

# app.py (Gradio app — copy paste)
This app uses a Hugging Face `text-generation` pipeline with small GPT-2 by default. It supports temperature, max_length, and top_p/top_k sampling. This runs locally without API keys. If you want to use OpenAI or a larger HF model, replace the pipeline/model name.

```python
# app.py
import gradio as gr
from transformers import pipeline, set_seed

# choose model: "gpt2" is small & works without large disk/ram
MODEL_NAME = "gpt2"

def get_generator(model_name=MODEL_NAME):
    try:
        gen = pipeline("text-generation", model=model_name, device=-1)  # CPU by default; set device=0 for GPU
        return gen
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")

generator = get_generator()

def generate_story(prompt: str, max_length: int = 200, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95, seed: int = None):
    if not prompt or prompt.strip() == "":
        return "Please provide a story starter prompt."
    try:
        if seed is not None:
            set_seed(int(seed))
        outputs = generator(prompt, max_length=int(max_length), do_sample=True, temperature=float(temperature),
                            top_k=int(top_k), top_p=float(top_p), num_return_sequences=1)
        text = outputs[0]["generated_text"]
        return text
    except Exception as e:
        return f"Generation failed: {e}"

with gr.Blocks(title="Interactive AI StoryTeller") as demo:
    gr.Markdown("# Interactive AI StoryTeller\nEnter a story starter and tweak sampling settings for creativity.")
    with gr.Row():
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(label="Story starter / prompt", placeholder="e.g. 'On a rainy night, Mira discovered...'",
                                      lines=6)
            generate_btn = gr.Button("Generate")
            output = gr.Textbox(label="Generated story", lines=12)
        with gr.Column(scale=1):
            max_length = gr.Slider(50, 1000, value=250, step=10, label="Max length (tokens)")
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
            top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k")
            top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p (nucleus)")
            seed = gr.Number(value=42, label="Seed (optional)")

    generate_btn.click(fn=generate_story, inputs=[prompt_input, max_length, temperature, top_k, top_p, seed], outputs=output)

if __name__ == "__main__":
    demo.launch()

