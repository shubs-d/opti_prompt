import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi",
    "uvicorn",
    "transformers",
    "torch",
    "tiktoken"
).run_commands(
    'python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained(\'distilgpt2\'); AutoModelForCausalLM.from_pretrained(\'distilgpt2\')"'
).add_local_dir(".", remote_path="/root/opti_prompt", ignore=[".git", "__pycache__"])

app = modal.App("optiprompt-backend")

@app.function(
    image=image
)
@modal.asgi_app()
def serve():
    import sys
    import os
    sys.path.append("/root/opti_prompt")
    from app.main import app as fastapi_app
    return fastapi_app
