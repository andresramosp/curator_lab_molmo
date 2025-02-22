from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
from io import BytesIO
import requests

# Iniciar la API
app = FastAPI()

# Cargar el modelo en RAM una sola vez al iniciar el pod
MODEL_REPO = "allenai/Molmo-7B-D-0924"
print("Cargando modelo en GPU...")
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
print("Modelo cargado.")

@app.post("/generate")
async def generate_text(image: UploadFile = File(...), prompt: str = "Describe this image."):

    inputs = processor.process(
        images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
        text="Describe this image."
    )

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return {"description": generated_text}
