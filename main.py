from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import requests

app = FastAPI()

MODEL_REPO = "allenai/Molmo-7B-D-0924"
MODEL_DIR = "/workspace/molmo_model/models--allenai--Molmo-7B-D-0924/snapshots/1721478b71306fb7dc671176d5c204dc7a4d27d7"

print("Cargando modelo en GPU...")
processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
print("Modelo cargado.")
torch.cuda.empty_cache()

@app.post("/generate")
async def generate_text(image: UploadFile = File(...), prompt: str = "Describe this image."):
    # Se puede usar la imagen subida o una imagen de ejemplo
    # Aquí se usa una imagen de ejemplo para simplificar:
    inputs = processor.process(
        images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
        text=prompt
    )
    # Convertir los inputs a batch y moverlos al mismo dispositivo que el modelo
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    
    # Convertir el modelo y la imagen a bfloat16 para reducir el consumo de memoria
    model.to(dtype=torch.bfloat16)
    if "images" in inputs:
        inputs["images"] = inputs["images"].to(torch.bfloat16)
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )
    
    # Decodificar únicamente los tokens generados
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return {"description": generated_text}
