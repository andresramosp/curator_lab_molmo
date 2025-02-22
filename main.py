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
processor = AutoProcessor.from_pretrained(MODEL_REPO, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("Modelo cargado.")

@app.post("/generate")
async def generate_text(image: UploadFile = File(...), prompt: str = "Describe this image."):
    # Leer la imagen
    # image_bytes = await image.read()
    # image = Image.open(BytesIO(image_bytes))

    # # Preprocesar la imagen
    # inputs = processor.process(images=[image], text=prompt)

    # # Verificar si inputs contiene tensores
    # print("Inputs antes de conversión:", {k: type(v) for k, v in inputs.items()})

    # # Convertir listas en tensores y moverlos a la GPU
    # inputs = {k: torch.tensor(v).to(model.device).unsqueeze(0) if isinstance(v, list) else v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # print("Inputs después de conversión:", {k: v.shape for k, v in inputs.items()})

    inputs = processor.process(
    images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
    text="Describe this image."
)

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generación de texto
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=500, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

    # Decodificar la salida
    generated_text = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return {"description": generated_text}
