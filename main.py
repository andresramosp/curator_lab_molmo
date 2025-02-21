from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
from io import BytesIO
import requests
import os

# Iniciar la API
app = FastAPI()

# Cargar el modelo en RAM una sola vez al iniciar el pod
MODEL_REPO = "allenai/Molmo-7B-D-0924"
CACHE_DIR = "/workspace"

print("Cargando modelo en GPU...")
processor = AutoProcessor.from_pretrained(
    MODEL_REPO,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
print("Modelo cargado.")

torch.cuda.empty_cache()
from io import BytesIO

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print("Tokenizer vocab size:", processor.tokenizer.vocab_size)


@app.post("/generate")
async def generate_text(image: UploadFile = File(...), prompt: str = "Describe this image."):
    # Leer imagen del UploadFile
    # image_bytes = await image.read()
    # pil_image = Image.open(BytesIO(image_bytes))

    # # Procesar la imagen y el prompt
    # inputs = processor.process(
    #     images=[pil_image],
    #     text=prompt
    # )

    # # Mover inputs al dispositivo correcto y crear un batch de tamaño 1
    # inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # for key, value in inputs.items():
    #     print(f"{key} shape: {value.shape}")
    #     print(f"{key} min: {value.min().item()}, max: {value.max().item()}")

    # # Generar la salida
    # with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
    #     output = model.generate_from_batch(
    #         inputs,
    #         GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    #         tokenizer=processor.tokenizer
    #     )

    # # Decodificar los tokens generados
    # generated_tokens = output[0, inputs['input_ids'].size(1):]
    # generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    inputs = processor.process(
        images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
        text="Describe this image."
    )

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    model.to(dtype=torch.bfloat16)
    inputs["images"] = inputs["images"].to(torch.bfloat16)
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # print the generated text
    print(generated_text)
    return {"description": generated_text}

