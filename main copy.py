
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
from io import BytesIO
import os

def image_to_text(image_bytes: bytes, prompt: str = "Describe this image.") -> str:
    """
    Procesa una imagen en bytes y genera una descripción usando un modelo de lenguaje.
    :param image_bytes: Imagen en formato de bytes
    :param prompt: Texto de entrada para el modelo
    :return: Texto generado como descripción de la imagen
    """
    MODEL_REPO = "allenai/Molmo-7B-D-0924"

    # Configurar Hugging Face para usar una caché en RAM
    os.environ["HF_HOME"] = "/dev/shm/huggingface_cache"

    processor = AutoProcessor.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Cargar imagen desde bytes
    image = Image.open(BytesIO(image_bytes))
    
    # Preprocesamiento
    inputs = processor.process(images=[image], text=prompt)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    
    # Generación de salida
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )
    
    # Decodificación
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text