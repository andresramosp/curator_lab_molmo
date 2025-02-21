import base64
import runpod
from main import run_molmo  # Importamos la función común

async def handler(job):
    input_data = job.get("input", {})
    # Se espera que input_data incluya "prompt" e "image" (en base64)
    prompt = input_data.get("prompt", "Describe this image.")
    image_b64 = input_data.get("image")
    if not image_b64:
        return {"error": "Missing 'image' in input"}
    
    try:
        image_bytes = base64.b64decode(image_b64)
        result = run_molmo(prompt, image_bytes)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# Inicia el servidor RunPod Serverless
runpod.serverless.start({"handler": handler})
