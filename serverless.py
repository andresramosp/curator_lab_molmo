import base64
import runpod
from main import image_to_text  # Importamos la función común

async def handler(job):
    input_data = job.get("input", {})
    prompt = input_data.get("prompt", "Describe this image.")
    
    image_b64 = input_data.get("image")
    image_file = input_data.get("image_file")
    
    if image_b64:
        image_bytes = base64.b64decode(image_b64)
    elif image_file:
        image_bytes = image_file.read()
    else:
        return {"error": "Missing 'image' or 'image_file' in input"}
    
    try:
        result = image_to_text(image_bytes, prompt)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# Inicia el servidor RunPod Serverless
runpod.serverless.start({"handler": handler})
