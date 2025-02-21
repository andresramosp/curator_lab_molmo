from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from transformers import pipeline
from PIL import Image
import io

app = FastAPI()

# Crea el pipeline de forma global para evitar recargas en cada petición
pipe = pipeline("image-text-to-text", model="allenai/Molmo-7B-D-0924", trust_remote_code=True)

def run_molmo(prompt: str, image_bytes: bytes):
    """
    Función que procesa la imagen (en bytes) y el prompt usando el modelo Molmo,
    y devuelve el resultado de la generación.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Prepara la entrada como una lista de mensajes (incluyendo la imagen)
        messages = [{"role": "user", "content": prompt, "image": image}]
        result = pipe(messages)
        return result
    except Exception as e:
        raise e

@app.post("/pipeline-image-to-text")
async def image_to_text(prompt: str = Form("Describe this image."), file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = run_molmo(prompt, contents)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
