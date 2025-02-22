# Imagen base con soporte CUDA (ajustar según necesidad)
FROM python:3.10

# Definir el directorio de trabajo
WORKDIR /app

# Copiar el código de la API
COPY . /app

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Configurar Hugging Face para almacenar modelos en una carpeta persistente dentro del contenedor
ENV HF_HOME="/app/model_cache"

# Pre-descargar el modelo Molmo para evitar descargas en cada arranque
RUN python -c "\
    from transformers import AutoModelForCausalLM, AutoProcessor;\
    model_id='allenai/Molmo-7B-D-0924';\
    AutoProcessor.from_pretrained(model_id, trust_remote_code=True);\
    AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto')"

# Definir el comando para ejecutar el handler en RunPod
CMD ["python", "main.py"]
