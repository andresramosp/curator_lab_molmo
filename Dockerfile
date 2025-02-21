# Usamos una imagen base de Python (puedes elegir una con soporte CUDA si lo requieres)
FROM python:3.10

# Definir el directorio de trabajo
WORKDIR /app

# Copiar el código de la API (incluyendo main.py, serverless.py, requirements.txt, etc.)
COPY . /app

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar el modelo Molmo para evitar descargas en cada arranque.
# Esto invoca el pipeline, lo que hará que se descarguen todos los shards del modelo.
RUN python -c "from transformers import pipeline; pipeline('image-text-to-text', model='allenai/Molmo-7B-D-0924', trust_remote_code=True)"

# Definir el comando para ejecutar el handler en RunPod
CMD ["python", "serverless.py"]
