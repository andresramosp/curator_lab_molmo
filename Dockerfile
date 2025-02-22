# Imagen base con soporte CUDA (PyTorch)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Definir el directorio de trabajo
WORKDIR /app

# Copiar el código de la API
COPY . /app

# Instalar las dependencias con logging
RUN echo "Instalando dependencias..." && \
    pip install --no-cache-dir -r requirements.txt | tee /app/install.log

# Configurar Hugging Face para almacenar modelos en una carpeta persistente
ENV HF_HOME="/app/model_cache"

# No descargar el modelo en el build para evitar timeout
# Se descargará en runtime dentro de `main.py`

# Definir el comando para ejecutar el handler en RunPod
CMD ["python", "main.py"]
