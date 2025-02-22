import time
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import torch

# Medición de tiempo total
start_total = time.perf_counter()

# Medición y carga del processor
start_processor = time.perf_counter()
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
end_processor = time.perf_counter()
print(f"Tiempo de carga del processor: {end_processor - start_processor:.2f} segundos")

# Medición y carga del modelo
start_model = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
end_model = time.perf_counter()
print(f"Tiempo de carga del modelo: {end_model - start_model:.2f} segundos")

# Medición de descarga y carga de la imagen
start_image = time.perf_counter()
image_url = "https://picsum.photos/id/237/536/354"
image = Image.open(requests.get(image_url, stream=True).raw)
end_image = time.perf_counter()
print(f"Tiempo de descarga y carga de imagen: {end_image - start_image:.2f} segundos")

# Medición del preprocesamiento (imagen y texto)
start_preprocess = time.perf_counter()
inputs = processor.process(
    images=[image],
    text="Describe this image."
)
# Mover inputs al dispositivo del modelo y crear batch de tamaño 1
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
end_preprocess = time.perf_counter()
print(f"Tiempo de preprocesamiento: {end_preprocess - start_preprocess:.2f} segundos")

# Medición de la generación de salida
start_generation = time.perf_counter()
with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
  output = model.generate_from_batch(
      inputs,
      GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
      tokenizer=processor.tokenizer
  )

end_generation = time.perf_counter()
print(f"Tiempo de generación: {end_generation - start_generation:.2f} segundos")

# Medición de decodificación
start_decode = time.perf_counter()
generated_tokens = output[0, inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
end_decode = time.perf_counter()
print(f"Tiempo de decodificación: {end_decode - start_decode:.2f} segundos")

end_total = time.perf_counter()
print(f"Tiempo total de ejecución: {end_total - start_total:.2f} segundos")

# Mostrar resultado generado
print("\n--- Resultado ---")
print(generated_text)
