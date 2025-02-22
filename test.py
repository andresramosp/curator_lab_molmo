import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

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



img = Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)
print("Tamaño de la imagen:", img.size)
print("Modo de la imagen:", img.mode)

inputs = processor.process(
    images=[img],
    text="Describe this image."
)


# move inputs to the correct device and make a batch of size 1
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

# generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
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

# >>>  This image features an adorable black Labrador puppy, captured from a top-down
#      perspective. The puppy is sitting on a wooden deck, which is composed ...
