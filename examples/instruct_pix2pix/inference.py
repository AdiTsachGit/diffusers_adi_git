import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, PaintByExamplePipeline
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/adi.tsach/diffusers_adi_git/src/diffusers/pipelines/stable_diffusion/')
import pipeline_stable_diffusion_instruct_pix2pix_image
from pipeline_stable_diffusion_instruct_pix2pix_image import StableDiffusionInstructPix2PixImagePipeline
from io import BytesIO
model_id = "instruct-pix2pix-model" # <- replace this
# pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, device_type="cuda", torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionInstructPix2PixImagePipeline.from_pretrained(model_id,device_type="cuda", torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

url= "https://www.waco-texas.com/files/sharedassets/public/v/1/departments/parks-amp-recreation/images/parks-pictures/cameron-park/cameron-park.jpg"
url2 = "https://cdn.britannica.com/79/232779-050-6B0411D7/German-Shepherd-dog-Alsatian.jpg"
# url2 = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/image/example_1.png"
# url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/reference/example_1.jpg"

def download_image(url):
   image = PIL.Image.open(requests.get(url, stream=True).raw)
   image = PIL.ImageOps.exif_transpose(image)
   image = image.convert("RGB")
   return image

def download_image_for_en(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

def preprocess_image(image, size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(image).unsqueeze(1)  # Add batch dimension

def get_image_encoding(url, encoder):
    image = download_image(url)
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        encoding = encoder(image_tensor)
    return encoding



image = download_image(url)
image2 = download_image(url)
width, height = image.size
black_image = Image.new("L", (width, height), 0)

# image2 =get_image_encoding(url2, pipe.image_encoder)
num_inference_steps = 20
image_guidance_scale = 5
guidance_scale = 10
# c = pipe.image_encoder(image2)
# c = pipe.proj_out(c)
edited_image = pipe( 
   prompt="",
   image=image,
   new_image =image2,
#    stength =0.75,
   num_inference_steps=num_inference_steps,
#    image_guidance_scale=image_guidance_scale,
#    guidance_scale=guidance_scale,
   generator=generator,
#    conditioning = c
).images[0]
print("worked")
edited_image.save("edited_image.png")