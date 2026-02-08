import torch
from diffusers import DiffusionPipeline
from PIL import Image

# 1. Load the pipeline
print("Loading pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    'ShinoharaHare/Waifu-Inpaint-XL',
    torch_dtype=torch.float16,
    use_safetensors=True,
)

# CRITICAL STEP: Move to GPU
# If you miss this, it runs on CPU and hangs at 0%
pipe.to("cuda") 

# 2. Create dummy inputs (Black image, White square mask)
# This lets you test the code without needing real files
print("Creating dummy inputs...")
init_image = Image.new("RGB", (1024, 1024), "black")
mask_image = Image.new("L", (1024, 1024), 0)
# Draw a white box in the middle of the mask (area to inpaint)
for x in range(400, 600):
    for y in range(400, 600):
        mask_image.putpixel((x, y), 255)

# 3. Run Inference
print("Running inference...")
prompt = "1girl, smiling, masterpiece, best quality"
negative_prompt = "low quality, worst quality"

# Using a generator ensures reproducibility
generator = torch.Generator("cuda").manual_seed(42)

image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt,
    image=init_image, 
    mask_image=mask_image, 
    num_inference_steps=20, # Reduced steps for a quick test
    guidance_scale=7.5,
    strength=1.0, # 1.0 = completely destroy masked area and redraw
    generator=generator
).images[0]

# 4. Save result
image.save("test_inpainting.png")
print("Done! Saved to test_inpainting.png")