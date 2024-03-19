import torch
from eva_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
from transformers import  CLIPImageProcessor
# from transformers import AutoModel, AutoConfig

# model_name = "EVA02-CLIP-L-14-InternVL-LLaMA-CN-7B" 
# pretrained = "/mnt/pfs-guan-ssai/cv/cjy/models/mindvit/2024_03_06/eva_clip_l_e10.bin" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"
model_name = "EVA02-CLIP-L-14-336-InternVL-LLaMA-CN-7B" 
pretrained = "/mnt/pfs-guan-ssai/cv/cjy/models/mindvit/2024_03_19/eva_clip_l_336_e4.bin"

image_path = "CLIP.png"
caption = ["a diagram", "a dog", "a cat"]

image = Image.open(image_path)
# processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
inputs = processor(images=image, return_tensors="pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
model = model.to(device)
image_features = model.encode_image(inputs['pixel_values'].to(device))
print(image_features)


# pytorch_dump_folder_path = "/mnt/pfs-guan-ssai/cv/cjy/models/models--QuanSun--EVA-CLIP/snapshots/11afd202f2ae80869d6cef18b1ec775e79bd8d12/EVA02-l-14"
# hf_config = AutoConfig.from_pretrained(pytorch_dump_folder_path, trust_remote_code=True)
# hf_model = AutoModel.from_pretrained(pytorch_dump_folder_path, config=hf_config, trust_remote_code=True)
# outputs = hf_model(**inputs)
# print(outputs)

# with torch.no_grad(), torch.cuda.amp.autocast():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)  # prints: [[0.8275, 0.1372, 0.0352]]