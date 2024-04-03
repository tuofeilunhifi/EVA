import torch
from eva_clip import create_model_and_transforms, get_tokenizer
from PIL import Image

# model_name = "EVA02-CLIP-B-16" 
# pretrained = "/mnt/pfs-guan-ssai/cv/cjy/models/models--QuanSun--EVA-CLIP/snapshots/11afd202f2ae80869d6cef18b1ec775e79bd8d12/EVA02_CLIP_B_psz16_s8B.pt"

model_name="EVA02-CLIP-L-14-InternVL-LLaMA-CN-7B"
pretrained=''
precision='amp'
device='cuda:0'
torchscript=False
force_quick_gelu=False
force_custom_clip=True
force_patch_dropout=None
pretrained_image="/mnt/pfs-guan-ssai/cv/cjy/models/mindvit/2024_04_01/eva_clip_l_e20.bin"
pretrained_text="/mnt/pfs-guan-ssai/cv/cjy/models/internvl_c_13b_224px.pth"
pretrained_visual_model="EVA02-CLIP-L-14"
pretrained_text_model="other"
image_mean=None
image_std=None
cache_dir=None
skip_list=[]

image_path = "CLIP.png"
caption = ["a diagram", "a dog", "a cat"]

device = "cuda" if torch.cuda.is_available() else "cpu"
# model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
model, _, preprocess = create_model_and_transforms(
    model_name,
    pretrained,
    precision=precision,
    device=device,
    jit=torchscript,
    force_quick_gelu=force_quick_gelu,
    force_custom_clip=force_custom_clip,
    force_patch_dropout=force_patch_dropout,
    pretrained_image=pretrained_image,
    pretrained_text=pretrained_text,
    pretrained_visual_model=pretrained_visual_model,
    pretrained_text_model=pretrained_text_model,
    image_mean=image_mean,
    image_std=image_std,
    cache_dir=cache_dir,
    skip_list=skip_list,
)
tokenizer = get_tokenizer(model_name)
model = model.to(device)

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = tokenizer(caption).to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[0.8275, 0.1372, 0.0352]]