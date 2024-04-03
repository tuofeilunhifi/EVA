import torch
# from eva_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
from transformers import  CLIPImageProcessor
from transformers import AutoModel, AutoConfig

from torch_to_hf.rope import resample_abs_pos_embed

# model_name = "EVA02-CLIP-L-14-336-InternVL-LLaMA-CN-7B" 
# pretrained = "/mnt/pfs-guan-ssai/cv/cjy/models/mindvit/2024_03_06/eva_clip_l_e10.bin" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"

image_path = "torch_to_hf/CLIP.png"
# caption = ["a diagram", "a dog", "a cat"]

device = "cuda" if torch.cuda.is_available() else "cpu"

image = Image.open(image_path)
# processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
inputs = processor(images=image, return_tensors="pt").to(device)

# model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
# model = model.to(device)
# image_features = model.encode_image(inputs['pixel_values'])
# print(image_features)


pytorch_dump_folder_path = "/mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/torch_to_hf/"
hf_config = AutoConfig.from_pretrained(pytorch_dump_folder_path, trust_remote_code=True)
hf_model = AutoModel.from_pretrained(pytorch_dump_folder_path, config=hf_config, trust_remote_code=True, ignore_mismatched_sizes=True).to(device)

# position_embedding_weight = resample_abs_pos_embed(
#     hf_model.vision_model.embeddings.position_embedding.weight.unsqueeze(0),
#     (hf_model.vision_model.embeddings.dynamic_image_size // hf_model.vision_model.embeddings.patch_size, hf_model.vision_model.embeddings.dynamic_image_size // hf_model.vision_model.embeddings.patch_size),
#     num_prefix_tokens=hf_model.vision_model.embeddings.num_prefix_tokens,
# ).squeeze(0)

# num_patches = (hf_model.vision_model.embeddings.dynamic_image_size // hf_model.vision_model.embeddings.patch_size) ** 2
# num_prefix_tokens = 1
# num_positions = num_patches + 1
# position_embedding = torch.nn.Embedding(num_positions, hf_model.vision_model.embeddings.embed_dim).to(device)
# hf_model.vision_model.embeddings.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)).to(device), persistent = False)
# position_embedding.weight = torch.nn.Parameter(position_embedding_weight)
# hf_model.vision_model.embeddings.position_embedding = position_embedding

torch_outputs = hf_model(**inputs)
print(torch_outputs)

# onnx_model_path = "/mnt/pfs-guan-ssai/cv/cjy/models/mindvit/2024_03_19/eva_clip_l_336_e4.onnx"

# torch.onnx.export(hf_model,  # model being run
#                   (inputs.pixel_values),  # model input (or a tuple for multiple inputs)
#                   onnx_model_path,   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=15,          # the ONNX version to export the model to
#                   do_constant_folding=False,  # whether to execute constant folding for optimization
#                   input_names=['pixel_values'],   # the model's input names
#                   # output_names=['output'],  # the model's output names
#                   # dynamic_axes={'pixel_values': {0: 'batch', 2: 'hight', 3: 'width'}},
#                   )


# import onnxruntime
# import onnx

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# ## onnx测试
# model_session = onnxruntime.InferenceSession(onnx_model_path)
# #compute ONNX Runtime output prediction
# inputs = {model_session.get_inputs()[0].name: to_numpy(inputs.pixel_values)}
# onnx_outputs = model_session.run(None, inputs)[0]
# onnx_outputs = torch.from_numpy(onnx_outputs).to(device)

# print("onnx weights", onnx_outputs)