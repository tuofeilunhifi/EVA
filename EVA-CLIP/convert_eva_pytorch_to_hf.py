# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Part of the code was taken from:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clap/convert_clap_original_pytorch_to_hf.py

import argparse

import torch
from PIL import Image
from transformers import AutoModel, AutoConfig
from transformers import  CLIPImageProcessor, pipeline, CLIPTokenizer
from torch_to_hf.configuration_evaclip import EvaCLIPConfig, EvaCLIPVisionConfig
from torch_to_hf.modeling_eva import EvaCLIPModel, EvaCLIPVisionModel, EvaCLIPVisionModelWithProjection


KEYS_TO_MODIFY_MAPPING = {
    "cls_token":"embeddings.class_embedding",
    "pos_embed":"embeddings.position_embedding.weight",
    "patch_embed.proj":"embeddings.patch_embedding",
    ".positional_embedding":".embeddings.position_embedding.weight",
    ".token_embedding":".embeddings.token_embedding",
    "text.text_projection":"text_projection.weight",
    "mlp.c_fc":"mlp.fc1",
    "mlp.c_proj":"mlp.fc2",
    ".proj.":".out_proj.",
    # "q_bias":"q_proj.bias",
    # "v_bias":"v_proj.bias",
    "out.":"out_proj.",
    "norm1":"layer_norm1",
    "norm2":"layer_norm2",
    "ln_1":"layer_norm1",
    "ln_2":"layer_norm2",
    "attn":"self_attn",
    "norm.":"post_layernorm.",
    "ln_final":"final_layer_norm",
    "visual.blocks":"vision_model.encoder.layers",
    "text.transformer.resblocks":"text_model.encoder.layers",
    "visual.head":"visual_projection",
    "visual.":"vision_model.",
    "text.":"text_model.",

}

def rename_state_dict(state_dict):
    model_state_dict = {}

    for key, value in state_dict.items():
        # check if any key needs to be modified
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        if "text_projection" in key:
            model_state_dict[key] = value.T
        elif "attn.qkv" in key:
            # split qkv into query key and value
            mixed_qkv = value
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]

            model_state_dict[key.replace("qkv", "q_proj")] = query_layer
            model_state_dict[key.replace("qkv", "k_proj")] = key_layer
            model_state_dict[key.replace("qkv", "v_proj")] = value_layer

        elif "attn.in_proj" in key:
            # split qkv into query key and value
            mixed_qkv = value
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]

            model_state_dict[key.replace("in_proj_", "q_proj.")] = query_layer
            model_state_dict[key.replace("in_proj_", "k_proj.")] = key_layer
            model_state_dict[key.replace("in_proj_", "v_proj.")] = value_layer

        elif "class_embedding" in key:
            model_state_dict[key] = value[0,0,:]
        elif "vision_model.embeddings.position_embedding" in key:
            model_state_dict[key] = value[0,:,:]

        else:
            model_state_dict[key] = value

    return model_state_dict

# This requires having a clone of https://github.com/baaivision/EVA/tree/master/EVA-CLIP as well as the right conda env
# Part of the code is copied from https://github.com/baaivision/EVA/blob/master/EVA-CLIP/README.md "Usage" section
def getevaclip(checkpoint_path, input_pixels, captions):
    from eva_clip import create_model_and_transforms, get_tokenizer
    model_name = "EVA02-CLIP-bigE-14-plus"
    model, _, _ = create_model_and_transforms(model_name, checkpoint_path, force_custom_clip=True)
    tokenizer = get_tokenizer(model_name)
    text = tokenizer(captions)

    with torch.no_grad():
        text_features = model.encode_text(text)
        image_features = model.encode_image(input_pixels)
        image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_normed = text_features / text_features.norm(dim=-1, keepdim=True)

        label_probs = (100.0 * image_features_normed @ text_features_normed.T).softmax(dim=-1)

    return  label_probs

def save_model_and_config(pytorch_dump_folder_path, hf_model, transformers_config):
    hf_model.save_pretrained(pytorch_dump_folder_path)
    transformers_config.save_pretrained(pytorch_dump_folder_path)

def check_loaded_model(pytorch_dump_folder_path, tokenizer, processor, image, captions):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_config = AutoConfig.from_pretrained(pytorch_dump_folder_path, trust_remote_code=True)
    hf_model = AutoModel.from_pretrained(pytorch_dump_folder_path, config=hf_config, trust_remote_code=True).to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = hf_model(**inputs)
    print(hf_config, hf_model)
    print(outputs)
    # detector = pipeline(model=hf_model, task="zero-shot-image-classification", tokenizer = tokenizer, image_processor=processor)
    # detector_probs = detector(image, candidate_labels=captions)
    # print(f"text_probs loaded hf_model using pipeline: {detector_probs}")

def convert_evaclip_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path, image_path, save=False, resize=224):
    if resize == 224:
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    else:
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    image = Image.open(image_path)
    captions = ["a diagram", "a dog", "a cat"]
    tokenizer = CLIPTokenizer.from_pretrained(pytorch_dump_folder_path)
    # input_ids = tokenizer(captions,  return_tensors="pt", padding=True).input_ids
    # input_pixels = processor( images=image, return_tensors="pt", padding=True).pixel_values

    # This requires having a clone of https://github.com/baaivision/EVA/tree/master/EVA-CLIP as well as the right conda env
    # original_evaclip_probs = getevaclip(checkpoint_path, input_pixels, captions)
    # print(f"original_evaclip label probs: {original_evaclip_probs}")

    # transformers_config = EvaCLIPConfig.from_pretrained(config_path)
    # hf_model = EvaCLIPModel(transformers_config)
    transformers_config = EvaCLIPVisionConfig.from_pretrained(config_path)
    hf_model = EvaCLIPVisionModel(transformers_config)
    # hf_model = EvaCLIPVisionModelWithProjection(transformers_config)
    pt_model_state_dict = torch.load(checkpoint_path)
    state_dict = rename_state_dict(pt_model_state_dict)

    # print(hf_model, state_dict.keys())
    hf_model.load_state_dict(state_dict, strict=False)
    
    
    # inputs = processor(images=image, return_tensors="pt")
    # outputs = hf_model(**inputs)
    # print(outputs)

    # with torch.no_grad():
    #     image_features = hf_model.get_image_features(input_pixels)
    #     text_features = hf_model.get_text_features(input_ids)
    #     image_features /= image_features.norm(dim=-1, keepdim=True)
    #     text_features /= text_features.norm(dim=-1, keepdim=True)

    # label_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # print(f"hf_model label probs: {label_probs}")

    if save:
        save_model_and_config(pytorch_dump_folder_path, hf_model, transformers_config)
    
    check_loaded_model(pytorch_dump_folder_path, tokenizer, processor, image, captions)

    # hf_model.push_to_hub("ORGANIZATION_NAME/EVA02_CLIP_E_psz14_plus_s9B")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default="EVA-CLIP/EVA02-CLIP-bigE-14-plus_s9B" ,type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default="EVA02_CLIP_E_psz14_plus_s9B.pt", type=str, help="Path to fairseq checkpoint" )
    parser.add_argument("--config_path", default='EVA-CLIP/EVA02-CLIP-bigE-14-plus_s9B', type=str, help="Path to hf config.json of model to convert")
    parser.add_argument("--image_path", default='EVA-CLIP/EVA02-CLIP-bigE-14-plus_s9B/CLIP.png', type=str, help="Path to image")
    parser.add_argument("--save", default=False, type=str, help="Path to image")
    parser.add_argument("--resize", default=224, type=int, help="Image resize")

    args = parser.parse_args()

    convert_evaclip_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.image_path, args.save, args.resize)

    # python convert_eva_pytorch_to_hf.py --pytorch_dump_folder_path /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/torch_to_hf/ --checkpoint_path /mnt/pfs-guan-ssai/cv/cjy/models/mindvit/2024_03_19/eva_clip_l_336_e4.bin --config_path /mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/torch_to_hf/ --image_path ./CLIP.png --save True --resize 336