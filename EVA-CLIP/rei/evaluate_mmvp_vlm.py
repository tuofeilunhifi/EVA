import csv
import os
from PIL import Image
import torch
from clip import load
import clip
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import csv

from eva_clip import create_model_and_transforms, get_tokenizer

# python evaluate_mmvp_vlm.py --directory /mnt/pfs-guan-ssai/cv/cjy/data/MMVP_VLM

def benchmark_model(model_name, benchmark_dir, device = "cpu"):
    # model, preprocess = load(model_name, device=device)
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

    tokenizer = get_tokenizer(model_name)
    
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

    image_dir = os.path.join(benchmark_dir, 'MLLM_VLM Images')
    csv_file = os.path.join(benchmark_dir, 'Questions.csv')
    

    csv_outfile = open('output.csv', 'w', newline='')
    csv_writer = csv.writer(csv_outfile)
    csv_writer.writerow(['qid1', 'qid2', 'pred1', 'pred2', 'gt1', 'gt2', 'q1score', 'q2score'])  # header

    categories = [
        'Orientation and Direction', 'Presence of Specific Features', 
        'State and Condition', 'Quantity and Count', 
        'Positional and Relational Context', 'Color and Appearance',
        'Structural Characteristics', 'Texts',
        'Viewpoint and Perspective'
    ]

    pair_accuracies = {category: 0 for category in categories}
    num_pairs = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            qid1, qtype1, statement1 = row
        
            # Get next row for the pair
            row = next(reader, None)
            if not row:
                break
            qid2, qtype2, statement2 = row
            
            qid1, qid2 = int(qid1), int(qid2)
            
            img1 = Image.open(os.path.join(image_dir, qtype1, f'{qid1}.jpg'))
            img2 = Image.open(os.path.join(image_dir, qtype1, f'{qid2}.jpg'))

            text1 = 'a photo of ' + statement1
            text2 = 'a photo of ' + statement2

            # text1 = clip.tokenize([text1]).to(device)
            # text2 = clip.tokenize([text2]).to(device)
            text1 = tokenizer(text1).to(device)
            text2 = tokenizer(text2).to(device)
            
            img1 = preprocess(img1).unsqueeze(0).to(device)
            img2 = preprocess(img2).unsqueeze(0).to(device)
            imgs = torch.cat((img1, img2), dim=0)


            with torch.no_grad():
                # logits_per_image1, logits_per_text1= model(imgs, text1)
                # logits_per_image2, logits_per_text2 = model(imgs, text2)
                # probs1 = logits_per_text1.softmax(dim=-1).cpu().numpy()
                # probs2 = logits_per_text2.softmax(dim=-1).cpu().numpy()

                # image_features = model.encode_image(imgs)
                # text_features1 = model.encode_text(text1)
                # text_features2 = model.encode_text(text2)
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                # text_features1 /= text_features1.norm(dim=-1, keepdim=True)
                # text_features2 /= text_features2.norm(dim=-1, keepdim=True)
                # probs1 = (100.0 * text_features1 @ image_features.T).softmax(dim=-1)
                # probs2 = (100.0 * text_features2 @ image_features.T).softmax(dim=-1)

                image_features1, text_features1, _ = model(imgs, text1)
                image_features2, text_features2, _ = model(imgs, text2)

                probs1 = (100.0 * text_features1 @ image_features1.T).softmax(dim=-1)
                probs2 = (100.0 * text_features2 @ image_features2.T).softmax(dim=-1)
                # print(probs1, probs2)

            img1_score1 = probs1[0][0]
            img1_score2 = probs2[0][0]
            
            pred1 = "img1" if img1_score1 > 0.5 else "img2"
            pred2 = "img1" if img1_score2 > 0.5 else "img2"

            gt1 = "img1" if qid1 % 2 == 1 else "img2"
            gt2 = "img1" if qid2 % 2 == 1 else "img2"

            
            csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])
                
            current_category = categories[num_pairs // 15]
            if pred1 == gt1 and pred2 == gt2:
                pair_accuracies[current_category] += 1
            num_pairs += 1

        csv_outfile.close()

    # Calculate percentage accuracies
    for category in pair_accuracies:
        pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100

    pair_accuracies["avg"] = sum(pair_accuracies.values()) / len(categories)
    return pair_accuracies


parser = argparse.ArgumentParser(description='Process a directory path.')
    
# Adding an argument for the directory path
parser.add_argument('--directory', type=str, help='The path to the directory')

# Parsing the arguments
args = parser.parse_args()

# OpenAI models
models = ['ViT-L/14']

results_openai = {f'openai-{model}': benchmark_model(model, args.directory) for model in models}


# Merge results
results = {**results_openai}

# Convert results to format suitable for star plot
categories = results[list(results.keys())[0]].keys()
data = {'Categories': list(categories)}
for model in list(results_openai.keys()):
    data[model] = [results[model][category] for category in categories]

print(results)
