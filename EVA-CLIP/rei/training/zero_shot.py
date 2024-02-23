import logging

import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eva_clip import get_cast_dtype, get_tokenizer
from .precision import get_autocast


def zero_shot_classifier(model, classnames, templates, args):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(c=classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(args.device)  # tokenize
            if args.distributed:
                class_embeddings = model.module.encode_text(texts)
            else:
                class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                if args.distributed:
                    image_features = model.module.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5

def _load_classnames_and_classification_templates(dataset_name, current_folder, language):
    with open(os.path.join(current_folder, language + '_classnames.json'), 'r') as f:
        class_names = json.load(f)
    DEFAULT_CLASSNAMES = class_names['imagenet1k']
    classnames = class_names.get(dataset_name, DEFAULT_CLASSNAMES)

    # Zero-shot classification templates, collected from a bunch of sources
    # - CLIP paper (https://github.com/openai/CLIP/blob/main/data/prompts.md)
    # - Lit Paper (https://arxiv.org/pdf/2111.07991.pdf)
    # - SLIP paper (https://github.com/facebookresearch/SLIP/blob/main/templates.json)
    # Some are fixed mnaually

    with open(os.path.join(current_folder, language + '_zeroshot_classification_templates.json'), 'r') as f:
        zeroshot_classification_templates = json.load(f)
    # default template to use when the dataset name does not belong to `zeroshot_classification_templates`
    DEFAULT_ZEROSHOT_CLASSIFICATION_TEMPLATES = zeroshot_classification_templates['imagenet1k']

    templates = zeroshot_classification_templates.get(dataset_name, DEFAULT_ZEROSHOT_CLASSIFICATION_TEMPLATES)
    return classnames, templates


def zero_shot_eval(model, data, epoch, args):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting zero-shot imagenet.')

    logging.info('Building zero-shot classifier')

    dataset_name = "imagenet1k"
    current_folder = os.path.join(os.path.dirname(__file__), 'datasets')
    classnames, templates = _load_classnames_and_classification_templates(dataset_name, current_folder, args.language)
    classifier = zero_shot_classifier(model, classnames, templates, args)

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['epoch'] = int(epoch)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['epoch'] = int(epoch)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results
