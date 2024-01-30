import math
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from timm.loss import LabelSmoothingCrossEntropy

from einops import rearrange, repeat


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            # all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features, async_op=True), dim=0)
            # all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features, async_op=True), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            smoothing=0.,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.label_smoothing_cross_entropy = LabelSmoothingCrossEntropy(smoothing=smoothing) if smoothing > 0 else None

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale=1.):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        # print(logits_per_image.shape, logits_per_text.shape, labels.shape)
        # torch.Size([2048, 8192]) torch.Size([2048, 8192]) torch.Size([2048])
        
        if self.label_smoothing_cross_entropy:
            total_loss = (
                self.label_smoothing_cross_entropy(logits_per_image, labels) +
                self.label_smoothing_cross_entropy(logits_per_text, labels)
                ) / 2
        else:
            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
                ) / 2
            
        acc = None
        i2t_acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
        t2i_acc = (logits_per_text.argmax(-1) == labels).sum() / len(logits_per_text)
        acc = {"i2t": i2t_acc, "t2i": t2i_acc}
        return total_loss, acc

def masked_pairwise_contrastive_loss(a, b, mask, logit_scale):
    batch_size, seq_len, _ = a.shape
    # mask = repeat(mask, 'b n -> b n m', m=seq_len)
    # mask = rearrange(mask, 'b n m -> (b n) m')
    labels = repeat(torch.eye(a.shape[1]), 'n s -> b n s', b=batch_size)
    # labels = rearrange(labels, 'b n s -> (b n) s')
    # labels = torch.where(mask > 0, labels, 0)
    logits = torch.einsum('bmd,bnd->bmn', a, b) * logit_scale
    # logits = rearrange(logits, 'b n m -> (b n) m')
    # logits = torch.where(repeat(mask, 'b n -> b n m', m=seq_len) > 0, logits, float("-inf"))
    loss = F.cross_entropy(logits, labels.to(device=logits.device), reduction='none')
    # print("debug0", loss, loss.shape, mask.shape)
    loss = torch.sum(loss * mask) / torch.sum(mask)
    # print("debug1", loss, loss.shape, mask.shape)
    return loss

class SPARCLoss(nn.Module):

    def __init__(
            self,
            # local_loss=False,
            # gather_with_grad=False,
            # cache_labels=False,
            # rank=0,
            # world_size=1,
            # use_horovod=False,
            # smoothing=0.,
    ):
        super().__init__()
        # self.local_loss = local_loss
        # self.gather_with_grad = gather_with_grad
        # self.cache_labels = cache_labels
        # self.rank = rank
        # self.world_size = world_size
        # self.use_horovod = use_horovod
        # self.label_smoothing_cross_entropy = LabelSmoothingCrossEntropy(smoothing=smoothing) if smoothing > 0 else None

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        self.similarity_threshold = 1 / 14**2

    def forward(self, v_patch_embed, l_token_embed, attn_mask, logit_scale=1.):
        # similarity calculation
        similarity = torch.einsum('btd,bpd->btp', l_token_embed, v_patch_embed)

        # print("debug0", v_patch_embed.shape, l_token_embed.shape, logit_scale, similarity.shape)
        # torch.Size([2048, 196, 512]) torch.Size([2048, 77, 512]) tensor(14.2985, device='cuda:1', grad_fn=<ExpBackward0>) torch.Size([2048, 77, 196])

        # min-max normalization
        similarity = (similarity - torch.min(similarity, dim=-1, keepdim=True)[0]) / (torch.max(similarity, dim=-1, keepdim=True)[0] - torch.min(similarity, dim=-1, keepdim=True)[0])
        # print("debug1", similarity.shape)

        # thresholding
        similarity = torch.where(similarity < self.similarity_threshold, 0.0, similarity)
        # print("debug2", similarity.shape)

        # alignment-weighting
        v_align_weights = similarity / torch.sum(similarity, dim=-1, keepdim=True)
        l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd', v_align_weights, v_patch_embed)
        # print("debug3", v_align_weights.shape, l_grouped_v_patch_embed.shape)

        image_features = F.normalize(l_grouped_v_patch_embed, dim=-1)
        text_features = F.normalize(l_token_embed, dim=-1)
        # print("debug4", image_features.shape, text_features.shape)
        # debug1 torch.Size([2048, 77, 196])
        # debug2 torch.Size([2048, 77, 196])
        # debug3 torch.Size([2048, 77, 196]) torch.Size([2048, 77, 512])
        # debug4 torch.Size([2048, 77, 512]) torch.Size([2048, 77, 512])

        logits_per_image = masked_pairwise_contrastive_loss(image_features, text_features, attn_mask, logit_scale)
        # print("debug5", logits_per_image.shape)
        logits_per_text = masked_pairwise_contrastive_loss(text_features, image_features, attn_mask, logit_scale)
        # print("debug6", logits_per_text.shape)

        total_loss = (logits_per_image + logits_per_text) / 2
        
        return total_loss

        # if self.world_size > 1:
        #     all_image_features, all_text_features = gather_features(
        #         image_features, text_features,
        #         self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

        #     if self.local_loss:
        #         logits_per_image = logit_scale * image_features @ all_text_features.T
        #         logits_per_text = logit_scale * text_features @ all_image_features.T
        #     else:
        #         logits_per_image = logit_scale * all_image_features @ all_text_features.T
        #         logits_per_text = logits_per_image.T
        # else:
        #     logits_per_image = logit_scale * image_features @ text_features.T
        #     logits_per_text = logit_scale * text_features @ image_features.T


        # calculated ground-truth and cache if enabled
        # num_logits = logits_per_image.shape[0]
        # if self.prev_num_logits != num_logits or device not in self.labels:
        #     labels = torch.arange(num_logits, device=device, dtype=torch.long)
        #     if self.world_size > 1 and self.local_loss:
        #         labels = labels + num_logits * self.rank
        #     if self.cache_labels:
        #         self.labels[device] = labels
        #         self.prev_num_logits = num_logits
        # else:
        #     labels = self.labels[device]
        
        # if self.label_smoothing_cross_entropy:
        #     total_loss = (
        #         self.label_smoothing_cross_entropy(logits_per_image, labels) +
        #         self.label_smoothing_cross_entropy(logits_per_text, labels)
        #         ) / 2
        # else:
        #     total_loss = (
        #         F.cross_entropy(logits_per_image, labels) +
        #         F.cross_entropy(logits_per_text, labels)
        #         ) / 2
            
        # acc = None
        # i2t_acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
        # t2i_acc = (logits_per_text.argmax(-1) == labels).sum() / len(logits_per_text)
        # acc = {"i2t": i2t_acc, "t2i": t2i_acc}
        # return total_loss, acc