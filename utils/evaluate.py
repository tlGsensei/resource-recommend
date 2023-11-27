from operator import imod
import torch
from collections import defaultdict
from utils.dataset import *

def evaluate(model, data_loader, device, Ks=[20]):
    model.eval()
    num_samples = 0
    max_K = max(Ks)
    results = defaultdict(float)
    with torch.no_grad():
        for batch in data_loader:
            batch_g, labels = prepare_batch(batch, device)
            logits, cor, self_supervised_loss = model(batch_g)
            batch_size = logits.size(0)
            num_samples += batch_size
            topk = torch.topk(logits, k=max_K, sorted=True)[1]
            labels = labels.unsqueeze(-1)
            for K in Ks:
                hit_ranks = torch.where(topk[:, :K] == labels)[1] + 1
                hit_ranks = hit_ranks.float().cpu()
                results[f'HR@{K}'] += hit_ranks.numel()
                results[f'MRR@{K}'] += hit_ranks.reciprocal().sum().item()
                results[f'NDCG@{K}'] += torch.log2(1 + hit_ranks).reciprocal().sum().item()
    for metric in results:
        results[metric] /= num_samples
    return results