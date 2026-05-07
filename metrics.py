import torch

@torch.no_grad()
def compute_metrics_torch(y_true, y_score):
    
    metrics_dict = {}

    return metrics