import torch
from tqdm import tqdm
from train_utils import *
from loss_functions import *
from metrics import *
import numpy as np
import gc

def run_epoch(model, data_loader, criterion, device, args, epoch, optimizer=None, is_train=True, header=''):
    """
    Internal helper to run a single epoch pass (either training or validation).
    """

    model.train(is_train) # Sets mode to train or eval    

    if header=='':
        if is_train:
            header = f'train [{epoch}]'
        else:
            header = 'validation'
    else:
        header = header

    metric_logger = MetricLogger(delimiter="  ")

    if is_train:
        optimizer.zero_grad()
    
    # Context manager: Enable gradients only if training
    with torch.set_grad_enabled(is_train):
        
        #for batch_idx, batch in tqdm(enumerate(metric_logger.log_every(data_loader, args.print_freq, header)), total=len(data_loader)):
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):

            ### UNPACK BATCH AND MOVE TO DEVICE ###

            for ky in batch:
                if torch.is_tensor(batch[ky]):
                    batch[ky] = batch[ky].to(device)
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)
                loss_dict = criterion(output, batch)

                metrics = compute_metrics_torch(output, batch)
                
            
            # 4. Backward Pass (Only if training)
            if is_train:
                loss = loss_dict['loss'] / args.grad_accum_iter
                loss.backward()

                is_accum_step = ((batch_idx + 1) % args.grad_accum_iter == 0) or ((batch_idx + 1) == len(data_loader))

                if is_accum_step:
                    
                    if args.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # Optional to save memory with time trade-off
                    #torch.cuda.empty_cache()
                    #gc.collect()

            
            for ky in loss_dict:
                if torch.is_tensor(loss_dict[ky])
                    loss_dict[ky] = loss_dict[ky].detach().item()
                metric_logger.meters[ky].update(loss_dict[ky], n=args.batch_size)

            for ky in metrics:
                if torch.is_tensor(metrics[ky])
                    metrics[ky] = metrics[ky].detach()..item()
                metric_logger.meters[ky].update(metrics[ky], n=args.batch_size)
            
            """
            #Optional to save memory with time trade-off
            for ky in batch:
                if torch.is_tensor(batch[ky]):
                    batch[ky] = batch[ky].to(device)
            for ky in output:
                if torch.is_tensor(output[ky]):
                    output[ky] = output[ky].to(device)
            """



    print(f"{header} loss {metric_logger.loss.global_avg:.4f}")
    
    
    nxt_ln = f"{header} "
    for ky in metric_logger.__dict__['meters']:
            if ky=='loss':
                continue
            if "loss" in ky:                
                nxt_ln += f"{ky} {metric_logger.__dict__['meters'][ky].global_avg:.3f} "        
    print(nxt_ln)

    nxt_ln = f"{header} "
    for ky in metric_logger.__dict__['meters']:            
            if "loss" not in ky:                
                nxt_ln += f"{ky} {metric_logger.__dict__['meters'][ky].global_avg:.3f} "
    print(nxt_ln)

    return metric_logger.loss.global_avg


