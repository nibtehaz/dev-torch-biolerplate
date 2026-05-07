

'''
CUDA_VISIBLE_DEVICES=1,2 python3 -m torch.distributed.run \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=172.18.132.71:29500 \
    train.py --batch-size 12 --grad_accum_iter 22 --lr 1e-3 --epochs 30 --output-dir experiments --printout --amp  --resume 'f' >> experiments/log.log

'''


import pickle
from model import *
from loss_functions import * 
from train_utils import *
from training_loop import *
from dataset import *
import pandas as pd

PARAMS = {
    
}

def main(args):

    
    device = torch.device(args.device)      
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir), exist_ok=True)
    
    if not args.printout:
        sys.stdout = open(f'{output_dir}/log.log', 'a' if args.resume=='true' else 'w')

    print(args)
    
    ########## 
    ## DATA ##
    ##########

    # PUT DATA LOADING CODE HERE
    train_dataset = TrainingDataset() 
    val_dataset = ValidationDataset()
        
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,sampler=train_sampler,num_workers=args.workers,pin_memory=True,drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,sampler=val_sampler,num_workers=args.workers,pin_memory=True,drop_last=False)
    
    print('Data loaded')
    
    ########### 
    ## MODEL ##
    ###########

    # PUT MODEL CODE HERE

    model = MODEL()
    model.to(device)
    print('Model loaded')

    print(PARAMS)


    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    
    ### PUT OPTIMIZER, LR SCHEDULER, CRITERION CODE HERE
    criterion = LossFunction()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler


    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
    model_without_ddp = model.module

    best_loss = 10000000

    BEST_LOSSES = {}
    
    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)


    if args.resume=='true':
        resume_path = output_dir + "/" + "checkpoint.pth"
        checkpoint = torch.load(resume_path, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])        
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])

        best_loss = checkpoint["best_loss"]

    

    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        run_epoch(model, train_dataloader, criterion, device, args, epoch, optimizer=optimizer, is_train=True)
     

        #if(epoch<lr_scheduler.T_max):
        lr_scheduler.step()

        val_loss = run_epoch(model, val_dataloader, criterion, device, args, epoch, optimizer=None, is_train=False)

        
                        

        if model_ema:
            val_loss = run_epoch(model_ema, val_dataloader, criterion, device, args, optimizer=None, is_train=False)
            
        if output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "best_loss": best_loss,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            
            if best_loss > val_loss:                    
                print(f"loss improved to {val_loss}@{epoch} from {best_loss}")
                best_loss = val_loss
                save_on_master(checkpoint, os.path.join(output_dir, "checkpoint.pth"))
                
            epoch_step = epoch//10
            if epoch_step not in BEST_LOSSES:
                BEST_LOSSES[epoch_step] = 10000000
            if BEST_LOSSES[epoch_step] > val_loss:
                print(f"Stage {epoch_step} loss improved to {val_loss} from {BEST_LOSSES[epoch_step]}")
                BEST_LOSSES[epoch_step] = val_loss
                save_on_master(checkpoint, os.path.join(output_dir, f"checkpoint_{epoch_step}.pth"))


    

if __name__ == "__main__":
    args = get_args_parser().parse_args()

    init_distributed_mode(args)    

    
    main(args)
