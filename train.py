import datetime
import os
from argparse import ArgumentParser

# from torch.utils.tensorboard import SummaryWriter

import torch
from dateutil import tz

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from datasets.data_module import DataModule
from datasets.chexpert_dataset import CheXpertPretrainingDataset, multimodal_collate_fn
from datasets.embed_dataset import EmbedPretrainingDataset
from datasets.eval_dataset import RSNADataset
from datasets.transforms import Moco2Transform

from model_cleft import CLEFT
from model_clip import CLIP

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


os.environ['WANDB_START_METHOD'] = 'thread'


@torch.no_grad()
def concat_all_gather(tensor):
    '''
    Performs all_gather operation on the provided tensors
    '''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def train(args, model, datamodule):
        # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    extension += f"_{args.experiment_name}"
    ckpt_dir = os.path.join(
        BASE_DIR, f"logs/ckpts/CLEFT/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=2),
        # EarlyStopping(monitor="val_loss", min_delta=0.,
        #               patience=5, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"./logs")
    os.makedirs(logger_dir, exist_ok=True)
    if args.img_cls_ft:
        if args.embed:
            project = "CLEFT_img_embed_ft"
        else:
            project = "CLEFT_img_cls_ft"
        if 'fft' in args.experiment_name:
            project.replace('ft', 'fft')
    elif args.embed:
        project = "CLEFT_Embed"
    else:
        project = "CLEFT_fix_step"
    wandb_logger = WandbLogger(
        project=project, save_dir=logger_dir, name=extension)
    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=wandb_logger,
        fast_dev_run=args.dev,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_ckpt,)
    trainer.test(model, datamodule=datamodule)


    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)
    return model


def eval(args, model, datamodule):
    model.eval()

    # Single GPU inference
    trainer = Trainer(
        accelerator=args.accelerator,
        precision=args.precision,
        devices=1,
        fast_dev_run=args.dev,
        max_epochs=1,
        deterministic=args.deterministic,
        inference_mode=True
    )

    trainer.test(model, datamodule=datamodule)


def cli_main():
    
    parser = ArgumentParser()
    parser.add_argument("--eval", action="store_true", 
                        help="Run evaluation")
    parser.add_argument("--llm_type", type=str, default="gpt", 
                        help="bert, gpt, or llama")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to the pretrained model, used for evaluation")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Path to the checkpoint to resume training")
    parser.add_argument("--rsna", action="store_true",
                        help="Use RSNA dataset")
    parser.add_argument("--embed", action="store_true",
                        help="Use Embed dataset")
    parser = CLEFT.add_model_specific_args(parser)

    args = parser.parse_args()
    if args.llm_type in ["gpt", "llama"]:
        args.deterministic = False
    else:
        args.img_encoder = 'vit_base'
        args.deterministic = True

    if args.eval:
        args.batch_size = 32
        args.data_pct = 1.0
        args.max_epoch = 1
        args.accumulate_grad_batches = 1
        args.dev = False
        args.strategy = None
        args.devices = 1
        args.grad_ckpt = False
        args.train_sub_set = False
    num_cores = len(os.sched_getaffinity(0))
    if args.num_workers > num_cores:
        args.num_workers = num_cores
        print('switching to maximum num_workers = ', num_cores)

    if args.use_flash_attention:
        os.environ["XFORMERS_DISABLED"] = "1"
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

    # seed
    seed_everything(args.seed)

    if args.rsna:
        dataset = RSNADataset
    elif args.embed:
        dataset = EmbedPretrainingDataset
    else:
        dataset = CheXpertPretrainingDataset
    transform_obj = Moco2Transform
    datamodule = DataModule(dataset, multimodal_collate_fn,
                            transform_obj, args.data_pct,
                            args.batch_size, args.num_workers, 
                            masked_lm_ratio=args.masked_lm_ratio,
                            llm_type=args.llm_type,
                            prompt_ft=args.prompt_ft,
                            train_split=args.train_split,
                            valid_split=args.valid_split, five_cls=args.five_cls,
                            train_sub_set=args.train_sub_set, structural_cap=args.structural_cap,
                            simple_cap=args.simple_cap, natural_cap=args.natural_cap,
                            instance_test_cap=args.instance_test_cap, 
                            balanced_test=args.balanced_test, 
                            balance_training=args.balance_training,
                            pred_density=args.pred_density, keep_size=args.keep_size)

    # Add load from checkpoint
    if args.llm_type in ['gpt', 'llama']:
        if args.pretrained_model is None:
            model = CLEFT(**args.__dict__)
        else:
            model = CLEFT.load_from_checkpoint(args.pretrained_model, map_location="cpu", strict=False, **args.__dict__)
    else:
        if args.pretrained_model is None:
            model = CLIP(**args.__dict__)
        else:
            model = CLIP.load_from_checkpoint(args.pretrained_model, map_location="cpu", strict=False, **args.__dict__)

    if args.eval:
        eval(args, model, datamodule)
    else:
        model = train(args, model, datamodule)



if __name__ == "__main__":
    cli_main()
