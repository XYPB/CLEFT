import datetime
import os
import random
from argparse import ArgumentParser
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT

# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import AUC, MulticlassAccuracy, MulticlassConfusionMatrix
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch.distributed as dist

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from lightning_fabric.strategies import FSDPStrategy
from backbones.encoder_clip import BertEncoder, ImageEncoder
from backbones.encoder_cleft import SwinEncoder, DinoEncoder, CausalLMEncoder, BaselineEncoder
from backbones.loss import DINOLoss
import utils_mae.lr_decay as lrd
from transformers import Adafactor
from sklearn.metrics import roc_auc_score, accuracy_score
from peft import (PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType)
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from memory_profiler import profile

from backbones.mgca_encoder import ImageEncoder as MGCAImageEncoder
from backbones.mgca_encoder import BertEncoder as MGCABertEncoder


torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHEXPERT_BASE_CAPTION = "this is a chest x ray of a patient with "


# os.environ['CUDA_VISIBLE_DEVICES']='0,1'

os.environ['WANDB_START_METHOD'] = 'thread'

class CLIP(LightningModule):

    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_llm: bool = False,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 144,
                 num_workers: int = 8,
                 num_heads: int = 1,
                 lamb: float = 0.75,
                 epsilon: float = 0.05,
                 img_mask_ratio: float = 0,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.confmat = MulticlassConfusionMatrix(self.hparams.num_classes)
        self.all_scores = None
        self.all_labels = None

        # init encoders
        if self.hparams.mgca_encoder:
            self.img_encoder_q = MGCAImageEncoder(text_feat_dim=128, output_dim=128)
            self.text_encoder_q = MGCABertEncoder(output_dim=128, freeze_bert=freeze_llm)
            mgca_model = torch.load('./pretrained/vit_base.ckpt', map_location='cpu')
            mgca_vision_state_dict = {}
            for k, v in mgca_model['state_dict'].items():
                if 'text_encoder' in k:
                    continue
                if 'local_atten_layer' in k or 'prototype_layer' in k:
                    continue
                k = k.replace('img_encoder_q.', '')
                mgca_vision_state_dict[k] = v
            self.img_encoder_q.load_state_dict(mgca_vision_state_dict)
            mgca_text_state_dict = {}
            for k, v in mgca_model['state_dict'].items():
                if 'img_encoder_q' in k:
                    continue
                if 'local_atten_layer' in k or 'prototype_layer' in k:
                    continue
                k = k.replace('text_encoder_q.', '')
                mgca_text_state_dict[k] = v
            self.text_encoder_q.load_state_dict(mgca_text_state_dict)
        else:
            self.img_encoder_q = ImageEncoder(
                model_name=img_encoder, output_dim=self.hparams.emb_dim, 
                pretrained_pth=self.hparams.pretrained_encoder,
                mae_ratio=self.hparams.img_mask_ratio,
                freeze_vit=self.hparams.freeze_vit)
            if not self.hparams.img_cls_ft:
                self.text_encoder_q = BertEncoder(
                    output_dim=self.hparams.emb_dim, freeze_llm=freeze_llm)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.hparams.softmax_temperature))
        
        self.zero_shot_text_feats = None
        # create a global classifier
        if self.hparams.img_cls_ft:
            self.img_encoder_q.global_embed = nn.Linear(self.img_encoder_q.feature_dim, self.hparams.num_classes)
            self.img_encoder_q.global_embed.weight.requires_grad = True
            self.img_encoder_q.global_embed.bias.requires_grad = True

    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''

        # Forward of query image encoder
        img_feat_q, patch_feat_q, loss_mae, pred_mae, mask_mae, pred_feat = self.img_encoder_q(batch["imgs"])
        # patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        # patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        img_emb_q = self.img_encoder_q.global_embed(pred_feat.mean(dim=1))
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        # Forward of query text encoder
        report_feat_q_full, word_feat_q_full, word_attn_q_full, sents_full = self.text_encoder_q(
            batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
        
        # word_emb_q = self.text_encoder_q.local_embed(word_feat_q_full)
        # word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q_full)
        report_emb_q = F.normalize(report_emb_q, dim=-1)

        ########### image-text contrastive loss ################
        bz = img_emb_q.size(0)
        labels = torch.arange(bz).type_as(report_emb_q).long()

        scores = img_emb_q.mm(report_emb_q.t())
        scores *= self.logit_scale.exp()
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_c = loss0 + loss1

        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(scores, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(scores1, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        return loss_c, acc1, acc5
    
    def zero_shot_inference(self, batch, batch_idx):
        '''Inference with zero shot setting'''

        # Forward of query image encoder
        img_feat_q, patch_feat_q, loss_mae, pred_mae, mask_mae, pred_feat = self.img_encoder_q(batch["imgs"])
        # Use classification token instead of averaged patch tokens
        if self.hparams.mgca_encoder:
            img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        else:
            img_emb_q = self.img_encoder_q.global_embed(pred_feat.mean(dim=1))
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        # Forward of query text encoder
        # N x CLS x S
        bsz = img_emb_q.size(0) # N x C
        batch_scores = []
        fixed_caption_ids = batch["caption_ids"][0] # 14 x S, get rid of batch dim
        fixed_attention_mask = batch["attention_mask"][0]
        fixed_token_type_ids = batch["token_type_ids"][0]
        for idx in range(bsz):
            if self.zero_shot_text_feats is None:
                report_feat_q_full, word_feat_q_full, word_attn_q_full, sents_full = self.text_encoder_q(
                    fixed_caption_ids, fixed_attention_mask, fixed_token_type_ids)
                report_emb_q = self.text_encoder_q.global_embed(report_feat_q_full)
                report_emb_q = F.normalize(report_emb_q, dim=-1)
                self.zero_shot_text_feats = report_emb_q
            scores = img_emb_q[idx:idx+1].mm(self.zero_shot_text_feats.t()) # 1 x CLS
            scores *= self.logit_scale.exp()
            batch_scores.append(scores.squeeze(0))
        scores = torch.stack(batch_scores, dim=0) # N x CLS

        ########### image-text zero-shot cls loss ################
        labels = batch["multi_hot_label"].type_as(self.zero_shot_text_feats) # N x CLS

        # Image to text classification loss
        loss0 = F.cross_entropy(scores, labels.argmax(dim=-1))


        if self.hparams.devices > 1:
            score_list = [torch.zeros_like(scores) for _ in range(dist.get_world_size())]
            dist.all_gather(score_list, scores)
            all_scores = torch.cat(score_list, dim=0)
            label_list = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
            dist.all_gather(label_list, labels)
            all_labels = torch.cat(label_list, dim=0)
        else:
            all_scores = scores
            all_labels = labels
        self.confmat.update(
            torch.argmax(all_scores, dim=-1), all_labels.argmax(dim=-1))
        all_scores = all_scores.detach().to(torch.float32)
        all_scores = torch.softmax(all_scores, dim=-1).cpu().numpy()
        all_labels = all_labels.detach().to(torch.float32).cpu().numpy()
        if self.all_scores is None:
            self.all_scores = all_scores
        else:
            self.all_scores = np.concatenate([self.all_scores, all_scores], axis=0)
        if self.all_labels is None:
            self.all_labels = all_labels
        else:
            self.all_labels = np.concatenate([self.all_labels, all_labels], axis=0)

        # compute retrieval accuracy
        i2t_acc1 = self.precision_at_k(scores, labels.argmax(dim=-1), top_k=(1,))[0]

        return loss0, i2t_acc1, 0.

    def visual_forward(self, batch, batch_idx, split="train"):
        # Forward of query image encoder
        img_feat_q, patch_feat_q, loss_mae, pred_mae, mask_mae, pred_feat = self.img_encoder_q(batch["imgs"])
        # Use classification token instead of averaged patch tokens
        img_emb_q = self.img_encoder_q.global_embed(pred_feat.mean(dim=1))
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        ########### Classification loss ################
        labels = batch["multi_hot_label"].type_as(img_emb_q) # N x CLS
        
        # Image classification loss
        loss0 = F.cross_entropy(img_emb_q, labels.argmax(dim=-1))

        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(img_emb_q, labels.argmax(dim=-1), top_k=(1, 2))

        if split == 'test':
            if self.hparams.devices > 1:
                img_emb_q_list = [torch.zeros_like(img_emb_q) for _ in range(dist.get_world_size())]
                dist.all_gather(img_emb_q_list, img_emb_q)
                all_img_emb_qs = torch.cat(img_emb_q_list, dim=0)
                label_list = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
                dist.all_gather(label_list, labels)
                all_labels = torch.cat(label_list, dim=0)
            else:
                all_img_emb_qs = img_emb_q
                all_labels = labels
            self.confmat.update(
                torch.argmax(all_img_emb_qs, dim=-1), all_labels.argmax(dim=-1))
            all_img_emb_qs = all_img_emb_qs.detach().to(torch.float32)
            all_img_emb_qs = torch.softmax(all_img_emb_qs, dim=-1).cpu().numpy()
            all_labels = all_labels.detach().to(torch.float32).cpu().numpy()
            if self.all_scores is None:
                self.all_scores = all_img_emb_qs
            else:
                self.all_scores = np.concatenate([self.all_scores, all_img_emb_qs], axis=0)
            if self.all_labels is None:
                self.all_labels = all_labels
            else:
                self.all_labels = np.concatenate([self.all_labels, all_labels], axis=0)

        return loss0, i2t_acc1, i2t_acc5

    def training_step(self, batch, batch_idx):
        if self.hparams.img_cls_ft:
            loss_c, acc1, acc5 = self.visual_forward(
                batch, batch_idx, "train")
        else:
            loss_c, acc1, acc5 = self(
                batch, batch_idx, "train")
        loss = loss_c
        
        log = {
            "train_loss": loss,
            "train_loss_c": loss_c,
            "train_acc1": acc1,
            "train_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.img_cls_ft:
            loss_c, acc1, acc5 = self.visual_forward(
                batch, batch_idx, "valid")
        else:
            loss_c, acc1, acc5 = self(
                batch, batch_idx, "valid")
        loss = loss_c

        log = {
            "val_loss": loss,
            "val_loss_c": loss_c,
            "val_acc1": acc1,
            "val_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        if self.hparams.img_cls_ft:
            loss_c, acc1, acc5 = self.visual_forward(
                batch, batch_idx, "test")
        else:
            loss_c, acc1, acc5 = self.zero_shot_inference(
                batch, batch_idx)
        loss = loss_c

        log = {
            "test_loss": loss,
            "test_loss_c": loss_c,
            "test_acc1": acc1,
            "test_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):

        # Calculate the confusion matrix using the accumulated predictions and targets
        conf_matrix = self.confmat.compute()
        print("### Confusion Matrix:\n", conf_matrix)
        # Calculate the accuracy using the accumulated predictions and targets
        acc = 100 * accuracy_score(np.argmax(self.all_labels, -1), np.argmax(self.all_scores, -1))
        try:
            if self.hparams.num_classes == 2:
                auc = 100 * roc_auc_score(self.all_labels, self.all_scores)
            else:
                auc = 100 * roc_auc_score(np.argmax(self.all_labels, -1), self.all_scores, multi_class="ovr")
        except Exception as e:
            print("### Warning: AUC calculation failed with error:", e)
            auc = 0
        print("### Accuracy: {:.4f}".format(acc))
        print("### AUC: {:.4f}".format(auc))

        # Reset metrics for the next test run
        self.confmat.reset()
        self.all_scores = None
        self.all_labels = None

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        
        parameters = lrd.param_groups_lrd_moco(self, self.hparams.weight_decay, no_weight_decay_list=[],
                                               lr_layer_wise="2e-5,2e-5,2e-5")
        debugc = 1
        optimizer = torch.optim.AdamW(
            parameters,
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_after_backward(self) -> None:
        pass
        # print("on_after_backward enter")
        # for name, p in self.named_parameters():
        #     if p.grad is None and p.requires_grad:
        #         print(name)
        # print("on_after_backward exit")


    @staticmethod
    def add_model_specific_args(parent_parser):
        pass

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, DDPStrategy)
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = trainer.num_devices
        
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return trainer.max_steps
