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


class CLEFT(LightningModule):

    def __init__(self,
                 img_encoder: str = "dinov2_vitb14_reg",
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
                 peft: str = None,
                 agg_tokens: bool = False,
                 grad_ckpt: bool = False,
                 use_flash_attention_2: bool = False,
                 img_cls_ft: bool = False,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.confmat = MulticlassConfusionMatrix(self.hparams.num_classes)
        self.all_scores = None
        self.all_labels = None

        # init encoders
        if self.hparams.baseline_vit:
            img_encoder_obj = BaselineEncoder
        else:
            img_encoder_obj = DinoEncoder
        self.img_encoder_q = img_encoder_obj(
            model_name=img_encoder, output_dim=self.hparams.emb_dim, 
            img_mask_ratio=self.hparams.img_mask_ratio, linear_proj=self.hparams.linear_proj,
            freeze_vit=self.hparams.freeze_vit)

        # Randomize the visual transformer
        if self.hparams.random_vit:
            self.img_encoder_q.model.init_weights()

        # Create a text encoder
        if not self.hparams.img_cls_ft:
            self.text_encoder_q = CausalLMEncoder(
                output_dim=self.hparams.emb_dim, freeze_llm=self.hparams.freeze_llm, 
                peft=self.hparams.peft, agg_tokens=self.hparams.agg_tokens, 
                grad_ckpt=self.hparams.grad_ckpt, llm_type=self.hparams.llm_type,
                use_flash_attention_2=self.hparams.use_flash_attention_2,
                linear_proj=self.hparams.linear_proj, train_embed=self.hparams.train_embed,
                unlock_ln=self.hparams.unlock_ln, prompt_ft=self.hparams.prompt_ft,
                total_steps=self.hparams.max_steps, num_freeze_blocks=self.hparams.num_freeze_blocks)
        
        # Load pre-trained vit parameter
        if self.hparams.pretrained_encoder != None:
            print("Loading pretrained model from {}".format(self.hparams.pretrained_encoder))
            state_dict = torch.load(self.hparams.pretrained_encoder, map_location="cpu")['state_dict']
            img_encoder_state_dict = {k.replace('img_encoder_q.', ''): v for k, v in state_dict.items() if k.startswith('img_encoder_q')}
            self.img_encoder_q.load_state_dict(img_encoder_state_dict)
            if not self.hparams.img_cls_ft:
                text_encoder_state_dict = {k.replace('text_encoder_q.', ''): v for k, v in state_dict.items() if k.startswith('text_encoder_q')}
                self.text_encoder_q.load_state_dict(text_encoder_state_dict)
        
        # Create a prompt learner
        if self.hparams.prompt_ft and not self.hparams.img_cls_ft:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            prompt_init = PromptTuningInit.TEXT if self.hparams.ctx_init == "caption" else PromptTuningInit.RANDOM
            config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                        num_virtual_tokens=self.hparams.ctx_length,
                                        prompt_tuning_init = prompt_init,
                                        prompt_tuning_init_text=CHEXPERT_BASE_CAPTION,
                                        tokenizer_name_or_path=self.text_encoder_q.llm_name)
            self.text_encoder_q.model = get_peft_model(self.text_encoder_q.model, config)
            
        # create a global classifier
        if self.hparams.img_cls_ft:
            self.img_encoder_q.global_embed = nn.Linear(self.img_encoder_q.feature_dim, self.hparams.num_classes)
            self.img_encoder_q.global_embed.weight.requires_grad = True
            self.img_encoder_q.global_embed.bias.requires_grad = True
        
        # Initialize the learnable logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.hparams.softmax_temperature))
        
        self.zero_shot_text_feats = None




    def get_data_keys(self, split="train"):
        # 50% of chance to use unpaired text
        # Only provide unpaired text for training
        keys =  ["imgs", "caption_ids", "attention_mask", "multi_hot_label"]
        return keys

    # @profile
    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''
        img_key, cap_key, attn_key, label_key = self.get_data_keys(split)

        # Forward of query image encoder
        img_feat_q, patch_feat_q, img_full = self.img_encoder_q(batch[img_key])
        # Following FLIP, use the average of patch features w/o layer norm
        if self.hparams.img_mask_ratio > 0 or self.hparams.pool_feat:
            img_feat_q = img_full.mean(dim=1)
        # Use classification token instead of averaged patch tokens
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        # Forward of query text encoder
        report_feat_q_full, word_feat_q_full, word_attn_q_full, sents_full = self.text_encoder_q(
            batch[cap_key], batch[attn_key])
        self.hparams.img_mask_ratio
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
    
    def ctx_tuning_forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method in context prompt tuning'''
        img_key, cap_key, attn_key, label_key = self.get_data_keys(split)

        # Forward of query image encoder
        img_feat_q, patch_feat_q, img_full = self.img_encoder_q(batch[img_key])
        # Following FLIP, use the average of patch features w/o layer norm
        if self.hparams.img_mask_ratio > 0 or self.hparams.pool_feat:
            img_feat_q = img_full.mean(dim=1)
        # Use classification token instead of averaged patch tokens
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        # Forward of query text encoder
        # Forward for each individual image
        bsz = img_emb_q.size(0) # N x C
        batch_scores = []
        fixed_caption_ids = batch[cap_key][0] # 14 x S, get rid of batch dim
        fixed_attention_mask = batch[attn_key][0]
        
        # Manually map input_ids to embeddings
        fix_inputs_embeds = self.text_encoder_q.model.get_input_embeddings()(fixed_caption_ids)
        
        for idx in range(bsz):
            inputs_embeds, attn_masks = self.prompt_learner(
                fix_inputs_embeds, fixed_attention_mask, patch_feat_q[idx:idx+1])
            report_feat_q_full, word_feat_q_full, word_attn_q_full, sents_full = self.text_encoder_q(
                fixed_caption_ids, attn_masks, inputs_embeds=inputs_embeds)
            report_emb_q = self.text_encoder_q.global_embed(report_feat_q_full)
            report_emb_q = F.normalize(report_emb_q, dim=-1)
            
            scores = img_emb_q[idx:idx+1].mm(report_emb_q.t()) # 1 x CLS
            scores *= self.logit_scale.exp()
            batch_scores.append(scores.squeeze(0))
        scores = torch.stack(batch_scores, dim=0) # N x CLS

        ########### image-text contrastive loss ################
        labels = batch[label_key].type_as(scores) # N x CLS

        # Multi-label image to text classification loss
        # Using raw unnormalized scores
        loss0 = F.binary_cross_entropy_with_logits(scores, labels)

        # compute retrieval accuracy
        acc1 = self.multi_label_precision(torch.sigmoid(scores), labels)

        return loss0, acc1
    
    def zero_shot_inference(self, batch, batch_idx, split="test"):
        '''Inference with zero shot setting'''
        img_key, cap_key, attn_key, label_key = self.get_data_keys(split)

        with torch.no_grad():
            # Forward of query image encoder
            img_feat_q, patch_feat_q, img_full = self.img_encoder_q(batch[img_key])
            # Following FLIP, use the average of patch features w/o layer norm
            if self.hparams.img_mask_ratio > 0 or self.hparams.pool_feat:
                img_feat_q = img_full.mean(dim=1)
            # Use classification token instead of averaged patch tokens
            img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
            img_emb_q = F.normalize(img_emb_q, dim=-1)

            # Forward of query text encoder
            # Forward for each individual image
            bsz = img_emb_q.size(0) # N x C
            batch_scores = []
            if batch[cap_key].shape[0] == 1:
                raise ValueError
            if not self.hparams.instance_test_cap:
                fixed_caption_ids = batch[cap_key][0] # CLS x S, get rid of batch dim
                fixed_attention_mask = batch[attn_key][0]
            
            for idx in range(bsz):
                if self.hparams.instance_test_cap:
                    fixed_caption_ids = batch[cap_key][idx]
                    fixed_attention_mask = batch[attn_key][idx]
                if self.zero_shot_text_feats is None or self.hparams.instance_test_cap:
                    report_feat_q_full, word_feat_q_full, word_attn_q_full, sents_full = self.text_encoder_q(
                        fixed_caption_ids, fixed_attention_mask)
                    report_emb_q = self.text_encoder_q.global_embed(report_feat_q_full)
                    report_emb_q = F.normalize(report_emb_q, dim=-1)
                    
                    self.zero_shot_text_feats = report_emb_q # CLS x C

                scores = img_emb_q[idx:idx+1].mm(self.zero_shot_text_feats.t()) # 1 x CLS
                scores *= self.logit_scale.exp()
                batch_scores.append(scores.squeeze(0))
            scores = torch.stack(batch_scores, dim=0) # N x CLS

            ########### image-text zero-shot cls loss ################
            labels = batch[label_key].type_as(scores) # N x CLS

            # Image to text classification loss
            loss0 = F.cross_entropy(scores, labels.argmax(dim=-1))

            # compute retrieval accuracy
            i2t_acc1 = self.precision_at_k(scores, labels.argmax(dim=-1), top_k=(1,))[0]

            if split == 'test':
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
            
            labels = labels.float().detach().cpu().numpy()
            scores = torch.softmax(scores.float().detach(), dim=1).cpu().numpy()
            auc = 0.
            

        return loss0, i2t_acc1, auc


    def visual_forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''
        img_key, cap_key, attn_key, label_key = self.get_data_keys(split)

        # Forward of query image encoder
        img_feat_q, patch_feat_q, img_full = self.img_encoder_q(batch[img_key])
        # Following FLIP, use the average of patch features w/o layer norm
        if self.hparams.img_mask_ratio > 0 or self.hparams.pool_feat:
            img_feat_q = img_full.mean(dim=1)
        # Use classification token instead of averaged patch tokens
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)

        ########### Classification loss ################
        labels = batch[label_key].type_as(img_emb_q) # N x CLS
        
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
                      sync_dist=True, prog_bar=True, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.img_cls_ft:
            loss_c, acc1, acc5 = self.visual_forward(
                batch, batch_idx, "val")
        else:
            loss_c, acc1, acc5 = self(
                batch, batch_idx, "val")
        loss = loss_c

        log = {
            "val_loss": loss,
            "val_loss_c": loss_c,
            "val_acc1": acc1,
            "val_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True, rank_zero_only=True)
        return loss

    def test_step(self, batch, batch_idx):

        if self.hparams.img_cls_ft:
            loss_c, acc1, auc = self.visual_forward(
                batch, batch_idx, "test")
        else:
            loss_c, acc1, auc = self.zero_shot_inference(batch, batch_idx, "test")
        loss = loss_c

        log = {
            "test_loss": loss,
            "test_loss_c": loss_c,
            "test_acc1": acc1,
            "test_auc": auc
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True, rank_zero_only=True)
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
        
    @staticmethod
    def multi_label_precision(output: torch.Tensor, target: torch.Tensor, threshold=0.5):
        ''' Compute the accuracy over the k top predictions for the specified values'''
        with torch.no_grad():
            # Applying threshold to prediction probabilities
            preds = output > threshold

            # Correct output are only those where prediction and label are equal
            correct_preds = (preds == target).float()

            # Compute accuracy across all target
            accuracy = 100 * correct_preds.sum() / (len(target) * target.size(1))

            return accuracy

    def configure_optimizers(self):
        if self.hparams.no_lrd:
            parameters = self.parameters()
        else:
            parameters = lrd.param_groups_lrd_moco(self, self.hparams.weight_decay, 
                                                   no_weight_decay_list=[],
                                                   lr_layer_wise="2e-5,2e-5,2e-5")
        if self.hparams.adafactor:
            optimizer = Adafactor(
                parameters,
                self.hparams.learning_rate,
                beta1=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
                relative_step=False,
                scale_parameter=False,
            )
        elif self.hparams.sgd:
            optimizer = torch.optim.SGD(
                parameters,
                self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                parameters,
                self.hparams.learning_rate,
                betas=(self.hparams.momentum, 0.999),
                weight_decay=self.hparams.weight_decay
            )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.hparams.max_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=self.hparams.min_lr,
            warmup_steps=self.hparams.warm_up
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
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        # Model args
        parser.add_argument("--emb_dim", type=int, default=512,
                            help="Embedding dimension")
        parser.add_argument("--prompt_ft", action="store_true",
                            help="Use prompt tuning")
        parser.add_argument("--ctx_length", type=int, default=30,
                            help="Context prompt length")
        parser.add_argument("--ctx_init", type=str, default="random", 
                            help="random caption (random|caption)")
        parser.add_argument("--linear_proj", action="store_true",
                            help="Use linear projection layer")
        parser.add_argument("--pool_feat", action="store_true",
                            help="Use global average pooling for patch features")
        ### Visual Model args 
        parser.add_argument("--img_encoder", type=str, default="dinov2_vitb14_reg",
                            help="Image encoder model")
        parser.add_argument("--freeze_vit", action="store_true",
                            help="Freeze visual transformer")
        parser.add_argument("--baseline_vit", action="store_true",
                            help="Use baseline visual transformer")
        parser.add_argument("--img_mask_ratio", type=float, default=0,
                            help="Ratio of masked tokens in image")
        parser.add_argument("--random_vit", action="store_true",
                            help="Randomize visual transformer, use for baseline")
        ### LLM args
        parser.add_argument("--freeze_llm", action="store_true",
                            help="Freeze language model")
        parser.add_argument("--train_embed", action="store_true",
                            help="Train word embeddings layer")
        parser.add_argument("--unlock_ln", action="store_true",
                            help="Unlock layer norm in LLM")
        parser.add_argument("--num_freeze_blocks", type=int, default=0,
                            help="Number of blocks to freeze in LLM")
        parser.add_argument("--masked_lm_ratio", type=float, default=0,
                            help="Ratio of masked tokens in text")
        parser.add_argument("--peft", type=str, default=None,
                            help="Use prompt tuning, (lora|ia3|prefix)")
        
        # Training args
        parser.add_argument("--num_workers", type=int, default=16,
                            help="Number of workers for dataloader")
        parser.add_argument("--batch_size", type=int, default=72,
                            help="Batch size")
        parser.add_argument("--max_epochs", type=int, default=50) # Unused
        parser.add_argument("--max_steps", type=int, default=40000)
        parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                            help="Number of gradient accumulation steps")
        parser.add_argument("--img_cls_ft", action="store_true",
                            help="Image classification finetuning")
        parser.add_argument("--num_classes", type=int, default=1000,
                            help="Number of classes for image classification")
        parser.add_argument("--num_heads", type=int, default=1,
                            help="Number of heads in multi-head attention")
        parser.add_argument("--experiment_name", type=str, default="",
                            help="Name of the experiment")
        parser.add_argument("--seed", type=int, default=42,
                            help="Seed for reproducibility")
        parser.add_argument("--devices", type=int, default=4,
                            help="Number of devices")
        parser.add_argument("--strategy", type=str, default="ddp",
                            help="Training strategy")
        parser.add_argument("--accelerator", type=str, default='gpu')
        parser.add_argument("--precision", type=str, default="32",
                            help="Precision for training")
        parser.add_argument("--dev", action="store_true")
        parser.add_argument("--grad_ckpt", action="store_true",
                            help="Use gradient checkpointing for LLM")
        parser.add_argument("--warm_up", type=int, default=16000)
        parser.add_argument("--balance_training", action="store_true",
                            help="Balance training data during fine-tuning")
        ### Hyperparameters
        parser.add_argument("--softmax_temperature", type=float, default=0.07,
                            help="Softmax temperature")
        parser.add_argument("--learning_rate", type=float, default=2e-5,
                            help="Learning rate")
        parser.add_argument("--min_lr", type=float, default=1e-8,
                            help="Minimum learning rate")
        parser.add_argument("--momentum", type=float, default=0.9,
                            help="Momentum for optimizer")
        parser.add_argument("--weight_decay", type=float, default=0.05,
                            help="Weight decay for optimizer")
        parser.add_argument("--no_lrd", action="store_true",
                            help="No layer-wise learning rate decay")
        ### Optimizer
        parser.add_argument("--adafactor", action="store_true",
                            help="Use Adafactor optimizer, save more memory")
        parser.add_argument("--sgd", action="store_true", 
                            help="Use SGD optimizer")
        ### Pretrained args
        parser.add_argument("--pretrained_encoder", type=str, default=None,
                            help="Path to the pretrained encoders, used for fine-tuning")
        
        # Data args
        parser.add_argument("--agg_tokens", action="store_true",
                            help="Aggregate tokens")
        parser.add_argument("--train_sub_set", action="store_true",
                            help="Use subset of training data")
        parser.add_argument("--data_pct", type=float, default=1.0,
                            help="Percentage of data to use")
        parser.add_argument("--train_split", type=str, default="train")
        parser.add_argument("--valid_split", type=str, default="valid")
        parser.add_argument("--keep_size", action="store_true",
                            help="Keep the size of the dataset")
        ### EMBED test set args
        parser.add_argument("--balanced_test", action="store_true",
                            help="Use balanced test set")
        parser.add_argument("--small_balanced_train", action="store_true",
                            help="Use small balanced train set")
        parser.add_argument("--pred_density", action="store_true",
                            help="Use prediction density")
        # Caption args
        parser.add_argument("--structural_cap", action="store_true")
        parser.add_argument("--simple_cap", action="store_true")
        parser.add_argument("--natural_cap", action="store_true")
        
        # Inference args
        parser.add_argument("--instance_test_cap", action="store_true",
                            help="Use instance test caption for zero-shot inference")
        parser.add_argument("--five_cls", action="store_true",
                            help="Use five classes for fine-tuning/evaluation")

        # Baseline args
        parser.add_argument("--mgca_encoder", action="store_true",
                            help="Use MGCA encoder")
        parser.add_argument("--mrm_encoder", action="store_true",
                            help="Use MRM encoder")

        parser.add_argument("--use_flash_attention", action="store_true")
        # Not supported!
        parser.add_argument("--use_flash_attention_2", action="store_true")

        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPStrategy, FSDPStrategy))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""

        return trainer.max_steps