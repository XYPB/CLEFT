import os
import types
from typing import Callable
import copy
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from transformers import AutoTokenizer, logging, AutoModelForCausalLM, BertTokenizer
from transformers import AutoModelForImageClassification

from peft import (IA3Config, LoraConfig, PrefixTuningConfig, AdaLoraConfig, get_peft_model, TaskType)

from backbones.gloria_backbone import ImageEncoder
from functools import partial
from timm.models.vision_transformer import VisionTransformer
from backbones.mgca_vit import create_vit
from medclip import MedCLIPModel, MedCLIPVisionModelViT

logging.set_verbosity_error()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MY_API_TOKEN = "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"


def get_tokenizer(llm_type):
    if llm_type == 'gpt':
        tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
        tokenizer.pad_token = tokenizer.eos_token
    elif llm_type == 'llama':
        tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b", token=MY_API_TOKEN,
                                                        padding_side='right')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return tokenizer


def random_masking(x, mask_ratio=0.50):
    N, S, D = x.shape

    mask = torch.rand(N, S, device=x.device)

    # sort noise for each sample
    ids_shuffle = torch.argsort(mask, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :int(S * (1 - mask_ratio))]

    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    return x_masked, ids_restore


def masked_only_prepare_tokens_with_masks(self, x, masks=None):
    B, nc, w, h = x.shape
    x = self.patch_embed(x)
    if masks is not None:
        x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + self.interpolate_pos_encoding(x, w, h)

    # Keep the CLS token and mask the rest
    x_masked, ids_restore = random_masking(x[:, 1:, :], self.mask_ratio)
    x = torch.cat((x[:, :1, :], x_masked), dim=1)

    if self.register_tokens is not None:
        x = torch.cat(
            (
                x[:, :1],
                self.register_tokens.expand(x.shape[0], -1, -1),
                x[:, 1:],
            ),
            dim=1,
        )

    return x


class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)

        return x.permute(0, 2, 1)


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(1).expand(-1, N, -1), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD


class SwinEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "microsoft/swin-base-patch4-window7-224-in22k",
                 text_feat_dim: int = 768,
                 output_dim: int = 512,
                 hidden_dim: int = 2048,
                 img_mask_ratio: float = 0,
                 freeze_vit: bool = False,
                 pretrained: bool = True,
                 linear_proj: bool = False,
                 ):
        super(SwinEncoder, self).__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.text_feat_dim = text_feat_dim
        
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        # Only use SwinBackbone
        self.model = self.model.swin
        if img_mask_ratio > 0:
            # TODO
            pass
        
        self.feature_dim = 1024
        
        if linear_proj:
            self.global_embed = nn.Linear(self.feature_dim, output_dim)
        else:
            self.global_embed = GlobalEmbedding(
                self.feature_dim, hidden_dim, output_dim
            )
        if freeze_vit is True:
            print("Freezing vit model")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.global_embed.parameters():
                param.requires_grad = False

    def vit_forward(self, x):
        return self.model(x)
    
    def forward(self, x, get_local=False):
        ret = self.vit_forward(x)
        return ret.pooler_output.contiguous(), ret.last_hidden_state.contiguous(), ret.last_hidden_state.contiguous()


class DinoEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "dinov2_vitb14_reg_lc",
                 text_feat_dim: int = 768,
                 output_dim: int = 512,
                 hidden_dim: int = 2048,
                 img_mask_ratio: float = 0,
                 freeze_vit: bool = False,
                 pretrained: bool = True,
                 linear_proj: bool = False,
                 num_freeze_blocks: int = 0,
                 ):
        super(DinoEncoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.text_feat_dim = text_feat_dim

        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        if img_mask_ratio > 0:
            # self.model.random_masking = types.MethodType(random_masking, self.model)
            self.model.prepare_tokens_with_masks = types.MethodType(
                masked_only_prepare_tokens_with_masks, self.model)
            self.model.mask_ratio = img_mask_ratio

        self.feature_dim = self.model.embed_dim

        if linear_proj:
            self.global_embed = nn.Linear(self.feature_dim, output_dim)
        else:
            self.global_embed = GlobalEmbedding(
                self.feature_dim, hidden_dim, output_dim
            )

        # Unused
        self.local_embed = LocalEmbedding(
            self.feature_dim, hidden_dim, output_dim
        )
        
        if freeze_vit:
            print("Freezing vit model")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.global_embed.parameters():
                param.requires_grad = False
            for param in self.local_embed.parameters():
                param.requires_grad = False
                
        if num_freeze_blocks > 0:
            pass #TODO

    def vit_forward(self, x):
        return self.model(x, is_training=True)

    def forward(self, x, get_local=False):
        ret = self.vit_forward(x)
        return ret['x_norm_clstoken'].contiguous(), ret['x_norm_patchtokens'].contiguous(), ret['x_prenorm'].contiguous()


class BaselineEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "gloria",
                 pretrained: bool = True,
                 linear_proj: bool = False,
                 freeze_vit: bool = False,
                 output_dim: int = 512,
                 hidden_dim: int = 2048,
                 **kwargs,
                 ):
        super(BaselineEncoder, self).__init__()
        self.model_name = model_name
        if model_name == 'gloria':
            print("Using Gloria's Image Encoder")
            self.model = ImageEncoder('resnet50', 512)
            if pretrained:
                state_dict = torch.load('./pretrained/chexpert_resnet50.ckpt', map_location='cpu')
                gloria_img_state_dict = {}
                for k, v in state_dict['state_dict'].items():
                    if 'gloria.img_encoder.model.' in k:
                        k = k.replace('gloria.img_encoder.model.', 'model.')
                        gloria_img_state_dict[k] = v
                self.model.load_state_dict(gloria_img_state_dict, strict=False)
            self.model = self.model.model
            self.feature_dim = 2048
        elif model_name == 'mrm':
            print("Using MRM's Vision Transformer")
            self.model = VisionTransformer(norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                           num_classes=14, drop_path_rate=0.1,
                                           global_pool="avg")
            if pretrained:
                state_dict = torch.load('./pretrained/MRM.pth', map_location='cpu')
                self.model.load_state_dict(state_dict['model'], strict=False)
            self.feature_dim = 768
            self.model.head = nn.Identity()
        elif model_name == 'mgca':
            print("Using MGCA's Vision Transformer")
            self.model, _ = create_vit('base', 224)
            if pretrained:
                state_dict = torch.load('./pretrained/mgca_vit_base_only.ckpt', map_location='cpu')
                self.model.load_state_dict(state_dict, strict=False)
            self.feature_dim = 768
        elif model_name == 'medclip':
            print("Using MedCLIP's Vision Transformer")
            model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
            model.from_pretrained()
            self.model = model.vision_model
            self.model.projection_head = nn.Identity()
            self.feature_dim = 768
        else:
            return NotImplementedError
        
        if linear_proj:
            self.global_embed = nn.Linear(self.feature_dim, output_dim)
        else:
            self.global_embed = GlobalEmbedding(
                self.feature_dim, hidden_dim, output_dim
            )
        self.local_embed = nn.Identity()
        
        if freeze_vit:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.global_embed.parameters():
                param.requires_grad = False

    def vit_forward(self, x):
        return self.model(x)

    def forward(self, x, get_local=False):
        # All three model only returns global feature
        ret = self.vit_forward(x)
        return ret, ret, ret



class CausalLMEncoder(nn.Module):
    def __init__(self,
                 tokenizer: AutoTokenizer = None,
                 emb_dim: int = 2560,
                 output_dim: int = 512,
                 hidden_dim: int = 2048,
                 freeze_llm: bool = True,
                 agg_tokens: bool = False,
                 peft: str = None,
                 grad_ckpt: bool = False,
                 use_flash_attention_2: bool = False,
                 llm_type: str = 'gpt',
                 prompt_ft: bool = False,
                 linear_proj: bool = False,
                 train_embed: bool = False,
                 unlock_ln: bool = False,
                 num_freeze_blocks: int = 0,
                 total_steps: int = 40000,):
        super(CausalLMEncoder, self).__init__()

        self.llm_type = llm_type
        if self.llm_type == 'gpt':
            self.llm_name = "stanford-crfm/BioMedLM"
            model_param = {
                # "torch_dtype": torch.bfloat16,
                # "low_cpu_mem_usage": True,
            }
            emb_dim = 2560
        elif self.llm_type == 'llama':
            self.llm_name = "epfl-llm/meditron-7b"
            model_param = {
                "torch_dtype": torch.bfloat16,
                # "load_in_4bit": True,
                # "bnb_4bit_compute_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                # "device_map": "auto"
            }
            emb_dim = 4096
        self.last_n_layers = 1
        self.aggregate_method = "sum"
        self.embedding_dim = emb_dim
        self.output_dim = output_dim
        self.freeze_llm = freeze_llm
        self.agg_tokens = agg_tokens
        self.prompt_ft = prompt_ft
        # self.max_sent_num = 10

        self.model = AutoModelForCausalLM.from_pretrained(self.llm_name, token=MY_API_TOKEN,
                                                          use_flash_attention_2=use_flash_attention_2,
                                                          **model_param)
        # Remove the LM head
        self.model.lm_head = nn.Identity()

        if peft:
            self.model = self.get_peft_model(peft, total_steps)

        # Default CKPT failed in forward pass
        if grad_ckpt:
            if self.llm_type == "gpt":
                self.model.transformer.gradient_checkpointing_enable()
            elif self.llm_type == "llama":
                self.model.model.gradient_checkpointing_enable()

        if train_embed:
            for name, param in self.model.named_parameters():
                if self.llm_type == "gpt":
                    if "wte" in name:
                        param.requires_grad = True
                if self.llm_type == "llama":
                    if "embed_tokens" in name:
                        param.requires_grad = True

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = get_tokenizer(self.llm_type)

        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        if self.freeze_llm is True:
            print("Freezing llm model")
            for param in self.model.parameters():
                param.requires_grad = False

        if linear_proj:
            self.global_embed = nn.Linear(self.embedding_dim, output_dim)
        else:
            self.global_embed = GlobalEmbedding(
                self.embedding_dim, hidden_dim, self.output_dim)
        # Unused
        self.local_embed = LocalEmbedding(
            self.embedding_dim, hidden_dim, self.output_dim)
        self.global_embed.to(self.model.dtype)
        self.local_embed.to(self.model.dtype)
        
        if self.prompt_ft:
            print("Freezing full llm model")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.global_embed.parameters():
                param.requires_grad = False
            for param in self.local_embed.parameters():
                param.requires_grad = False
                
        if unlock_ln:
            print("Unlocking LayerNorm within pre-trained LLM")
            for name, param in self.model.named_parameters():
                if "ln" in name:
                        param.requires_grad = True
                        
        if num_freeze_blocks > 0:
            if self.llm_type == "gpt":
                print("Freeze first {} blocks in GPT model".format(num_freeze_blocks))
                for name, param in self.model.named_parameters():
                    for i in range(num_freeze_blocks):
                        if f"h.{i}." in name:
                            param.requires_grad = False

    def get_peft_model(self, peft, total_steps=40000):
        print(f"Using PEFT: {peft}")
        if self.llm_type == "gpt":
            target_modules = ["c_attn", "mlp.c_proj"]
            feedforward_modules=["mlp.c_proj"]
        elif self.llm_type == "llama":
            target_modules = ["q_proj", "v_proj"]
            feedforward_modules = ["down_proj"]

        inference_mode = self.prompt_ft
        if peft == "ia3":
            config = IA3Config(peft_type="ia3", task_type=TaskType.CAUSAL_LM, inference_mode=inference_mode,
                                target_modules=target_modules, feedforward_modules=feedforward_modules)
        elif peft == "lora":
            config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=inference_mode, r=8, 
                                lora_alpha=32, lora_dropout=0.1)
        elif peft == "prefix":
            config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, inference_mode=inference_mode,
                                        num_virtual_tokens=20)
        elif peft == "adalora":
            config = AdaLoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=inference_mode, target_r=8,
                                   init_r=12, tinit=500, tfinal=1500, beta1=0.85, beta2=0.85, 
                                   total_step=total_steps)
        peft_model = get_peft_model(self.model, config)
        print(peft_model.print_trainable_parameters())
        return peft_model

    def find_last_word_token(self, embeddings, caption_ids):
        """
        :param embeddings: bz, 1, S, C
        :param caption_ids: bz, S
        """

        bz, _, _, _ = embeddings.shape
        last_word_tokens = []
        eos_token = self.tokenizer.eos_token_id
        # print(caption_ids.shape)
        # print(embeddings.shape)
        for i in range(bz):
            # print(caption_ids[i, :])
            # Need to consider the prepending Tokens
            last_word_idx = 0
            for j in range(1, len(caption_ids[i, :]) + 1):
                # First eos token
                if caption_ids[i, -j] != eos_token:
                    break
                last_word_idx -= 1
            # last_word_idx = torch.argwhere(caption_ids[i, :] == eos_token)[0][0].item()
            # print(last_word_idx, caption_ids[i, last_word_idx])
            # print(embeddings[i, 0, last_word_idx, :])
            last_word_tokens.append(embeddings[i, 0, last_word_idx, :].unsqueeze(0))
        return torch.stack(last_word_tokens, dim=0)


    def forward(self, ids, attn_mask, inputs_embeds=None, get_local=False):
        if len(ids.shape) == 1:
            ids = ids.unsqueeze(0)
        outputs = self.model(input_ids=ids, attention_mask=attn_mask, 
                             output_attentions=True, return_dict=True)
        target_dtype = self.model.dtype

        last_layer_attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1).to(target_dtype)
        all_feat = outputs.logits.unsqueeze(1).to(target_dtype)

        sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]
        last_atten_pt = last_layer_attn.contiguous()

        # Causal LM: only the last word token is used as the report feature
        report_feat = self.find_last_word_token(all_feat, ids).contiguous()
        word_feat = all_feat[:, :, :].contiguous()

        if self.last_n_layers == 1:
            report_feat = report_feat[:, 0]
            word_feat = word_feat[:, 0]

        return report_feat, word_feat, last_atten_pt, sents