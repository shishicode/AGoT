import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

import sys

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)                 # CLS
        n_ctx = cfg.TRAINER.DPCLIP.N_CTX        # 4
        ctx_init = cfg.TRAINER.DPCLIP.CTX_INIT  # 'a photo of a'
        dtype = clip_model.dtype                # torch.float16 

        ctx_dim = clip_model.ln_final.weight.shape[0]     # 512
        vis_dim = clip_model.visual.output_dim            # 512

        clip_imsize = clip_model.visual.input_resolution  # 224
        cfg_imsize = cfg.INPUT.SIZE[0]                    # 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
       
        # num_domain_tokens = cfg.TRAINER.DPLCLIP.NUM_DOMAIN_TOKENS
        num_domain_tokens = 16
        classnames = [name.replace("_", " ") for name in classnames]
        # prompt_dplclip = torch.cat([clip.tokenize(f'a photo of a{ppt}') for ppt in classnames])
        # print('prompt_dplclip',prompt_dplclip)

        prompt_prefix = ' '.join(['X'] * num_domain_tokens )
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]  # XXXXXXXXXXXXXXXX Alarm Clock.
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # [CLS,77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # [CLS,77,512]

        self.register_buffer("token_prefix", embedding[:, :1, :])           # SOS      [CLS,1,512]
        self.register_buffer("token_suffix", embedding[:, 1 + num_domain_tokens:, :])  # CLS, EOS [CLS,72,512]

        self.domain_network = nn.Sequential(OrderedDict([
            ("input", nn.Linear(vis_dim, vis_dim )),  # vis_dim:[512,512]
            ("dropout", nn.Dropout(p=0.1, inplace=False)),
            ("relu1", nn.ReLU(inplace=True)),
            ("hiddens", nn.Linear(vis_dim, vis_dim )),
            ("relu2", nn.ReLU(inplace=True)),
            ("output", nn.Linear(vis_dim, vis_dim * num_domain_tokens ))  # [512,8192]
        ]))

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.domain_network.apply(init_weights)

        if cfg.TRAINER.DPCLIP.PREC == "fp16":
            self.domain_network.half()

        self.n_cls = n_cls  # 100
        self.n_ctx = n_ctx  # 16
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor [CLS,77]
        self.num_domain_tokens = num_domain_tokens
        # self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix  #[CLS,1,512]
        suffix = self.token_suffix  #[CLS,72,512]
        # ctx = self.ctx   # [4,512], # (n_ctx, ctx_dim)
                            
        domain_features = self.domain_network(im_features)  #[B,512] -> [B,8192]
        mean_domain_features = domain_features.mean(dim=0, keepdim=True)

        _mean_domain_features = mean_domain_features.repeat_interleave(self.n_cls, dim=0)  # [CLS,8192]
        domain_feature = _mean_domain_features.reshape(-1, self.num_domain_tokens, im_features.shape[1])  # [CLS,16,512]
        prompts = self.construct_prompts(domain_feature, prefix, suffix)  # (n_cls, n_tkn, ctx_dim) [CLS,77,512]
        # prompts.append(pts_i)  # -> 1,n_ctx
        # prompts = torch.stack(prompts)  # -> batch,n_ctx [CLS,77,512]->[1,CLS,77,512]
        
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts  # [CLS,77]
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts # [CLS,77]
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))  #[B,512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)  #[CLS,77,512]
        
        # logits = []
        # for pts_i, imf_i in zip(prompts, image_features):
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # [CLS,512]
        logits = logit_scale * image_features @ text_features.t()  # [B,CLS]
        # logits.append(l_i)
        # logits = torch.stack(logits)
        
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)
        
        return logits


@TRAINER_REGISTRY.register()
class DPLCLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.DPCLIP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.DPCLIP.PREC == "fp32" or cfg.TRAINER.DPCLIP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            #print("name,param:", name)  
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.DPCLIP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.DPCLIP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
