import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from typing import List
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
        # print("x:", x.shape)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )
        return x


class ChainPromptLearner(nn.Module):
    def __init__(
        self,
        cfg,
        classnames,
        clip_model,
        chain_length: int = 3,
        chain_width: int = 3,
        chain_dynamic: bool = False,
    ):
        super().__init__()
        n_cls = len(classnames)  # 101
        n_ctx = cfg.TRAINER.COCOOP.N_CTX  # 4
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT  # 'a photo of a'
        dtype = clip_model.dtype  # torch.float16

        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512
        vis_dim = clip_model.visual.output_dim  # 512

        clip_imsize = clip_model.visual.input_resolution  # 224
        cfg_imsize = cfg.INPUT.SIZE[0]  # 224
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))  # 4
            prompt = clip.tokenize(ctx_init)  # [1,77]

            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)  # [1,77,512]
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]  # [4,512]
            prompt_prefix = ctx_init
        else:
            # NOTE this task we use
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)  # [4,512]
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.n_ctx = n_ctx
        self.clip_model_token_embedding = clip_model.token_embedding
        self.dtype = dtype

        self.chain_dynamic = chain_dynamic
        self.chain_length = chain_length
        self.chain_width = chain_width

        # learned chain prompts
        # self.ctx = nn.Parameter(ctx_vectors)  #[4,512]
        # self.chain_ctx = nn.ModuleList([
        #     nn.Parameter(ctx_vectors) for _ in range(num_chains)
        # ])
        for i in range(self.chain_length):  # sequential
            for j in range(self.chain_width):  # parallel
                setattr(self, f"ctx_s{i}_p{j}", nn.Parameter(ctx_vectors))

        self.parallel_chain_weights = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 16, bias=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(vis_dim // 16, vis_dim, bias=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(vis_dim, self.chain_width, bias=False),
            nn.Sigmoid(),
        )

        # meta token
        # self.meta_net = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(vis_dim, vis_dim // 16)),  # vis_dim:512
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        # ]))  # [512,32],[32,512]

        self.meta_net = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "linear1",
                                nn.Linear(vis_dim, vis_dim // 16),
                            ),  # vis_dim:512
                            ("relu", nn.ReLU(inplace=True)),
                            ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                        ]
                    )
                )
            ]
            * self.chain_length
        )

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()
            self.parallel_chain_weights.half()
            self.sequential_chain_weights.half()


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
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features, label_text: List[str]):
        # (n_ctx, ctx_dim)

        # single prompt
        # ctx = self.ctx   # [4,512]
        # ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        # ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)

        tokenized_prompts = torch.cat(
            [clip.tokenize(p, truncate=True) for p in label_text]
        ).cuda()  # (n_sample, n_tkn) [101,77]  -> n_cls -> n_sample

        with torch.no_grad():
            embedding = self.clip_model_token_embedding(tokenized_prompts).type(
                self.dtype
            )  # (n_cls, n_tkn, vis_dim)[101,77,512]
        # print("tokenized_prompts:", tokenized_prompts.shape)  # [2, 77]
        # print("embedding:", embedding.shape)  # [2, 77, 512]

        token_prefix = embedding[:, :1, :]
        token_suffix = embedding[:, 1 + self.n_ctx :, :]

        seq_weights = self.sequential_chain_weights(im_features)
        para_weights = self.parallel_chain_weights(im_features)

        return prompts, tokenized_prompts


def cross_entropy(preds, targets, reduction="none"):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.prompt_learner = ChainPromptLearner(
            cfg,
            classnames,
            clip_model,
            chain_length=2, # 
            chain_width=4, # [2,2], [2,1]
            chain_dynamic=False,
        )
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))  # [B,dim],[1,512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts, tokenized_prompts = self.prompt_learner(
            image_features,
            label,
        )  # [1,101,77,512]


        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = text_features @ image_features.T  # (2,2)

        # TODO if test, 1 text should compute sim with all images.

        # logits = logit_scale * logits
        if self.prompt_learner.training:
            images_similarity = image_features @ image_features.T
            texts_similarity = text_features @ text_features.T
            targets = F.softmax((images_similarity + texts_similarity) / 2, dim=-1)
            texts_loss = cross_entropy(logits, targets, reduction="none")
            images_loss = cross_entropy(logits.T, targets.T, reduction="none")

            # TODO what is loss func
            return ((images_loss + texts_loss) / 2.0).mean()
        return logits


@TRAINER_REGISTRY.register()
class CoCoOp_MulNetCOTMetaNet_TextRevieal(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        # classnames = self.dm.dataset.classnames
        classnames = []

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name or "clip_model_token_embedding" in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            # print("name,param:", name)
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "prompt_learner", self.model.prompt_learner, self.optim, self.sched
        )

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        device_count = 1
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)  # here we pass in label(text)
            optim.zero_grad()
            loss.sum().backward()  # loss.mean()
            optim.step()

        loss_summary = {"loss": loss.sum().item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        return input, label

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        # label = label.to(self.device)
        return input, label

    def model_inference(self, input, label):
        return self.model(input, label)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            # model_file = "model.pth.tar-" + str(epoch)
            model_file = model_file
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

            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    # override dataloader
    def build_data_loader(self):
        from dassl.data.data_manager import (
            build_data_loader,
            build_sampler,
            build_transform,
        )
        from datasets.flickr30k import Flickr30k

        trainset = Flickr30k(split="train", ratio=0.02)
        testset = Flickr30k(split="test")

        tfm_train = build_transform(self.cfg, is_train=True)
        tfm_test = build_transform(self.cfg, is_train=False)

        self.train_loader = build_data_loader(
            self.cfg,
            sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=trainset,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=None,
        )
        self.test_loader = build_data_loader(
            self.cfg,
            sampler_type=self.cfg.DATALOADER.TEST.SAMPLER,
            data_source=testset,
            batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None,
        )

    def run_epoch(self):
        from dassl.utils import MetricMeter, AverageMeter
        import datetime
        import time

        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            # zeroshot
            # loss_summary = self.model_inference(batch)

            # Normal conclude: train val test, 1shot, 2shot, 4shot, 8shot, 16shot
            loss_summary = self.forward_backward(batch)
            # print("loss_summary:",loss_summary)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
