import os
import sys
import argparse
import torch

from clip.simple_tokenizer import SimpleTokenizer
from clip import clip


def load_clip_to_cpu(backbone_name="RN50"):
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


parser = argparse.ArgumentParser()
parser.add_argument("--fpath", type=str, default="/home/dell/codes/CoOp_DG/output/coop/caltech101_normal/coop_caltech101_lr0.002_rn50_ep200_batch32_20230321/prompt_learner/model.pth.tar-200",
                    help="Path to the learned prompt")
parser.add_argument("--topk", type=int, default=3,
                    help="Select top-k similar words")
args = parser.parse_args()

fpath = args.fpath
topk = args.topk

assert os.path.exists(fpath)

print(f"Return the top-{topk} matched words")

tokenizer = SimpleTokenizer()
clip_model = load_clip_to_cpu()
token_embedding = clip_model.token_embedding.weight  #[49408,512]
print(f"Size of token embedding: {token_embedding.shape}")

prompt_learner = torch.load(fpath, map_location="cpu")["state_dict"]  #ctx:[16,512],token_prefix:[100,1,512],token_suffix:[100,60,512]
ctx = prompt_learner["ctx"]  #[16,512]
ctx = ctx.float()
print(f"Size of context: {ctx.shape}")

if ctx.dim() == 2:
    # Generic context
    distance = torch.cdist(ctx, token_embedding) #[16,49408]
    print(f"Size of distance matrix: {distance.shape}")
    sorted_idxs = torch.argsort(distance, dim=1) #[16,49408]
    sorted_idxs = sorted_idxs[:, :topk]          #[16,3]

    for m, idxs in enumerate(sorted_idxs):
        words = [tokenizer.decoder[idx.item()] for idx in idxs]   # 49408
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs] # 1.0563，1.0575，1.0577
        print(f"{m+1}: {words} {dist}")

elif ctx.dim() == 3:
    # Class-specific context
    raise NotImplementedError

# result:
# 1: ['âľ¨</w>', 'ðŁĴĸðŁĴĸðŁĴĸ</w>', 'nemo</w>'] ['1.0563', '1.0575', '1.0577']
# 2: ['descrip', 'broker', 'dareto'] ['1.2646', '1.2665', '1.2677']
# 3: ['auga</w>', 'paleo', 'resur'] ['1.2009', '1.2032', '1.2056']
# 4: ['caulfield</w>', 'foursquare</w>', 'cougar'] ['1.0588', '1.0646', '1.0651']
# 5: ['horseshoe</w>', 'rnc</w>', 'influenza</w>'] ['1.1550', '1.1553', '1.1556']
# 6: ['august</w>', 'alizer</w>', 'swim'] ['1.0366', '1.0398', '1.0398']
# 7: ['advises</w>', 'âĶ', 'tutor'] ['1.0071', '1.0220', '1.0236']
# 8: ['seeing</w>', 'à¸±</w>', 'oring</w>'] ['0.8449', '0.8473', '0.8483']
# 9: ['newyear', 'week', 'years'] ['0.8573', '0.8595', '0.8596']
# 10: ['thisi', 'paper', 'breakout</w>'] ['1.1700', '1.1727', '1.1729']
# 11: ['storms</w>', 'dits</w>', 'blackandwhite</w>'] ['1.1628', '1.1641', '1.1683']
# 12: ['bookreview</w>', 'blend</w>', 'broch'] ['1.5339', '1.5384', '1.5387']
# 13: ['diff', 'kla', 'ðŁĺĬðŁĺĬðŁĺĬðŁĺĬ'] ['1.1238', '1.1251', '1.1254']
# 14: ['attacking</w>', 'sively</w>', 'st'] ['1.3890', '1.3895', '1.3958']
# 15: ['ships</w>', 'w</w>', 'seaport</w>'] ['1.4673', '1.4738', '1.4793']
# 16: ['digitally</w>', 'aul', 'advant'] ['1.4459', '1.4510', '1.4516']