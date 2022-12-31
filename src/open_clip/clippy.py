import torch
from torch import nn
from transformers import AutoModel
from .hf_model import MeanPooler

class ClippyVisionModel(nn.Module):
    """
    Same as modified resnet, but with maxpool instead of attention module.
    """

    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        self.model = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                nn.Conv2d(2048, output_dim, kernel_size=1, stride=1))
        self.image_size = 256

    def forward(self, x):
        features = self.model(x)
        return features.amax(dim=[2, 3])

class ClippyTextEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        #TODO: try sentence-t5
        self.transformer = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        # model = SentenceTransformer("sentence-transformers/sentence-t5-base")
        self.proj = nn.Linear(384, embed_dim, bias=False)
        self.pooler = MeanPooler()
        self.config = self.transformer.config

    def forward(self, x):
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        return self.proj(pooled_out)

