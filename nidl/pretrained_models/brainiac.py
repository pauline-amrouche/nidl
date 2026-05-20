##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import torch
from torch import nn

from nidl.volume.backbones import VisionTransformer3D


class BrainIAC(nn.Module):
    '''Implements the BrainIAC backbone vision encoder
    (https://www.nature.com/articles/s41593-026-02202-6)
    using nidl library. This does not include BrainIAC preprocessing
    nor finetuned models'''

    def __init__(self, checkpoint_path):
        '''checkpoint_path: path to BrainIAC backbone saved checkpoint which
        can be downloaded from https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&e=1&st=b9cnvwh8&dl=0'''
        super().__init__()

        # Inititalize ViT backbone
        self.backbone = VisionTransformer3D(
            img_size=96,
            patch_size=16,
            in_chans=1,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=False,
            class_token=False,  # disable CLS token, in BrainIAC the first token is used
            num_classes=0,  # backbone only
        )

        # Load state dict
        model_state_dict = torch.load(checkpoint_path)['state_dict']

        # Map original keys (MONAI) to nidl keys (timm)
        # Remap keys
        new_state_dict = {}

        for k, v in model_state_dict.items():

            # Ignore training projection head
            if k.startswith('projection_head'):
                continue

            # First strip 'backbone' from MONAI keys
            k = k.replace('backbone.', '')

            # patch embedding
            k = k.replace(
                "patch_embedding.patch_embeddings.",
                "patch_embed.proj."
            )

            # positional embeddings
            k = k.replace(
                "patch_embedding.position_embeddings",
                "pos_embed"
            )

            # attention projection
            k = k.replace(
                "attn.out_proj.",
                "attn.proj."
            )

            # Transformer block mlp
            k = k.replace(
                ".mlp.linear",
                ".mlp.fc"
            )

            new_state_dict[k] = v

        print('Loading checkpoint for BrainIAC pretrained model.')
        missing, unexpected = self.backbone.load_state_dict(
                    new_state_dict,
                    strict=True,
                    )
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing keys in checkpoint: {missing}")
        print("BrainIAC checkpoint loaded.", end='\n')

    def forward(self, x):
        features = self.backbone.forward_features(x)
        # Get features for first token as done in the BrainIAC model
        first_token = features[:, 0, :]  # Shape: [batch_size, 768]
        return first_token