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
    """
    BrainIAC vision backbone encoder.

    Loads pretrained BrainIAC backbone weights using nidl. The checkpoint
    must be downloaded separately from the
    `official BrainIAC release <https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&e=1&st=b9cnvwh8&dl=0>`_.
    Only the pretrained backbone is included; preprocessing pipelines and
    task-specific finetuned weights are not provided.

    References
    ----------
    Tak et al., "A generalizable foundation model for analysis of
    human brain MRI", Nature Neuroscience (2026).
    """

    def __init__(self, checkpoint_path):
        '''Initialize the BrainIAC backbone from a pretrained checkpoint.

        Parameters
        ----------
        checkpoint_path: str
            Path to the downloaded BrainIAC pretrained backbone checkpoint.
        '''
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
            class_token=False,  # BrainIAC uses first token as representation
            num_classes=0,
        )

        # Load state dict
        state_dict = torch.load(checkpoint_path)['state_dict']

        # Map original keys (MONAI) to nidl keys (timm)
        # Remap keys
        mapped_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith("projection_head"):
                continue
            
            new_key = key.replace("backbone.", "")
            new_key = new_key.replace("patch_embedding.patch_embeddings.",
                                      "patch_embed.proj.")
            new_key = new_key.replace("patch_embedding.position_embeddings",
                                      "pos_embed")
            new_key = new_key.replace("attn.out_proj.", "attn.proj.")
            new_key = new_key.replace(".mlp.linear", ".mlp.fc")

            mapped_state_dict[new_key] = value

        print('Loading checkpoint for BrainIAC pretrained model.')
        missing, unexpected = self.backbone.load_state_dict(
                    mapped_state_dict, strict=True)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing keys in checkpoint: {missing}")
        print("BrainIAC checkpoint loaded.", end='\n')

    def forward(self, x):
        '''Return the first token embedding from ViT backbone'''
        features = self.backbone.forward_features(x)
        first_token = features[:, 0, :]  # Shape: [batch_size, 768]
        return first_token