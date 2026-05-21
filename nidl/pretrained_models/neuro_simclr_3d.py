##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import torch
from torch import nn

from nidl.volume.backbones import resnet18


class NeuroSimCLR3D(nn.Module):
    """
    3D-Neuro-SimCLR vision backbone encoder.

    Loads pretrained 3D-Neuro-SimCLR backbone weights using nidl. The
    checkpoint must be downloaded separately from the
    `official 3D-Neuro-SimCLR release <https://github.com/emilykaczmarek/3D-Neuro-SimCLR/releases/download/v1.0.0/simclr_3d_brain_foundation.tar>`_.
    The tar file should NOT be extracted.

    References
    ----------
    Kaczmarek et al., "Building a General SimCLR Self-Supervised Foundation
    Model Across Neurological Diseases to Advance 3D Brain MRI Diagnoses",
    ICCV Workshop CVAMD (2025).
    """

    def __init__(self, checkpoint_path):
        '''Initialize the 3D-Neuro-SimCLR backbone from a pretrained
        checkpoint.

        Parameters
        ----------
        checkpoint_path: str
            Path to the downloaded 3D-Neuro-SimCLR pretrained backbone
            checkpoint.
        '''
        super().__init__()

        # Inititalize ResNet18 backbone and match 3D-Neuro-SimCLR configuration
        self.backbone = resnet18(in_channels=1)
        # We need to modify the first convolutional layer to set the stride
        # to 1 to match the original configuration (vs 2 by default in nidl)
        self.backbone.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7),
                                    stride=(1, 1, 1),
                                    padding=(3, 3, 3), bias=False)
        # The original configuration has no final fully connected layer
        self.backbone.embedding = nn.Identity()

        # Load checkpoint and weights
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = ckpt['model_state_dict']

        # Map original keys to nidl keys
        mapped_state_dict = {}
        # Checkpoint keys: "module.encoder.<key>" and "module.projector.<key>"
        # Strip both "module." and "encoder." prefixes; skip projector weights
        for key, value in state_dict.items():
            if "projector" in key:
                continue
            new_key = key.replace("module.encoder.", "")
            mapped_state_dict[new_key] = value

        missing, unexpected = self.backbone.load_state_dict(
            mapped_state_dict, strict=False)

        # There should be no missing keys
        if missing:
            raise RuntimeError(f"Missing keys in checkpoint: {missing}")
        # The bias in downsample layers placed right before a batchnorm layer
        # is omitted in the nidl implementation. This does not change the
        # result as the batch norm layer centers values and therefore the bias
        # is useless.
        if unexpected:
            truly_unexpected = [key for key in unexpected if
                                'downsample.0.bias' not in key]
            if truly_unexpected:
                raise RuntimeError("Unexpected keys in checkpoint:"
                                f"{truly_unexpected}")
        
        print("3D-Neuro-SimCLR checkpoint loaded.", end='\n')

    def forward(self, x):
        return self.backbone(x)