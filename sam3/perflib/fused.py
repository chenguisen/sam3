# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch

addmm_act_op = torch.ops.aten._addmm_activation


def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
    x = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        x = torch.nn.functional.relu(x)
    elif activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        x = torch.nn.functional.gelu(x)
    else:
        raise ValueError(f"Unexpected activation {activation}")
    return x
