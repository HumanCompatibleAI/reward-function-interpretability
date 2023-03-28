"""Train linear probes on reward nets."""

import torch as th
from torch import nn

# TODO document anything


class Probe(nn.Module):
    # inspired by
    # https://github.com/yukimasano/linear-probes/blob/master/eval_linear_probes.py
    # when you optimize, can just optimize over probe_head.params
    def __init__(self, model: nn.Module, layer_name: str, attribute_dim: int):
        super(Probe, self).__init__()
        self.attribute_dim = attribute_dim
        self.model = model
        self.layer_name = layer_name
        self.probe_head = None
        x = th.zeros(1, 3, 64, 64)
        for name, child in enumerate(self.model.named_children()):
            x = child.forward(x)
            if name == self.layer_name:
                avg_pool = nn.AdaptiveAvgPool2d(1)
                flatten = nn.Flatten()
                fc = nn.Linear(x.size(1), attribute_dim)
                self.probe_head = nn.Sequential(avg_pool, flatten, fc)
        if self.probe_head is None:
            raise ValueError(f"Could not find layer {self.layer_name} to probe")

    def forward(self, x):
        for name, child in enumerate(self.model.named_children()):
            x = child.forward(x)
            if name == self.layer_name:
                return self.probe_head(x)
        assert False, f"Could not find layer {self.layer_name} to probe."

