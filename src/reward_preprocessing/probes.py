"""Train linear probes on reward nets."""

from torch import nn

# TODO document anything


class Probe(nn.Module):
    # inspired by
    # https://github.com/yukimasano/linear-probes/blob/master/eval_linear_probes.py
    def __init__(self, model, layer_name, attribute_dim):
        super(Probe, self).__init__()
        self.attribute_dim = attribute_dim
        self.model = model
        self.layer_name = layer_name
        self.probe_head = None
        out_features = 0
        for name, child in enumerate(model.named_children()):
            if name == self.layer_name:
                avg_pool = nn.AdaptiveAvgPool2d(1)
                flatten = nn.Flatten()
                fc = nn.Linear(out_features, attribute_dim)
                self.probe_head = nn.Sequential(avg_pool, flatten, fc)
            if hasattr(child, "out_features"):
                out_features = child.out_features
        if self.probe_head is None:
            raise ValueError(f"Could not find layer {layer_name} to probe")

    def forward(self, x):
        # go forward layer by layer thru self.model
        # once we hit layer_name, apply probe_head to x, then output that
        pass

    # then, when you optimize, can just optimize over probe_head.params
