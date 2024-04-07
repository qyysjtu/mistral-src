import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate # nn.Linear(input_dim, num_experts), gate is a linear layer
        self.args = moe_args

    
    # foward pass of the Mixture of Experts layer
    def forward(self, inputs: torch.Tensor):
        # inputs is the output of the previous layer, gate_logits is the output of the gate
        gate_logits = self.gate(inputs)
        # topk returns the top k values and their indices
        # selected_experts is the indices of the top k values
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        # weights is the softmax of the gate logits, so the weights sum to 1
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        # Iterate over the experts and apply them to the inputs
        for i, expert in enumerate(self.experts):
            # Find the batch indices and the indices of the nth expert
            batch_idx, nth_expert = torch.where(selected_experts == i)
            # Apply the expert to the inputs for each batch
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        return results
