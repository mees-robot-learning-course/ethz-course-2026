"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn

VALID_BACKBONES = {"mlp"}  # TODO: Add more backbones here

class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        activation: type[nn.Module] = nn.GELU,
        use_layernorm: bool = False
        ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_layernorm = use_layernorm

        self.layers = nn.ModuleList()

        for i in range(depth):
            d_out = out_dim if i == depth - 1 else hidden_dim
            self.layers.append(nn.Linear(in_dim, d_out))
            if i != depth - 1:
                if use_layernorm:
                    self.layers.append(nn.LayerNorm(d_out))
                self.layers.append(activation())
            in_dim = d_out

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        chunk_size: int,
        backbone: str = "mlp",
        d_model: int = 128,
        depth: int = 2,
        ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        assert isinstance(backbone, str) and backbone, (
            "ObstaclePolicy requires a non-empty backbone string."
        )
        
        self.d_model = int(d_model)
        self.depth = int(depth)

        # Define loss function
        self.loss_fn = nn.MSELoss()

        # Select backbone
        if backbone not in VALID_BACKBONES:
            raise ValueError(
                f"Unknown backbone: {backbone}. Supported backbones: {sorted(VALID_BACKBONES)}"
            )
        if backbone == "mlp":
            self.backbone = MLP(
                state_dim,
                chunk_size * action_dim,
                hidden_dim=self.d_model,
                depth=self.depth,
                use_layernorm=True,
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        assert state.ndim == 2 and state.shape[1] == self.state_dim, "State must have shape (B, state_dim)"
            
        B = state.shape[0]
        pred = self.backbone(state)
        pred = pred.reshape(B, self.chunk_size, self.action_dim)
        return pred

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        # Forward pass
        pred_action_chunk = self(state)
        assert pred_action_chunk.shape == action_chunk.shape, "Predicted and actual action chunks must have the same shape"
        
        # Compute loss
        loss = self.loss_fn(pred_action_chunk, action_chunk)
        return loss

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self(state)



# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        raise NotImplementedError


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    backbone: str | None = None,
    d_model: int = 128,
    depth: int = 2,
    # TODO,
) -> BasePolicy:
    if policy_type == "obstacle":
        if backbone is None:
            backbone = "mlp"
        assert isinstance(backbone, str) and backbone, (
            "build_policy() requires a non-empty 'backbone' for obstacle policy."
        )
        return ObstaclePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            backbone=backbone,
            d_model=d_model,
            depth=depth,
            # TODO: Build with your chosen specifications
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
