"""
Reward Models for GFlowNet Peptide Generation.

Implements various reward components based on public data:
- StabilityReward: Trained on FLIP stability task
- BindingReward: Trained on Propedia binding data
- NaturalnessReward: Based on language model perplexity
- CompositeReward: Combines multiple rewards multiplicatively
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESMBackbone(nn.Module):
    """
    ESM-2 backbone for sequence encoding.

    Provides fixed embeddings from a pretrained protein language model.
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        freeze: bool = True,
    ):
        """
        Args:
            model_name: ESM-2 model to use
            freeze: If True, freeze ESM parameters
        """
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze

        # Lazy loading - will be initialized on first forward
        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self._device = None  # Track target device for lazy loading

        # ESM-2 embedding dimensions
        self._embed_dims = {
            "esm2_t6_8M_UR50D": 320,
            "esm2_t12_35M_UR50D": 480,
            "esm2_t30_150M_UR50D": 640,
            "esm2_t33_650M_UR50D": 1280,
            "esm2_t36_3B_UR50D": 2560,
        }
        self.embed_dim = self._embed_dims.get(model_name, 1280)

        # Representation layer to extract (based on model)
        self._repr_layers = {
            "esm2_t6_8M_UR50D": 6,
            "esm2_t12_35M_UR50D": 12,
            "esm2_t30_150M_UR50D": 30,
            "esm2_t33_650M_UR50D": 33,
            "esm2_t36_3B_UR50D": 36,
        }
        self._repr_layer = self._repr_layers.get(model_name, 33)

    def _load_model(self, device: torch.device):
        """Lazy load ESM model to specified device."""
        if self._model is None or self._device != device:
            try:
                import esm

                self._model, self._alphabet = esm.pretrained.load_model_and_alphabet(
                    self.model_name
                )
                self._batch_converter = self._alphabet.get_batch_converter()

                # Move to target device
                self._model = self._model.to(device)
                self._device = device

                if self.freeze:
                    self._model.eval()
                    for param in self._model.parameters():
                        param.requires_grad = False

            except ImportError:
                raise ImportError(
                    "ESM is required. Install with: pip install fair-esm"
                )

    def _apply(self, fn):
        """Override to track target device for lazy loading.

        PyTorch's .to() calls _apply() internally, so we intercept here
        to detect device changes even when called via parent module.
        """
        # Try to detect device from the function
        # PyTorch's .to() creates a function that moves tensors
        try:
            test_tensor = torch.zeros(1)
            result = fn(test_tensor)
            if result.device != test_tensor.device:
                self._device = result.device
                if self._model is not None:
                    self._model = self._model.to(self._device)
        except Exception:
            pass  # Not all _apply functions move tensors

        return super()._apply(fn)

    def to(self, device):
        """Override to track target device for lazy loading."""
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, torch.device):
            pass
        else:
            # Could be dtype or something else, let parent handle
            return super().to(device)

        self._device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return super().to(device)

    def forward(self, sequences: list[str]) -> torch.Tensor:
        """
        Encode sequences using ESM-2.

        Args:
            sequences: List of peptide sequences

        Returns:
            embeddings: Mean-pooled ESM embeddings [batch, embed_dim]
        """
        # Determine target device
        device = self._device if self._device is not None else torch.device("cpu")
        self._load_model(device)

        # Prepare batch
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self._batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        # Get ESM representations
        with torch.set_grad_enabled(not self.freeze):
            results = self._model(
                batch_tokens, repr_layers=[self._repr_layer], return_contacts=False
            )
            token_embeddings = results["representations"][self._repr_layer]

        # Mean pool over sequence (excluding special tokens)
        # ESM adds <cls> at start and <eos> at end
        sequence_embeddings = token_embeddings[:, 1:-1, :].mean(dim=1)

        return sequence_embeddings


class RewardHead(nn.Module):
    """MLP head for predicting a scalar property from embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        activation: str = "relu",
        output_transform: str = "exp",
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            n_layers: Number of MLP layers
            activation: Activation function ("relu", "gelu")
            output_transform: Transform for non-negative output ("exp", "softplus", "relu")
        """
        super().__init__()

        self.output_transform = output_transform

        # Build MLP
        layers = []
        current_dim = input_dim

        for i in range(n_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            current_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict scalar reward from embeddings.

        Args:
            embeddings: Input embeddings [batch, embed_dim]

        Returns:
            rewards: Non-negative rewards [batch]
        """
        raw = self.mlp(embeddings).squeeze(-1)

        # Apply non-negativity transform
        if self.output_transform == "exp":
            return torch.exp(raw)
        elif self.output_transform == "softplus":
            return F.softplus(raw)
        elif self.output_transform == "relu":
            return F.relu(raw) + 1e-6  # Small epsilon to avoid zero
        elif self.output_transform == "sigmoid":
            return torch.sigmoid(raw)
        else:
            return raw


class StabilityReward(nn.Module):
    """
    Stability reward based on FLIP benchmark.

    Predicts thermal stability (ΔΔG) from sequence.
    """

    def __init__(
        self,
        esm_model: str = "esm2_t33_650M_UR50D",
        hidden_dim: int = 256,
        freeze_esm: bool = True,
    ):
        super().__init__()

        self.backbone = ESMBackbone(esm_model, freeze=freeze_esm)
        self.head = RewardHead(
            input_dim=self.backbone.embed_dim,
            hidden_dim=hidden_dim,
            n_layers=2,
            output_transform="exp",
        )

    def forward(self, sequences: Union[str, list[str]]) -> torch.Tensor:
        """
        Compute stability reward.

        Args:
            sequences: Single sequence or list of sequences

        Returns:
            rewards: Stability rewards [batch]
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        embeddings = self.backbone(sequences)
        return self.head(embeddings)


class BindingReward(nn.Module):
    """
    Binding reward based on Propedia data.

    Predicts binding affinity from peptide sequence.
    Note: This is a simplified version that doesn't use target information.
    """

    def __init__(
        self,
        esm_model: str = "esm2_t33_650M_UR50D",
        hidden_dim: int = 256,
        freeze_esm: bool = True,
    ):
        super().__init__()

        self.backbone = ESMBackbone(esm_model, freeze=freeze_esm)
        self.head = RewardHead(
            input_dim=self.backbone.embed_dim,
            hidden_dim=hidden_dim,
            n_layers=2,
            output_transform="softplus",
        )

    def forward(self, sequences: Union[str, list[str]]) -> torch.Tensor:
        """Compute binding reward."""
        if isinstance(sequences, str):
            sequences = [sequences]

        embeddings = self.backbone(sequences)
        return self.head(embeddings)


class NaturalnessReward(nn.Module):
    """
    Naturalness reward based on language model perplexity.

    Low perplexity = more natural/protein-like = higher reward.
    """

    def __init__(
        self,
        esm_model: str = "esm2_t33_650M_UR50D",
        temperature: float = 10.0,
    ):
        """
        Args:
            esm_model: ESM model for perplexity computation
            temperature: Scaling factor for perplexity -> reward conversion
        """
        super().__init__()
        self.temperature = temperature

        # Will compute pseudo-perplexity from ESM
        self.backbone = ESMBackbone(esm_model, freeze=True)

    def forward(self, sequences: Union[str, list[str]]) -> torch.Tensor:
        """
        Compute naturalness reward.

        Args:
            sequences: Peptide sequence(s)

        Returns:
            rewards: Naturalness rewards in (0, 1]
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        # Get embeddings (as proxy for sequence quality)
        embeddings = self.backbone(sequences)

        # Use embedding norm as simple naturalness proxy
        # (well-embedded sequences are more natural)
        embedding_norms = torch.norm(embeddings, dim=-1)

        # Normalize and convert to reward
        # Higher norm = more typical protein = higher reward
        rewards = torch.sigmoid(embedding_norms / self.temperature)

        return rewards


class CompositeReward(nn.Module):
    """
    Composite reward combining multiple properties.

    R(x) = stability^w1 * binding^w2 * naturalness^w3

    Geometric mean ensures all properties must be good for high reward.
    """

    def __init__(
        self,
        esm_model: str = "esm2_t33_650M_UR50D",
        stability_weight: float = 1.0,
        binding_weight: float = 1.0,
        naturalness_weight: float = 0.5,
        freeze_esm: bool = True,
        share_backbone: bool = True,
    ):
        """
        Args:
            esm_model: ESM-2 model name
            stability_weight: Exponent for stability reward
            binding_weight: Exponent for binding reward
            naturalness_weight: Exponent for naturalness reward
            freeze_esm: Freeze ESM parameters
            share_backbone: Share ESM backbone across components
        """
        super().__init__()

        # Weights as buffer (not trainable by default)
        self.register_buffer(
            "weights",
            torch.tensor([stability_weight, binding_weight, naturalness_weight]),
        )

        # Shared backbone for efficiency
        if share_backbone:
            self.backbone = ESMBackbone(esm_model, freeze=freeze_esm)
            embed_dim = self.backbone.embed_dim

            self.stability_head = RewardHead(embed_dim, output_transform="exp")
            self.binding_head = RewardHead(embed_dim, output_transform="softplus")
            self.naturalness_head = RewardHead(embed_dim, output_transform="sigmoid")

            self._use_shared = True
        else:
            self.stability_model = StabilityReward(esm_model, freeze_esm=freeze_esm)
            self.binding_model = BindingReward(esm_model, freeze_esm=freeze_esm)
            self.naturalness_model = NaturalnessReward(esm_model)

            self._use_shared = False

    def forward(
        self,
        sequences: Union[str, list[str]],
        return_components: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """
        Compute composite reward.

        Args:
            sequences: Peptide sequence(s)
            return_components: If True, also return individual components

        Returns:
            rewards: Composite rewards [batch]
            components: Dict of individual rewards (if return_components=True)
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        if self._use_shared:
            # Shared backbone - single forward pass
            embeddings = self.backbone(sequences)

            stability = self.stability_head(embeddings)
            binding = self.binding_head(embeddings)
            naturalness = self.naturalness_head(embeddings)
        else:
            # Separate models
            stability = self.stability_model(sequences)
            binding = self.binding_model(sequences)
            naturalness = self.naturalness_model(sequences)

        # Geometric mean with weights
        # R = s^w1 * b^w2 * n^w3
        composite = (
            stability ** self.weights[0]
            * binding ** self.weights[1]
            * naturalness ** self.weights[2]
        )

        if return_components:
            components = {
                "stability": stability,
                "binding": binding,
                "naturalness": naturalness,
            }
            return composite, components

        return composite

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "CompositeReward":
        """Load pretrained reward model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract config if available
        config = checkpoint.get("config", {})
        config.update(kwargs)

        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    def save_pretrained(self, checkpoint_path: str, config: Optional[dict] = None):
        """Save reward model to checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {},
        }
        torch.save(checkpoint, checkpoint_path)
