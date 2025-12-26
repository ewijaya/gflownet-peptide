"""ESM-2 based stability predictor for peptide/protein sequences.

This module implements a stability predictor using ESM-2 embeddings as input
to an MLP regression head. The model predicts thermal stability (melting
temperature) based on sequence information.

Architecture:
    Input: Peptide/protein sequence
        ↓
    ESM-2 (frozen): Extract mean-pooled embeddings
        ↓
    MLP: embedding_dim → hidden_dims → 1
        ↓
    Output: Stability score (normalized)
"""

import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StabilityPredictor(nn.Module):
    """ESM-2 based stability predictor.

    Uses frozen ESM-2 embeddings as input to an MLP head for predicting
    thermal stability. The model is designed to be trained on datasets
    like FLIP stability (meltome) task.

    Attributes:
        esm: ESM-2 language model (frozen)
        alphabet: ESM-2 tokenizer
        head: MLP regression head
        embed_dim: Dimension of ESM-2 embeddings
        repr_layer: Which layer to extract representations from
    """

    def __init__(
        self,
        esm_model: str = "esm2_t6_8M_UR50D",
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        freeze_esm: bool = True,
        device: Optional[str] = None,
    ):
        """Initialize stability predictor.

        Args:
            esm_model: Name of ESM-2 model to use. Options:
                - esm2_t6_8M_UR50D (8M params, 320 dim, fastest)
                - esm2_t12_35M_UR50D (35M params, 480 dim)
                - esm2_t33_650M_UR50D (650M params, 1280 dim, most accurate)
            hidden_dims: Hidden layer dimensions for MLP head.
                Default: [256, 128]
            dropout: Dropout rate for MLP layers
            freeze_esm: Whether to freeze ESM-2 parameters
            device: Device to load model on. Auto-detected if None.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.esm_model_name = esm_model
        self.freeze_esm = freeze_esm
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load ESM-2
        self._load_esm(esm_model)

        if freeze_esm:
            for param in self.esm.parameters():
                param.requires_grad = False
            self.esm.eval()

        # Build MLP head
        layers = []
        in_dim = self.embed_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))

        self.head = nn.Sequential(*layers)

        logger.info(f"StabilityPredictor initialized with {esm_model}, "
                    f"hidden_dims={hidden_dims}, freeze_esm={freeze_esm}")

    def _load_esm(self, esm_model: str):
        """Load ESM-2 model and set embedding dimensions."""
        import esm

        logger.info(f"Loading ESM-2 model: {esm_model}")

        if esm_model == "esm2_t6_8M_UR50D":
            self.esm, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.embed_dim = 320
            self.repr_layer = 6
        elif esm_model == "esm2_t12_35M_UR50D":
            self.esm, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.embed_dim = 480
            self.repr_layer = 12
        elif esm_model == "esm2_t33_650M_UR50D":
            self.esm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.embed_dim = 1280
            self.repr_layer = 33
        else:
            raise ValueError(f"Unknown ESM model: {esm_model}")

        self.batch_converter = self.alphabet.get_batch_converter()

    def get_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """Extract ESM-2 embeddings for sequences.

        Args:
            sequences: List of amino acid sequences

        Returns:
            Tensor of shape (batch_size, embed_dim) with mean-pooled embeddings
        """
        # Prepare data for ESM
        data = [(f"seq{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(next(self.esm.parameters()).device)

        # Forward pass through ESM-2
        with torch.no_grad() if self.freeze_esm else torch.enable_grad():
            results = self.esm(
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=False
            )

        # Mean pool over sequence positions (excluding BOS and EOS tokens)
        embeddings = []
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            # Tokens are: [BOS, seq..., EOS, PAD...]
            # We want indices 1 to seq_len+1 (exclusive of EOS)
            emb = results["representations"][self.repr_layer][i, 1:seq_len+1, :]
            embeddings.append(emb.mean(dim=0))

        return torch.stack(embeddings)

    def forward(self, sequences: Union[str, List[str]]) -> torch.Tensor:
        """Predict stability for sequences.

        Args:
            sequences: Single sequence or list of sequences

        Returns:
            Tensor of shape (batch_size,) with stability predictions
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        embeddings = self.get_embeddings(sequences)
        predictions = self.head(embeddings).squeeze(-1)

        return predictions

    def predict(self, sequences: Union[str, List[str]]) -> List[float]:
        """Predict stability scores (convenience method).

        Args:
            sequences: Single sequence or list of sequences

        Returns:
            List of stability scores
        """
        self.eval()
        with torch.no_grad():
            preds = self.forward(sequences)
        return preds.cpu().tolist()

    def to(self, device: Union[str, torch.device]) -> 'StabilityPredictor':
        """Move model to device."""
        self.device = str(device)
        self.esm = self.esm.to(device)
        self.head = self.head.to(device)
        return super().to(device)


class BindingPredictor(StabilityPredictor):
    """ESM-2 based binding predictor.

    Same architecture as StabilityPredictor but intended for binding
    affinity prediction. Currently only supports binary classification
    (binder vs non-binder) due to Propedia dataset limitations.

    For regression tasks, additional data with continuous binding affinities
    (e.g., from PDBbind) would be needed.
    """

    def __init__(
        self,
        esm_model: str = "esm2_t6_8M_UR50D",
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        freeze_esm: bool = True,
        device: Optional[str] = None,
        use_sigmoid: bool = True,
    ):
        """Initialize binding predictor.

        Args:
            esm_model: Name of ESM-2 model to use
            hidden_dims: Hidden layer dimensions for MLP head
            dropout: Dropout rate
            freeze_esm: Whether to freeze ESM-2
            device: Device to load model on
            use_sigmoid: Whether to apply sigmoid for binary classification
        """
        super().__init__(
            esm_model=esm_model,
            hidden_dims=hidden_dims,
            dropout=dropout,
            freeze_esm=freeze_esm,
            device=device,
        )
        self.use_sigmoid = use_sigmoid
        logger.info(f"BindingPredictor initialized, use_sigmoid={use_sigmoid}")

    def forward(self, sequences: Union[str, List[str]]) -> torch.Tensor:
        """Predict binding score for sequences.

        Args:
            sequences: Single sequence or list of sequences

        Returns:
            Tensor of shape (batch_size,) with binding predictions.
            If use_sigmoid=True, values are in [0, 1].
        """
        preds = super().forward(sequences)
        if self.use_sigmoid:
            preds = torch.sigmoid(preds)
        return preds
