"""
E(n)-Equivariant Graph Neural Network (EGNN)

Implements the core equivariant message passing layers for molecular generation.

Key Properties:
- E(3) equivariance: Rotations and translations of input preserve output structure
- Coordinate updates: x_i^{l+1} = x_i^l + Σ_j (x_i - x_j) * φ(m_ij)
- Invariant features: h_i^{l+1} = φ(h_i^l, Σ_j m_ij)

Reference: E(n) Equivariant Graph Neural Networks (Satorras et al., 2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch import Tensor


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embeddings for diffusion timesteps.
    
    Maps scalar timestep t to a vector of dimension dim using:
        emb[2i] = sin(t / 10000^(2i/dim))
        emb[2i+1] = cos(t / 10000^(2i/dim))
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B,) or (B, 1) timesteps in [0, T]
        
        Returns:
            (B, dim) sinusoidal embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Compute embedding frequencies
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Reshape t if needed
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2:
            t = t.squeeze(-1)
        
        # Compute embeddings
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


class EGNN_Layer(nn.Module):
    """
    Single E(n)-Equivariant Graph Neural Network layer.
    
    Message Passing:
        m_ij = φ_e(h_i, h_j, ||x_i - x_j||², edge_attr)
        x_i' = x_i + Σ_j (x_i - x_j) * φ_x(m_ij)  [EQUIVARIANT UPDATE]
        h_i' = φ_h(h_i, Σ_j m_ij)                  [INVARIANT UPDATE]
    
    The coordinate update is key to equivariance:
    - (x_i - x_j) is a relative vector (translation invariant)
    - Multiplying by scalar φ_x(m_ij) preserves rotation equivariance
    """
    
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 16,
        act_fn: nn.Module = nn.SiLU(),
        attention: bool = True,
        normalize: bool = True,
        tanh: bool = False,
        coords_weight: float = 1.0
    ):
        """
        Args:
            hidden_dim: Node feature dimension
            edge_dim: Edge feature dimension
            act_fn: Activation function
            attention: Use attention mechanism
            normalize: Normalize coordinate updates
            tanh: Apply tanh to coordinate updates (for stability)
            coords_weight: Scale factor for coordinate updates
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        self.coords_weight = coords_weight
        
        # Edge MLP: combines sender, receiver, distance, edge features
        edge_input_dim = hidden_dim * 2 + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        
        # Coordinate update MLP: outputs scalar
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False)
        )
        # Initialize to small values for stability
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)
        
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention
        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of EGNN layer.
        
        Args:
            h: (N, hidden_dim) node features
            x: (N, 3) node coordinates
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) edge features
        
        Returns:
            Tuple of (h_out, x_out):
                h_out: (N, hidden_dim) updated node features
                x_out: (N, 3) updated coordinates
        """
        row, col = edge_index
        
        # Compute relative vectors and distances
        coord_diff = x[row] - x[col]  # (E, 3)
        radial = (coord_diff ** 2).sum(dim=-1, keepdim=True)  # (E, 1)
        
        # Build edge input
        if edge_attr is not None:
            edge_input = torch.cat([h[row], h[col], radial, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([h[row], h[col], radial], dim=-1)
        
        # Compute messages
        m_ij = self.edge_mlp(edge_input)  # (E, hidden_dim)
        
        # Apply attention if enabled
        if self.attention:
            att = self.att_mlp(m_ij)  # (E, 1)
            m_ij = m_ij * att
        
        # === Coordinate Update (Equivariant) ===
        coord_weights = self.coord_mlp(m_ij)  # (E, 1)
        
        if self.tanh:
            coord_weights = torch.tanh(coord_weights)
        
        # Normalize by distance to prevent explosions
        if self.normalize:
            norm = torch.sqrt(radial + 1e-8)
            coord_diff = coord_diff / norm
        
        # Compute coordinate update
        coord_update = coord_diff * coord_weights  # (E, 3)
        
        # Aggregate coordinate updates
        x_out = x.clone()
        x_out = x_out.index_add(0, row, coord_update * self.coords_weight)
        
        # === Node Update (Invariant) ===
        # Aggregate messages
        m_i = torch.zeros_like(h)
        m_i = m_i.index_add(0, row, m_ij)
        
        # Update node features
        h_input = torch.cat([h, m_i], dim=-1)
        h_out = h + self.node_mlp(h_input)  # Residual connection
        
        return h_out, x_out


class EGNN(nn.Module):
    """
    Full E(n)-Equivariant Graph Neural Network.
    
    Stacks multiple EGNN layers with residual connections.
    Includes timestep conditioning for diffusion models.
    """
    
    def __init__(
        self,
        in_node_dim: int,
        hidden_dim: int,
        out_node_dim: int,
        n_layers: int,
        edge_dim: int = 16,
        attention: bool = True,
        normalize: bool = True,
        tanh: bool = False,
        dropout: float = 0.0,
        time_emb_dim: int = 64
    ):
        """
        Args:
            in_node_dim: Input node feature dimension
            hidden_dim: Hidden dimension
            out_node_dim: Output node feature dimension
            n_layers: Number of EGNN layers
            edge_dim: Edge feature dimension
            attention: Use attention in layers
            normalize: Normalize coordinate updates
            tanh: Apply tanh to coordinate updates
            dropout: Dropout rate
            time_emb_dim: Timestep embedding dimension
        """
        super().__init__()
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.node_embed = nn.Linear(in_node_dim, hidden_dim)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # EGNN layers
        self.layers = nn.ModuleList([
            EGNN_Layer(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                attention=attention,
                normalize=normalize,
                tanh=tanh
            )
            for _ in range(n_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Output projection
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_node_dim)
        )
        
        # Coordinate output (scale to predict noise)
        self.coord_out = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.coord_out.weight)
    
    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        t: Tensor,
        edge_attr: Optional[Tensor] = None,
        return_features: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of EGNN.
        
        Args:
            h: (N, in_node_dim) input node features
            x: (N, 3) input coordinates
            edge_index: (2, E) edge indices
            t: (B,) timesteps
            edge_attr: (E, edge_dim) edge features
            return_features: Also return hidden features
        
        Returns:
            Tuple of (h_out, x_out):
                h_out: (N, out_node_dim) output node features (type predictions)
                x_out: (N, 3) output coordinates (noise predictions)
        """
        # Input embedding
        h = self.node_embed(h)
        
        # Add timestep embedding (broadcast to all nodes)
        # Assume batch info available via node assignment
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        
        # For single graph or batched, broadcast appropriately
        if t_emb.size(0) == 1:
            h = h + t_emb
        else:
            # Need batch assignment - for now broadcast
            h = h + t_emb.mean(dim=0, keepdim=True)
        
        # Original coordinates for residual
        x_init = x
        
        # Process through layers
        for i, (layer, ln) in enumerate(zip(self.layers, self.layer_norms)):
            h_new, x = layer(h, x, edge_index, edge_attr)
            h = ln(h_new)
            h = self.dropout(h)
        
        # Output projections
        h_out = self.out_mlp(h)  # (N, out_node_dim)
        
        # Coordinate prediction as noise
        x_out = x - x_init  # Predict change from input
        
        if return_features:
            return h_out, x_out, h
        
        return h_out, x_out


class ConditionalEGNN(nn.Module):
    """
    Conditional EGNN for pocket-conditioned ligand generation.
    
    Architecture:
    1. Pocket encoder (shallow, cached)
    2. Ligand denoiser (deep)
    3. Cross-attention between ligand and pocket
    """
    
    def __init__(
        self,
        in_node_dim: int,
        hidden_dim: int,
        out_node_dim: int,
        n_layers: int,
        pocket_layers: int = 2,
        edge_dim: int = 16,
        attention: bool = True,
        dropout: float = 0.1,
        time_emb_dim: int = 64
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Pocket encoder (shallow - runs once per sample)
        self.pocket_encoder = EGNN(
            in_node_dim=in_node_dim,
            hidden_dim=hidden_dim,
            out_node_dim=hidden_dim,
            n_layers=pocket_layers,
            edge_dim=edge_dim,
            attention=attention,
            dropout=0.0,  # No dropout for pocket
            time_emb_dim=time_emb_dim
        )
        
        # Ligand denoiser (deep)
        self.ligand_denoiser = EGNN(
            in_node_dim=in_node_dim,
            hidden_dim=hidden_dim,
            out_node_dim=out_node_dim,
            n_layers=n_layers,
            edge_dim=edge_dim,
            attention=attention,
            dropout=dropout,
            time_emb_dim=time_emb_dim
        )
        
        # Cross-attention layers
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
            for _ in range(n_layers // 2)
        ])
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output heads
        self.type_head = nn.Linear(hidden_dim, out_node_dim)
        self.coord_head = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.coord_head.weight)
        nn.init.zeros_(self.coord_head.bias)
        
        # Pocket cache
        self._pocket_cache = None
    
    def encode_pocket(
        self,
        pocket_h: Tensor,
        pocket_x: Tensor,
        pocket_edge_index: Tensor,
        pocket_edge_attr: Optional[Tensor] = None
    ) -> Tensor:
        """
        Encode pocket features (cached after first call).
        """
        # Use dummy timestep for pocket (t=0)
        t = torch.zeros(1, device=pocket_h.device)
        
        _, _, pocket_emb = self.pocket_encoder(
            pocket_h, pocket_x, pocket_edge_index, t, pocket_edge_attr,
            return_features=True
        )
        
        return pocket_emb
    
    def forward(
        self,
        lig_h: Tensor,
        lig_x: Tensor,
        lig_edge_index: Tensor,
        pocket_h: Tensor,
        pocket_x: Tensor,
        pocket_edge_index: Tensor,
        t: Tensor,
        lig_edge_attr: Optional[Tensor] = None,
        pocket_edge_attr: Optional[Tensor] = None,
        use_pocket_cache: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with pocket conditioning.
        
        Args:
            lig_h: (N, in_dim) ligand node features
            lig_x: (N, 3) ligand coordinates (noised)
            lig_edge_index: (2, E) ligand edges
            pocket_h: (M, in_dim) pocket node features
            pocket_x: (M, 3) pocket coordinates
            pocket_edge_index: (2, E') pocket edges
            t: (B,) timesteps
            use_pocket_cache: Use cached pocket encoding
        
        Returns:
            Tuple of (type_pred, coord_pred):
                type_pred: (N, out_dim) noise prediction for types
                coord_pred: (N, 3) noise prediction for coordinates
        """
        # Encode pocket
        if use_pocket_cache and self._pocket_cache is not None:
            pocket_emb = self._pocket_cache
        else:
            pocket_emb = self.encode_pocket(pocket_h, pocket_x, pocket_edge_index, pocket_edge_attr)
            if use_pocket_cache:
                self._pocket_cache = pocket_emb.detach()
        
        # Encode ligand
        _, _, lig_emb = self.ligand_denoiser(
            lig_h, lig_x, lig_edge_index, t, lig_edge_attr,
            return_features=True
        )
        
        # Cross-attention (ligand attends to pocket)
        for attn in self.cross_attention:
            lig_emb_attn, _ = attn(
                lig_emb.unsqueeze(0),
                pocket_emb.unsqueeze(0),
                pocket_emb.unsqueeze(0)
            )
            lig_emb = lig_emb + lig_emb_attn.squeeze(0)
        
        # Output predictions
        type_pred = self.type_head(lig_emb)
        coord_pred = self.coord_head(lig_emb)
        
        return type_pred, coord_pred
    
    def clear_pocket_cache(self):
        """Clear the pocket cache (call between different pockets)."""
        self._pocket_cache = None
