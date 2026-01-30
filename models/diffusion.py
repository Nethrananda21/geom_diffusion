"""
DDPM Diffusion Core

Implements Denoising Diffusion Probabilistic Models (DDPM) with:
- Cosine noise schedule (better for molecules than linear)
- Epsilon parameterization (predict noise)
- Classifier-free guidance for conditional generation
- Separate handling for coordinates (equivariant) and types (invariant)

Reference: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from torch import Tensor
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Configuration for diffusion process."""
    timesteps: int = 500  # T4: 500, A100: 1000
    schedule: str = "cosine"  # "linear" or "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    parameterization: str = "eps"  # "eps" (predict noise) or "x0" (predict data)
    
    # Classifier-free guidance
    guidance_dropout: float = 0.1  # Dropout pocket conditioning during training
    guidance_scale: float = 2.0  # Scale at inference


def get_linear_schedule(timesteps: int, beta_start: float, beta_end: float) -> Tensor:
    """Linear noise schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


def get_cosine_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    """
    Cosine noise schedule.
    
    Better than linear for molecular generation:
    - Slower noise increase at start preserves structure longer
    - Gentler transition to full noise
    
    From "Improved Denoising Diffusion Probabilistic Models"
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    
    # Cosine schedule
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    # Convert to betas
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    
    return betas


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process for molecular generation.
    
    Handles both:
    - Coordinate diffusion (continuous, E(3)-equivariant)
    - Type diffusion (categorical, can use continuous approximation)
    
    Training: noise data → model predicts noise → loss = MSE(pred, true_noise)
    Inference: sample noise → denoise T steps → clean data
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.timesteps = config.timesteps
        
        # Create noise schedule
        if config.schedule == "linear":
            betas = get_linear_schedule(config.timesteps, config.beta_start, config.beta_end)
        elif config.schedule == "cosine":
            betas = get_cosine_schedule(config.timesteps)
        else:
            raise ValueError(f"Unknown schedule: {config.schedule}")
        
        # Precompute diffusion quantities
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For forward process (adding noise)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
        # FIX Bug #1: Clamp alphas_cumprod before computing reciprocals
        # At high timesteps, alphas_cumprod can be ~1e-30, causing inf/NaN
        alphas_cumprod_clamped = torch.clamp(alphas_cumprod, min=1e-10)
        
        # For reverse process (removing noise) - use clamped values
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1 / alphas))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1 / alphas_cumprod_clamped))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1 / alphas_cumprod_clamped - 1))
        
        # Posterior mean coefficients
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
        
        # Posterior variance
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', torch.log(posterior_variance.clamp(min=1e-20)))
    
    def q_sample(
        self,
        x_0: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward diffusion: add noise to data.
        
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x_0: (N, D) clean data
            t: (B,) timesteps
            noise: Optional pre-sampled noise
        
        Returns:
            Tuple of (x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get coefficients for each sample
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # Add noise
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
        return x_t, noise
    
    def predict_x0_from_eps(
        self,
        x_t: Tensor,
        t: Tensor,
        eps: Tensor
    ) -> Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        """
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip * x_t - sqrt_recipm1 * eps
    
    def q_posterior_mean_variance(
        self,
        x_0: Tensor,
        x_t: Tensor,
        t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0).
        
        Returns mean, variance, log_variance.
        """
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        
        mean = coef1 * x_0 + coef2 * x_t
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        log_variance = self._extract(self.posterior_log_variance, t, x_t.shape)
        
        return mean, variance, log_variance
    
    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        clip_denoised: bool = True,
        **model_kwargs
    ) -> Dict[str, Tensor]:
        """
        Compute p(x_{t-1} | x_t) using model predictions.
        
        Args:
            model: Denoising model
            x_t: (N, D) noisy data
            t: (B,) timesteps
            clip_denoised: Clip predicted x_0 to [-1, 1]
            **model_kwargs: Additional model arguments
        
        Returns:
            Dict with 'mean', 'variance', 'log_variance', 'pred_x0'
        """
        # Model prediction
        model_output = model(x_t, t, **model_kwargs)
        
        if self.config.parameterization == "eps":
            # Model predicts noise
            if isinstance(model_output, tuple):
                eps_pred = model_output[0]
            else:
                eps_pred = model_output
            pred_x0 = self.predict_x0_from_eps(x_t, t, eps_pred)
        else:
            # Model predicts x0 directly
            pred_x0 = model_output
        
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1, 1)
        
        # Compute posterior
        mean, variance, log_variance = self.q_posterior_mean_variance(pred_x0, x_t, t)
        
        return {
            'mean': mean,
            'variance': variance,
            'log_variance': log_variance,
            'pred_x0': pred_x0
        }
    
    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        **model_kwargs
    ) -> Tensor:
        """
        Single reverse diffusion step: x_t → x_{t-1}.
        """
        out = self.p_mean_variance(model, x_t, t, **model_kwargs)
        
        noise = torch.randn_like(x_t)
        
        # No noise at t=0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        x_prev = out['mean'] + nonzero_mask * torch.exp(0.5 * out['log_variance']) * noise
        
        return x_prev
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        **model_kwargs
    ) -> Tensor:
        """
        Full reverse diffusion process: x_T → x_0.
        
        Args:
            model: Denoising model
            shape: Shape of output (N, D)
            device: Device to sample on
            **model_kwargs: Additional model arguments
        
        Returns:
            (N, D) generated samples
        """
        # Start with pure noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, **model_kwargs)
        
        return x
    
    def training_losses(
        self,
        model: nn.Module,
        x_0: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
        **model_kwargs
    ) -> Dict[str, Tensor]:
        """
        Compute training losses.
        
        Args:
            model: Denoising model
            x_0: (N, D) clean data
            t: (B,) timesteps
            noise: Optional pre-sampled noise
            **model_kwargs: Additional model arguments
        
        Returns:
            Dict with 'loss', 'mse_loss'
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Forward diffusion
        x_t, _ = self.q_sample(x_0, t, noise)
        
        # Model prediction
        model_output = model(x_t, t, **model_kwargs)
        
        if isinstance(model_output, tuple):
            eps_pred = model_output[0]
        else:
            eps_pred = model_output
        
        # MSE loss
        if self.config.parameterization == "eps":
            target = noise
        else:
            target = x_0
        
        mse_loss = F.mse_loss(eps_pred, target)
        
        return {
            'loss': mse_loss,
            'mse_loss': mse_loss
        }
    
    def _extract(self, a: Tensor, t: Tensor, x_shape: Tuple[int, ...]) -> Tensor:
        """Extract values from schedule array for given timesteps."""
        batch_size = t.shape[0]
        
        # Handle both batched and non-batched cases
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        out = a.gather(-1, t)
        
        # Reshape for broadcasting
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class MolecularDiffusion(GaussianDiffusion):
    """
    Molecular-specific diffusion handling both coordinates and atom types.
    
    Coordinates: Standard Gaussian diffusion (E(3)-equivariant with EGNN)
    Types: Gaussian diffusion on continuous relaxation, argmax at end
    """
    
    def __init__(
        self,
        config: DiffusionConfig,
        n_atom_types: int = 10
    ):
        super().__init__(config)
        self.n_atom_types = n_atom_types
    
    def q_sample_mol(
        self,
        coords: Tensor,
        types: Tensor,
        t: Tensor,
        coord_noise: Optional[Tensor] = None,
        type_noise: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward diffusion for molecules.
        
        Args:
            coords: (N, 3) atom coordinates
            types: (N, n_types) one-hot atom types
            t: (B,) timesteps
        
        Returns:
            Tuple of (coords_t, types_t, coord_noise, type_noise)
        """
        # Coordinate diffusion
        coords_t, coord_noise = self.q_sample(coords, t, coord_noise)
        
        # Type diffusion (continuous relaxation)
        types_t, type_noise = self.q_sample(types, t, type_noise)
        
        return coords_t, types_t, coord_noise, type_noise
    
    def training_losses_mol(
        self,
        model: nn.Module,
        coords: Tensor,
        types: Tensor,
        t: Tensor,
        lambda_pos: float = 1.0,
        lambda_type: float = 0.5,
        **model_kwargs
    ) -> Dict[str, Tensor]:
        """
        Compute molecular training losses.
        
        Args:
            model: Denoising model (returns type_pred, coord_pred)
            coords: (N, 3) clean coordinates
            types: (N, n_types) one-hot atom types
            t: (B,) timesteps
            lambda_pos: Weight for coordinate loss
            lambda_type: Weight for type loss
        
        Returns:
            Dict with losses
        """
        # Sample noise
        coord_noise = torch.randn_like(coords)
        type_noise = torch.randn_like(types)
        
        # Forward diffusion
        coords_t, types_t, _, _ = self.q_sample_mol(coords, types, t, coord_noise, type_noise)
        
        # Model prediction
        type_pred, coord_pred = model(types_t, coords_t, t, **model_kwargs)
        
        # Losses
        coord_loss = F.mse_loss(coord_pred, coord_noise)
        
        # Type loss can be MSE (continuous) or cross-entropy (discrete target)
        type_loss = F.mse_loss(type_pred, type_noise)
        
        # Combined loss
        total_loss = lambda_pos * coord_loss + lambda_type * type_loss
        
        return {
            'loss': total_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss
        }
    
    @torch.no_grad()
    def sample_mol(
        self,
        model: nn.Module,
        n_atoms: int,
        device: torch.device,
        pocket_data: Optional[Dict] = None,
        guidance_scale: float = 2.0
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate molecules via reverse diffusion.
        
        Args:
            model: Denoising model
            n_atoms: Number of atoms to generate
            device: Device to sample on
            pocket_data: Optional pocket conditioning data
            guidance_scale: Classifier-free guidance scale
        
        Returns:
            Tuple of (coords, types)
        """
        # Initialize with noise
        coords = torch.randn(n_atoms, 3, device=device)
        types = torch.randn(n_atoms, self.n_atom_types, device=device)
        
        # Reverse diffusion
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
            
            # Model prediction
            if pocket_data is not None:
                type_pred, coord_pred = model(
                    types, coords, t_batch,
                    **pocket_data
                )
                
                # Classifier-free guidance
                if guidance_scale > 1.0:
                    # FIX Bug #3: Unconditional pass needs dummy pocket data, not missing args
                    # Create zeroed pocket tensors with same shapes as conditional
                    pocket_h_uncond = torch.zeros_like(pocket_data['pocket_h'])
                    pocket_x_uncond = torch.zeros_like(pocket_data['pocket_x'])
                    # Pass the same edge structure but zeroed features
                    type_pred_uncond, coord_pred_uncond = model(
                        types, coords, t_batch,
                        pocket_h=pocket_h_uncond,
                        pocket_x=pocket_x_uncond,
                        pocket_edge_index=pocket_data['pocket_edge_index'],
                        pocket_edge_attr=pocket_data.get('pocket_edge_attr'),
                        pocket_id=None  # Don't cache unconditional
                    )
                    type_pred = type_pred_uncond + guidance_scale * (type_pred - type_pred_uncond)
                    coord_pred = coord_pred_uncond + guidance_scale * (coord_pred - coord_pred_uncond)
            else:
                type_pred, coord_pred = model(types, coords, t_batch)
            
            # Denoise step
            coords = self._denoise_step(coords, coord_pred, t)
            types = self._denoise_step(types, type_pred, t)
        
        # Convert types to discrete
        types = F.softmax(types, dim=-1)
        
        return coords, types
    
    def _denoise_step(
        self,
        x_t: Tensor,
        eps_pred: Tensor,
        t: int
    ) -> Tensor:
        """Single denoising step."""
        t_tensor = torch.tensor([t], device=x_t.device)
        
        # Predict x_0
        x_0_pred = self.predict_x0_from_eps(x_t, t_tensor, eps_pred)
        
        if t > 0:
            # Compute posterior mean
            mean, _, log_var = self.q_posterior_mean_variance(x_0_pred, x_t, t_tensor)
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.exp(0.5 * log_var) * noise
        else:
            x_prev = x_0_pred
        
        return x_prev
