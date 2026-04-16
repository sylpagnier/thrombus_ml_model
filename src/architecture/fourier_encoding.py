import torch


def build_fourier_features(features_to_encode: torch.Tensor, fourier_freqs: torch.Tensor) -> torch.Tensor:
    """Apply shared sinusoidal Fourier encoding over selected node features."""
    num_nodes = features_to_encode.shape[0]
    x_proj = (features_to_encode.unsqueeze(-1) * fourier_freqs).contiguous()
    x_proj = x_proj.view(num_nodes, -1)
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
