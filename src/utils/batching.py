import torch


def get_batch_tensor(data, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Return per-node batch ids, defaulting to a single-graph tensor."""
    batch = getattr(data, "batch", None)
    if batch is not None:
        return batch
    return torch.zeros(int(num_nodes), dtype=torch.long, device=device)
