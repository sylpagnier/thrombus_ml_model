import random
import torch
from torch.utils.data import Sampler


class StratifiedAnchorSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.anchor_indices = []
        self.physics_indices = []

        # Start in warmup mode by default
        self.warmup_mode = True

        for i, d in enumerate(dataset):
            if hasattr(d, "is_anchor"):
                ia = d.is_anchor
                if torch.is_tensor(ia):
                    is_sup = bool(ia.any().item())
                else:
                    is_sup = bool(ia)
            else:
                is_sup = False
            if is_sup:
                self.anchor_indices.append(i)
            else:
                self.physics_indices.append(i)

        if not self.anchor_indices:
            raise ValueError("Dataset has no anchor graphs. Warmup impossible.")

        self.num_batches = len(dataset) // batch_size
        self.num_anchors_per_batch = batch_size // 2
        self.num_physics_per_batch = batch_size - self.num_anchors_per_batch

    def set_warmup_mode(self, is_warmup: bool):
        """Toggles whether to sample ONLY anchors or a mix."""
        self.warmup_mode = is_warmup

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []

            if self.warmup_mode or not self.physics_indices:
                # WARMUP: 100% of the batch comes from anchored ground truth
                batch_indices.extend(random.choices(self.anchor_indices, k=self.batch_size))
            else:
                # PHYSICS ACTIVE: 50/50 mix of anchors and unanchored graphs
                batch_indices.extend(random.choices(self.anchor_indices, k=self.num_anchors_per_batch))
                batch_indices.extend(random.choices(self.physics_indices, k=self.num_physics_per_batch))

            random.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self):
        return self.num_batches * self.batch_size