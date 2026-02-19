import random
from torch.utils.data import Sampler

class StratifiedAnchorSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.anchor_indices = []
        self.physics_indices = []

        for i, d in enumerate(dataset):
            if hasattr(d, 'is_anchor') and d.is_anchor.item() is True:
                self.anchor_indices.append(i)
            else:
                self.physics_indices.append(i)

        if not self.anchor_indices or not self.physics_indices:
            raise ValueError("Dataset split failed. Ensure you have both anchor and physics graphs.")

        self.num_batches = len(dataset) // batch_size
        self.num_anchors_per_batch = batch_size // 2
        self.num_physics_per_batch = batch_size - self.num_anchors_per_batch

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            batch_indices.extend(random.choices(self.anchor_indices, k=self.num_anchors_per_batch))
            batch_indices.extend(random.choices(self.physics_indices, k=self.num_physics_per_batch))
            random.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self):
        return self.num_batches * self.batch_size