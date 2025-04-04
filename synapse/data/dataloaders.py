import torch
from torch.utils.data import DataLoader

class MaskedDataLoader:
    """Wrapper that applies random masking to batches"""
    def __init__(self, dataset, batch_size, mask_prob):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=False,
            persistent_workers=False,
        )
        self.num_numerical = dataset.num_numerical
        self.num_categorical = dataset.num_categorical
        self.mask_prob = mask_prob
        self.dims_categorical = torch.tensor([
            [dim,] for dim in dataset.cardinalities
        ]).repeat(batch_size, 1)
        self.dims_numerical = torch.tensor([1]).repeat(batch_size, 1)

    def __iter__(self):
        for x_num, x_cat in self.dataloader:
            # Move to GPU
            x_num = x_num.cuda()
            x_cat = x_cat.cuda()

            yield x_num, x_cat

    def __len__(self):
        return len(self.dataloader)
