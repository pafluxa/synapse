import torch
import numpy as np
from scipy.stats import special_ortho_group
from torch.utils.data import Dataset, DataLoader


class SphereMembershipClassifier(nn.Module):
    """Classifier to distinguish true rotations from perturbations"""
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Example usage pipeline
def train_sphere_classifier(trained_model, legit_data, dim=64):
    # Step 1: Extract codecs from legitimate data
    with torch.no_grad():
        codecs = trained_model(legit_data[0], legit_data[1])['codec'].cpu()

    # Step 2: Create augmentation and dataset
    rot_augmenter = RotationAugmentation(dim=dim, perturbation_scale=0.15)
    contrastive_dataset = ContrastiveSphereDataset(codecs, rot_augmenter, pairs_per_codec=5)

    # Step 3: Initialize and train classifier
    classifier = SphereMembershipClassifier(input_dim=dim)
    trainer = ContrastiveTrainer(classifier, contrastive_dataset)
    trainer.train(num_epochs=10)

    return classifier
