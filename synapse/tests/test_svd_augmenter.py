import unittest
import torch
from synapse.utils.rotation_augmenter import SVDRotationAugmenter

class TestSVDAugmenter(unittest.TestCase):
    def setUp(self):
        self.augmenter = SVDRotationAugmenter(dim=64, epsilon=0.2)

    def test_perturbation_bounds(self):
        R, P = self.augmenter.generate_pair()

        # Verify perturbation preserves shape
        self.assertEqual(R.shape, (64, 64))
        self.assertEqual(P.shape, (64, 64))

        # Check orthogonality measures
        rot_error = self.augmenter.orthogonality_measure(R)
        pert_error = self.augmenter.orthogonality_measure(P)

        self.assertAlmostEqual(rot_error, 0, delta=1e-4)
        self.assertGreater(pert_error, 1e-3)

    def test_singular_value_bounds(self):
        _, P = self.augmenter.generate_pair()
        S = torch.linalg.svdvals(P)

        self.assertTrue(torch.all(S >= self.augmenter.min_singular))
        self.assertTrue(torch.all(S <= self.augmenter.max_singular))
