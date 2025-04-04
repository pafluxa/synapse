from itertools import tee
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from torch.utils.data import DataLoader
from synapse.data.datasets import CSVDataset  # Replace with actual import

@pytest.fixture
def sample_csv():
    # Generate sample data with mixed types and missing values
    np.random.seed(42)
    num_samples = 100

    data = {
        'var1': np.random.randint(18, 80, num_samples),
        'var2': np.round(np.random.normal(50000, 15000, num_samples), 2),
        'var3': np.round(np.random.normal(1.8, 0.4, num_samples), 2),
        'var4': np.round(np.random.normal(1.0, 1.0, num_samples), 2),
        'var5': np.round(np.random.normal(1.0, 1.0, num_samples), 2),
        'var6': np.round(np.random.normal(1.0, 1.0, num_samples), 2),
        'var7': np.round(np.random.normal(1.0, 1.0, num_samples), 2),
        'var8': np.round(np.random.normal(1.0, 1.0, num_samples), 2),
        'cat1': np.random.choice(['M', 'F'], num_samples, p=[0.6, 0.4]),
        'cat2': np.random.choice(['A', 'B', 'C'], num_samples),
        'cat3': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], num_samples),
        'cat4': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], num_samples),
        'cat5': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], num_samples),
        'cat6': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], num_samples),
        'cat7': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], num_samples),
        'cat8': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], num_samples),
    }
    df = pd.DataFrame(data)
    # Create temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        df.to_csv(f, index=False)
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)  # Cleanup after test

def test_csv_dataset_loading(sample_csv):
    # Test basic loading functionality
    dataset = CSVDataset(
        file_path=sample_csv,
        numerical_cols=[f"var{i}" for i in range(1, 9)],
        categorical_cols=[f"cat{i}" for i in range(1, 9)],
    )

    # Check basic properties
    assert len(dataset) == 100

def test_train_val_split(sample_csv):
    # Test dataset splitting
    dataset = CSVDataset(
        file_path=sample_csv,
        numerical_cols=[f"var{i}" for i in range(1, 9)],
        categorical_cols=[f"cat{i}" for i in range(1, 9)],
    )
    train_dataset, test_dataset, val_dataset = dataset.prepare_splits()
    # Check split sizes
    assert len(train_dataset) == 80
    assert len(test_dataset) == 10
    assert len(val_dataset) == 10

def test_dataset_iteration(sample_csv):
    dataset = CSVDataset(
        file_path=sample_csv,
        numerical_cols=[f"var{i}" for i in range(1, 9)],
        categorical_cols=[f"cat{i}" for i in range(1, 9)],
    )
    dataset, _, _ = dataset.prepare_splits(test_size=0.01)
    loader = DataLoader(dataset, batch_size=32)

    # Test iteration
    for num_data, cat_data in loader:
        x_num, num_mask_idx = num_data
        x_cat, cat_mask_idx = cat_data
        assert x_num.shape == (32, 7)
        assert x_cat.shape == (32, 7)
        break  # Just test first batch

def test_dataset_length(sample_csv):
    dataset = CSVDataset(
        file_path=sample_csv,
        numerical_cols=[f"var{i}" for i in range(1, 9)],
        categorical_cols=[f"cat{i}" for i in range(1, 9)],
    )

    train_dataset, test_dataset, val_dataset = dataset.prepare_splits()
    assert len(dataset) == 100
    assert len(train_dataset) + len(test_dataset) + len(val_dataset) == 100
