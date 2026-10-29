import json
from pathlib import Path
import unittest
import pandas as pd
from train_models import generate_dataset, FEATURES, FAULTS, LINES

class GridGuardDataTests(unittest.TestCase):
    def test_dataset_is_reproducible_and_has_expected_columns(self):
        a = generate_dataset(n=100, seed=42)
        b = generate_dataset(n=100, seed=42)
        self.assertTrue(a.equals(b))
        self.assertTrue(set(FEATURES + ["fault_type", "faulted_line"]).issubset(a.columns))
        self.assertEqual(len(a), 100)

    def test_labels_are_valid(self):
        df = generate_dataset(n=300, seed=10)
        self.assertTrue(set(df["fault_type"]).issubset(set(FAULTS)))
        self.assertTrue(set(df["faulted_line"]).issubset(set(LINES)))

if __name__ == "__main__":
    unittest.main()
