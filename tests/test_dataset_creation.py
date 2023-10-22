import unittest
from src.data_preprocessing import SequenceDataset, word2id, fam2label, seq_max_len, data_dir


class TestDatasetCreation(unittest.TestCase):
    def test_dataset_shapes(self):

        train_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "train")

        train_shape = next(iter(train_dataset))['sequence'].shape
        self.assertEqual(train_shape, (22, 120))  # Define your_expected_shape


if __name__ == '__main__':
    unittest.main()