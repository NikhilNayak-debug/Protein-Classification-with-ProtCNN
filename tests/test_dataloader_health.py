import unittest
import torch
from src.data_preprocessing import SequenceDataset, word2id, fam2label, seq_max_len, data_dir


class TestDataLoaderHealth(unittest.TestCase):
    def test_dataloader_shapes(self):
        batch_size = 1
        num_workers = 8

        train_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "train")
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=num_workers)

        batch = next(iter(dataloader))
        self.assertEqual(batch['sequence'].shape, (1, 22, 120))  # Define your_expected_shape
        self.assertEqual(batch['target'].shape, (1,))  # Define your_expected_shape

        # Similarly, test dev_dataset and test_dataset dataloader shapes

        dev_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "dev")
        dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers)

        batch = next(iter(dataloader))
        self.assertEqual(batch['sequence'].shape, (1, 22, 120))  # Define your_expected_shape
        self.assertEqual(batch['target'].shape, (1,))  # Define your_expected_shape

        test_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "test")
        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers)

        batch = next(iter(dataloader))
        self.assertEqual(batch['sequence'].shape, (1, 22, 120))  # Define your_expected_shape
        self.assertEqual(batch['target'].shape, (1,))  # Define your_expected_shape


if __name__ == '__main__':
    unittest.main()