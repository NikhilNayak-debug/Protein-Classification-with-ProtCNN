import unittest
import torch
from your_project.model_definition import ProtCNN
from src.data_preprocessing import SequenceDataset, word2id, fam2label, seq_max_len, data_dir


class TestModelHealth(unittest.TestCase):
    def test_model_on_mini_batch(self):
        batch_size = 1
        num_workers = 8

        train_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "train")
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=num_workers)

        num_classes = len(fam2label)
        num_res_blocks = 2
        num_filters = 128
        kernel_size = 3
        learning_rate = 1e-2
        optimizer = 'adam'
        prot_cnn = ProtCNN(num_classes, num_res_blocks, num_filters, kernel_size, learning_rate, optimizer)  # Define your_num_classes and other arguments

        batch = next(iter(dataloader))
        batch_shape = batch["sequence"].shape
        output_shape = prot_cnn(batch["sequence"]).shape
        self.assertEqual(batch_shape, (1, 22, 120))
        self.assertEqual(output_shape, (1, 17930))


if __name__ == '__main__':
    unittest.main()