import unittest
from torch.utils.data import DataLoader
from src.data_preprocessing import SequenceDataset, word2id, fam2label, seq_max_len, data_dir  # Import your data preprocessing functions
from src.predict_evaluate import load_model, predict, evaluate


class TestModelHealth(unittest.TestCase):
    def test_model(self):

        model_path = 'model/'

        # Load the trained model
        model = load_model(model_path)

        # Load data for prediction/evaluation
        dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, 'test')
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

        # Make predictions
        predictions = predict(model, data_loader)

        # Load true labels (assuming they are available)
        true_labels = dataset.label.tolist()

        # Evaluate the model
        accuracy, report = evaluate(predictions, true_labels)

        self.assertGreater(accuracy, 0.9)


if __name__ == "__main__":
    unittest.main()