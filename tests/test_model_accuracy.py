import unittest
import torch
from torch.utils.data import DataLoader
from src.model_definition import ProtCNN
from src.data_preprocessing import SequenceDataset, word2id, fam2label, seq_max_len, data_dir  # Import your data preprocessing functions


def load_model(model_path):
    model = ProtCNN.load_from_checkpoint(model_path, num_classes=len(fam2label), num_res_blocks=1, kernel_size=3, learning_rate=1e-2, optimizer='adam')  # Change parameters here according to what the model was trained on.
    model.eval()  # Set the model to evaluation mode
    return model


def predict(model, data_loader):
    predictions = []
    for batch in data_loader:
        sequence = batch['sequence']
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sequence = sequence.to(device)
        with torch.no_grad():
            output = model(sequence)
        predicted_labels = torch.argmax(output, dim=1)
        predictions.extend(predicted_labels.tolist())
    return predictions


def evaluate(predictions, true_labels):
    # Add your evaluation metrics here (e.g., accuracy, F1-score, confusion matrix)
    # You can use scikit-learn or other libraries for evaluation
    # For example:
    from sklearn.metrics import accuracy_score, classification_report
    true_labels = [fam2label.get(x, fam2label['<unk>']) for x in true_labels]
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    return accuracy, report

class TestModelHealth(unittest.TestCase):
    def test_model(self):

        model_path = 'models/model-epoch=00-val_loss=0.00.ckpt'

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