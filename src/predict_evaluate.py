import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from model_definition import ProtCNN  # Import your model definition
from data_preprocessing import SequenceDataset, word2id, fam2label, seq_max_len  # Import your data preprocessing functions


def load_model(model_path):
    model = ProtCNN.load_from_checkpoint(model_path)
    model.eval()  # Set the model to evaluation mode
    return model


def predict(model, data_loader):
    predictions = []
    for batch in data_loader:
        sequence = batch['sequence']
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
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    return accuracy, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Prediction and Evaluation Script')

    # Model and Data Paths
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data for prediction/evaluation')
    parser.add_argument('--filename', type=str, required=True, help='Filename for prediction/evaluation')

    args = parser.parse_args()

    # Load the trained model
    model = load_model(args.model_path)

    # Load data for prediction/evaluation
    dataset = SequenceDataset(word2id, fam2label, seq_max_len, args.data_path, args.filename)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Make predictions
    predictions = predict(model, data_loader)

    print('Model predictions are:\n', predictions)

    # Load true labels (assuming they are available)
    true_labels = dataset.label.tolist()

    # Evaluate the model
    accuracy, report = evaluate(predictions, true_labels)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)