# Protein Classification with ProtCNN

## Overview

This codebase provides a framework for implementing a protein classifier that assigns Pfam family labels to proteins. The model architecture is inspired by ProtCNN, a convolutional neural network designed for protein classification.

## Data Source

The data used for this project is sourced from the PFAM dataset, which is available [here](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split/).

## Features

- Build a local development environment using Docker.
- Refactor the code into modular Python scripts, allowing users to adjust training and model hyperparameters from the command-line interface (CLI).
- Implement a command-line interface for model prediction and evaluation.
- Ensure code quality and consistency.
- Comprehensive documentation of the repository.
- Include unit tests for critical components.
- Utilize PyTorch Lightning as the deep learning framework for model development.

## Dependencies

The codebase has been tested with the following Python version and dependencies:

- Python 3.7.4
- matplotlib 3.4.1
- numpy 1.18.5
- pandas 1.2.3
- pytorch-lightning 1.5.3
- seaborn 0.11.1
- tensorboard 2.2.2
- torch 1.8.1
- torchmetrics 0.6.0

The dependencies can also be found in requirements.txt.

## Repository Structure

The project directory structure is organized as follows:

- **data**: This directory should contain the PFAM dataset or data used for training and evaluation.

- **src**: This directory contains the core source code for data preprocessing, model definition, and other utility functions.

- **tests**: Unit tests for various components of the codebase, including data preprocessing, model evaluation, and more.

- **notebooks**: This original notebook providing code snippets for a baseline model and some util functions.

- **models**: Model checkpoints are saved here after training.

## Setup and Usage

1. **Environment Setup**: You can set up a local development environment using Docker. Refer to the provided Dockerfile for environment setup.

2. **Data Preprocessing**: Data preprocessing is handled in the `data_preprocessing.py` script. Ensure that you have the data in the appropriate format in the data directory.

3. **Model Definition**: The model architecture is defined in the `model_definition.py` script. You can customize the model's hyperparameters in this script.

4. **Training**: To train the model, execute the `train.py` script. You can adjust training hyperparameters such as the number of epochs, learning rate, and batch size via command-line arguments.

    Example CLI usage:
       ```
       python train.py --epochs 25 --learning_rate 0.01 --optimizer sgd --kernel_size 5
       ```

5. **Prediction and Evaluation**: Use the `predict_evaluate.py` script to load a trained model and make predictions. This script also includes model evaluation functionality. Provide the path to the trained model checkpoint and the data for prediction/evaluation as command-line arguments.

    Example CLI usage:
       ```
       python predict_evaluate.py --model_path models/model_checkpoint.ckpt --data_path data/random_split --dir_name test
       ```

6. **Visualization of Training**: The training progress can be visualized using TensorBoard or the `visualize_training.py` script. Run `tensorboard --logdir "./lightning_logs/"` to start TensorBoard and monitor training metrics.

## Unit Tests

Unit tests are available in the `tests` directory to validate the functionality of key components of the codebase. Use these tests to ensure that data preprocessing, model evaluation, and other critical parts of the code are functioning as expected.

## Acknowledgments

This project was made possible with the contributions of many open-source libraries, including PyTorch, PyTorch Lightning, and others.