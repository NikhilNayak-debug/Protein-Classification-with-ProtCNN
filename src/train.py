import argparse
import torch
import pytorch_lightning as pl
from data_preprocessing import fam2label, dataloaders
from model_definition import ProtCNN

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Script for Protein Classification Model')

    # Training-related hyperparameters
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate for the optimizer')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer type (Adam or SGD)')

    # Model-related hyperparameters
    parser.add_argument('--num_res_blocks', type=int, default=3, help='Number of residual blocks in the model')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for convolutional layers')

    args = parser.parse_args()

    num_classes = len(fam2label)

    prot_cnn = ProtCNN(num_classes, args.num_res_blocks, args.kernel_size, args.learning_rate, args.optimizer)

    print('Peek at the model architecture:\n', prot_cnn)

    pl.seed_everything(0)

    if torch.cuda.is_available():
        gpus = 1
        accelerator = 'gpu'
        trainer = pl.Trainer(devices=gpus, accelerator="gpu", max_epochs=args.epochs)
    else:
        trainer = pl.Trainer(max_epochs=args.epochs)

    trainer.fit(prot_cnn, dataloaders['train'], dataloaders['dev'])