


# Building the classification model

class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class ResidualBlock(torch.nn.Module):
    """
    The residual block used by ProtCNN (https://www.biorxiv.org/content/10.1101/626507v3.full).

    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        dilation: Dilation rate of the first convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()

        # Initialize the required layers
        self.skip = torch.nn.Sequential()

        self.bn1 = torch.nn.BatchNorm1d(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, bias=False, dilation=dilation, padding=dilation)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, bias=False, padding=1)

    def forward(self, x):
        # Execute the required layers and functions
        activation = F.relu(self.bn1(x))
        x1 = self.conv1(activation)
        x2 = self.conv2(F.relu(self.bn2(x1)))

        return x2 + self.skip(x)

class ProtCNN(pl.LightningModule):

    def __init__(self, num_classes, num_res_blocks, num_filters, kernel_size, learning_rate, optimizer):
        super().__init__()
        residual_blocks = [ResidualBlock(num_filters, num_filters, kernel_size, dilation=i + 2) for i in
                           range(num_res_blocks)]
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(22, num_filters, kernel_size=1, padding=0, bias=False),
            *residual_blocks,
            torch.nn.MaxPool1d(3, stride=2, padding=1),
            Lambda(lambda x: x.flatten(start_dim=1)),
            torch.nn.Linear(7680, num_classes)
        )

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.learning_rate = learning_rate
        self.optimizer = optimizer


    def forward(self, x):
        return self.model(x.float())

    def training_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        pred = torch.argmax(y_hat, dim=1)
        self.train_acc(pred, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)
        acc = self.valid_acc(pred, y)
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)

        return acc

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-2)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8, 10, 12, 14, 16, 18, 20], gamma=0.9)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }