import torch.nn as nn

class BCBaseline(nn.Module):
    """
    A simple pixel-wise baseline classifier for binary classification.

    This model uses a single 1x1 convolution to perform logistic regression
    on the input channels for each pixel. It takes in a tensor of shape
    (B, C, H, W) and outputs a logit tensor of shape (B, 1, H, W).
    """
    def __init__(self, in_channels=37, out_channels=1, **kwargs):
        super().__init__()
        # A 1x1 convolution is equivalent to a fully connected layer applied
        # independently to each pixel.
        self.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.classifier(x)