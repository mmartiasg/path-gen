import torch


class MSELoss:
    def __call__(self, y_hat, y):
        return torch.mean((y_hat - y) ** 2)
