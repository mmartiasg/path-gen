import torch


class BaseTrainer:
    def __init__(self, epochs, device="mps", loss=None):
        self.epochs = epochs
        self.device = device
        self.loss_f = loss

    def train(self, model, dataloader):
        if self.loss_f is None:
            raise Exception("Loss function not specified")

        optimizer = torch.optim.Adam(model.parameters())
        model = model.to(self.device)

        loss_mean = []
        for _ in range(self.epochs):
            loss_batch = []
            for batch in dataloader:
                batch = batch.to(self.device)

                y_hat = model(batch)
                loss = self.loss_f(y_hat, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_batch.append(loss.item())
            loss_mean.append(sum(loss_batch) / len(loss_batch))

        return loss_mean
