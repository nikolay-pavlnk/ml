import torch


class BaseLinear:
    def __init__(self, model, device, loss_func, optimizer, epochs):
        self.model = model
        self.device = device
        self.epochs = int(epochs)
        self.loss_func = loss_func
        self.optimizer = optimizer

    def fit(self, loader):
        self.loss = []
        for _ in range(int(self.epochs)):
            for data in loader:
                X, y = data
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model.forward(X)
                loss = self.loss_func(pred, y[:, None])
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.loss.append(loss.item())


class LinearRegressionTorch(torch.nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.linear = self.linear.type(torch.float64)
        self.linear = self.linear.to(device)

    def forward(self, x):
        out = self.linear(x)
        return out


class LogisticRegressionTorch(torch.nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.linear = self.linear.type(torch.float64)
        self.linear = self.linear.to(device)

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)
