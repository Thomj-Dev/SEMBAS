from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader


class ConcreteLinear(nn.Module):
    def __init__(self, in_features, out_features, weight=None, bias=None):
        super(ConcreteLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        if weight is not None and bias is not None:
            with torch.no_grad():
                self.linear.weight = nn.Parameter(weight)
                self.linear.bias = nn.Parameter(bias)

    def forward(self, x):
        return self.linear(x)


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()

        # Weight mean and variance (log variance) parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.zeros(out_features, in_features))

        # Bias mean and variance (log variance) parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Sample weights and biases from Normal distributions
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)

        weight_dist = Normal(self.weight_mu, weight_std)
        bias_dist = Normal(self.bias_mu, bias_std)

        weight_sample = weight_dist.rsample()  # Reparametrization trick
        bias_sample = bias_dist.rsample()

        # KL divergence between posterior and prior
        kl_weight = kl_divergence(weight_dist, Normal(0, 1)).sum()
        kl_bias = kl_divergence(bias_dist, Normal(0, 1)).sum()

        self.kl = kl_weight + kl_bias

        # Linear transformation
        return torch.matmul(x, weight_sample.t()) + bias_sample

    def sample_concrete_weights(self):
        """Sample and return concrete weights and biases from the posterior."""
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)

        weight_sample = Normal(self.weight_mu, weight_std).rsample()
        bias_sample = Normal(self.bias_mu, bias_std).rsample()

        return weight_sample, bias_sample


# Define the Bayesian Neural Network
class BayesianNN(nn.Module):
    def __init__(self):
        super(BayesianNN, self).__init__()
        self.bayesian_layer1 = BayesianLinear(2, 50)  # 2 inputs (a, b), 50 hidden units
        self.bayesian_layer2 = BayesianLinear(50, 1)  # 50 hidden units, 1 output (y)

    def forward(self, x):
        x = torch.relu(self.bayesian_layer1(x))
        return self.bayesian_layer2(x)

    def sample_network(self) -> nn.Module:
        """Sample a concrete neural network from the BNN's distribution"""
        l1_w, l1_b = self.bayesian_layer1.sample_concrete_weights()
        l1 = ConcreteLinear(2, 50, l1_w, l1_b)

        l2_w, l2_b = self.bayesian_layer2.sample_concrete_weights()
        l2 = ConcreteLinear(50, 1, l2_w, l2_b)

        return nn.Sequential(l1, nn.ReLU(), l2)


def loss_fn(predictions, targets, model, kl_weight):
    mse_loss = nn.MSELoss()(predictions, targets)
    kl_loss = sum(
        [layer.kl for layer in model.modules() if isinstance(layer, BayesianLinear)]
    )
    return mse_loss + kl_weight * kl_loss


def train_bnn(model, optimizer, dataset, kl_weight=1e-6, epochs=1):
    train_data, test_data = train_test_split(dataset, test_size=0.1, shuffle=True)
    test_loss = []
    train_loss = []

    test_x = torch.vstack([row[0] for row in test_data])
    test_y = torch.vstack([row[1] for row in test_data])

    loader = DataLoader(train_data, 32, True)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        for i, batch in enumerate(loader):
            inputs, labels = batch

            optimizer.zero_grad()

            y_hat = model(inputs)
            loss = loss_fn(y_hat, labels, model, kl_weight)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                yhat = model(test_x)
                test_loss.append(loss_fn(yhat, test_y, model, kl_weight).item())
                model.train()

        print(
            f"Epoch #{epoch + 1}: Loss = {sum(train_loss[-i:]) / i}, Test Loss = {sum(test_loss[-i:]) / i}"
        )

    return test_loss, train_loss
