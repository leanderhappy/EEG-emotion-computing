import numpy as np
import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


class FeatureExtractor(nn.Module):
    def __init__(self, in_dim, hidden=32, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class EmotionClassifier(nn.Module):
    def __init__(self, latent, n_classes):
        super().__init__()
        self.net = nn.Linear(latent, n_classes)

    def forward(self, x):
        return self.net(x)


class SubjectDiscriminator(nn.Module):
    def __init__(self, latent, n_domains, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, x, alpha=1.0):
        return self.net(grad_reverse(x, alpha))


class DANNModel(nn.Module):
    def __init__(self, in_dim, n_classes, n_domains, hidden=32, latent=32):
        super().__init__()
        self.feature_extractor = FeatureExtractor(in_dim, hidden, latent)
        self.emotion_classifier = EmotionClassifier(latent, n_classes)
        self.domain_discriminator = SubjectDiscriminator(latent, n_domains)

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        y_hat = self.emotion_classifier(features)
        d_hat = self.domain_discriminator(features, alpha)
        return y_hat, d_hat


def dann_lambda(p, max_lambda=0.5):
    return max_lambda * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
