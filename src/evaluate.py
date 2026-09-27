"""
Метрики для оценки моделей.

Основная метрика задачи: 1 - Brier, где Brier = mean((p - y)^2).
Кроме нее смотрим logloss, accuracy и ECE (насколько вероятности откалиброваны).
"""
import numpy as np
import torch


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    brier = np.mean((p - y) ** 2)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    logloss = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    acc = np.mean((p > 0.5) == (y > 0.5))
    return {'score': 1 - brier, 'brier': brier, 'logloss': logloss, 'acc': acc, 'ece': ece(y, p)}


def ece(y, p, bins=15):
    # expected calibration error: разбиваем на корзины по уверенности и смотрим разницу с реальной частотой
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return err


@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x).numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)
