"""
Датасеты для обучения и валидации.

Все картинки в пачках (data/synth/*.npz, data/real/*.npz) стоят ПРАВИЛЬНО и уже приведены к высоте 32.
Метку делаем сами: с вероятностью 0.5 переворачиваем картинку на 180 и ставим label = 1.
Поворот делаем до всех паддингов, так же как (скорее всего) делали с тестом.

На валидации поворот фиксированный (по сиду), чтоб метрики между запусками можно было сравнивать.
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.common import ROOT, Pack, rotate180, to_model_input


def load_packs(names, folder):
    return [Pack(os.path.join(ROOT, 'data', folder, n + '.npz')) for n in names]


def light_aug(arr, rng):
    """
    Легкие аугментации прямо на картинке 32 x w (дешево, можно на лету).
    Все они симметричные относительно поворота, т.е. не подсказывают сетке ответ.
    """
    h, w = arr.shape
    # обрезаем немного слева/справа (детектор мог отрезать кусок текста)
    if rng.random() < 0.3 and w > 16:
        l = int(rng.integers(0, int(w * 0.15) + 1))
        r = int(rng.integers(0, int(w * 0.15) + 1))
        arr = arr[:, l:w - r] if w - r - l > 8 else arr
    # немного сжимаем/растягиваем по ширине
    if rng.random() < 0.3:
        k = rng.uniform(0.8, 1.25)
        arr = cv2.resize(arr, (max(4, int(arr.shape[1] * k)), h), interpolation=cv2.INTER_LINEAR)
    # сдвиг по вертикали на пару пикселей (срезаем сверху или снизу и растягиваем обратно)
    if rng.random() < 0.2:
        cut = int(rng.integers(1, 4))
        arr = arr[cut:] if rng.random() < 0.5 else arr[:-cut]
        arr = cv2.resize(arr, (arr.shape[1], h), interpolation=cv2.INTER_LINEAR)
    # инверсия (светлый текст на темном <-> темный на светлом)
    if rng.random() < 0.2:
        arr = 255 - arr
    # гамма
    if rng.random() < 0.3:
        g = rng.uniform(0.6, 1.6)
        arr = (255 * (arr / 255.0) ** g).astype(np.uint8)
    # блюр и шум
    if rng.random() < 0.1:
        arr = cv2.GaussianBlur(arr, (0, 0), rng.uniform(0.4, 1.0))
    if rng.random() < 0.2:
        noise = rng.normal(0, rng.uniform(3, 12), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class RotDataset(Dataset):
    """
    Склеивает несколько пачек в один датасет.
    train=True: случайный поворот + легкие аугментации
    train=False: поворот фиксирован сидом, без аугментаций
    """

    def __init__(self, packs, width, train=True, seed=0):
        self.packs = packs
        self.width = width
        self.train = train
        # глобальный индекс -> (номер пачки, индекс внутри)
        self.index = [(pi, i) for pi, p in enumerate(packs) for i in range(len(p))]
        self.seed = seed
        rng = np.random.RandomState(seed)
        self.fixed_rot = rng.rand(len(self.index)) < 0.5
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pi, i = self.index[idx]
        arr = self.packs[pi][i]
        if self.train:
            arr = light_aug(arr, self.rng)
            rot = self.rng.random() < 0.5
        else:
            rot = self.fixed_rot[idx]
        if rot:
            arr = rotate180(arr)
        x = to_model_input(arr, self.width)
        return torch.from_numpy(x), torch.tensor(float(rot))


def worker_init(worker_id):
    # у каждого воркера даталоадера свой генератор, иначе аугментации будут одинаковые
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds.rng = np.random.default_rng(ds.seed * 1000 + worker_id + 1 + torch.initial_seed() % 100000)


def mix_weights(packs_a, packs_b, share_b):
    """Веса для семплера: чтоб доля картинок из packs_b в батчах была примерно share_b."""
    na = sum(len(p) for p in packs_a)
    nb = sum(len(p) for p in packs_b)
    wa = (1 - share_b) / na
    wb = share_b / nb
    return torch.tensor([wa] * na + [wb] * nb, dtype=torch.double)
