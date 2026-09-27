"""
Общие штуки которые нужны и в обучении и в предикте.

Самое важное тут это препроцессинг картинки. Он ОДИН для всего: для синтетики, для реальных кропов и для теста,
иначе можно получить тихий баг когда обучались на одном а предиктим на другом.

Как готовим картинку:
    1. переводим в серый (цвет для ориентации текста почти не нужен)
    2. ресайзим до высоты 32 с сохранением пропорций. Ширину ограничиваем MAX_STORE_W,
       если картинка длиннее то просто сжимаем по горизонтали
    3. (уже перед сеткой) сжимаем до ширины модели если надо, нормируем и ставим ПО ЦЕНТРУ пустого холста

Почему по центру: тогда поворот на 180 готового тензора = препроцессинг повернутой картинки.
Это удобно для TTA (можно просто перевернуть тензор) и нет никакой разницы между классами в том где паддинг.
"""
import os
import random

import cv2
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEIGHT = 32          # высота входа сетки
MAX_STORE_W = 512    # до такой ширины храним заготовленные картинки (потом можно сжать сильнее)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(path=None):
    path = path or os.path.join(ROOT, 'config.yaml')
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def read_image(path):
    # cv2.imread не любит кириллицу в путях, поэтому через imdecode
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'не смог прочитать {path}')
    return img  # BGR


def to_gray(img):
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_to_height(gray, height=HEIGHT, max_w=MAX_STORE_W):
    """Серую картинку любого размера приводим к высоте height, ширина пропорционально (но не больше max_w)."""
    h, w = gray.shape[:2]
    new_w = int(round(w * height / h))
    new_w = max(4, min(new_w, max_w))
    # уменьшаем через INTER_AREA (меньше артефактов), увеличиваем линейной
    interp = cv2.INTER_AREA if h > height else cv2.INTER_LINEAR
    return cv2.resize(gray, (new_w, height), interpolation=interp)


def prepare_image(img, height=HEIGHT, max_w=MAX_STORE_W):
    """Главная функция препроцессинга: на входе BGR/серая картинка, на выходе uint8 массив (height, w)."""
    return resize_to_height(to_gray(img), height, max_w)


def to_model_input(arr, width):
    """
    uint8 (32, w) в float32 (1, 32, width) для сетки.
    Если картинка шире чем width, то сжимаем по горизонтали. Потом нормируем по самой картинке
    (минус среднее, делим на std) и ставим по центру холста из нулей.
    """
    h, w = arr.shape
    if w > width:
        arr = cv2.resize(arr, (width, h), interpolation=cv2.INTER_AREA)
        w = width
    x = arr.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1.0)  # +1 чтобы не делить на ноль на пустых картинках
    out = np.zeros((1, h, width), dtype=np.float32)
    left = (width - w) // 2
    out[0, :, left:left + w] = x
    return out


def rotate180(arr):
    # поворот на 180 это просто разворот по обеим осям
    return np.ascontiguousarray(arr[::-1, ::-1])


# ---- хранение заготовленных картинок ----
# все картинки одной высоты, поэтому склеиваем их по ширине в один большой массив
# и отдельно храним ширины. Так быстро грузится и мало места занимает.

def save_pack(path, arrays, **extra):
    widths = np.array([a.shape[1] for a in arrays], dtype=np.int32)
    data = np.concatenate(arrays, axis=1) if arrays else np.zeros((HEIGHT, 0), np.uint8)
    np.savez(path, data=data, widths=widths, **extra)


class Pack:
    """Читалка того что сохранили через save_pack."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.data = z['data']
        self.widths = z['widths']
        self.offsets = np.concatenate([[0], np.cumsum(self.widths)])
        self.extra = {k: z[k] for k in z.files if k not in ('data', 'widths')}

    def __len__(self):
        return len(self.widths)

    def __getitem__(self, i):
        return self.data[:, self.offsets[i]:self.offsets[i + 1]]
