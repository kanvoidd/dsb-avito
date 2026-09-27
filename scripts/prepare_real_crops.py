"""
Готовим реальные кропы текста из открытых датасетов (huggingface).

Идея: любой кроп из OCR датасета стоит правильно (label=0), а перевернутый пример получаем поворотом на 180.
Так что реальные фотки текста это бесплатные размеченные данные. Жаль что в основном на английском,
но зато тут настоящий блюр, шум, перспектива, освещение и тд, чего в синтетике нормально не сделать.

Что берем:
    COCO-Text v2      - слова с фоток, английский        (Bekhouche/COCO-Text_V2_STR)
    ICDAR 2015        - слова с фоток, очень размытые    (MiXaiLL76/ICDAR2015_OCR)
    TextOCR           - слова с фоток, один кусок трейна (MiXaiLL76/TextOCR_OCR)
    рукописный русский + номера машин                    (Foximaz/russian_ocr_small)

Все картинки сразу прогоняем через общий препроцессинг (серый, высота 32) и сохраняем в data/real/*.npz

Запуск:
    python scripts/prepare_real_crops.py
"""
import os
import sys
import urllib.request

import cv2
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import ROOT, prepare_image, save_pack  # noqa: E402

RAW_DIR = os.path.join(ROOT, 'data', 'raw')
OUT_DIR = os.path.join(ROOT, 'data', 'real')

HF = 'https://huggingface.co/datasets'
# имя: (ссылка на parquet, что с ним делать)
# train / val - целиком в обучение или в валидацию, split - делим сами 80/20
SOURCES = {
    'coco_train': (f'{HF}/Bekhouche/COCO-Text_V2_STR/resolve/main/data/train-00000-of-00001.parquet', 'train'),
    'coco_val': (f'{HF}/Bekhouche/COCO-Text_V2_STR/resolve/main/data/valid-00000-of-00001.parquet', 'val'),
    'ic15_train': (f'{HF}/MiXaiLL76/ICDAR2015_OCR/resolve/main/data/train-00000-of-00001.parquet', 'train'),
    'ic15_val': (f'{HF}/MiXaiLL76/ICDAR2015_OCR/resolve/main/data/test-00000-of-00001.parquet', 'val'),
    'textocr_train': (f'{HF}/MiXaiLL76/TextOCR_OCR/resolve/main/data/train-00000-of-00005.parquet', 'train'),
    'textocr_val': (f'{HF}/MiXaiLL76/TextOCR_OCR/resolve/main/data/test-00000-of-00001.parquet', 'val'),
    'hw_ru': (f'{HF}/Foximaz/russian_ocr_small/resolve/main/handwriting/test-00000.parquet', 'split'),
    'plates_ru': (f'{HF}/Foximaz/russian_ocr_small/resolve/main/car_plate/test-00000.parquet', 'split'),
}


def download(name, url):
    path = os.path.join(RAW_DIR, name + '.parquet')
    if not os.path.exists(path):
        print('качаю', url)
        urllib.request.urlretrieve(url, path)
    return path


def good_crop(img):
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return False
    # вертикальный текст и одиночные буквы выкидываем, в тесте такого почти нет.
    # плюс в coco-text часть слов повернута на 90, они как раз узкие и высокие
    return w >= h


def process(path):
    table = pq.read_table(path)
    images = table.column('image').to_pylist()
    texts = table.column('text').to_pylist()
    arrs, keep_texts = [], []
    for im, text in tqdm(zip(images, texts), total=len(texts), desc=os.path.basename(path)):
        if not text or not text.strip():
            continue
        img = cv2.imdecode(np.frombuffer(im['bytes'], np.uint8), cv2.IMREAD_COLOR)
        if img is None or not good_crop(img):
            continue
        arrs.append(prepare_image(img))
        keep_texts.append(text.strip())
    return arrs, keep_texts


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(0)
    for name, (url, role) in SOURCES.items():
        path = download(name, url)
        arrs, texts = process(path)
        if role == 'split':
            idx = rng.permutation(len(arrs))
            cut = int(len(idx) * 0.8)
            parts = {name + '_train': idx[:cut], name + '_val': idx[cut:]}
        else:
            parts = {name: np.arange(len(arrs))}
        for part, ids in parts.items():
            save_pack(os.path.join(OUT_DIR, part + '.npz'), [arrs[i] for i in ids],
                      texts=np.array([texts[i] for i in ids]))
            print(f'{part}: {len(ids)} картинок')


if __name__ == '__main__':
    main()
