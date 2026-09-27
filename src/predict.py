"""
Предсказания для теста, на выходе submission.csv

    python -m src.predict                                 # финальная модель из weights/
    python -m src.predict --weights runs/base_w1/best.pt  # любой чекпоинт из экспериментов

Картинки берутся из data/test/images, порядок id как в data/sample_submission.csv.
Препроцессинг тот же что и в обучении (src/common.py), вероятность это просто sigmoid от логита.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.common import ROOT, prepare_image, read_image, to_model_input
from src.evaluate import sigmoid
from src.models import build_model


def load_predictor(path, threads=4):
    """Возвращает функцию: по батчу (N, 1, 32, W) numpy выдает логиты (N,). Умеет onnx и торчевые .pt"""
    if path.endswith('.onnx'):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        name = sess.get_inputs()[0].name
        return lambda x: sess.run(None, {name: x})[0].reshape(-1)

    torch.set_num_threads(threads)
    cfg = json.load(open(os.path.join(os.path.dirname(path), 'config.json')))
    model = build_model(cfg)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()

    def run(x):
        with torch.no_grad():
            return model(torch.from_numpy(x)).numpy()
    return run


def load_test(img_dir, ids, width):
    xs = []
    for i in tqdm(ids, desc='читаю картинки'):
        arr = prepare_image(read_image(os.path.join(img_dir, i + '.png')))
        xs.append(to_model_input(arr, width))
    return np.stack(xs)


def predict_logits(predictor, x, batch_size=256):
    out = []
    for s in tqdm(range(0, len(x), batch_size), desc='предикт'):
        out.append(predictor(x[s:s + batch_size]))
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=os.path.join(ROOT, 'weights', 'model.onnx'))
    parser.add_argument('--images', default=os.path.join(ROOT, 'data', 'test', 'images'))
    parser.add_argument('--sample', default=os.path.join(ROOT, 'data', 'sample_submission.csv'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'submission.csv'))
    parser.add_argument('--width', type=int, default=256)
    args = parser.parse_args()

    ids = pd.read_csv(args.sample)['image_id'].tolist()
    x = load_test(args.images, ids, args.width)
    p = sigmoid(predict_logits(load_predictor(args.weights), x))

    sub = pd.DataFrame({'image_id': ids, 'p_180': p})
    sub.to_csv(args.out, index=False)
    print(f'сохранил {args.out}: {len(sub)} строк, среднее p_180 = {p.mean():.4f}')


if __name__ == '__main__':
    main()
