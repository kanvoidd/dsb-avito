"""
Экспорт модели в onnx и замер скорости.

    python -m src.export --weights runs/base_w1/best.pt --out weights/model.onnx

В проде модель будет гоняться на каждом боксе, поэтому смотрим:
    - сколько параметров и сколько весит файл
    - сколько миллисекунд на один бокс на ОДНОМ ядре cpu (батч 1 и батч 64)
onnxruntime на cpu заметно быстрее торча, поэтому финальная модель в onnx.
"""
import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import torch

from src.models import build_model, count_params


def export(weights, out, width=256):
    cfg = json.load(open(os.path.join(os.path.dirname(weights), 'config.json')))
    model = build_model(cfg)
    model.load_state_dict(torch.load(weights, map_location='cpu'))
    model.eval()
    x = torch.randn(1, 1, 32, width)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.onnx.export(model, x, out, input_names=['image'], output_names=['logit'],
                      dynamic_axes={'image': {0: 'batch'}, 'logit': {0: 'batch'}}, opset_version=17, dynamo=False)
    return model


def benchmark(fn, x, n_warmup=5, n_iter=30):
    """Сколько миллисекунд занимает один вызов fn(x)."""
    for _ in range(n_warmup):
        fn(x)
    t = time.perf_counter()
    for _ in range(n_iter):
        fn(x)
    return (time.perf_counter() - t) / n_iter * 1000


def speed(onnx_path, width=256, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    res = {}
    for bs in (1, 64):
        x = np.random.randn(bs, 1, 32, width).astype(np.float32)
        ms = benchmark(lambda a: sess.run(None, {name: a}), x, n_iter=50 if bs == 1 else 10)
        res[f'ms_per_box_bs{bs}'] = ms / bs
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--width', type=int, default=256)
    args = parser.parse_args()

    model = export(args.weights, args.out, args.width)

    # проверяем что onnx выдает то же самое что торч
    x = np.random.randn(8, 1, 32, args.width).astype(np.float32)
    sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
    y_onnx = sess.run(None, {'image': x})[0]
    with torch.no_grad():
        y_torch = model(torch.from_numpy(x)).numpy()
    print('макс. разница onnx vs torch:', float(np.abs(y_onnx - y_torch).max()))

    info = {'params': count_params(model), 'size_kb': os.path.getsize(args.out) / 1024}
    info.update(speed(args.out, args.width))
    print(json.dumps(info, indent=2))
    with open(os.path.splitext(args.out)[0] + '_info.json', 'w') as f:
        json.dump(info, f, indent=2)


if __name__ == '__main__':
    main()
