"""
Обучение модели.

    python -m src.train --name base_w1
    python -m src.train --name w05 --width_mult 0.5   # сеть в 2 раза уже, для сравнения

Что происходит:
    - трейн: синтетика + реальные кропы (доля реальных в батче задается в конфиге),
      метка появляется из случайного поворота на 180
    - валидация: отдельная синтетика (другие шрифты и тексты) + реальные кропы из тестовых частей датасетов
    - лучшую эпоху выбираем по среднему скору (1 - Brier) на синт. валидации и на реальной валидации
    - все сохраняем в runs/<name>/
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.common import ROOT, load_config, seed_everything
from src.data import RotDataset, load_packs, mix_weights, worker_init
from src.evaluate import metrics, predict_logits, sigmoid
from src.models import build_model, count_params


def make_val_loaders(cfg, width):
    loaders = {}
    for name in cfg['data']['synth_val']:
        loaders['synth_' + name] = load_packs([name], 'synth')
    for name in cfg['data']['real_val']:
        loaders[name] = load_packs([name], 'real')
    return {k: DataLoader(RotDataset(v, width, train=False, seed=1), batch_size=512, num_workers=0)
            for k, v in loaders.items()}


def validate(model, val_loaders):
    res = {}
    all_real_y, all_real_p = [], []
    for name, loader in val_loaders.items():
        logits, y = predict_logits(model, loader)
        p = sigmoid(logits)
        res[name] = metrics(y, p)
        if not name.startswith('synth'):
            all_real_y.append(y)
            all_real_p.append(p)
    res['real_all'] = metrics(np.concatenate(all_real_y), np.concatenate(all_real_p))
    synth_keys = [k for k in res if k.startswith('synth')]
    res['select'] = (np.mean([res[k]['score'] for k in synth_keys]) + res['real_all']['score']) / 2
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(ROOT, 'config.yaml'))
    parser.add_argument('--name', required=True)
    parser.add_argument('--width_mult', type=float, default=None, help='поменять ширину сети (по умолчанию из конфига)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.width_mult is not None:
        cfg['model']['width_mult'] = args.width_mult
    seed_everything(cfg['seed'])
    tc = cfg['train']
    torch.set_num_threads(tc['threads'])
    out_dir = os.path.join(ROOT, 'runs', args.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    width = cfg['input']['width']
    synth = load_packs(cfg['data']['synth_train'], 'synth')
    real = load_packs(cfg['data']['real_train'], 'real')
    train_ds = RotDataset(synth + real, width, train=True, seed=cfg['seed'])
    weights = mix_weights(synth, real, cfg['data']['real_share'])
    gen = torch.Generator().manual_seed(cfg['seed'])
    sampler = WeightedRandomSampler(weights, num_samples=tc['samples_per_epoch'], replacement=True, generator=gen)
    train_loader = DataLoader(train_ds, batch_size=tc['batch_size'], sampler=sampler, num_workers=tc['workers'],
                              worker_init_fn=worker_init, drop_last=True, persistent_workers=tc['workers'] > 0)
    val_loaders = make_val_loaders(cfg, width)
    print(f'трейн: синтетика {sum(map(len, synth))}, реальные {sum(map(len, real))}')

    model = build_model(cfg)
    # channels_last на cpu дает заметное ускорение сверток (проверял, примерно на треть)
    model = model.to(memory_format=torch.channels_last)
    print('параметров:', count_params(model))

    steps = tc['epochs'] * len(train_loader)
    opt = torch.optim.AdamW(model.parameters(), lr=tc['lr'], weight_decay=tc['weight_decay'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc['lr'], total_steps=steps, pct_start=0.15)
    ls = tc['label_smoothing']

    best, history = -1, []
    for epoch in range(tc['epochs']):
        model.train()
        t0 = time.time()
        losses = []
        for step, (x, y) in enumerate(train_loader):
            x = x.to(memory_format=torch.channels_last)
            target = y * (1 - ls) + 0.5 * ls  # label smoothing
            loss = F.binary_cross_entropy_with_logits(model(x), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            if step % 100 == 0:
                print(f'  эпоха {epoch} шаг {step}/{len(train_loader)} loss {np.mean(losses[-100:]):.4f}', flush=True)

        res = validate(model, val_loaders)
        res['epoch'] = epoch
        res['train_loss'] = float(np.mean(losses))
        res['time'] = time.time() - t0
        history.append(res)
        line = ' | '.join(f"{k} {v['score']:.4f}" for k, v in res.items() if isinstance(v, dict))
        print(f'эпоха {epoch}: loss {res["train_loss"]:.4f}, {line}, select {res["select"]:.4f}, {res["time"]:.0f}с', flush=True)

        torch.save(model.state_dict(), os.path.join(out_dir, 'last.pt'))
        if res['select'] > best:
            best = res['select']
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
        with open(os.path.join(out_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    print('лучший select:', best)


if __name__ == '__main__':
    main()
