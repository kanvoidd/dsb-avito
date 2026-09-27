"""
Модель.

Маленькая сверточная сеть TinyCNN. Вход серый 1 x 32 x W, выход 1 логит (перевернуто или нет).
Устроена как обычная vgg-шная сетка из OCR (типа CRNN), только без рекуррентной части:
    - несколько блоков conv-bn-relu, по высоте постепенно сжимаем с 32 до 1
    - по ширине в конце просто усредняем (global average pooling), так сетке все равно какой ширины картинка
    - линейный слой выдает логит

Ширину сети можно менять множителем, чтоб сравнить качество и скорость для разных размеров.
"""
import torch.nn as nn


def conv_bn(cin, cout, k=3, stride=1, padding=None):
    if padding is None:
        padding = k // 2 if isinstance(k, int) else (k[0] // 2, k[1] // 2)
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, padding, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TinyCNN(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        c = [max(8, int(round(x * width_mult))) for x in (16, 32, 64, 96, 128)]
        self.features = nn.Sequential(
            conv_bn(1, c[0]),                  # 32 x W
            nn.MaxPool2d(2),                   # 16 x W/2
            conv_bn(c[0], c[1]),
            nn.MaxPool2d(2),                   # 8 x W/4
            conv_bn(c[1], c[2]),
            conv_bn(c[2], c[2]),
            nn.MaxPool2d(2),                   # 4 x W/8
            conv_bn(c[2], c[3]),
            conv_bn(c[3], c[3]),
            nn.MaxPool2d((2, 1)),              # 2 x W/8
            # последняя свертка на всю оставшуюся высоту, получаем 1 x W/8.
            # тут сетка видит сразу и верх и низ строки, это как раз то что нужно для ориентации
            conv_bn(c[3], c[4], k=(2, 3), padding=(0, 1)),
        )
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(c[4], 1)

    def forward(self, x):
        f = self.features(x)          # B x C x 1 x W/8
        f = f.mean(dim=(2, 3))        # B x C
        return self.head(self.dropout(f)).squeeze(1)


def build_model(cfg):
    return TinyCNN(width_mult=cfg['model'].get('width_mult', 1.0))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
