"""
Генератор синтетических боксов с текстом.

Каждый пример делается так:
    1. берем строку (из корпуса ozon, шаблоны типа телефонов/цен, или фразы из объявлений)
    2. берем случайный шрифт с кириллицей и рисуем текст на каком нибудь фоне
       (иногда с обводкой, тенью, соседними строками сверху/снизу и тд)
    3. немного крутим/искажаем перспективу и вырезаем бокс вокруг строки, как это делал бы детектор
    4. портим: уменьшаем до маленького разрешения, блюр, шум, jpeg
    5. прогоняем через общий препроцессинг (серый, высота 32)

ВАЖНО: картинка на выходе всегда стоит правильно. Поворот на 180 и метку делаем уже при обучении,
так одна и та же картинка может быть и в классе 0 и в классе 1, и сетке не за что зацепиться кроме самого текста.
Все "асимметричные" штуки (тень вниз, куски соседних строк и тд) делаются ДО поворота, как и в реальности.

На cpu рисовать на лету во время обучения медленно (ядер всего 4), поэтому генерим заранее:
    python -m src.synth --split train --n 300000 --out data/synth/train.npz
    python -m src.synth --split val --n 10000 --out data/synth/val.npz
"""
import argparse
import os
import zlib
from multiprocessing import Pool

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.common import ROOT, prepare_image, save_pack

FONTS_DIR = os.path.join(ROOT, 'data', 'fonts')
FONTS_LIST = os.path.join(ROOT, 'scripts', 'fonts.txt')
TEXTS_DIR = os.path.join(ROOT, 'data', 'texts')

# как часто берем шрифт из каждой категории (рукописных шрифтов мало, но в тесте такое бывает)
CATEGORY_P = {'Sans Serif': 0.45, 'Serif': 0.14, 'Display': 0.22, 'Monospace': 0.05, 'Handwriting': 0.14}

# типичные фразы из объявлений, их часто пишут на картинках
AD_PHRASES = [
    'Продам', 'Продается', 'Продаю', 'Срочно', 'Торг', 'Торг уместен', 'В наличии', 'Под заказ', 'Новый', 'Новое',
    'Б/у', 'Как новый', 'Отличное состояние', 'Хорошее состояние', 'Доставка', 'Доставка по всей России',
    'Бесплатная доставка', 'Самовывоз', 'Гарантия', 'Гарантия 1 год', 'Звоните', 'Пишите', 'Пишите в WhatsApp',
    'Звоните в любое время', 'Аренда', 'Сдается', 'Сдам', 'Сниму', 'Куплю', 'Обмен', 'Скидка', 'Скидки', 'Акция',
    'Распродажа', 'Оригинал', 'Опт и розница', 'Без посредников', 'Собственник', 'Кредит', 'Рассрочка',
    'Рассрочка без банка', 'Установка', 'Ремонт', 'Ремонт под ключ', 'Выезд мастера', 'Недорого', 'Дешево',
    'Цена договорная', 'Вопросы в лс', 'Подробнее по телефону', 'Все вопросы по телефону', 'Фото реальные',
    'Выгодно', 'Хит продаж', 'Новинка', 'Лучшая цена', 'Работаем без выходных', 'Ежедневно', 'Круглосуточно',
    'SALE', 'NEW', 'Made in Russia', 'Made in China', 'Made in Japan', 'Original', 'Premium', 'Free delivery',
]
CITIES = ['Москва', 'Санкт-Петербург', 'Казань', 'Новосибирск', 'Екатеринбург', 'Самара', 'Омск', 'Уфа', 'Пермь',
          'Краснодар', 'Воронеж', 'Ростов-на-Дону', 'Нижний Новгород', 'Челябинск', 'Тюмень', 'Барнаул', 'Сочи']
STREETS = ['Ленина', 'Мира', 'Советская', 'Гагарина', 'Пушкина', 'Кирова', 'Садовая', 'Лесная', 'Молодежная',
           'Центральная', 'Школьная', 'Октябрьская', 'Набережная', 'Заводская', 'Строителей', 'Победы']
PLATE_LETTERS = 'АВЕКМНОРСТУХ'


# ---------------------------------------------------------------------------------------------
# шрифты
# ---------------------------------------------------------------------------------------------

def load_fonts(split):
    """Список шрифтов для train или val (разбиение по семействам задано в scripts/fonts.txt)."""
    fonts = {}
    for line in open(FONTS_LIST, encoding='utf-8'):
        family, category, variants, sp = line.strip().split(';')
        if sp != split:
            continue
        for v in variants.split(','):
            path = os.path.join(FONTS_DIR, f"{family.replace(' ', '_')}-{v}.ttf")
            if os.path.exists(path):
                fonts.setdefault(category, []).append(path)
    return fonts


_font_cache = {}
_cmap_cache = {}


def get_font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def get_cmap(path):
    # какие символы есть в шрифте, чтоб не рисовать квадратики вместо букв
    if path not in _cmap_cache:
        _cmap_cache[path] = set(TTFont(path).getBestCmap().keys())
    return _cmap_cache[path]


# ---------------------------------------------------------------------------------------------
# текст
# ---------------------------------------------------------------------------------------------

_lines_cache = {}


def load_lines(name, split):
    # делим строки на train/val по хешу, чтоб в валидации были и незнакомые шрифты и незнакомые тексты
    if (name, split) in _lines_cache:
        return _lines_cache[(name, split)]
    lines = []
    for l in open(os.path.join(TEXTS_DIR, name), encoding='utf-8'):
        l = l.strip()
        if not l:
            continue
        is_val = zlib.crc32(l.encode('utf-8')) % 10 == 0
        if is_val == (split == 'val'):
            lines.append(l)
    _lines_cache[(name, split)] = lines
    return lines


class TextSampler:
    def __init__(self, split, rng):
        self.rng = rng
        self.ru = load_lines('ru.txt', split)
        self.latin = load_lines('latin.txt', split)

    def digits(self, n):
        return ''.join(str(d) for d in self.rng.integers(0, 10, n))

    def template(self):
        """Всякие телефоны, цены, номера, адреса и тд. Такого на фотках объявлений очень много."""
        r = self.rng
        d = self.digits
        kind = r.integers(0, 12)
        if kind == 0:
            return r.choice([f'+7 (9{d(2)}) {d(3)}-{d(2)}-{d(2)}', f'8 9{d(2)} {d(3)} {d(2)} {d(2)}',
                             f'8-9{d(2)}-{d(3)}-{d(2)}-{d(2)}', f'+79{d(9)}', f'тел. 8 (9{d(2)}) {d(3)}-{d(2)}-{d(2)}',
                             f'8 (800) {d(3)}-{d(2)}-{d(2)}'])
        if kind == 1:
            price = int(r.choice([r.integers(1, 100) * 10, r.integers(1, 300) * 100, r.integers(1, 999) * 1000]))
            p = f'{price:,}'.replace(',', ' ')
            return r.choice([f'{p} ₽', f'{p} руб.', f'{p}р', f'от {p} руб', f'Цена: {p} руб.',
                             f'{p} rub', f'${price}', f'{p},00', f'-{r.integers(5, 70)}%', f'скидка {r.integers(5, 70)}%'])
        if kind == 2:
            l = lambda: r.choice(list(PLATE_LETTERS))
            return f'{l()}{d(3)}{l()}{l()} {r.choice([d(2), d(3)])}'
        if kind == 3:
            return r.choice([f'{r.integers(1, 29):02d}.{r.integers(1, 13):02d}.{r.integers(1995, 2026)}',
                             f'{r.integers(1990, 2026)} г.', f'с {r.integers(7, 11)}:00 до {r.integers(17, 23)}:00',
                             f'ПН-ПТ {r.integers(8, 11)}:00-{r.integers(17, 21)}:00', f'{r.integers(0, 24):02d}:{r.integers(0, 60):02d}'])
        if kind == 4:
            name = ''.join(r.choice(list('abcdefghijklmnopqrstuvwxyz'), r.integers(3, 10)))
            return r.choice([f'www.{name}.ru', f'{name}.com', f'@{name}', f'@{name}_shop', f'{name}@mail.ru',
                             f'https://{name}.ru', f'vk.com/{name}', f't.me/{name}'])
        if kind == 5:
            alnum = list('ABCDEFGHJKLMNPRSTUVWXYZ0123456789')
            code = ''.join(r.choice(alnum, r.integers(5, 17)))
            return r.choice([code, f'SN: {code}', f'S/N {code}', f'Арт. {d(r.integers(4, 9))}', f'арт {d(6)}-{d(2)}',
                             f'ИНН {d(10)}', f'ГОСТ {d(4)}-{r.integers(1990, 2025)}', f'VIN {code}', f'Model: {code[:6]}'])
        if kind == 6:
            return r.choice([f'{r.integers(38, 56)}-{r.integers(40, 58)}', r.choice(['XS', 'S', 'M', 'L', 'XL', 'XXL']),
                             f'{r.integers(10, 300)}x{r.integers(10, 300)} см', f'{r.integers(1, 99)} мм',
                             f'{r.integers(1, 5)}.{r.integers(0, 9)} кВт', f'220V {r.integers(1, 3000)}W',
                             f'{r.integers(1, 99)} кг', f'{r.integers(1, 20)} л', f'{r.integers(16, 1024)} GB'])
        if kind == 7:
            return r.choice([f'г. {r.choice(CITIES)}', f'ул. {r.choice(STREETS)}, д. {r.integers(1, 150)}',
                             f'{r.choice(CITIES)}, ул. {r.choice(STREETS)} {r.integers(1, 99)}',
                             f'пр-т {r.choice(STREETS)} {r.integers(1, 99)}к{r.integers(1, 5)}'])
        if kind == 8:
            return d(r.integers(1, 8))
        return str(r.choice(AD_PHRASES))

    def take_words(self, line):
        # сколько слов берем: чаще короткие куски, в тесте много одиночных слов
        n = int(self.rng.choice([1, 2, 3, 4, 5, 6, 8, 10], p=[.26, .22, .15, .1, .08, .08, .06, .05]))
        words = line.split()
        if len(words) > n:
            start = self.rng.integers(0, len(words) - n + 1)
            words = words[start:start + n]
        text = ' '.join(words)
        return text[:60]

    def sample(self):
        r = self.rng.random()
        if r < 0.55:
            text = self.take_words(self.ru[self.rng.integers(len(self.ru))])
        elif r < 0.72:
            text = self.take_words(self.latin[self.rng.integers(len(self.latin))])
        else:
            text = self.template()

        # регистр: капсом на вывесках пишут очень часто
        r = self.rng.random()
        if r < 0.28:
            text = text.upper()
        elif r < 0.36:
            text = text.lower()
        elif r < 0.44:
            text = text.title()
        return text


# ---------------------------------------------------------------------------------------------
# рисование
# ---------------------------------------------------------------------------------------------

def luminance(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def random_colors(rng):
    """Цвет текста и фона. Следим чтобы контраст был, но иногда делаем и слабый (такое тоже бывает)."""
    if rng.random() < 0.45:
        # самый частый вариант: темный текст на светлом или наоборот
        dark = rng.integers(0, 70, 3)
        light = rng.integers(170, 256, 3)
        return (dark, light) if rng.random() < 0.65 else (light, dark)
    min_contrast = 25 if rng.random() < 0.15 else 60
    for _ in range(50):
        fg = rng.integers(0, 256, 3)
        bg = rng.integers(0, 256, 3)
        if abs(luminance(fg) - luminance(bg)) >= min_contrast:
            return fg, bg
    return np.array([0, 0, 0]), np.array([255, 255, 255])


def make_background(h, w, color, rng):
    """Фон: однотонный, градиент, шумная текстура или из двух частей."""
    color = color.astype(np.float32)
    kind = rng.choice(['solid', 'gradient', 'noise', 'split'], p=[0.45, 0.25, 0.2, 0.1])
    bg = np.empty((h, w, 3), np.float32)
    bg[:] = color
    if kind == 'gradient':
        other = np.clip(color + rng.normal(0, 40, 3), 0, 255)
        if rng.random() < 0.5:
            t = np.linspace(0, 1, w)[None, :, None]
        else:
            t = np.linspace(0, 1, h)[:, None, None]
        bg = (bg * (1 - t) + other * t).astype(np.float32)
    elif kind == 'noise':
        # низкочастотный шум: маленькую случайную картинку растягиваем до нужного размера
        small = rng.normal(0, rng.uniform(10, 40), (max(2, h // 8), max(2, w // 8), 3)).astype(np.float32)
        bg = bg + cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    elif kind == 'split':
        other = np.clip(color + rng.normal(0, 50, 3), 0, 255)
        if rng.random() < 0.5:
            bg[:, int(rng.integers(0, w)):] = other
        else:
            bg[int(rng.integers(0, h)):, :] = other
    if rng.random() < 0.3:
        # чуть мелкого шума чтоб фон не был идеально гладким
        bg += rng.standard_normal(bg.shape, dtype=np.float32) * rng.uniform(1, 6)
    return bg


def draw_line_mask(text, font, spacing=0.0, stroke=0):
    """Рисует строку белым по черному, возвращает маску (uint8) обрезанную по тексту."""
    if spacing == 0:
        l, t, r, b = font.getbbox(text, stroke_width=stroke)
        img = Image.new('L', (r - l + 4, b - t + 4), 0)
        ImageDraw.Draw(img).text((2 - l, 2 - t), text, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
    else:
        # рисуем по одной букве с доп. расстоянием (разрядка, бывает на вывесках)
        size = font.size
        widths = [font.getlength(ch) for ch in text]
        total = int(sum(widths) + spacing * size * len(text) + size + 4)
        img = Image.new('L', (total, int(size * 2)), 0)
        d = ImageDraw.Draw(img)
        x = 2
        for ch, cw in zip(text, widths):
            d.text((x, size // 2), ch, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
            x += cw + spacing * size
    arr = np.array(img)
    ys, xs = np.nonzero(arr > 0)
    if len(ys) == 0:
        return None
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


class SynthGenerator:
    def __init__(self, split, seed):
        self.rng = np.random.default_rng(seed)
        self.fonts = load_fonts(split)
        self.categories = [c for c in CATEGORY_P if c in self.fonts]
        p = np.array([CATEGORY_P[c] for c in self.categories])
        self.cat_p = p / p.sum()
        self.texts = TextSampler(split, self.rng)

    def pick_font(self):
        cat = self.rng.choice(self.categories, p=self.cat_p)
        paths = self.fonts[cat]
        return paths[self.rng.integers(len(paths))]

    def pick_text(self, font_path):
        # выкидываем символы которых нет в шрифте. Если выкинули много, то берем другой текст
        cmap = get_cmap(font_path)
        for _ in range(10):
            text = self.texts.sample()
            clean = ''.join(ch for ch in text if ch == ' ' or ord(ch) in cmap).strip()
            if clean and len(clean) >= 0.8 * len(text):
                return clean
        return None

    def render_block(self, font, texts, main_idx, spacing, stroke, line_gap):
        """
        Рисует основную строку и (если есть) соседние строки над и под ней.
        Возвращает маску заливки, маску с обводкой и маску основной строки (по ней потом режем бокс).
        """
        masks = []
        for t in texts:
            m_fill = draw_line_mask(t, font, spacing, 0)
            if m_fill is None:
                return None
            m_stroke = draw_line_mask(t, font, spacing, stroke) if stroke else m_fill
            masks.append((m_fill, m_stroke))
        pad = stroke + 2
        width = max(ms.shape[1] for _, ms in masks) + 2 * pad
        heights = [ms.shape[0] for _, ms in masks]
        gap = int(line_gap * font.size)
        height = sum(heights) + gap * (len(masks) - 1) + 2 * pad
        fill = np.zeros((height, width), np.uint8)
        outline = np.zeros((height, width), np.uint8)
        main = np.zeros((height, width), np.uint8)
        y = pad
        for i, (mf, ms) in enumerate(masks):
            # выравнивание строк: по левому краю или по центру
            x = pad if self.rng.random() < 0.5 else (width - ms.shape[1]) // 2
            dy = (ms.shape[0] - mf.shape[0]) // 2
            dx = (ms.shape[1] - mf.shape[1]) // 2
            outline[y:y + ms.shape[0], x:x + ms.shape[1]] = np.maximum(outline[y:y + ms.shape[0], x:x + ms.shape[1]], ms)
            fill[y + dy:y + dy + mf.shape[0], x + dx:x + dx + mf.shape[1]] = mf
            if i == main_idx:
                main[y:y + ms.shape[0], x:x + ms.shape[1]] = ms
            y += ms.shape[0] + gap
        return fill, outline, main

    def sample(self):
        for _ in range(20):
            out = self._sample()
            if out is not None:
                return out
        raise RuntimeError('не получилось сгенерить картинку')

    def _sample(self):
        rng = self.rng
        font_path = self.pick_font()
        text = self.pick_text(font_path)
        if text is None:
            return None
        size = int(rng.choice([18, 22, 26, 30, 36, 42, 48]))
        font = get_font(font_path, size)

        spacing = rng.uniform(0.05, 0.4) if rng.random() < 0.12 else 0.0
        stroke = int(rng.integers(1, max(2, size // 10) + 1)) if rng.random() < 0.15 else 0

        # иногда рисуем соседние строки, потом при обрезке от них останутся куски сверху/снизу
        texts, main_idx = [text], 0
        r = rng.random()
        if r < 0.12:
            texts, main_idx = [self.pick_text(font_path), text], 1
        elif r < 0.2:
            texts, main_idx = [text, self.pick_text(font_path)], 0
        elif r < 0.26:
            texts, main_idx = [self.pick_text(font_path), text, self.pick_text(font_path)], 1
        if any(t is None for t in texts):
            texts, main_idx = [text], 0
        block = self.render_block(font, texts, main_idx, spacing, stroke, line_gap=rng.uniform(0.15, 0.6))
        if block is None:
            return None
        fill, outline, main = block

        # растяжение/сжатие по горизонтали (узкие и широкие начертания)
        if rng.random() < 0.3:
            k = rng.uniform(0.7, 1.35)
            new_w = max(4, int(fill.shape[1] * k))
            fill, outline, main = [cv2.resize(m, (new_w, m.shape[0]), interpolation=cv2.INTER_LINEAR)
                                   for m in (fill, outline, main)]
        # наклон как у курсива
        if rng.random() < 0.12:
            fill, outline, main = self.shear([fill, outline, main], rng.uniform(-0.3, 0.3))

        # добавляем поля, чтобы было куда вращать и откуда резать бокс
        m = size // 2 + 2
        fill, outline, main = [cv2.copyMakeBorder(x, m, m, m, m, cv2.BORDER_CONSTANT, value=0)
                               for x in (fill, outline, main)]
        h, w = fill.shape

        # ---- собираем цветную картинку ----
        fg, bgc = random_colors(rng)
        img = make_background(h, w, bgc, rng)

        if rng.random() < 0.08:
            # плашка под текстом (цветной прямоугольник)
            plate = np.clip(bgc + rng.normal(0, 60, 3), 0, 255)
            ys, xs = np.nonzero(main)
            y0, y1 = ys.min() - int(rng.integers(0, m // 2 + 1)), ys.max() + int(rng.integers(0, m // 2 + 1))
            x0, x1 = xs.min() - int(rng.integers(0, m + 1)), xs.max() + int(rng.integers(0, m + 1))
            img[max(0, y0):y1, max(0, x0):x1] = plate

        a_fill = fill.astype(np.float32)[..., None] / 255
        a_out = outline.astype(np.float32)[..., None] / 255
        fg = fg.astype(np.float32)
        if rng.random() < 0.12:
            # тень: сдвинутая размытая копия текста, чаще вниз-вправо (свет обычно сверху)
            dx, dy = int(rng.integers(-1, 4)), int(rng.integers(0, 4))
            sh = np.roll(np.roll(outline, dy, axis=0), dx, axis=1).astype(np.float32)
            sh = cv2.GaussianBlur(sh, (0, 0), rng.uniform(0.5, 2.0))[..., None] / 255
            img = img * (1 - sh * 0.8) + np.float32(20) * sh * 0.8
        if stroke:
            stroke_color = rng.integers(0, 256, 3).astype(np.float32)
            img = img * (1 - a_out) + stroke_color * a_out
        # иногда текст не одним цветом а с легким градиентом
        if rng.random() < 0.15:
            text_color = fg + np.linspace(-40, 40, h, dtype=np.float32)[:, None, None] * rng.choice([-1, 1])
        else:
            text_color = fg
        img = img * (1 - a_fill) + text_color * a_fill

        if rng.random() < 0.05:
            # подчеркивание под основной строкой
            ys, xs = np.nonzero(main)
            yy = min(h - 1, ys.max() + int(rng.integers(1, 4)))
            img[yy:yy + max(1, size // 15), xs.min():xs.max()] = fg

        img = np.clip(img, 0, 255).astype(np.uint8)

        # ---- геометрия: небольшой поворот / перспектива ----
        img, main = self.warp(img, main)

        # ---- вырезаем бокс вокруг основной строки ----
        crop = self.crop_box(img, main)
        if crop is None:
            return None

        # ---- портим качество ----
        crop = self.degrade(crop)
        return prepare_image(crop), text

    def shear(self, masks, k):
        h, w = masks[0].shape
        shift = abs(k) * h
        mat = np.float32([[1, k, shift if k < 0 else 0], [0, 1, 0]])
        new_w = int(w + shift) + 1
        return [cv2.warpAffine(x, mat, (new_w, h)) for x in masks]

    def warp(self, img, main):
        rng = self.rng
        h, w = main.shape
        if rng.random() < 0.5:
            angle = rng.normal(0, 1.5)
            mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
            main = cv2.warpAffine(main, mat, (w, h))
        if rng.random() < 0.25:
            d = 0.06 * min(h, w)
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = src + rng.uniform(-d, d, (4, 2)).astype(np.float32)
            mat = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
            main = cv2.warpPerspective(main, mat, (w, h))
        return img, main

    def crop_box(self, img, main):
        """Режем прямоугольник вокруг основной строки со случайными полями (как у неидеального детектора)."""
        rng = self.rng
        ys, xs = np.nonzero(main > 64)
        if len(ys) == 0:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        th = y1 - y0
        # поля сверху и снизу независимые, распределение одинаковое -> никакой подсказки про ориентацию
        top = int(th * rng.uniform(-0.08, 0.35))
        bottom = int(th * rng.uniform(-0.08, 0.35))
        left = int(th * rng.uniform(-0.05, 0.6))
        right = int(th * rng.uniform(-0.05, 0.6))
        H, W = img.shape[:2]
        y0, y1 = max(0, y0 - top), min(H, y1 + bottom)
        x0, x1 = max(0, x0 - left), min(W, x1 + right)
        if y1 - y0 < 4 or x1 - x0 < 4:
            return None
        return img[y0:y1, x0:x1]

    def degrade(self, img):
        """Уменьшаем до случайной высоты (в тесте много мелких картинок), блюр, шум, jpeg."""
        rng = self.rng
        # высота как в тесте: от ~12 до ~100 пикселей, чаще маленькие
        target_h = int(np.exp(rng.uniform(np.log(12), np.log(100))))
        h, w = img.shape[:2]
        if target_h < h:
            new_w = max(4, int(w * target_h / h))
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        if rng.random() < 0.35:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
        if rng.random() < 0.08:
            # смаз от движения по горизонтали
            k = int(rng.integers(2, 5))
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0 / k
            img = cv2.filter2D(img, -1, kernel)
        img = img.astype(np.float32)
        if rng.random() < 0.4:
            # яркость / контраст
            img = img * rng.uniform(0.6, 1.3) + rng.uniform(-40, 40)
        if rng.random() < 0.15:
            # неравномерное освещение
            g = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.9, 1.3), img.shape[1])[None, :, None]
            img = img * g
        if rng.random() < 0.4:
            img = img + rng.normal(0, rng.uniform(2, 12), img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        if rng.random() < 0.5:
            q = int(rng.integers(15, 90))
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img


# ---------------------------------------------------------------------------------------------
# генерация пачки в несколько процессов
# ---------------------------------------------------------------------------------------------

CHUNK = 2000


def _gen_chunk(args):
    split, seed, n = args
    gen = SynthGenerator(split, seed)
    arrs, texts = [], []
    for _ in range(n):
        a, t = gen.sample()
        arrs.append(a)
        texts.append(t)
    return arrs, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', choices=['train', 'val'])
    parser.add_argument('--n', type=int, default=300000)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    # каждая пачка со своим сидом -> результат не зависит от числа процессов
    n_chunks = (args.n + CHUNK - 1) // CHUNK
    jobs = [(args.split, args.seed * 100000 + i + (50000 if args.split == 'val' else 0),
             min(CHUNK, args.n - i * CHUNK)) for i in range(n_chunks)]
    arrs, texts = [], []
    with Pool(args.workers) as pool:
        for a, t in tqdm(pool.imap(_gen_chunk, jobs), total=len(jobs)):
            arrs += a
            texts += t
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_pack(args.out, arrs, texts=np.array(texts))
    print('сохранил', len(arrs), 'картинок в', args.out)


if __name__ == '__main__':
    main()
