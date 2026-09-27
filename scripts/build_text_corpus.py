"""
Собираем корпус строк для генератора синтетики.

Источник: открытый датасет товаров ozon с huggingface (evgmaslov/ozon_ecup, конфиг cleaned).
Там названия товаров, описания и характеристики, по смыслу это очень близко к тому что пишут
на фотках объявлений: бренды, модели, размеры, цвета, "в наличии", и тд.
Берем только текстовые колонки, картинки и эмбеддинги не нужны.

На выходе:
    data/texts/ru.txt     - строки где есть кириллица
    data/texts/latin.txt  - строки только латиницей/цифрами (бренды, модели, англ текст)

Телефоны, цены, номера машин и тд не отсюда, их генерим шаблонами прямо в src/synth.py

Запуск:
    python scripts/build_text_corpus.py
"""
import html
import json
import os
import random
import re
import urllib.request

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
OUT_DIR = os.path.join(ROOT, 'data', 'texts')

OZON_URL = 'https://huggingface.co/datasets/evgmaslov/ozon_ecup/resolve/main/cleaned/cleaned-00000-of-00002.parquet'
OZON_FILE = os.path.join(RAW_DIR, 'ozon_cleaned_0.parquet')

# оставляем только нормальные символы, всякие эмодзи и редкие значки выкидываем
ALLOWED = re.compile('^[0-9A-Za-z\u0410-\u044f\u0401\u0451\\s.,:;!?()\\[\\]"\'\u00ab\u00bb%\u2116+\\-\u2013\u2014/&*#@_=\u20bd$\u20ac\u00b0\u00d7]+$')

random.seed(0)


def download():
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(OZON_FILE):
        return
    print('качаю', OZON_URL)
    urllib.request.urlretrieve(OZON_URL, OZON_FILE)


def clean(s):
    # убираем html теги и лишние пробелы
    s = re.sub(r'<[^>]+>', ' ', s)
    s = html.unescape(s)
    s = s.replace(' ', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def chunks(sentence, max_words=8):
    # длинные предложения режем на куски, на картинках обычно короткие строки
    words = sentence.split()
    i = 0
    while i < len(words):
        n = random.randint(2, max_words)
        yield ' '.join(words[i:i + n])
        i += n


def main():
    download()
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ['name', 'description', 'categories', 'characteristic_attributes_mapping']
    df = pq.read_table(OZON_FILE, columns=cols).to_pandas()
    # товары для взрослых убираем, там текст специфический
    df = df[~df.categories.fillna('').str.contains('для взрослых')]
    print('товаров:', len(df))

    lines = set()
    df = df.fillna('')  # в описаниях бывают пропуски (nan)
    for name, desc, cats, chars in df.itertuples(index=False):
        if name:
            name = clean(name)
            lines.add(name)
            # название часто через запятую: "Рюкзак на молнии, 2 кармана, цвет синий"
            for part in name.split(','):
                lines.add(part.strip())
        if desc:
            for sent in re.split(r'(?<=[.!?;])\s+', clean(desc)):
                for ch in chunks(sent):
                    lines.add(ch)
        if cats:
            for c in json.loads(cats).values():
                if c != 'EPG':
                    lines.add(c)
        if chars:
            for k, vals in json.loads(chars).items():
                for v in vals[:3]:
                    v = clean(str(v))
                    lines.add(v)
                    lines.add(f'{k}: {v}')

    lines = [l for l in lines if 1 <= len(l) <= 80 and ALLOWED.match(l)]
    ru = sorted(l for l in lines if re.search('[\u0410-\u044f\u0401\u0451]', l))
    latin = sorted(l for l in lines if not re.search('[\u0410-\u044f\u0401\u0451]', l) and re.search('[A-Za-z]', l))
    # из русских строк еще вытаскиваем латинские слова (бренды типа Samsung, модели и тд)
    extra = set()
    for l in ru:
        for tok in re.findall(r'[A-Za-z][A-Za-z0-9\-+.]{1,20}', l):
            extra.add(tok)
    latin = sorted(set(latin) | extra)

    with open(os.path.join(OUT_DIR, 'ru.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(ru) + '\n')
    with open(os.path.join(OUT_DIR, 'latin.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(latin) + '\n')
    print('ru строк:', len(ru), 'latin строк:', len(latin))


if __name__ == '__main__':
    main()
