"""
Качаем шрифты с кириллицей из Google Fonts в data/fonts/.

Список семейств лежит рядом в fonts.txt, формат строки:
    Название;категория;начертания через запятую;train/val

Список я один раз собрал из метаданных google fonts (https://fonts.google.com/metadata/fonts):
оставил семейства где есть кириллица, выкинул японские/китайские/корейские (они огромные)
и совсем нечитаемые декоративные. Примерно каждое 7е семейство в каждой категории помечено как val,
эти шрифты не участвуют в обучении, на них проверяем что сетка не запомнила конкретные шрифты.

Фишка: если зайти на css api гугла без браузерного user-agent, он отдает ссылки на обычные ttf файлы.

Запуск:
    python scripts/download_fonts.py
"""
import os
import re
import sys
import time
import urllib.request

from fontTools.ttLib import TTFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS_LIST = os.path.join(ROOT, 'scripts', 'fonts.txt')
OUT_DIR = os.path.join(ROOT, 'data', 'fonts')

# буквы которые обязательно должны быть в шрифте, иначе PIL нарисует квадратики
MUST_HAVE = ('АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя'
             'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')


def css_url(family, variant):
    # variant вида '400', '700', '400i'
    italic = variant.endswith('i')
    weight = variant.rstrip('i')
    fam = family.replace(' ', '+')
    if italic:
        return f'https://fonts.googleapis.com/css2?family={fam}:ital,wght@1,{weight}'
    return f'https://fonts.googleapis.com/css2?family={fam}:wght@{weight}'


def fetch(url, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:
            print('  ошибка', e, 'пробую еще раз')
            time.sleep(2 * (i + 1))
    return None


def has_all_glyphs(path):
    try:
        cmap = TTFont(path).getBestCmap()
    except Exception:
        return False
    return all(ord(ch) in cmap for ch in MUST_HAVE)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = [l.strip() for l in open(FONTS_LIST, encoding='utf-8') if l.strip()]
    ok, bad = 0, 0
    for line in lines:
        family, category, variants, split = line.split(';')
        for v in variants.split(','):
            fname = f"{family.replace(' ', '_')}-{v}.ttf"
            path = os.path.join(OUT_DIR, fname)
            if os.path.exists(path):
                ok += 1
                continue
            css = fetch(css_url(family, v))
            if css is None:
                bad += 1
                continue
            m = re.search(r'url\((https://[^)]+\.ttf)\)', css.decode('utf-8'))
            if not m:
                print('не нашел ttf для', family, v)
                bad += 1
                continue
            data = fetch(m.group(1))
            if data is None:
                bad += 1
                continue
            with open(path, 'wb') as f:
                f.write(data)
            # проверяем что реально есть все буквы, если нет то удаляем
            if not has_all_glyphs(path):
                print('в шрифте нет нужных букв, выкидываю:', fname)
                os.remove(path)
                bad += 1
                continue
            ok += 1
        print(f'{family} готово')
    print(f'скачано {ok}, не получилось {bad}')


if __name__ == '__main__':
    sys.exit(main())
