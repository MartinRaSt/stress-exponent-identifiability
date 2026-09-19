#!/usr/bin/env bash
set -euo pipefail
# Projekt: Sammonova projekce znovu - adaptivni vahovani, skalovatelny resic
#          a temporalni rozsireni (alfa-Sammon)
# Autori:  Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Afiliace: Katedra informatiky, Fakulta elektrotechniky a informatiky,
#           VSB - Technicka univerzita Ostrava, 17. listopadu 2172/15,
#           708 00 Ostrava-Poruba, Ceska republika
# Vytvoreno: 2026-09-09
# Licence: viz soubor LICENSE v korenu repozitare
#
# Znovu vygeneruje clanek/img/method_overview.pdf (graficky abstrakt /
# konceptualni schema metody) z rucne editovatelneho zdroje
# clanek/img/src/method_overview.svg.
# Konvertor: svglib + reportlab uvnitr conda prostredi projektu (venv),
# rizeny skriptem clanek/img/src/svg2pdf.py (Inkscape na tomto stroji neni
# na PATH; na Linuxu/macOS lze misto svg2pdf.py pouzit primo "inkscape",
# pokud je k dispozici - zde vsak zachovavame stejnou cestu pres svg2pdf.py
# jako na Windows kvuli shodnemu vystupu).
# Pouziti: make_method_overview.sh   (z libovolneho adresare; cesty se
#          odvozuji relativne k tomuto souboru: <repo>/clanek/img/src/)

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SRC_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-$REPO/venv/bin/python}"
SVG="$SRC_DIR/method_overview.svg"
PDF="$SRC_DIR/../method_overview.pdf"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python interpreter not found: \"$PYTHON\"" >&2
    echo "Nastavte promennou prostredi PYTHON na cestu k venv Pythonu projektu." >&2
    exit 1
fi
if [ ! -e "$SVG" ]; then
    echo "ERROR: SVG source not found: \"$SVG\"" >&2
    exit 1
fi

echo "[method_overview] converting SVG to PDF ..."
export PYTHONIOENCODING=utf-8
if ! "$PYTHON" "$SRC_DIR/svg2pdf.py" "$SVG" "$PDF"; then
    echo "ERROR: SVG to PDF conversion failed for method_overview.svg" >&2
    exit 1
fi
echo "[method_overview] done: \"$PDF\""
exit 0
