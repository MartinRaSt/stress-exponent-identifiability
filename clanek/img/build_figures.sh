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
# Znovu vygeneruje VSECHNY obrazky clanku v clanek/img/ volanim dilcich
# build skriptu. Schematicke (SVG) obrazky jsou v clanek/img/src/make_<jmeno>.sh;
# datove obrazky (generovane python-coderem) se pridavaji nize, jak
# pribyvaji, jeden radek na obrazek, ve stejnem vzoru volani.
# Zastavi se na prvnim selhavajicim obrazku (nenulovy navratovy kod), aby
# se chyby nikdy tise nepreskocily.
# Pouziti: build_figures.sh   (z libovolneho adresare)

IMG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$IMG_DIR/src"

echo "========================================"
echo "Building article figures"
echo "========================================"

# --- schematicke obrazky (rucne editovatelne SVG -> PDF) -------------------
if ! "$SRC_DIR/make_method_overview.sh"; then
    echo "========================================"
    echo "Figure build FAILED. See messages above."
    echo "========================================"
    exit 1
fi

if ! "$SRC_DIR/make_graphical_abstract.sh"; then
    echo "========================================"
    echo "Figure build FAILED. See messages above."
    echo "========================================"
    exit 1
fi

if ! "$SRC_DIR/make_study_design.sh"; then
    echo "========================================"
    echo "Figure build FAILED. See messages above."
    echo "========================================"
    exit 1
fi

# --- datove obrazky (pridavejte sem: volani "<skript>.sh" + kontrola navr. kodu) ---

echo "========================================"
echo "All figures built successfully."
echo "========================================"
exit 0
