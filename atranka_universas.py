#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIVERSO ATRANKA  —  2026-09-28
===============================

Vienintelis tikslas: patikrinti kandidatus TIKRAIS duomenimis ir pasakyti,
kurie tinka. Nieko nespeja. Nepriima nieko "todel, kad tai zinoma bendrove".

Kodel taip: universas.py jau kartą kainavo — 42 tikeriai is 254 negrazino
duomenu, nes buvo speti arba pasene. Todel kandidatu sarasas cia yra tik
PRASYMAS PATIKRINTI, o ne sarasas pridejimui. Kas nepraeina — nepridedama,
ir loge parasyta, kodel.

Paleidimas:
    python atranka_universas.py --rinka eu   --out kandidatai_eu.txt
    python atranka_universas.py --rinka us   --out kandidatai_us.txt
    python atranka_universas.py --rinka eu --tik-esamus   # esamo universo auditas

Kriterijai — KONSTANTOS ZEMIAU. Kiekviena isvesta is Manto mechanikos
(pozicija 18 000, sanaudos, 1% apyvartos dalis), ne is prielaidu apie
"kokybiskas bendroves". Jei keiciam kriteriju, keiciam viena eilute cia.
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("KLAIDA: reikia yfinance")

import detektorius as D

# ============================================================== KRITERIJAI

# --- 1 grupe: MECHANIKA (isvesta is pozicijos dydzio ir sanaudu) ---------

# Ribos PAGAL RINKA, imamos tiesiai is detektoriaus - kad universas ir
# detektorius niekada neisskirtu. JAV sanaudos perpus mazesnes, tad ir ribos
# perpus zemesnes; tai ne nuolaida, o ta pati dalyba is kitos sanaudu sumos.
#
# Apyvarta: pozicija ne daugiau 1% dienos apyvartos. Reikalaujam, kad riba
# butu praeita net RAMIAUSIU laikotarpiu - kitaip akcija kas kelias savaites
# iskristu is universo ir vel sugriztu. Riba kotiravimo valiuta.
def min_apyvarta_blogiausia(rinka):
    return D.min_apyvarta(rinka)


def min_apyvarta_mediana(rinka):
    return D.min_apyvarta(rinka) * 3


# Tikslas = 0.5 ATR ir turi buti bent 10x uz luzio taska.
# EU: 0.556% / 0.5 = 1.11%.  JAV: 0.278% / 0.5 = 0.556%.
# Akcija su mazesniu ATR FIZISKAI negali duoti tikslo, verto sandorio.
def min_atr_pct_mediana(rinka):
    return D.min_atr_pct(rinka)

# Zema kaina = didesnis santykinis spredas ir tick'o itaka. Spredo is dienos
# duomenu nematyti, tad tai artimiausias patikrinamas pakaitalas.
MIN_KAINA = 5.0

# --- 2 grupe: DUOMENU KOKYBE (tiksliai tai, kas pralauze universa anksciau) --
MIN_DIENU = 250                  # 2 m. atsiuntimas turi duoti tikra istorija
MIN_5MIN_SESIJU_DALIS = 0.90     # 5 min barai bent 90% paskutiniu sesiju
MIN_BARU_SESIJOJE = 50           # sesija su 12 baru yra duomenu skyle
MAX_NEJUDANCIU_DIENU = 0.05      # >5% dienu be pokycio = neprekiaujama

# --- 3 grupe: ELGSENA (ar akcija apskritai daro Manto scenarijus) --------
# Tikrinama TOMIS PACIOMIS funkcijomis, kaip detektorius. Ne panasiomis.
MIN_S1_DIENU = 6                 # kritimas >=1 ATR + uzdare apacioje, per 6 men.
MIN_S2_DIENU = 3                 # tarpas >=1 ATR, per 6 men.

MENESIAI = 6

# Ko CIA NERA ir sazinngai nebus: sektoriu balanso, indekso nariu saraso,
# kapitalizacijos ribos, "kokybes". Nei vienas ju neisvedamas is Manto
# mechanikos, o butent tokios paveldetos prielaidos nuzude reitinguotoja.


# ============================================================== patikra

def tikrinti(t, rinka):
    """Grazina (ar_tinka, eilute su skaiciais, atmetimo priezastis)."""
    r = dict(tickeris=t)
    try:
        raw = yf.download(t, period="2y", interval="1d", auto_adjust=False,
                          progress=False, group_by="ticker", actions=True)
        d = D._vienas(raw, t)
    except Exception as e:
        return False, r, f"atsiuntimas nepavyko ({type(e).__name__})"
    if d is None or len(d) < MIN_DIENU:
        return False, r, f"dienu {0 if d is None else len(d)} < {MIN_DIENU}"

    d = d.tail(MENESIAI * 21)
    c, v, h, l = d["Close"], d["Volume"], d["High"], d["Low"]
    r["kaina"] = float(c.iloc[-1])

    nejud = float(((c.diff() == 0) | (v == 0)).mean())
    r["nejud"] = nejud * 100

    apyv = (c * v).rolling(20).median().dropna()
    r["apyv_med"] = float(apyv.median()) if len(apyv) else np.nan
    r["apyv_min"] = float(apyv.min()) if len(apyv) else np.nan

    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr_pct = (tr.rolling(20).mean() / c * 100).dropna()
    r["atr_med"] = float(atr_pct.median()) if len(atr_pct) else np.nan

    # elgsena: tos pacios salygos, kaip scenarijuose, tik dienos lygmeniu
    rod = D.dienos_rodikliai(d, d["Dividends"] if "Dividends" in d else None)
    if rod is None:
        return False, r, "dienos rodikliai neskaiciuojami"
    rod = rod.dropna(subset=["atr_abs", "virsune_n", "uzd_vieta"])
    krit = rod["virsune_n"] - rod["uzdarymas"] - rod["div_lange"]
    s1 = ((krit >= D.S1_MIN_KRITIMAS_ATR * rod["atr_abs"]) &
          (rod["uzd_vieta"] <= D.S1_MAX_UZD_VIETA))
    r["s1_dienu"] = int(s1.sum())
    # rod jau nuvalytas nuo NaN, tad indeksai nebesutampa su d - reikia
    # aiskaus suderinimo, kitaip pandas meta ValueError (ir tai gerai:
    # tyliai suderinta lentele butu skaiciavusi ne tas dienas).
    tarpas = d["Open"].reindex(rod.index) - rod["uzdarymas"]
    r["s2_dienu"] = int((tarpas >= D.S2_MIN_TARPAS_ATR * rod["atr_abs"]).sum())

    # 5 min duomenys — butent cia luzdavo tyliai
    try:
        raw5 = yf.download(t, period="30d", interval="5m", auto_adjust=False,
                           progress=False, group_by="ticker", prepost=False)
        b5 = D._vienas(raw5, t)
    except Exception:
        b5 = None
    if b5 is None or not len(b5):
        r["ses_5min"] = 0
        r["dalis_5min"] = 0.0
    else:
        b5 = D.sesijos_rodikliai(b5, rinka)
        pilnos = b5.groupby("sesija").size()
        pilnos = pilnos[pilnos >= MIN_BARU_SESIJOJE]
        # Vardiklis - TIKROS prekybos dienos tame paciame lange pagal dienos
        # duomenis. Fiksuotas "22" duodavo virs 100%, t.y. matavimas buvo
        # beprasmis kaip tik tada, kai duomenu daugiau nei tiketasi.
        nuo = min(b5["sesija"])
        laukta = len([x for x in d.index if x.date() >= nuo])
        r["ses_5min"] = int(len(pilnos))
        r["dalis_5min"] = len(pilnos) / max(1, laukta)

    # --- sprendimas ---
    for salyga, tekstas in (
        (r["kaina"] >= MIN_KAINA, f"kaina {r['kaina']:.1f} < {MIN_KAINA}"),
        (nejud <= MAX_NEJUDANCIU_DIENU,
         f"nejudanciu dienu {nejud*100:.0f}% > {MAX_NEJUDANCIU_DIENU*100:.0f}%"),
        (r["apyv_min"] >= min_apyvarta_blogiausia(rinka),
         f"ramiausia apyvarta {r['apyv_min']/1e6:.1f} mln < "
         f"{min_apyvarta_blogiausia(rinka)/1e6:.1f}"),
        (r["apyv_med"] >= min_apyvarta_mediana(rinka),
         f"tipine apyvarta {r['apyv_med']/1e6:.1f} mln < "
         f"{min_apyvarta_mediana(rinka)/1e6:.1f}"),
        (r["atr_med"] >= min_atr_pct_mediana(rinka),
         f"ATR {r['atr_med']:.2f}% < {min_atr_pct_mediana(rinka):.2f}% "
         f"(tikslas nevertas sanaudu)"),
        (r["dalis_5min"] >= MIN_5MIN_SESIJU_DALIS,
         f"5 min sesiju tik {r['dalis_5min']*100:.0f}%"),
        (r["s1_dienu"] >= MIN_S1_DIENU,
         f"atsistatymo dienu {r['s1_dienu']} < {MIN_S1_DIENU}"),
        (r["s2_dienu"] >= MIN_S2_DIENU,
         f"ralio dienu {r['s2_dienu']} < {MIN_S2_DIENU}"),
    ):
        if not salyga:
            return False, r, tekstas
    return True, r, ""


def paleisti(kandidatai, rinka, out):
    print(f"\nTikrinama {len(kandidatai)} kandidatu ({rinka.upper()}), "
          f"{MENESIAI} men. duomenys")
    val = "USD" if rinka == "us" else "EUR"
    print(f"  ribos: apyvarta min {min_apyvarta_blogiausia(rinka)/1e6:.2f} / "
          f"tipine {min_apyvarta_mediana(rinka)/1e6:.2f} mln {val}, "
          f"ATR >= {min_atr_pct_mediana(rinka):.2f}%, "
          f"sanaudos {D.sanaudos(rinka):.0f} EUR")
    if rinka == "us":
        print(f"  EUR/USD kursas: {D.eur_usd():.4f}")
    print()
    print(f"  {'tickeris':<12}{'kaina':>8}{'apyv.med':>10}{'apyv.min':>10}"
          f"{'ATR%':>7}{'5min':>7}{'S1':>5}{'S2':>5}  verdiktas")
    print("  " + "-" * 92)
    geri, blogi = [], []
    for t in kandidatai:
        ok, r, kodel = tikrinti(t, rinka)
        (geri if ok else blogi).append((t, r, kodel))
        print(f"  {t:<12}{r.get('kaina', float('nan')):>8.1f}"
              f"{r.get('apyv_med', float('nan'))/1e6:>10.1f}"
              f"{r.get('apyv_min', float('nan'))/1e6:>10.1f}"
              f"{r.get('atr_med', float('nan')):>7.2f}"
              f"{r.get('dalis_5min', 0)*100:>6.0f}%"
              f"{r.get('s1_dienu', 0):>5}{r.get('s2_dienu', 0):>5}"
              f"  {'TINKA' if ok else kodel}")

    print(f"\n  TINKA: {len(geri)}   atmesta: {len(blogi)}")
    priez = {}
    for _, _, k in blogi:
        priez[k.split("(")[0].split()[0]] = priez.get(k.split("(")[0].split()[0], 0) + 1
    if priez:
        print("  atmetimo priezastys:")
        for k, n in sorted(priez.items(), key=lambda x: -x[1]):
            print(f"    {n:>4}x  {k}")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            for t, _, _ in geri:
                f.write(t + "\n")
        print(f"\n  tinkami irasyti: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rinka", choices=["eu", "us"], required=True)
    ap.add_argument("--failas", help="kandidatu sarasas, po viena eiluteje")
    ap.add_argument("--tik-esamus", action="store_true",
                    help="tikrinti dabartini universa (auditas)")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.tik_esamus:
        kand = D.universas(a.rinka)
    elif a.failas:
        kand = [x.strip() for x in open(a.failas, encoding="utf-8")
                if x.strip() and not x.startswith("#")]
    else:
        sys.exit("reikia --failas arba --tik-esamus")
    paleisti(kand, a.rinka, a.out)
