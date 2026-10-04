#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5: VIDUTINIU IR MAZU AKCIJU ARCHYVAS (5 min nuo 2020-10)  -  2026-10-04
=======================================================================
Registracija: claude/strategijos-testai-2-registracija.md (T5).

Universas parenkamas pagal 2020 m. rugsejo pabaigos apyvarta (EODHD bulk
dienos kainos), o ne pagal siandiena - taip i ji patenka ir velau isbrauktos
akcijos (mazesnis isgyvenusiuju iskraipymas).
  JAV: paprastos akcijos, kaina > 5 $, mediana dienos apyvarta 10-100 mln. $,
       iki 300 akciju (didziausios apyvartos is juostos).
  EU (XETRA, PA, AS, MC, BR, IR): paprastos akcijos, apyvarta 1-20 mln. EUR,
       iki 250 akciju.
  Dabartinio (dideliu akciju) universo akcijos neimamos.
Toliau - tas pats parsisiuntimas kaip archyvas.py (5 min, dienos, splitai).

    python archyvas_mid.py --savitikra
    python archyvas_mid.py [--tik N]          (reikia EODHD_TOKEN)
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import archyvas as AR
from archyvas import gauti

DATOS = ["2020-09-21", "2020-09-23", "2020-09-25", "2020-09-29", "2020-10-01"]
JUOSTOS = {"US": (10e6, 100e6, 300), "EU": (1e6, 20e6, 250)}
EU_BIRZOS = ["XETRA", "PA", "AS", "MC", "BR", "IR"]
YAHOO_GALUNE = {"XETRA": "DE", "PA": "PA", "AS": "AS", "MC": "MC", "BR": "BR", "IR": "IR"}
APLANKAS = "archyvas_mid"


def paprastos(sarasai):
    """{kodas} paprastu akciju is exchange-symbol-list (dabartiniu ir isbrauktu)."""
    out = set()
    for sl in sarasai:
        for x in sl or []:
            if str(x.get("Type", "")).lower() == "common stock":
                out.add(x.get("Code"))
    return out


def atrinkti(bulk_dienos, bendros, dabartiniai, juosta, min_kaina=0.0):
    """bulk_dienos: [ [ {code, close, volume}, ... ], ... ] -> rikiuotas kodu sarasas."""
    ap = {}
    kaina = {}
    for diena in bulk_dienos:
        for x in diena or []:
            k = x.get("code")
            try:
                c, v = float(x.get("close")), float(x.get("volume"))
            except (TypeError, ValueError):
                continue
            if not (c > 0 and v >= 0):
                continue
            ap.setdefault(k, []).append(c * v)
            kaina[k] = c
    lo, hi, n = juosta
    eil = [(k, float(np.median(v))) for k, v in ap.items()
           if k in bendros and k not in dabartiniai and kaina.get(k, 0) > min_kaina and len(v) >= 3]
    eil = [(k, m) for k, m in eil if lo <= m <= hi]
    eil.sort(key=lambda x: -x[1])
    return [k for k, _ in eil[:n]]


def universas_mid(token):
    import universas as U
    dab_eu = {t.rsplit(".", 1)[0] for t in U.visi_tikeriai()}
    dab_us = {t for l in U.UNIVERSAS_US.values() for t in l}
    out = []
    # JAV
    sl = [gauti("exchange-symbol-list/US", token)[0], gauti("exchange-symbol-list/US", token, delisted=1)[0]]
    bulk = [gauti("eod-bulk-last-day/US", token, date=d)[0] for d in DATOS]
    us = atrinkti(bulk, paprastos(sl), dab_us, JUOSTOS["US"], min_kaina=5.0)
    out += [(k, f"{k}.US", "us") for k in us]
    print(f"  JAV: paprastu akciju {len(paprastos(sl))}, atrinkta {len(us)}")
    # EU - kiekviena birza atskirai, tada bendra juosta per visas
    kand = []
    for b in EU_BIRZOS:
        sl = [gauti(f"exchange-symbol-list/{b}", token)[0], gauti(f"exchange-symbol-list/{b}", token, delisted=1)[0]]
        bulk = [gauti(f"eod-bulk-last-day/{b}", token, date=d)[0] for d in DATOS]
        k = atrinkti(bulk, paprastos(sl), dab_eu, (JUOSTOS["EU"][0], JUOSTOS["EU"][1], 10 ** 6))
        ap = {}
        for diena in bulk:
            for x in diena or []:
                try:
                    ap.setdefault(x["code"], []).append(float(x["close"]) * float(x["volume"]))
                except (KeyError, TypeError, ValueError):
                    pass
        kand += [(c, b, float(np.median(ap[c]))) for c in k]
        print(f"  {b}: atrinkta juostoje {len(k)}")
    kand.sort(key=lambda x: -x[2])
    for c, b, _ in kand[:JUOSTOS["EU"][2]]:
        out.append((f"{c}.{YAHOO_GALUNE[b]}", f"{c}.{b}", "eu"))
    return out


def paleisti(tik=None):
    token = os.environ.get("EODHD_TOKEN", "").strip()
    if not token:
        sys.exit("KLAIDA: EODHD_TOKEN nenustatytas")
    print("1) UNIVERSAS (pagal 2020-09 apyvarta)")
    sar = universas_mid(token)
    if tik:
        sar = [x for x in sar if x[2] == "us"][:tik] + [x for x in sar if x[2] == "eu"][:tik]
    os.makedirs(APLANKAS, exist_ok=True)
    pd.DataFrame(sar, columns=["tikeris", "eodhd", "rinka"]).to_csv(f"{APLANKAS}/universas.csv", index=False)
    AR.APLANKAS = APLANKAS
    for r in ("eu", "us"):
        os.makedirs(f"{APLANKAS}/5m/{r}", exist_ok=True)
    os.makedirs(f"{APLANKAS}/dienos", exist_ok=True)
    print(f"\n2) PARSISIUNTIMAS: {len(sar)} akciju")
    t0, eil = time.time(), []
    with ThreadPoolExecutor(AR.SRAUTAI) as ex:
        for k, (r, _) in enumerate(ex.map(AR.vienas, [(t, s, rn, token) for t, s, rn in sar]), 1):
            eil.append(r)
            if k % 50 == 0 or k == len(sar):
                print(f"   {k}/{len(sar)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    rep = pd.DataFrame(eil)
    rep.to_csv(f"{APLANKAS}/eodhd_ataskaita.csv", index=False)
    AR.santrauka(rep, [], [])


def savitikra():
    ok = True

    def t(s, a, b):
        nonlocal ok
        g = a == b
        print(f"  {'OK ' if g else 'BLOGAI'}  {s}{'' if g else f'  (gauta {a}, laukta {b})'}")
        ok = ok and g
    sl = [[dict(Code="A", Type="Common Stock"), dict(Code="E", Type="ETF")], [dict(Code="D", Type="Common Stock")]]
    t("paprastos (su isbrauktomis), be ETF", sorted(paprastos(sl)), ["A", "D"])
    diena = [dict(code="A", close=10, volume=2e6), dict(code="D", close=20, volume=1e6),
             dict(code="E", close=10, volume=5e6), dict(code="L", close=50, volume=1e7),
             dict(code="P", close=3, volume=5e6)]
    bendros = {"A", "D", "L", "P"}
    t("juosta 10-100 mln., be dabartiniu (L), be ETF (E), kaina > 5 (P)",
      atrinkti([diena] * 5, bendros, {"L"}, (10e6, 100e6, 300), min_kaina=5), ["A", "D"])
    t("rikiuota pagal apyvarta, riba n=1", atrinkti([diena] * 5, bendros, set(), (10e6, 1e9, 1)), ["L"])
    t("mazai dienu (<3) - neimama", atrinkti([diena] * 2, bendros, set(), (1, 1e12, 10)), [])
    print("SAVITIKRA: VISKAS GERAI" if ok else "SAVITIKRA: YRA KLAIDU")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--savitikra", action="store_true")
    ap.add_argument("--tik", type=int, default=None)
    a = ap.parse_args()
    sys.exit(savitikra() if a.savitikra else paleisti(a.tik))
