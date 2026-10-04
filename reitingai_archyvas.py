#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALITIKU REITINGU ARCHYVAS (Yahoo upgrades_downgrades)  -  2026-10-05
=====================================================================
Registracija: claude/reitingai-registracija.md.
Kiekvienam universo tikeriui parsiuncia Yahoo reitingu pakeitimu istorija ir
iraso reitingai/reitingai.csv: tikeris, data, firma, is, i, veiksmas (+ kainos
tikslai, jei Yahoo juos duoda). Parodo padengima pagal rinka ir metus.
Detektoriaus nekeicia.

    python reitingai_archyvas.py --savitikra
    python reitingai_archyvas.py [--tik N]
"""
import argparse
import os
import sys
import time

import pandas as pd

APLANKAS = "reitingai"
STULP = {"Firm": "firma", "FromGrade": "is", "ToGrade": "i", "Action": "veiksmas",
         "priceTargetAction": "tikslo_veiksmas", "currentPriceTarget": "tikslas", "priorPriceTarget": "ankstesnis_tikslas"}


def sutvarkyti(df, t, rinka):
    """yfinance upgrades_downgrades -> standartine lentele (gali buti tuscia ar None)."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["tikeris", "rinka", "data"] + list(STULP.values()))
    d = df.copy()
    if "GradeDate" in d.columns:
        d = d.set_index("GradeDate")
    idx = pd.to_datetime(d.index, errors="coerce", utc=True)
    d = d.rename(columns=STULP)
    out = pd.DataFrame({"tikeris": t, "rinka": rinka, "data": idx.strftime("%Y-%m-%d"),
                        "laikas_utc": idx.strftime("%H:%M")})
    for c in STULP.values():
        out[c] = d[c].values if c in d.columns else None
    out = out[pd.notna(idx)]
    out["veiksmas"] = out["veiksmas"].astype(str).str.lower().str.strip()
    return out.reset_index(drop=True)


def parsiusti(t, bandymai=3):
    import yfinance as yf
    for k in range(bandymai):
        try:
            return yf.Ticker(t).upgrades_downgrades, None
        except Exception as e:
            kl = type(e).__name__
            time.sleep(3 * (k + 1))
    return None, kl


def paleisti(tik=None):
    import universas as U
    sar = [(t, "eu") for t in U.visi_tikeriai()] + [(t, "us") for l in U.UNIVERSAS_US.values() for t in l]
    sar = list(dict.fromkeys(sar))
    if tik:
        sar = [x for x in sar if x[1] == "eu"][:tik] + [x for x in sar if x[1] == "us"][:tik]
    os.makedirs(APLANKAS, exist_ok=True)
    dalys, klaidos = [], {}
    for k, (t, r) in enumerate(sar, 1):
        df, kl = parsiusti(t)
        if kl:
            klaidos[t] = kl
        dalys.append(sutvarkyti(df, t, r))
        time.sleep(0.3)
        if k % 50 == 0:
            print(f"  {k}/{len(sar)}", flush=True)
    R = pd.concat(dalys, ignore_index=True)
    R.to_csv(f"{APLANKAS}/reitingai.csv", index=False)
    print(f"\nIraso: {len(R)}; tikeriu su bent vienu: {R['tikeris'].nunique()} is {len(sar)}; klaidu: {len(klaidos)}"
          + (f" (pvz. {list(klaidos.items())[:3]})" if klaidos else ""))
    if R.empty:
        return
    R["metai"] = R["data"].str[:4]
    print("\nIrasu pagal metus ir rinka (veiksmai up / down / visi):")
    for r in ("eu", "us"):
        g = R[R["rinka"] == r]
        n_t = sum(1 for t, rr in sar if rr == r)
        print(f"  {r.upper()}: tikeriu su irasais {g['tikeris'].nunique()} is {n_t}; pirmas irasas {g['data'].min() if len(g) else '-'}")
        for m in sorted(g["metai"].unique()):
            x = g[g["metai"] == m]
            print(f"     {m}: up {int((x.veiksmas == 'up').sum()):>5}  down {int((x.veiksmas == 'down').sum()):>5}  visi {len(x):>6}")
    print("\nLaiko dalis (ar Yahoo duoda valanda, ne tik diena):",
          f"{(R['laikas_utc'] != '00:00').mean() * 100:.0f}% irasu su ne-nuliniu laiku")
    print("Veiksmu reiksmes:", R["veiksmas"].value_counts().head(8).to_dict())


def savitikra():
    ok = True

    def tk(s, a, b):
        nonlocal ok
        g = a == b
        print(f"  {'OK ' if g else 'BLOGAI'}  {s}{'' if g else f'  (gauta {a}, laukta {b})'}")
        ok = ok and g
    idx = pd.DatetimeIndex(["2024-01-05 12:30", "2023-06-01"], name="GradeDate")
    df = pd.DataFrame({"Firm": ["MS", "GS"], "ToGrade": ["Buy", "Sell"], "FromGrade": ["Hold", "Hold"],
                       "Action": ["up", "Down "]}, index=idx)
    s = sutvarkyti(df, "AAPL", "us")
    tk("eiluciu 2", len(s), 2)
    tk("data ir laikas", (s.loc[0, "data"], s.loc[0, "laikas_utc"]), ("2024-01-05", "12:30"))
    tk("veiksmas sunormintas", list(s["veiksmas"]), ["up", "down"])
    tk("truksta tikslo stulpeliu -> None", s.loc[0, "tikslas"], None)
    tk("None -> tuscia lentele", len(sutvarkyti(None, "X", "eu")), 0)
    tk("GradeDate kaip stulpelis", len(sutvarkyti(df.reset_index(), "AAPL", "us")), 2)
    print("SAVITIKRA: VISKAS GERAI" if ok else "SAVITIKRA: YRA KLAIDU")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--savitikra", action="store_true")
    ap.add_argument("--tik", type=int, default=None)
    a = ap.parse_args()
    sys.exit(savitikra() if a.savitikra else paleisti(a.tik))
