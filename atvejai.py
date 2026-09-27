#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATVEJU PATIKRA v4  —  2026-09-27
================================

Vienas klausimas: ar v4 detektorius ATPAZISTA Manto nurodytus atvejus?

Ne pelningumas. Ne statistika. Tik: suveikia ar nesuveikia, kada, ir jei ne -
kuri salyga pasake "ne".

Etalonai (Manto nurodyti, datos +/-2 d., nes IBKR grafikas buvo netikslus):
    ADYEN   2026-08-10 ..08-14   -> tikrasis judesys 08-13 (tarpas +5.7%, diena +10.1%)
    AMD     2026-08-12 ..08-18   -> IBIS, JAV bendrove Xetroje
    CAP     2026-07-22 ..07-28   -> 07-23 kapituliacija, 07-24.. atsistatymas +18%
    PTX     2026-08-03 ..08-05   -> Palantir, 08-04 tarpas +16.4%, diena +8.8%
    SAP     2026-07-22 ..07-28   -> 07-23 kapituliacija, 07-24.. atsistatymas +24%

Pirmoji sio failo versija tikrino v3 (trys scenarijai, paveldeti filtrai).
v4 turi DU scenarijus ir tik isvestas salygas, tad viskas perrasyta.
"""

import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("KLAIDA: reikia yfinance")

import detektorius as D

ATVEJAI = [
    ("ADYEN",    ["ADYEN.AS"],        "2026-08-10", "2026-08-14"),
    ("AMD IBIS", ["AMD.DE", "AMD.F"], "2026-08-12", "2026-08-18"),
    ("CAP SBF",  ["CAP.PA"],          "2026-07-22", "2026-07-28"),
    ("PTX IBIS", ["PTX.DE", "PTX.F"], "2026-08-03", "2026-08-05"),
    ("SAP",      ["SAP.DE"],          "2026-07-22", "2026-07-28"),
]

UZD = D.RINKOS["eu"]["uzdarymas"]


# ------------------------------------------------------------ salygu sekimas

def s1_salygos(langas, kd):
    """Tos pacios salygos kaip scenarijus_1, su etiketemis.

    Pabaigoje tikrinama, ar diagnostika sutaria su tikraja funkcija. Jei ne -
    tai kodo dubliavimo klaida, ir skriptas apie tai rekia.
    """
    atr = kd["atr_abs"]
    vu = kd["uzdarymas"]
    kaina = float(langas["Close"].iloc[-1])
    atid = float(langas["ses_atidarymas"].iloc[-1])
    orb = langas.iloc[:D.S2_ORB_BARU]
    krit = kd["virsune_n"] - vu - kd["div_lange"]
    sant = float(langas["apyv_santykis"].iloc[-1])
    vwap = float(langas["vwap"].iloc[-1])
    R_p = float(kd["virsune_n"])
    return [
        ("baru pakanka", len(langas) >= D.S2_ORB_BARU + 2, f"{len(langas)}"),
        ("kritimas >= 1 ATR", krit >= D.S1_MIN_KRITIMAS_ATR * atr,
         f"{krit/atr:.2f} ATR"),
        ("vakar uzdare apacioje", kd["uzd_vieta"] <= D.S1_MAX_UZD_VIETA,
         f"{kd['uzd_vieta']:.2f} <= {D.S1_MAX_UZD_VIETA}"),
        ("NE tarpo diena", atid - vu < D.S2_MIN_TARPAS_ATR * atr,
         f"tarpas {(atid-vu)/atr:+.2f} ATR"),
        ("atsieme vakar uzd.", kaina > vu, f"{kaina:.2f} > {vu:.2f}"),
        ("virs atidarymo diapaz.", kaina > float(orb["High"].max()),
         f"{kaina:.2f} > {float(orb['High'].max()):.2f}"),
        ("apyvarta", bool(np.isfinite(sant) and sant >= D.S1_MIN_APYV_SANTYKIS),
         f"x{sant:.2f} >= {D.S1_MIN_APYV_SANTYKIS}"),
        ("virs VWAP", bool(np.isfinite(vwap) and kaina > vwap),
         f"{kaina:.2f} > {vwap:.2f}"),
        ("lygis dar virs kainos", R_p > kaina, f"{R_p:.2f} > {kaina:.2f}"),
    ]


def s2_salygos(langas, kd):
    atr = kd["atr_abs"]
    vu = kd["uzdarymas"]
    kaina = float(langas["Close"].iloc[-1])
    atid = float(langas["ses_atidarymas"].iloc[-1])
    orb = langas.iloc[:D.S2_ORB_BARU]
    sant = float(pd.to_numeric(orb["apyv_santykis"], errors="coerce").median())
    vwap = float(langas["vwap"].iloc[-1])
    return [
        ("baru pakanka", len(langas) >= D.S2_ORB_BARU + 2, f"{len(langas)}"),
        ("tarpas >= 1 ATR", atid - vu >= D.S2_MIN_TARPAS_ATR * atr,
         f"{(atid-vu)/atr:+.2f} ATR"),
        ("30 min apyvarta >= 2x",
         bool(np.isfinite(sant) and sant >= D.S2_MIN_APYVARTA_X), f"x{sant:.2f}"),
        ("virs atidarymo diapaz.", kaina > float(orb["High"].max()),
         f"{kaina:.2f} > {float(orb['High'].max()):.2f}"),
        ("virs VWAP", bool(np.isfinite(vwap) and kaina > vwap),
         f"{kaina:.2f} > {vwap:.2f}"),
        ("tarpas neuzpildytas", float(langas["Low"].min()) >= vu,
         f"dugnas {float(langas['Low'].min()):.2f} >= {vu:.2f}"),
    ]


def sekti(sd, kd, ses):
    print(f"\n    --- {ses} ({len(sd)} baru) ---")
    print(f"    kontekstas: ATR {kd['atr_abs']:.2f} ({kd['atr_pct']:.2f}%)  "
          f"vakar uzd. {kd['uzdarymas']:.2f}  uzd_vieta {kd['uzd_vieta']:.2f}  "
          f"5 d. virsune {kd['virsune_n']:.2f}  apyvarta {kd['apyvarta']/1e6:.1f} mln")

    suveike, geriausi = {}, {1: (-1, None), 2: (-1, None)}
    for i in range(D.S2_ORB_BARU + 1, len(sd)):
        langas = sd.iloc[:i + 1]
        iki = UZD - int(langas["minute"].iloc[-1])
        m = int(langas["minute"].iloc[-1])
        laikas = f"{m//60:02d}:{m%60:02d}"
        for nr, sal in ((1, s1_salygos), (2, s2_salygos)):
            try:
                s = sal(langas, kd)
            except Exception:
                continue
            n = sum(1 for _, ok, _ in s if ok)
            if n > geriausi[nr][0]:
                geriausi[nr] = (n, (laikas, i, s))
        for s in D.aptikti(langas, kd, iki):
            nr = s["tipas"]
            if nr in suveike:
                continue
            suveike[nr] = s
            print(f"    SUVEIKE {nr} ({s['scenarijus']}) {laikas}: "
                  f"ieina {s['ieina']:.2f}  tikslas "
                  f"{s['tikslas'] if s['tikslas'] else '-'}  stop {s['stop']:.2f}"
                  + (f"  R:R {s['rr']:.2f}" if s.get("rr") else "")
                  + (f"  BLOKUOTA: {', '.join(s['kliutys'])}" if s["kliutys"]
                     else "  PRALEISTA"))

    for nr in (1, 2):
        if nr in suveike:
            continue
        n, det = geriausi[nr]
        print(f"    scenarijus {nr}: NESUVEIKE")
        if det:
            laikas, i, s = det
            # savitikra: jei visos salygos praeina, funkcija PRIVALO duoti signala
            langas = sd.iloc[:i + 1]
            tikras = (D.scenarijus_1(langas, kd) if nr == 1
                      else D.scenarijus_2(langas, kd, UZD - int(langas["minute"].iloc[-1])))
            if all(ok for _, ok, _ in s) and tikras is None:
                print("      !!! diagnostika nesutinka su funkcija - kodo dubliavimas")
            print(f"      arciausiai {laikas} ({n}/{len(s)} salygu):")
            for nm, ok, d in s:
                if not ok:
                    print(f"        NE  {nm:<24} {d}")
    return suveike


def atvejis(etikete, kandidatai, nuo_s, iki_s):
    nuo, iki = date.fromisoformat(nuo_s), date.fromisoformat(iki_s)
    print("\n" + "=" * 78)
    print(f"{etikete}   {nuo_s} .. {iki_s}")
    print("=" * 78)

    t = dien = None
    for kand in kandidatai:
        raw = yf.download(kand, period="2y", interval="1d", auto_adjust=False,
                          progress=False, group_by="ticker", actions=True)
        d = D._vienas(raw, kand)
        if d is not None and len(d) > 250:
            t, dien = kand, d
            print(f"    {kand}: {len(d)} dienu")
            break
    if t is None:
        print("    tikerio nerasta")
        return

    rod = D.dienos_rodikliai(dien, dien["Dividends"] if "Dividends" in dien else None)
    raw5 = yf.download(t, period="60d", interval="5m", auto_adjust=False,
                       progress=False, group_by="ticker", prepost=False)
    b5 = D._vienas(raw5, t)
    if b5 is None or len(b5) < 100:
        print("    5 min baru negauta")
        return
    visos = D.sesijos_rodikliai(b5, "eu")
    turim = sorted({s for s in visos["sesija"].unique() if nuo <= s <= iki})
    if not turim:
        print(f"    5 min duomenyse nera {nuo}..{iki} "
              f"(anksciausia {min(visos['sesija'])})")
        return

    rado = {}
    for ses, sd in visos.groupby("sesija", sort=True):
        if ses not in turim or len(sd) < D.S2_ORB_BARU + 4:
            continue
        kd = D._kd(rod, ses)
        if kd is None:
            print(f"\n    --- {ses}: dienos konteksto nera (NaN) ---")
            continue
        for nr in sekti(sd, kd, ses):
            rado.setdefault(nr, []).append(str(ses))

    print(f"\n    SANTRAUKA {etikete}: "
          + (", ".join(f"scenarijus {nr} -> {', '.join(v)}" for nr, v in sorted(rado.items()))
             if rado else "NE VIENAS scenarijus nesuveike"))


if __name__ == "__main__":
    print("ATVEJU PATIKRA v4 - ar detektorius atpazista Manto nurodytus atvejus")
    print(f"scenarijai: 1 = {D.S1_MIN_KRITIMAS_ATR} ATR kritimas + atsiemimas, "
          f"2 = {D.S2_MIN_TARPAS_ATR} ATR tarpas + eiga")
    for a in ATVEJAI:
        try:
            atvejis(*a)
        except Exception as e:
            import traceback
            print(f"\n    KLAIDA {a[0]}: {e}")
            traceback.print_exc()
    print("\n" + "=" * 78 + "\nBAIGTA")
