#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATVEJU PATIKRA v5 — ATPAZINIMAS  —  2026-09-28
==============================================

Vienas klausimas: ar detektorius ATPAZISTA Manto nurodytus atvejus, o jei ne -
KODEL. Ne pelningumas.

Kas nauja lyginant su v4:
  1. AMD ir PTX tikrinami IR Xetroje (kur Mantas prekiauja), IR PIRMINEJE
     JAV biržoje (kur formuojasi kaina). AMD (JAV) yra JAV universe, bet niekas
     niekada nepatikrino, ar jis suveike tomis dienomis.
  2. Kiekvienai dienai spausdinamas DIENOS vaizdas, kuriam 5 min duomenu
     nereikia (tarpas, uzdarymo vieta, dienos graza, apyvarta, atstumas iki
     52 sav. aukstumos, ataskaitos data). Jis veikia ir tada, kai 5 min
     duomenys jau pasibaige (yahoo ~60 d.).
  3. Xetroje prekiaujamoms JAV akcijoms apyvarta matuojama DVIEM langais:
     09:00-09:30 (Xetros atidarymas) ir 15:30-16:00 (JAV atidarymas). Hipoteze:
     AMD judesys vyksta JAV sesijoje, o 2 scenarijus ziuri i Xetros ryta.
  4. Informacinis "ralis be tarpo" pozymis (literatura: George & Hwang 2004,
     Huddart ir kt. 2009 - 52 sav. aukstuma su apyvartos suoliu). Tik
     spausdinamas - detektoriaus nekeicia.

Etalonai (Manto, datos +/-1-2 d.):
    ADYEN   08-10..08-14   (judesys 08-13)
    AMD     08-12..08-21   Xetra AMD.DE + JAV AMD
    CAP     07-22..07-28
    PTX     08-03..08-05   Xetra PTX.DE + JAV PLTR
    SAP     07-22..07-28
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

# (etikete, [(tickeris, rinka)], nuo, iki)
ATVEJAI = [
    ("ADYEN",       [("ADYEN.AS", "eu")],                "2026-08-10", "2026-08-14"),
    ("AMD Xetra",   [("AMD.DE", "eu"), ("AMD.F", "eu")], "2026-08-12", "2026-08-21"),
    ("AMD JAV",     [("AMD", "us")],                     "2026-08-12", "2026-08-21"),
    ("CAP",         [("CAP.PA", "eu")],                  "2026-07-22", "2026-07-28"),
    ("PTX Xetra",   [("PTX.DE", "eu"), ("PTX.F", "eu")], "2026-08-03", "2026-08-05"),
    ("PTX JAV",     [("PLTR", "us")],                    "2026-08-03", "2026-08-05"),
    ("SAP",         [("SAP.DE", "eu")],                  "2026-07-22", "2026-07-28"),
]

# Xetros laiku JAV atidarymas 15:30 (Berlyno laikas; vasara ir ziema sutampa
# su 15:30 visus metus, isskyrus kelias pereinamojo laikotarpio savaites).
JAV_ATID_BERLYNE = 15 * 60 + 30


# ------------------------------------------------------------ dienos vaizdas

def ataskaitos(t):
    try:
        e = yf.Ticker(t).earnings_dates
        if e is None or not len(e):
            return []
        return sorted({x.date() for x in e.index})
    except Exception:
        return []


def dienos_vaizdas(dien, nuo, iki, atask):
    """Viena eilute dienai. Visi skaiciai tik is tos dienos ir praeities."""
    c, o, h, l, v = (dien[k] for k in ("Close", "Open", "High", "Low", "Volume"))
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20).mean().shift(1)             # iki vakar
    relvol = v / v.rolling(20).median().shift(1)
    h52 = h.rolling(252, min_periods=120).max().shift(1)
    h20 = h.rolling(20).max().shift(1)
    print(f"    {'diena':<11}{'tarpas':>8}{'diena %':>9}{'uzd.vieta':>10}"
          f"{'apyv. x':>9}{'iki 52s':>9}{'>20d virs':>10}  ataskaita")
    for ix in dien.index:
        d = ix.date()
        if not (nuo <= d <= iki):
            continue
        a = atr.loc[ix]
        tarp = (o.loc[ix] - prev.loc[ix]) / a if a > 0 else np.nan
        diap = h.loc[ix] - l.loc[ix]
        uzd = (c.loc[ix] - l.loc[ix]) / diap if diap > 0 else np.nan
        artim = [x for x in atask if abs((x - d).days) <= 3]
        at = ", ".join(f"{x} ({(x - d).days:+d} d.)" for x in artim) or "-"
        print(f"    {str(d):<11}{tarp:>+7.2f}A{(c.loc[ix] / prev.loc[ix] - 1) * 100:>+8.1f}%"
              f"{uzd:>10.2f}{relvol.loc[ix]:>8.2f}x"
              f"{(c.loc[ix] / h52.loc[ix] - 1) * 100:>+8.1f}%"
              f"{'TAIP' if c.loc[ix] > h20.loc[ix] else 'ne':>10}  {at}")


# ------------------------------------------------------------ 5 min dalis

def s1_salygos(langas, kd):
    atr = kd["atr_abs"]
    vu = kd["uzdarymas"]
    kaina = float(langas["Close"].iloc[-1])
    atid = float(langas["ses_atidarymas"].iloc[-1])
    orb = langas.iloc[:D.S2_ORB_BARU]
    krit = kd["virsune_n"] - vu - kd["div_lange"]
    sant = float(langas["apyv_santykis"].iloc[-1])
    vwap = float(langas["vwap"].iloc[-1])
    return [
        ("kritimas >= 1 ATR", krit >= D.S1_MIN_KRITIMAS_ATR * atr, f"{krit/atr:.2f} ATR"),
        ("vakar uzdare apacioje", kd["uzd_vieta"] <= D.S1_MAX_UZD_VIETA,
         f"{kd['uzd_vieta']:.2f} <= {D.S1_MAX_UZD_VIETA}"),
        ("NE tarpo diena", atid - vu < D.S2_MIN_TARPAS_ATR * atr,
         f"tarpas {(atid-vu)/atr:+.2f} ATR"),
        ("atsieme vakar uzd.", kaina > vu, f"{kaina:.2f} > {vu:.2f}"),
        ("virs atidarymo diapaz.", kaina > float(orb["High"].max()),
         f"{kaina:.2f} > {float(orb['High'].max()):.2f}"),
        ("apyvarta", bool(np.isfinite(sant) and sant >= D.S1_MIN_APYV_SANTYKIS),
         f"x{sant:.2f}"),
        ("virs VWAP", bool(np.isfinite(vwap) and kaina > vwap), f"{kaina:.2f} > {vwap:.2f}"),
        ("lygis dar virs kainos", float(kd["virsune_n"]) > kaina,
         f"{float(kd['virsune_n']):.2f} > {kaina:.2f}"),
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
        ("tarpas >= 1 ATR", atid - vu >= D.S2_MIN_TARPAS_ATR * atr,
         f"{(atid-vu)/atr:+.2f} ATR"),
        ("30 min apyvarta >= 2x", bool(np.isfinite(sant) and sant >= D.S2_MIN_APYVARTA_X),
         f"x{sant:.2f}"),
        ("virs atidarymo diapaz.", kaina > float(orb["High"].max()),
         f"{kaina:.2f} > {float(orb['High'].max()):.2f}"),
        ("virs VWAP", bool(np.isfinite(vwap) and kaina > vwap), f"{kaina:.2f} > {vwap:.2f}"),
        ("tarpas neuzpildytas", float(langas["Low"].min()) >= vu,
         f"dugnas {float(langas['Low'].min()):.2f} >= {vu:.2f}"),
    ]


def apyvartos_langai(sd, rinka):
    """Apyvartos santykis dviejuose languose: atidarymas ir (Xetroje) JAV atidarymas."""
    out = {}
    a = D.RINKOS[rinka]["atidarymas"]
    for pav, nuo in (("atidarymas", a), ("JAV atid. 15:30", JAV_ATID_BERLYNE)):
        if rinka != "eu" and pav != "atidarymas":
            continue
        w = sd[(sd["minute"] >= nuo) & (sd["minute"] < nuo + 30)]
        if len(w):
            out[pav] = float(pd.to_numeric(w["apyv_santykis"], errors="coerce").median())
    return out


def sekti(sd, kd, ses, rinka):
    uzd = D.RINKOS[rinka]["uzdarymas"]
    lang = apyvartos_langai(sd, rinka)
    print(f"\n    --- {ses} ({len(sd)} baru)  ATR {kd['atr_abs']:.2f}  "
          f"vakar {kd['uzdarymas']:.2f}  uzd.vieta {kd['uzd_vieta']:.2f}  "
          f"apyvarta {kd['apyvarta']/1e6:.1f} mln  "
          + "  ".join(f"{k}: x{v:.2f}" for k, v in lang.items()))
    suveike, geriausi = {}, {1: (-1, None), 2: (-1, None)}
    for i in range(D.S2_ORB_BARU + 3, len(sd)):
        langas = sd.iloc[:i + 1]
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
        for s in D.aptikti(langas, kd, uzd - m, rinka):
            nr = s["tipas"]
            if nr in suveike:
                continue
            suveike[nr] = s
            print(f"    SUVEIKE {nr} ({s['scenarijus']}) {laikas}: ieina {s['ieina']:.2f}"
                  + (f"  BLOKUOTA: {', '.join(s['kliutys'])}" if s["kliutys"]
                     else "  TINKAMAS"))
    for nr in (1, 2):
        if nr in suveike:
            continue
        n, det = geriausi[nr]
        print(f"    scenarijus {nr}: NESUVEIKE")
        if det:
            laikas, i, s = det
            langas = sd.iloc[:i + 1]
            tikras = (D.scenarijus_1(langas, kd, rinka) if nr == 1
                      else D.scenarijus_2(langas, kd, uzd - int(langas["minute"].iloc[-1]), rinka))
            if all(ok for _, ok, _ in s) and tikras is None:
                print("      !!! diagnostika nesutinka su funkcija - patikrink kodą")
            print(f"      arciausiai {laikas} ({n}/{len(s)} salygu):")
            for nm, ok, d in s:
                if not ok:
                    print(f"        NE  {nm:<24} {d}")
    return suveike


def atvejis(etikete, kandidatai, nuo_s, iki_s):
    nuo, iki = date.fromisoformat(nuo_s), date.fromisoformat(iki_s)
    print("\n" + "=" * 96)
    print(f"{etikete}   {nuo_s} .. {iki_s}")
    print("=" * 96)
    t = rinka = dien = None
    for kand, rk in kandidatai:
        raw = yf.download(kand, period="2y", interval="1d", auto_adjust=False,
                          progress=False, group_by="ticker", actions=True)
        d = D._vienas(raw, kand)
        if d is not None and len(d) > 250:
            t, rinka, dien = kand, rk, d
            break
    if t is None:
        print("    tikerio nerasta")
        return {}
    print(f"    {t} ({rinka.upper()}): {len(dien)} dienu")

    atask = ataskaitos(t)
    print("\n    DIENOS VAIZDAS (be 5 min duomenu):")
    dienos_vaizdas(dien, nuo, iki, atask)

    rod = D.dienos_rodikliai(dien, dien["Dividends"] if "Dividends" in dien else None)
    raw5 = yf.download(t, period="60d", interval="5m", auto_adjust=False,
                       progress=False, group_by="ticker", prepost=False)
    b5 = D._vienas(raw5, t)
    if b5 is None or len(b5) < 100:
        print("\n    5 min baru negauta")
        return {}
    visos = D.sesijos_rodikliai(b5, rinka)
    turim = sorted({s for s in visos["sesija"].unique() if nuo <= s <= iki})
    if not turim:
        print(f"\n    5 min duomenyse sio laikotarpio jau nera "
              f"(anksciausia {min(visos['sesija'])}) - lieka tik dienos vaizdas")
        return {}
    print("\n    5 MIN: DETEKTORIUS")
    rado = {}
    for ses, sd in visos.groupby("sesija", sort=True):
        if ses not in turim or len(sd) < D.S2_ORB_BARU + 4:
            continue
        kd = D._kd(rod, ses)
        if kd is None:
            print(f"\n    --- {ses}: dienos konteksto nera ---")
            continue
        for nr in sekti(sd, kd, ses, rinka):
            rado.setdefault(nr, []).append(str(ses))
    print(f"\n    SANTRAUKA {etikete}: "
          + (", ".join(f"scenarijus {nr} -> {', '.join(v)}" for nr, v in sorted(rado.items()))
             if rado else "NE VIENAS scenarijus nesuveike"))
    return rado


if __name__ == "__main__":
    D._KURSAS.clear()
    print("ATVEJU PATIKRA v5 - atpazinimas ir priezastys")
    visi = {}
    for a in ATVEJAI:
        try:
            visi[a[0]] = atvejis(*a)
        except Exception as e:
            import traceback
            print(f"\n    KLAIDA {a[0]}: {e}")
            traceback.print_exc()
    print("\n" + "=" * 96 + "\nGALUTINE SANTRAUKA")
    for k, v in visi.items():
        print(f"  {k:<12} " + (", ".join(f"sc.{nr}: {', '.join(d)}" for nr, d in sorted(v.items()))
                               if v else "nesuveike / nera duomenu"))
    print("=" * 96)
