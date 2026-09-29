#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZYMU PATIKRA 3 - KRITIMO "TIESUMAS" (EFFICIENCY RATIO)  -  2026-09-29
=====================================================================

Tas pats klausimas ir tas pats matas kaip zymos.py: ar pozymis padeda
atskirti TIKRA 1 scenariju (korteles tikslas 0.5 ATR anksciau nei stop'as per
3 sesijas) nuo panasaus. Signalai - tie patys, kuriuos rodo detektorius.
Niekas nefiltruojama, detektorius.py ir korteles nelieciami.

Naudojama zymos.py masina (surinkimas, keturi matavimai, bootstrap,
verdiktas) be pakeitimu - pakeiciamas TIK pozymiu sarasas.
Saltinis: claude/sablonai-ir-cnn-2026-09-29.md.

UZREGISTRUOTA PRIES PALEIDIMA (ribos is saltinio, ne derintos):

  Z12 "tiesus kritimas"   ER10 >= 0.20
        ER10 = |C(D-1) - C(D-11)| / sum_{i=D-10..D-1} |C(i) - C(i-1)|
        (Kaufman efficiency ratio; tik dienos uzdarymai iki D-1)
        Alvarez 2023 (Alvarez Quant Trading, "Efficiency Ratio and Mean
        Reversion"): kai ER >= 20 (skale 0-100), vidutinis pelnas ~15%
        didesnis; tiesesnis kritimas atsistato geriau -> hipoteze +1.
        Alvarez ER dar suglodindavo, bet glodinimo ilgio neatskleide -
        todel naudojama NEGLODINTA reiksme (vienintelis neapibreztas
        pasirinkimas, uzrasytas cia pries paleidima).
        Irodymas silpnas (B): saltinis neatskleide sanaudu ir nematytos imties
        testo, o 3-iai jo strategijai filtras nepadejo.

DUBLIO TAISYKLE (pries paleidima): jei Spearman |rho| > 0.7 su jau tikrintu
pozymiu - kritimo dydziu ATR vienetais, 63 d. trendu (R_Q) arba atstumu iki
52 sav. aukstumos - pozymis laikomas dubliu ir nenaudojamas, net jei praeitu.

Verdiktas "PADEDA" - kaip zymos.py: ta pati kryptis visose keturiose pusese
>= 5 p.p., sujungtu nematytu pusiu p < 0.05, abiejose grupese >= 30 signalu.
Papildomai (tik informacija): tikru dalis pagal ER10 intervalus.

Paleidimas:
    python zymos3.py --savitikra
    python zymos3.py
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import detektorius as D
import zymos as Z
import zymos2 as Z2

ER_N = 10
ER_RIBA = 0.20
DUBLIO_RIBA = 0.7
INTERVALAI = [0.0, 0.1, 0.2, 0.4, 1.0001]

NAUJOS_ZYMOS = [("Z12", 1, "tiesus kritimas (ER10 >= 0.20)", +1)]

DUBLIAI = [
    ("_er", "_krit", "kritimo dydis (ATR)"),
    ("_er", "_rq", "63 d. trendas (R_Q)"),
    ("_er", "_52s", "atstumas iki 52 sav. aukstumos"),
]

_originalus_kontekstas = Z.dienos_kontekstas


# ============================================================ pozymiai

def er_serija(close, n=ER_N):
    """ER dienai D is uzdarymu iki D-1 (eilute D pati D nemato)."""
    c = pd.Series(close, dtype=float)
    kelias = c.diff().abs().rolling(n).sum()
    grynas = (c - c.shift(n)).abs()
    er = (grynas / kelias.replace(0, np.nan)).clip(0, 1)
    return er.shift(1)


def kritimas_atr(d):
    """Kritimas nuo 5 d. uzdarymu virsunes iki vakar, ATR vienetais (iki D-1).

    Tik dublio patikrai; tas pats ATR kaip detektoriuje (20 d. TR vidurkis),
    be dividendu pataisos.
    """
    c, h, l = d["Close"], d["High"], d["Low"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20).mean()
    k = (c.rolling(D.S1_LANGAS_D).max() - c) / atr
    return k.shift(1)


def dienos_kontekstas3(dienos, sekt):
    out = _originalus_kontekstas(dienos, sekt)       # jau paslinktas (iki D-1)
    for t, d in dienos.items():
        extra = Z2.kritimo_pozymiai(d)[["rq"]]       # jau "iki D-1"
        extra["er10"] = er_serija(d["Close"]).values
        extra["kritimas_atr"] = kritimas_atr(d).values
        out[t] = out[t].join(extra)
    return out


def zymos_signalui3(tipas, k, atask_datos, ses):
    if tipas != 1:
        return {}
    er = k.get("er10", np.nan)
    return {
        "Z12": np.nan if pd.isna(er) else float(er >= ER_RIBA),
        "_er": er, "_krit": k.get("kritimas_atr", np.nan),
        "_rq": k.get("rq", np.nan), "_52s": k.get("iki_52s", np.nan),
    }


def ijungti():
    """zymos.py masina su nauju pozymiu (nieko kito nekeiciant)."""
    Z.ZYMOS = NAUJOS_ZYMOS
    Z.dienos_kontekstas = dienos_kontekstas3
    Z.zymos_signalui = zymos_signalui3


# ============================================================ ataskaita

def dublio_patikra(dalys):
    visi = pd.concat(list(dalys.values()))
    s1 = visi[visi["tipas"] == 1]
    dub = False
    print(f"\n{'='*84}\nDUBLIO PATIKRA (Spearman, 1 scenarijus; riba |rho| > {DUBLIO_RIBA})\n{'='*84}")
    for a, b, pav in DUBLIAI:
        for rinka in ("eu", "us"):
            g = s1[s1["rinka"] == rinka][[a, b]].dropna()
            if len(g) < 30:
                print(f"  ER10 vs {pav:<32} {rinka.upper():<4} per mazai ({len(g)})")
                continue
            rho = g[a].rank().corr(g[b].rank())
            ar = abs(rho) > DUBLIO_RIBA
            dub = dub or ar
            print(f"  ER10 vs {pav:<32} {rinka.upper():<4} rho={rho:+.2f}  N={len(g)}"
                  + ("   <<< DUBLIS" if ar else ""))
    return dub


def intervalai(dalys):
    visi = pd.concat(list(dalys.values()))
    s1 = visi[visi["tipas"] == 1]
    print(f"\n{'='*84}\nINFORMACIJA: tikru dalis pagal ER10\n{'='*84}")
    print(f"  {'ER10':<14}" + "".join(f"{r.upper():>22}" for r in ("eu", "us")))
    for lo, hi in zip(INTERVALAI[:-1], INTERVALAI[1:]):
        t = ""
        for rinka in ("eu", "us"):
            g = s1[(s1["rinka"] == rinka) & (s1["_er"] >= lo) & (s1["_er"] < hi)]
            t += (f"{g['tikras'].mean()*100:>12.1f}% (N={len(g):>4})" if len(g)
                  else f"{'-':>22}")
        print(f"  {f'{lo:.2f}-{min(hi, 1.0):.2f}':<14}{t}")
    for rinka in ("eu", "us"):
        g = s1[s1["rinka"] == rinka]["_er"].dropna()
        if len(g):
            print(f"  {rinka.upper()}: ER10 mediana {g.median():.2f}; "
                  f">= {ER_RIBA}: {(g >= ER_RIBA).mean()*100:.0f}% signalu")


def paleisti(dienos5):
    ijungti()
    dalys = {}
    for rinka in ("eu", "us"):
        df = Z.surinkti(rinka, dienos5)
        if df.empty:
            continue
        zyme = D.RINKOS[rinka]["zyme"]
        ses = sorted(df["sesija"].unique())
        riba = ses[len(ses) // 2]
        dalys[f"{zyme} matyta"] = df[df["sesija"] <= riba]
        dalys[f"{zyme} NEMATYTA"] = df[df["sesija"] > riba]
    if len(dalys) < 4:
        sys.exit("truksta duomenu - keturiu matavimu nera")
    intervalai(dalys)
    dub = dublio_patikra(dalys)
    pad = Z.vertinti(dalys)
    print("\nGALUTINIS: " + ("Z12 PADEDA (ziurek krypti lenteleje)" if pad and not dub else
                             "Z12 yra dublis" if pad else
                             "Z12 itaka NEIRODYTA (priezastis - verdikto stulpelyje)"))


# ============================================================ savitikra

def savitikra():
    ok = True

    def tikrinti(s, a, b):
        nonlocal ok
        g = (a == b)
        print(f"  {'OK ' if g else 'BLOGAI'}  {s}{'' if g else f'  (gauta {a}, laukta {b})'}")
        ok = ok and g

    # 1) tiesi linija -> ER = 1 (paskutine eilute mato 10 zingsniu iki D-1)
    c = pd.Series(np.linspace(120, 100, 15))
    tikrinti("tiesus kritimas -> ER10 = 1.0", round(float(er_serija(c).iloc[-1]), 6), 1.0)
    # 2) zigzagas su grynu 0 -> ER = 0
    z = pd.Series([100, 101] * 8, dtype=float)
    tikrinti("zigzagas be grynojo judesio -> ER10 = 0.0",
             round(float(er_serija(z).iloc[-1]), 6), 0.0)
    # 3) zinoma reiksme: 10 zingsniu, kelias 10 x 1.0, grynas 4 -> 0.4
    v = [100.0]
    for s_ in [-1, -1, -1, +1, -1, -1, +1, -1, +1, -1]:
        v.append(v[-1] + s_)
    v.append(v[-1])                                  # diena D (pati nematoma)
    tikrinti("kelias 10, grynas 4 -> ER10 = 0.4",
             round(float(er_serija(pd.Series(v)).iloc[-1]), 6), 0.4)
    # 4) ateitis: D eilute nepriklauso nuo D uzdarymo; D+1 - priklauso
    a = pd.Series(v)
    b = a.copy(); b.iloc[-1] = 50.0
    tikrinti("ER10 dienai D nepriklauso nuo D uzdarymo",
             float(er_serija(a).iloc[-1]) == float(er_serija(b).iloc[-1]), True)
    a2 = pd.concat([a, pd.Series([100.0])], ignore_index=True)
    b2 = pd.concat([b, pd.Series([100.0])], ignore_index=True)
    tikrinti("ER10 dienai D+1 jau mato D (testas ne tuscias)",
             float(er_serija(a2).iloc[-1]) != float(er_serija(b2).iloc[-1]), True)
    # 5) per trumpa istorija -> NaN
    tikrinti("per trumpa istorija -> NaN",
             bool(np.isnan(er_serija(pd.Series([1.0] * 10)).iloc[-1])), True)
    # 6) riba
    tikrinti("ER10 = 0.20 -> Z12 = 1 (riba imtinai)",
             zymos_signalui3(1, pd.Series(dict(er10=0.20)), [], None)["Z12"], 1.0)
    tikrinti("ER10 = 0.1999 -> Z12 = 0",
             zymos_signalui3(1, pd.Series(dict(er10=0.1999)), [], None)["Z12"], 0.0)
    tikrinti("2 scenarijui zymos nera", zymos_signalui3(2, pd.Series(dtype=float), [], None), {})
    # 7) kritimas_atr dublio patikrai: iki D-1, tas pats ATR
    idx = pd.bdate_range("2026-01-01", periods=40)
    cc = np.r_[np.full(34, 100.0), [100, 98, 96, 94, 92, 90]]
    dd = pd.DataFrame(dict(Open=cc, High=cc + 1, Low=cc - 1, Close=cc), index=idx)
    ka = kritimas_atr(dd)
    prev = dd["Close"].shift(1)
    tr = pd.concat([dd.High - dd.Low, (dd.High - prev).abs(), (dd.Low - prev).abs()], axis=1).max(axis=1)
    laukta = (dd["Close"].iloc[-6:-1].max() - dd["Close"].iloc[-2]) / tr.rolling(20).mean().iloc[-2]
    tikrinti("kritimas_atr = (5 d. max uzd. - vakar) / ATR20, iki D-1",
             round(float(ka.iloc[-1]), 10), round(float(laukta), 10))
    # 8) verdiktas per zymos.py masina
    import io, contextlib
    ijungti()

    def dal(sk):
        rng = np.random.default_rng(3)
        return pd.DataFrame([{"sesija": f"2026-0{1 + i % 8}-{1 + i % 27:02d}", "tipas": 1,
                              "tikras": float(rng.random() < 0.5 + (sk / 200 if i % 2 else -sk / 200)),
                              "Z12": float(i % 2)} for i in range(300)])
    for pav, sk, laukta_v in (("aiskus skirtumas -> PADEDA", 30, True),
                              ("jokio skirtumo -> nepadeda", 0, False)):
        dd4 = {m: dal(sk) for m in ("EU matyta", "EU NEMATYTA", "JAV matyta", "JAV NEMATYTA")}
        with contextlib.redirect_stdout(io.StringIO()):
            r = Z.vertinti(dd4)
        tikrinti(f"verdiktas: {pav}", bool(r), laukta_v)
    # 9) dublio taisykle
    rng = np.random.default_rng(1)
    x = rng.random(120)
    df = pd.DataFrame(dict(rinka="eu", tipas=1, _er=x, _krit=x + rng.normal(0, 0.01, 120),
                           _rq=rng.random(120), _52s=rng.random(120)))
    with contextlib.redirect_stdout(io.StringIO()):
        d1 = dublio_patikra({"a": df})
    tikrinti("dublis su kritimo dydziu -> suveikia", d1, True)
    df2 = df.assign(_krit=rng.random(120))
    with contextlib.redirect_stdout(io.StringIO()):
        d2 = dublio_patikra({"a": df2})
    tikrinti("nepriklausomi -> ne dublis", d2, False)

    print("-" * 60)
    print("SAVITIKRA: VISKAS GERAI" if ok else "SAVITIKRA: YRA KLAIDU")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--savitikra", action="store_true")
    ap.add_argument("--dienos", type=int, default=60)
    a = ap.parse_args()
    if a.savitikra:
        sys.exit(savitikra())
    D._KURSAS.clear()
    paleisti(a.dienos)
