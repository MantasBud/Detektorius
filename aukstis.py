#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEJIMO AUKSCIO PATIKRA - AR "PER AUKSTAS" IEJIMAS BLOGINA 1 SCENARIJU  -  2026-09-29
===================================================================================

Manto klausimas: ar 1 scenarijaus iejimo kaina nebuna per auksta (per toli
nuo dienos dugno)? Atsakome duomenimis, ne ispudziu.

Matas tas pats kaip zymos.py: tikras 1 scenarijus = korteles tikslas (0.5 ATR)
pasiekiamas anksciau nei stop'as (0.5 ATR) per 3 sesijas. Signalai - tie
patys, kuriuos rodo detektorius (D.aptikti, pirmas tinkamas per sesija).
Tikslas ir stop'as skaiciuojami NUO IEJIMO, tad aukstesnis iejimas matas
neiskraipo - tik klausia, ar po jo dar lieka vietos.

UZREGISTRUOTA PRIES PALEIDIMA:

  Z11 "iejimas > 0.5 ATR virs dugno"
      aukstis = (iejimo kaina - sesijos dugnas iki signalo) / ATR
      Riba 0.5 ATR - ne derinta: tai stop'o atstumas. Virs jos stop'as
      atsiduria VIRS dienos dugno (sokio viduje).
      Hipoteze -1 (Mantas): aukstesnis iejimas -> maziau tikru.

  Verdiktas - zymos.py masina be pakeitimu: ta pati kryptis visose keturiose
  pusese >= 5 p.p., sujungtu nematytu pusiu p < 0.05, abiejose grupese >= 30.

  Dublio taisykle: jei Spearman |rho| > 0.7 su "atsiemimu" (iejimas - vakar
  uzdarymas, ATR) - jau tikrintu ribos.py - rezultatas laikomas dubliu.

  Papildomai (tik informacija, verdikto nekeicia): tikru dalis pagal auksti
  intervaluose 0-0.25 / 0.25-0.5 / 0.5-1.0 / >1.0 ATR.

Niekas nefiltruojama, detektorius.py ir korteles nelieciami.

Paleidimas:
    python aukstis.py --savitikra
    python aukstis.py
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import detektorius as D
import zymos as Z

RIBA_ATR = 0.5
DUBLIO_RIBA = 0.7
INTERVALAI = [0.0, 0.25, 0.5, 1.0, np.inf]
ZYMA = [("Z11", 1, "iejimas > 0.5 ATR virs dugno", -1)]


def aukscio_pozymiai(langas, kd, ieina):
    atr = float(kd["atr_abs"])
    dugnas = float(langas["Low"].min())
    a = (ieina - dugnas) / atr
    return {"Z11": float(a > RIBA_ATR), "_aukstis": a,
            "_atsiemimas": (ieina - float(kd["uzdarymas"])) / atr}


def surinkti(rinka, dienos5, duomenys=None):
    zyme = D.RINKOS[rinka]["zyme"]
    print(f"\n{'='*84}\n{zyme}\n{'='*84}")
    dd, rod, barai = duomenys if duomenys else Z.atsisiusti(rinka, dienos5)
    print(f"  akciju: {len(barai)}")
    eil = []
    for t, d in barai.items():
        sesijos = list(d.groupby("sesija", sort=True))
        for si, (ses, sd) in enumerate(sesijos):
            if len(sd) < D.S2_ORB_BARU + 4:
                continue
            kd = D._kd(rod[t], ses)
            if kd is None:
                continue
            for i in range(D.S2_ORB_BARU + 3, len(sd)):
                iki = D.RINKOS[rinka]["uzdarymas"] - int(sd["minute"].iloc[i])
                langas = sd.iloc[:i + 1]
                s = next((x for x in D.aptikti(langas, kd, iki, rinka)
                          if x["tipas"] == 1 and x["tinkamas"]), None)
                if s is None:
                    continue
                toliau = pd.concat(
                    [sd.iloc[i + 1:]] +
                    [sesijos[si + j][1] for j in range(1, D.HORIZONTAS_SESIJU)
                     if si + j < len(sesijos)])
                b = D.baigtis(toliau, s)
                e = dict(rinka=rinka, tickeris=t, sesija=str(ses), tipas=1,
                         tikras=float(b["baigtis"] == "tikslas"))
                e.update(aukscio_pozymiai(langas, kd, float(s["ieina"])))
                eil.append(e)
                break
    df = pd.DataFrame(eil)
    if len(df):
        print(f"  1 scenarijus: {len(df)} signalu, tikru {df['tikras'].mean()*100:.1f}%")
    return df


def intervalai(dalys):
    visi = pd.concat(list(dalys.values()))
    print(f"\n{'='*84}\nINFORMACIJA: tikru dalis pagal iejimo auksti virs dienos dugno\n{'='*84}")
    print(f"  {'aukstis (ATR)':<16}" + "".join(f"{r.upper():>22}" for r in ("eu", "us")))
    for lo, hi in zip(INTERVALAI[:-1], INTERVALAI[1:]):
        pav = f"{lo:.2f}-{hi:.2f}" if np.isfinite(hi) else f"> {lo:.2f}"
        t = ""
        for rinka in ("eu", "us"):
            g = visi[(visi["rinka"] == rinka) & (visi["_aukstis"] >= lo) & (visi["_aukstis"] < hi)]
            t += (f"{g['tikras'].mean()*100:>12.1f}% (N={len(g):>4})" if len(g)
                  else f"{'-':>22}")
        print(f"  {pav:<16}{t}")
    for rinka in ("eu", "us"):
        g = visi[visi["rinka"] == rinka]
        if len(g):
            print(f"  {rinka.upper()}: aukscio mediana {g['_aukstis'].median():.2f} ATR; "
                  f"stop'as virs dienos dugno {(g['_aukstis'] > RIBA_ATR).mean()*100:.0f}% signalu")


def dublis(dalys):
    visi = pd.concat(list(dalys.values()))
    ar = False
    for rinka in ("eu", "us"):
        g = visi[visi["rinka"] == rinka][["_aukstis", "_atsiemimas"]].dropna()
        if len(g) < 30:
            continue
        rho = g["_aukstis"].rank().corr(g["_atsiemimas"].rank())
        d = abs(rho) > DUBLIO_RIBA
        ar = ar or d
        print(f"  dublio patikra {rinka.upper()}: aukstis vs atsiemimas rho={rho:+.2f}"
              + ("   <<< DUBLIS" if d else ""))
    return ar


def paleisti(dienos5):
    Z.ZYMOS = ZYMA
    dalys = {}
    for rinka in ("eu", "us"):
        df = surinkti(rinka, dienos5)
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
    dub = dublis(dalys)
    pad = Z.vertinti(dalys)
    print("\nGALUTINIS: " + ("aukstis PADEDA atskirti (ziurek krypti lenteleje)" if pad and not dub else
                             "aukstis yra dublis" if pad else
                             "itaka NEIRODYTA (priezastis - verdikto stulpelyje)"))


# ============================================================ savitikra

def savitikra():
    ok = True

    def tikrinti(s, a, b):
        nonlocal ok
        g = (a == b)
        print(f"  {'OK ' if g else 'BLOGAI'}  {s}{'' if g else f'  (gauta {a}, laukta {b})'}")
        ok = ok and g

    langas = pd.DataFrame({"Low": [100.0, 98.0, 99.0], "Close": [100.0, 99.0, 101.0]})
    kd = pd.Series({"atr_abs": 4.0, "uzdarymas": 100.0})
    p = aukscio_pozymiai(langas, kd, 101.0)
    tikrinti("aukstis = (101-98)/4 = 0.75 ATR", p["_aukstis"], 0.75)
    tikrinti("0.75 > 0.5 -> Z11 = 1", p["Z11"], 1.0)
    tikrinti("atsiemimas = (101-100)/4 = 0.25", p["_atsiemimas"], 0.25)
    p2 = aukscio_pozymiai(langas, kd, 100.0)
    tikrinti("aukstis 0.5 tiksliai -> Z11 = 0 (grieztai >)", p2["Z11"], 0.0)
    # dugnas imamas tik is lango (iki signalo), ne is ateities
    tikrinti("dugnas tik iki signalo: velesnis zemesnis baras neitraukiamas",
             aukscio_pozymiai(langas.iloc[:1], kd, 101.0)["_aukstis"], 0.25)

    import io, contextlib
    Z.ZYMOS = ZYMA

    def dal(sk):
        rng = np.random.default_rng(3)
        return pd.DataFrame([{"sesija": f"2026-0{1 + i % 8}-{1 + i % 27:02d}", "tipas": 1,
                              "tikras": float(rng.random() < 0.5 + (sk / 200 if i % 2 else -sk / 200)),
                              "Z11": float(i % 2)} for i in range(300)])
    for pav, sk, laukta in (("aiskus skirtumas -> PADEDA", -30, True),
                            ("jokio skirtumo -> nepadeda", 0, False)):
        dd = {m: dal(sk) for m in ("EU matyta", "EU NEMATYTA", "JAV matyta", "JAV NEMATYTA")}
        with contextlib.redirect_stdout(io.StringIO()):
            r = Z.vertinti(dd)
        tikrinti(f"verdiktas: {pav}", bool(r), laukta)

    rng = np.random.default_rng(1)
    x = rng.random(100)
    df = pd.DataFrame(dict(rinka="eu", _aukstis=x, _atsiemimas=x + rng.normal(0, 0.01, 100)))
    with contextlib.redirect_stdout(io.StringIO()):
        tikrinti("dublio taisykle suveikia", dublis({"a": df}), True)
    df2 = pd.DataFrame(dict(rinka="eu", _aukstis=x, _atsiemimas=rng.random(100)))
    with contextlib.redirect_stdout(io.StringIO()):
        tikrinti("nepriklausomi -> ne dublis", dublis({"a": df2}), False)

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
