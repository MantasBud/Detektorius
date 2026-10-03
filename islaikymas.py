#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISSILAIKYMO PATIKRA - 1 SCENARIJUS  -  2026-10-03
=================================================

Klausimas (Manto): ar 1 scenarijus pataikys dazniau, jei kortele rodysim tik
tada, kai signalas ISSILAIKO 10 min (2 barus), ir iejimas bus tada?

Uzuominos (claude/zurnalo-analize-2026-10-03.md): savaite EU kortele del
dalinio baro pagaudavo tik ilgiau issilaikiusius / stiprios apyvartos
signalus - ju pasiteisino 45%, korteles praleistu 24%; nepasiteisine signalai
"neuzsikuria" (stop'u MFE mediana 0.12 ATR). Tai TIK hipoteze is 4 dienu.

UZREGISTRUOTA PRIES PALEIDIMA
-----------------------------
Signalai - tie patys kaip kalibracijoje ir zymos.py: pirmas TINKAMAS
1 scenarijaus signalas akcijos sesijoje, baras i, D.aptikti().

  A (dabartine taisykle): visi signalai, iejimas bare i (jo uzdarymo kaina),
     tikslas/stop - kaip kortelėje.
  B (issilaikymas): signalas laikomas patvirtintu, jei bare i+2 (po 10 min)
     visos 1 scenarijaus salygos VIS DAR galioja, ISSKYRUS apyvartos santyki
     (kaina virs vakar uzdarymo, virs 30 min diapazono, virs VWAP, buves
     lygis dar virs kainos, ne tarpo diena) ir signalas tinkamas (be kliuciu,
     R:R >= 1). Iejimas - baro i+2 uzdarymo kaina; tikslas/stop perskaiciuojami
     detektoriaus taisyklemis nuo naujo iejimo. Jei i+2 iseina uz sesijos -
     nepatvirtintas (kortele jo nebutu parodziusi).
  "Tikras" - korteles tikslas pasiekiamas anksciau nei stop'as per 3 sesijas,
     matuojant nuo SAVO iejimo (A - nuo i, B - nuo i+2), ta pacia
     D.baigtis().

VERDIKTAS "ISSILAIKYMAS PADEDA" - VISOS trys:
  1. tikru dalis B minus tikru dalis A >= +5 p.p. visose keturiose pusese
     (EU/JAV x matyta/nematyta, pusiau pagal sesiju data);
  2. sujungtose nematytose pusese bootstrap pagal diena p < 0.05;
  3. kiekvienoje puseje B turi >= 30 signalu.
Papildomai (tik informacija, verdikto nekeicia): vid. EUR A ir B, kiek
signalu issilaiko, nepatvirtintu tikru dalis, ir patvirtintu tikru dalis,
jei butu iejusi bare i (kad matytusi, kiek lemia atranka, kiek velesnis
iejimas).

Niekas nefiltruojama, detektorius.py ir korteles nelieciami.

Paleidimas:
    python islaikymas.py --savitikra
    python islaikymas.py
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import detektorius as D
import zymos as Z

PO_BARU = 2
MIN_PP = 5.0
MIN_N = 30
APYV_APEJIMAS = 1e6     # apyvartos salyga patvirtinimo bare netikrinama


def _s1_tinkamas(langas, kd, iki, rinka):
    for s in D.aptikti(langas, kd, iki, rinka):
        if s["tipas"] == 1 and s["tinkamas"]:
            return s
    return None


def _eur(b, rinka):
    return b["pelnas_pct"] / 100 * D.POZICIJA - D.sanaudos(rinka)


def _toliau(sesijos, si, sd, j):
    return pd.concat([sd.iloc[j + 1:]] +
                     [sesijos[si + k][1] for k in range(1, D.HORIZONTAS_SESIJU)
                      if si + k < len(sesijos)])


def ivertinti(sesijos, si, i, kd, rinka, sigA):
    """A ir B baigtys vienam signalui (baras i sesijoje si)."""
    sd = sesijos[si][1]
    bA = D.baigtis(_toliau(sesijos, si, sd, i), sigA)
    e = dict(tikrasA=float(bA["baigtis"] == "tikslas"), eurA=_eur(bA, rinka),
             islaike=0.0, tikrasB=np.nan, eurB=np.nan)
    j = i + PO_BARU
    if j >= len(sd):
        return e
    w = sd.iloc[:j + 1].copy()
    w.iloc[-1, w.columns.get_loc("apyv_santykis")] = APYV_APEJIMAS
    iki = D.RINKOS[rinka]["uzdarymas"] - int(sd["minute"].iloc[j])
    sigB = _s1_tinkamas(w, kd, iki, rinka)
    if sigB is None:
        return e
    bB = D.baigtis(_toliau(sesijos, si, sd, j), sigB)
    e.update(islaike=1.0, tikrasB=float(bB["baigtis"] == "tikslas"), eurB=_eur(bB, rinka),
             ieinaB=sigB["ieina"])
    return e


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
                s = _s1_tinkamas(sd.iloc[:i + 1], kd, iki, rinka)
                if s is None:
                    continue
                e = dict(rinka=rinka, tickeris=t, sesija=str(ses), ieinaA=s["ieina"])
                e.update(ivertinti(sesijos, si, i, kd, rinka, s))
                eil.append(e)
                break
    df = pd.DataFrame(eil)
    if len(df):
        print(f"  1 scenarijus: {len(df)} signalu, A tikru {df['tikrasA'].mean()*100:.1f}%, "
              f"issilaike {df['islaike'].mean()*100:.0f}%")
    return df


# ============================================================ verdiktas

def p_dienomis(df, n=4000, seed=7):
    """Dvipusis bootstrap p skirtumui (B dalis - A dalis), persamplinant dienas."""
    dienos = df["sesija"].unique()
    if len(dienos) < 8:
        return float("nan")
    gr = {d: g for d, g in df.groupby("sesija")}
    r = np.random.default_rng(seed)
    sk = []
    for _ in range(n):
        x = pd.concat([gr[d] for d in r.choice(dienos, len(dienos), True)])
        b = x.loc[x["islaike"] == 1.0, "tikrasB"]
        if not len(b):
            continue
        sk.append(b.mean() - x["tikrasA"].mean())
    sk = np.array(sk)
    return 2 * min((sk <= 0).mean(), (sk >= 0).mean())


def vertinti(dalys):
    print(f"\n{'='*104}\nA (visi, iejimas signalo bare) vs B (issilaike 10 min, iejimas tada)\n{'='*104}")
    print(f"  {'puse':<14}{'A N':>6}{'A tikru':>9}{'A EUR':>8}{'B N':>6}{'B tikru':>9}{'B EUR':>8}"
          f"{'B-A p.p.':>10}{'nepatv.':>9}{'patv.@i':>9}")
    sk, nn, nem = [], [], []
    for m, df in dalys.items():
        b = df[df["islaike"] == 1.0]
        ne = df[df["islaike"] == 0.0]
        a_t, b_t = df["tikrasA"].mean(), b["tikrasB"].mean() if len(b) else np.nan
        d = (b_t - a_t) * 100 if len(b) else np.nan
        sk.append(d); nn.append(len(b))
        if "NEMATYTA" in m:
            nem.append(df)
        print(f"  {m:<14}{len(df):>6}{a_t*100:>8.1f}%{df['eurA'].mean():>8.0f}{len(b):>6}"
              f"{(b_t*100 if len(b) else float('nan')):>8.1f}%{b['eurB'].mean() if len(b) else float('nan'):>8.0f}"
              f"{d:>+10.1f}{(ne['tikrasA'].mean()*100 if len(ne) else float('nan')):>8.1f}%"
              f"{(b['tikrasA'].mean()*100 if len(b) else float('nan')):>8.1f}%")
    p = p_dienomis(pd.concat(nem)) if nem else float("nan")
    if any(n < MIN_N for n in nn):
        v = "PER MAZAI SIGNALU"
    elif not all(np.isfinite(x) and x >= MIN_PP for x in sk):
        v = "NEPADEDA (nelaikosi visose keturiose pusese >= +5 p.p.)"
    elif not np.isfinite(p) or p >= 0.05:
        v = f"KRYPTIS SUTAMPA, BET p={p:.3f}"
    else:
        v = "ISSILAIKYMAS PADEDA"
    print(f"\n  nepatv. = neissilaikiusiu tikru dalis (iejimas i); patv.@i = issilaikiusiu, jei iejimas i")
    print(f"  bootstrap p (nematytos puses, pagal diena): {p:.3f}")
    print(f"\nGALUTINIS: {v}")
    print("=" * 104)
    return v == "ISSILAIKYMAS PADEDA"


def paleisti(dienos5):
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
    vertinti(dalys)


# ============================================================ savitikra

def savitikra():
    ok = True

    def tikrinti(s, a, b):
        nonlocal ok
        g = (a == b)
        print(f"  {'OK ' if g else 'BLOGAI'}  {s}{'' if g else f'  (gauta {a}, laukta {b})'}")
        ok = ok and g

    import io, contextlib
    ix = pd.date_range("2026-07-24 09:00", periods=20, freq="5min", tz="Europe/Berlin")

    def ses(close):
        x = pd.DataFrame({"Open": close, "High": np.array(close) + 0.2,
                          "Low": np.array(close) - 0.2, "Close": close,
                          "apyv_santykis": 1.5, "minute": [9 * 60 + 5 * k for k in range(20)]},
                         index=ix)
        return x

    kd = pd.Series({"atr_abs": 2.0})
    base = dict(tipas=1, tinkamas=True, ieina=100.0, tikslas=101.0, stop=99.0, atr_abs=2.0)
    tikras_apt = D.aptikti

    # netikras aptikti: signalas galioja, kol paskutinis close >= 100; apyvartos
    # salyga tikrinama (>= 1.2), kad matytume, ar patvirtinimo bare ji apeinama
    def fake(langas, kd_, iki=None, rinka="eu"):
        c = float(langas["Close"].iloc[-1])
        if c >= 100 and float(langas["apyv_santykis"].iloc[-1]) >= 1.2:
            return [dict(base, ieina=c, tikslas=c + 1, stop=c - 1)]
        return []
    D.aptikti = fake
    try:
        # 1) issilaiko: i=10 close 100, i+2 close 100.5 -> B iejimas 100.5
        c = [99.0] * 10 + [100.0, 100.2, 100.5] + [100.6] * 7
        s1 = ses(c)
        e = ivertinti([(ix[0].date(), s1)], 0, 10, kd, "eu", dict(base))
        tikrinti("issilaike, kai i+2 salygos galioja", e["islaike"], 1.0)
        tikrinti("B iejimas = baro i+2 uzdarymas", e.get("ieinaB"), 100.5)
        # 2) apyvarta patvirtinimo bare NEtikrinama
        s2 = s1.copy(); s2.iloc[12, s2.columns.get_loc("apyv_santykis")] = 0.5
        e2 = ivertinti([(ix[0].date(), s2)], 0, 10, kd, "eu", dict(base))
        tikrinti("patvirtinimo bare apyvartos salyga apeinama", e2["islaike"], 1.0)
        # 3) neissilaiko: i+2 grizta zemiau
        c3 = [99.0] * 10 + [100.0, 99.5, 99.0] + [99.0] * 7
        e3 = ivertinti([(ix[0].date(), ses(c3))], 0, 10, kd, "eu", dict(base))
        tikrinti("neissilaike, kai i+2 kaina grizo", (e3["islaike"], np.isnan(e3["tikrasB"])),
                 (0.0, True))
        # 4) i+2 uz sesijos -> nepatvirtintas
        e4 = ivertinti([(ix[0].date(), s1)], 0, 18, kd, "eu", dict(base))
        tikrinti("signalas paskutiniuose 2 baruose -> nepatvirtintas", e4["islaike"], 0.0)
        # 5) baigtis B matuojama nuo baru PO i+2: tikslas pasiektas bare i+1
        #    (101.0) -> A tikras; B iejimas 100.5, tikslas 101.5, toliau krenta -> B ne
        c5 = [99.0] * 10 + [100.0, 101.0, 100.5] + [99.0] * 7
        s5 = ses(c5); s5["High"] = s5["Close"] + 0.05; s5["Low"] = s5["Close"] - 0.05
        e5 = ivertinti([(ix[0].date(), s5)], 0, 10, kd, "eu", dict(base))
        tikrinti("A tikras (tikslas i+1), B ne (matuojama po i+2)",
                 (e5["tikrasA"], e5["tikrasB"]), (1.0, 0.0))
        # 6) B baigtis matuojama TIK nuo baru po i+2: bare i+1 kaina buvo
        #    zemiau B stop'o (99.5), bet tai ivyko PRIES B iejima -> neskaiciuojama
        c6 = [99.0] * 10 + [100.0, 99.3, 100.5] + [101.6] * 7
        s6 = ses(c6); s6["High"] = s6["Close"] + 0.05; s6["Low"] = s6["Close"] - 0.05
        e6 = ivertinti([(ix[0].date(), s6)], 0, 10, kd, "eu", dict(base))
        tikrinti("B: judesys PRIES i+2 neiskaiciuojamas (B tikras)", e6["tikrasB"], 1.0)
    finally:
        D.aptikti = tikras_apt

    # 6) verdiktas
    def dal(pa, pb, n=200, seed=1):
        r = np.random.default_rng(seed)
        return pd.DataFrame([dict(sesija=f"2026-0{1 + k % 8}-{1 + k % 27:02d}",
                                  tikrasA=float(r.random() < pa), eurA=0.0,
                                  islaike=1.0, tikrasB=float(r.random() < pb), eurB=0.0)
                             for k in range(n)])
    for pav, pa, pb, laukta in (("B aiskiai geriau -> PADEDA", 0.4, 0.7, True),
                                ("jokio skirtumo -> ne", 0.5, 0.5, False)):
        dd = {m: dal(pa, pb, seed=i) for i, m in enumerate(
            ("EU matyta", "EU NEMATYTA", "JAV matyta", "JAV NEMATYTA"))}
        with contextlib.redirect_stdout(io.StringIO()):
            r = vertinti(dd)
        tikrinti(f"verdiktas: {pav}", r, laukta)
    dd = {m: dal(0.0, 1.0, n=20) for m in ("EU matyta", "EU NEMATYTA", "JAV matyta", "JAV NEMATYTA")}
    with contextlib.redirect_stdout(io.StringIO()):
        r = vertinti(dd)
    tikrinti("verdiktas: < 30 signalu puseje -> ne (net su didziuliu skirtumu)", r, False)
    # priesinga MATYTA puse: nematytu p butu geras, tad sustabdyti gali tik
    # "visose keturiose" taisykle
    dd = {m: dal(0.0, 1.0) for m in ("EU matyta", "EU NEMATYTA", "JAV NEMATYTA")}
    dd["JAV matyta"] = dal(1.0, 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        r = vertinti(dd)
    tikrinti("verdiktas: viena puse priesinga -> ne (reikia visu keturiu)", r, False)

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
