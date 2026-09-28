#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZYMU PATIKRA 2 - DU NAUJI POZYMIAI 1 SCENARIJUI  -  2026-09-28
===============================================================

Tas pats klausimas ir tas pats matas kaip zymos.py: ar pozymis padeda
atskirti TIKRA 1 scenariju (korteles tikslas 0.5 ATR anksciau nei stop'as
per 3 sesijas) nuo panasaus. Signalai - tie patys, kuriuos rodo detektorius.
Niekas nefiltruojama, detektorius.py ir korteles nelieciami.

Naudojama zymos.py masina (surinkimas, keturi matavimai, bootstrap,
verdiktas) be pakeitimu - pakeiciamas TIK pozymiu sarasas.
Saltinis: claude/nauji-pozymiai-2026-09-28.md.

Kritimo langas (kaip detektoriuje): 5 sesijos iki D-1. p = paskutine diena su
didziausiu uzdarymu lange; kritimas = dienos p+1 .. D-1.

UZREGISTRUOTA PRIES PALEIDIMA (ribos naturalios, ne derintos):

  Z9  "ketvirtis krito"      R_Q = ln(Close_p / Close_{p-63}) < 0
        Cheng, Hameed, Subrahmanyam, Titman 2017 (JFQA): po ketvircio
        kritimo atsokimai stipresni -> hipoteze +1 (daugiau tikru).
        Praktiku taisykle ("pirk tik kylanciame trende") sako priesingai,
        todel vertinama dvipusiai; priesinga kryptis butu pazymeta.

  Z10 "kritimas naktimis"    nakties dalis kritime > 0.5
        ON = sum ln((Open_d + dividendas_d) / Close_{d-1})   (d = p+1..D-1)
        ID = sum ln(Close_d / Open_d)
        dalis = ON / (ON + ID), apribota [0, 1]
        Lou, Polk, Skouras 2019 (JFE) + Chan 2003 / Savor 2012: nakties
        kritimas dazniau yra naujiena, o naujienu kritimai reciau
        atsistato -> hipoteze -1 (maziau tikru). Dividendas grazinamas,
        kad mechaninis kritimas nebutu laikomas naujiena.

DUBLIO TAISYKLE (irgi pries paleidima): jei pozymio Spearman |rho| > 0.7 su
jau tikrintu pozymiu (Z9 - atstumas iki 52 sav. aukstumos; Z10 - kritimo
dienos apyvarta ir ataskaita per [D-2, D]), jis laikomas dubliu ir
nenaudojamas, net jei praeitu.

Verdiktas "PADEDA" - kaip zymos.py: ta pati kryptis visose keturiose pusese
>= 5 p.p., sujungtu nematytu pusiu p < 0.05, abiejose grupese >= 30 signalu.

Paleidimas:
    python zymos2.py --savitikra
    python zymos2.py
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import detektorius as D
import zymos as Z

RQ_SESIJU = 63
DUBLIO_RIBA = 0.7

NAUJOS_ZYMOS = [
    ("Z9", 1, "ketvirtis krito", +1),
    ("Z10", 1, "kritimas naktimis", -1),
]

# kas su kuo tikrinama del dublio: (zyma, zalias stulpelis, su kuo, pavadinimas)
DUBLIAI = [
    ("Z9", "_rq", "_52s", "atstumas iki 52 sav. aukstumos"),
    ("Z10", "_naktis", "_relvol", "kritimo dienos apyvarta"),
    ("Z10", "_naktis", "Z1", "ataskaita per [D-2, D]"),
]

_originalus_kontekstas = Z.dienos_kontekstas


# ============================================================ pozymiai

def kritimo_pozymiai(d, langas=D.S1_LANGAS_D, rq_n=RQ_SESIJU):
    """Eilute D turi TIK informacija iki D-1 (be pacios D).

    Grazina DataFrame(index=d.index) su stulpeliais rq, naktis.
    """
    c = d["Close"].values.astype(float)
    o = d["Open"].values.astype(float)
    if "Dividends" in d:
        dv = pd.to_numeric(d["Dividends"], errors="coerce").fillna(0.0).values
    else:
        dv = np.zeros(len(d))
    n = len(d)
    rq = np.full(n, np.nan)
    nak = np.full(n, np.nan)
    for i in range(langas, n):                       # i = diena D
        lang = c[i - langas:i]                       # D-5 .. D-1
        if not np.all(np.isfinite(lang)):
            continue
        # paskutine diena su didziausiu uzdarymu
        p = i - langas + int(np.flatnonzero(lang == lang.max())[-1])
        if p - rq_n >= 0 and c[p - rq_n] > 0:
            rq[i] = np.log(c[p] / c[p - rq_n])
        if p >= i - 1:                               # virsune vakar - kritimo nera
            continue
        j = np.arange(p + 1, i)                      # p+1 .. D-1
        if np.any(o[j] <= 0) or np.any(c[j - 1] <= 0):
            continue
        on = np.log((o[j] + dv[j]) / c[j - 1]).sum()
        idr = np.log(c[j] / o[j]).sum()
        vis = on + idr
        if not np.isfinite(vis) or vis >= 0:
            continue
        nak[i] = float(np.clip(on / vis, 0.0, 1.0))
    return pd.DataFrame({"rq": rq, "naktis": nak}, index=d.index)


def dienos_kontekstas2(dienos, sekt):
    out = _originalus_kontekstas(dienos, sekt)       # jau paslinktas (iki D-1)
    for t, d in dienos.items():
        kp = kritimo_pozymiai(d)                     # jau "iki D-1" pagal konstrukcija
        out[t] = out[t].join(kp)
    return out


def zymos_signalui2(tipas, k, atask_datos, ses):
    if tipas != 1:
        return {}
    z1 = Z.ataskaita_lange(atask_datos, ses, -2, 0)
    rq, nk = k.get("rq", np.nan), k.get("naktis", np.nan)
    return {
        "Z9": np.nan if pd.isna(rq) else float(rq < 0),
        "Z10": np.nan if pd.isna(nk) else float(nk > 0.5),
        "Z1": z1,
        "_rq": rq, "_naktis": nk,
        "_52s": k.get("iki_52s", np.nan), "_relvol": k.get("relvol", np.nan),
    }


def ijungti():
    """zymos.py masina su naujais pozymiais (nieko kito nekeiciant)."""
    Z.ZYMOS = NAUJOS_ZYMOS
    Z.dienos_kontekstas = dienos_kontekstas2
    Z.zymos_signalui = zymos_signalui2


# ============================================================ dublio patikra

def dublio_patikra(dalys):
    visi = pd.concat(list(dalys.values()))
    s1 = visi[visi["tipas"] == 1]
    dubliai = set()
    print(f"\n{'='*84}\nDUBLIO PATIKRA (Spearman, visi 1 scenarijaus signalai; riba |rho| > {DUBLIO_RIBA})\n{'='*84}")
    for kod, a, b, pav in DUBLIAI:
        for rinka in ("eu", "us"):
            g = s1[s1["rinka"] == rinka][[a, b]].dropna()
            if len(g) < 30:
                print(f"  {kod} vs {pav:<32} {rinka.upper():<4} per mazai ({len(g)})")
                continue
            rho = g[a].rank().corr(g[b].rank())
            ar = abs(rho) > DUBLIO_RIBA
            if ar:
                dubliai.add(kod)
            print(f"  {kod} vs {pav:<32} {rinka.upper():<4} rho={rho:+.2f}"
                  f"  N={len(g)}{'   <<< DUBLIS' if ar else ''}")
    for rinka in ("eu", "us"):
        g = s1[s1["rinka"] == rinka]
        for kod in ("Z9", "Z10"):
            x = g[kod].dropna()
            print(f"  {kod} {rinka.upper()}: zyma yra {int(x.sum())} is {len(x)} signalu"
                  f" ({(x.mean()*100 if len(x) else 0):.0f}%)")
    return dubliai


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
    dubliai = dublio_patikra(dalys)
    padeda = Z.vertinti(dalys)
    galut = [x for x in padeda if x.split()[0] not in dubliai]
    print(f"\nGALUTINIS: {', '.join(galut) if galut else 'nei vienas pozymis nepraejo'}"
          + (f"  (atmesti kaip dubliai: {', '.join(sorted(dubliai))})" if dubliai else ""))


# ============================================================ savitikra

def savitikra():
    ok = True

    def tikrinti(s, a, b):
        nonlocal ok
        gerai = (a == b)
        print(f"  {'OK ' if gerai else 'BLOGAI'}  {s}"
              f"{'' if gerai else f'  (gauta {a}, laukta {b})'}")
        ok = ok and gerai

    idx = pd.bdate_range("2025-01-01", periods=120)

    def serija(o, c, dv=None):
        d = pd.DataFrame(dict(Open=o, High=np.maximum(o, c) * 1.001,
                              Low=np.minimum(o, c) * 0.999, Close=c,
                              Volume=np.full(len(c), 1e6)), index=idx[:len(c)])
        d["Dividends"] = 0.0 if dv is None else dv
        return d

    n = 100
    # 1) kilusi 63 sesijas, tada 3 dienos kritimo TIK naktimis (atsidaro zemiau, uzdaro = atidarymas)
    c = np.linspace(80, 100, n)
    o = c.copy()
    for k, j in enumerate(range(n - 4, n - 1)):         # D-3..D-1 (D = n-1)
        c[j] = 100 * (0.97 ** (k + 1)); o[j] = c[j]
    o[n - 4] = c[n - 4]; c[n - 5] = 100; o[n - 5] = 100
    d1 = serija(o.copy(), c.copy())
    kp = kritimo_pozymiai(d1).iloc[-1]
    tikrinti("tik nakties kritimas -> nakties dalis 1.0", round(float(kp["naktis"]), 6), 1.0)
    tikrinti("pries kritima kilo -> R_Q > 0 (Z9 = 0)", bool(kp["rq"] > 0), True)
    rq_laukta = np.log(100 / c[n - 5 - RQ_SESIJU])
    tikrinti("R_Q = ln(Close_p / Close_{p-63}) tiksliai",
             round(float(kp["rq"]), 10), round(float(rq_laukta), 10))

    # 2) tas pats kritimas, bet TIK sesijos metu (atsidaro ties vakar uzdarymu)
    o2 = o.copy()
    for j in range(n - 4, n - 1):
        o2[j] = c[j - 1]
    kp2 = kritimo_pozymiai(serija(o2, c.copy())).iloc[-1]
    tikrinti("tik dienos kritimas -> nakties dalis 0.0", round(float(kp2["naktis"]), 6), 0.0)

    # 3) pusiau: nakti -1%, diena -1% kiekviena diena -> ~0.5
    c3, o3 = c.copy(), o.copy()
    for k, j in enumerate(range(n - 4, n - 1)):
        o3[j] = c3[j - 1] * 0.99; c3[j] = o3[j] * 0.99
    kp3 = kritimo_pozymiai(serija(o3, c3)).iloc[-1]
    tikrinti("puse nakti, puse diena -> 0.5", round(float(kp3["naktis"]), 6), 0.5)

    # 4) dividendas: nakties tarpas lygus dividendui nelaikomas kritimu
    c4, o4 = c.copy(), o2.copy()          # dienos kritimas
    dv = np.zeros(n)
    o4[n - 3] = c4[n - 4] - 2.0; dv[n - 3] = 2.0   # ex-div: atsidaro -2, dividendas 2
    c4[n - 3] = o4[n - 3] * (c[n - 3] / c[n - 4])
    c4[n - 2] = c4[n - 3] * (c[n - 2] / c[n - 3]); o4[n - 2] = c4[n - 3]
    kp4 = kritimo_pozymiai(serija(o4, c4, dv)).iloc[-1]
    tikrinti("dividendo tarpas nakties kritimu nelaikomas -> 0.0",
             round(float(kp4["naktis"]), 6), 0.0)
    kp4b = kritimo_pozymiai(serija(o4, c4, np.zeros(n))).iloc[-1]
    tikrinti("be dividendo korekcijos tas pats tarpas jau matomas (testas ne tuscias)",
             bool(kp4b["naktis"] > 0.05), True)

    # 5) ateitis: pakeitus pacia D diena, D eilute nesikeicia; D+1 - keiciasi
    d5 = serija(o.copy(), c.copy())
    d5b = d5.copy()
    d5b.iloc[-1, d5b.columns.get_loc("Close")] = 50.0
    d5b.iloc[-1, d5b.columns.get_loc("Open")] = 50.0
    a5 = kritimo_pozymiai(d5).iloc[-1].fillna(-9).values
    b5 = kritimo_pozymiai(d5b).iloc[-1].fillna(-9).values
    tikrinti("D eilute nepriklauso nuo pacios D", bool(np.allclose(a5, b5)), True)
    d5c = pd.concat([d5b, serija(np.array([50.0]), np.array([50.0])).set_axis([idx[n]])])
    tikrinti("D+1 eilute jau mato D (testas ne tuscias)",
             bool(kritimo_pozymiai(d5c).iloc[-1]["naktis"] > 0.5), True)

    # 6) virsune vakar (kritimo nera) -> NaN; R_Q be 63 sesiju istorijos -> NaN
    c6 = np.linspace(90, 100, n)
    kp6 = kritimo_pozymiai(serija(c6.copy(), c6.copy())).iloc[-1]
    tikrinti("nera kritimo -> nakties dalis NaN", bool(np.isnan(kp6["naktis"])), True)
    kp7 = kritimo_pozymiai(serija(o[:40].copy(), c[:40].copy())).iloc[-1]
    tikrinti("per trumpa istorija -> R_Q NaN", bool(np.isnan(kp7["rq"])), True)

    # 7) zymos is reiksmiu
    zz = zymos_signalui2(1, pd.Series(dict(rq=-0.1, naktis=0.8, iki_52s=0.9, relvol=1.0)),
                         [], idx[-1].date())
    tikrinti("R_Q < 0 -> Z9 = 1", zz["Z9"], 1.0)
    tikrinti("nakties dalis 0.8 -> Z10 = 1", zz["Z10"], 1.0)
    zz = zymos_signalui2(1, pd.Series(dict(rq=0.0, naktis=0.5, iki_52s=0.9, relvol=1.0)),
                         [], idx[-1].date())
    tikrinti("R_Q = 0 -> Z9 = 0 (riba grieztai < 0)", zz["Z9"], 0.0)
    tikrinti("nakties dalis 0.5 -> Z10 = 0 (riba grieztai > 0.5)", zz["Z10"], 0.0)
    tikrinti("2 scenarijui zymu nera", zymos_signalui2(2, pd.Series(dtype=float), [], None), {})

    # 8) verdiktas per zymos.py masina su naujomis zymomis
    import io, contextlib
    ijungti()

    def dal(sk_pp, kod):
        eil = []
        rng = np.random.default_rng(3)
        for i in range(300):
            s = f"2026-0{1 + i % 8}-{1 + i % 27:02d}"
            zy = float(i % 2)
            p = 0.5 + (sk_pp / 200 if zy else -sk_pp / 200)
            eil.append({"sesija": s, "tipas": 1, "tikras": float(rng.random() < p),
                        "Z9": zy if kod == "Z9" else np.nan,
                        "Z10": zy if kod == "Z10" else np.nan})
        return pd.DataFrame(eil)
    for pav, sk, kod, laukta in (("Z9 aiskus skirtumas -> PADEDA", 30, "Z9", True),
                                 ("Z10 aiskus skirtumas -> PADEDA", -30, "Z10", True),
                                 ("Z9 be skirtumo -> nepadeda", 0, "Z9", False)):
        dd = {m: dal(sk, kod) for m in ("EU matyta", "EU NEMATYTA",
                                        "JAV matyta", "JAV NEMATYTA")}
        with contextlib.redirect_stdout(io.StringIO()):
            pad = Z.vertinti(dd)
        tikrinti(f"verdiktas: {pav}", any(x.split()[0] == kod for x in pad), laukta)

    # 9) dublio taisykle suveikia
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    eil = pd.DataFrame(dict(rinka=["eu"] * 100 + ["us"] * 100, tipas=1,
                            _rq=x, _52s=x + rng.normal(0, 0.05, 200),
                            _naktis=rng.random(200), _relvol=rng.random(200),
                            Z1=0.0, Z9=1.0, Z10=0.0))
    with contextlib.redirect_stdout(io.StringIO()):
        dub = dublio_patikra({"a": eil})
    tikrinti("dublis su 52 sav. -> Z9 atmetamas, Z10 ne", sorted(dub), ["Z9"])

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
