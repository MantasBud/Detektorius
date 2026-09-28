#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALTERNATYVIU IEJIMU PATIKRA  —  2026-09-28
==========================================

Klausimas
---------
ribos.py parode, kad dabartinio iejimo RIBU keitimas rezultato nepajudina:
32 kandidatai, 6 145 EU ir 3 879 JAV signalai, nepraejo ne vienas. Vadinasi,
jei pranasumas yra, jis ne ribose, o pačiame IEJIMO momente.

Sitas failas tikrina tris kitus iejimus:

  A. PIRKTI KRITIMO DIENOS UZDARYME, parduoti kita ryta / kitos dienos uzdaryme.
     Pagrindas: senoje kalibracijoje "rytas%" +0.11% EU ir +0.31% JAV, abiejose
     pusese; reitinguotojo laiku "nakties langas" patvirtintas dviejose rinkose;
     Lou, Polk & Skouras (2019) - didele grazos dalis ivyksta per nakti.
     DEMESIO: senieji skaiciai matuoti senu kodu - cia jie matuojami is naujo.

  B. PIRKTI ANKSCIAU, BE PATVIRTINIMO. Tas pats kontekstas kaip dabartinio
     1 scenarijaus, bet iejimas sesijos atidaryme arba po 30 min - nelaukiant,
     kol kaina atsiims vakarykscio uzdarymo. Tikrina hipoteze, kad patvirtinimo
     laukimas "suvalgo" laikinaja kritimo dali (Collin-Dufresne & Daniel 2014:
     laikina tik ~10% kritimo).

  C. INTRADAY SOKAS (Zawadowski, Andor & Kertesz 2006, Quantitative Finance).
     Per 60 min kaina nukrenta >= X% IR tas judesys >= K kartu didesnis uz
     iprasta TO PACIO paros meto 60 min svyravima. Vienintelis paskelbtas
     trigeris musu dazniu. Originale: 4% ir 8x, NASDAQ +1.93%, NYSE +1.02%
     per kita valanda, 2000-2002 m. duomenys.

Su kuo lyginama - svarbiausia sio failo dalis
---------------------------------------------
Kiekvienas iejimas lyginamas su KONTROLE: tokiu paciu laiku, bet BE SALYGOS.

  A lyginamas su "pirkti BET KURIA likvidzia akcija uzdaryme".
  B1 (atidaryme) - su "pirkti bet kuria atidaryme".
  B2 (po 30 min) - su "pirkti bet kuria po 30 min".
  B3 (dabartinis) ir C (sokas) - su "pirkti bet kuria atsitiktiniu baru".

Be kontroles rinkos kilimas atrodytu kaip pranasumas: jei visa rinka tuo
laikotarpiu kilo, bet koks pirkimas atrodo gerai. Kontrole atima butent tai.
Todel verdiktas vertina SKIRTUMA nuo kontroles, o ne gryna rezultata.

Verdiktas "TINKA" reikalauja VISU triju:
  1. skirtumas nuo kontroles > 10 EUR visuose keturiuose matavimuose;
  2. grynas rezultatas po sanaudu > 0 visuose keturiuose matavimuose;
  3. sujungtose nematytose pusese bootstrap p < 0.05 (persamplinama pagal DIENA).

Paleidimas:
    python iejimai.py --savitikra
    python iejimai.py --triuksmas          # pirma: ar masina nemeluoja ant triuksmo
    python iejimai.py --rinka abi --dienos 60
"""

import argparse
import json
import os
import sys
import warnings
import zlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import detektorius as D
import ribos as R


# ============================================================ IS ANKSTO
# ============================================================ UZRASYTI KANDIDATAI
#
# Sarasas uzrasytas PRIES paleidima. Pakeitus po rezultatu tai jau paieska.

SOKO_LANGAS = 12          # 12 baru x 5 min = 60 min, kaip Zawadowski
SOKO_ISTORIJA = 20        # "iprastas" svyravimas - 20 ankstesniu sesiju

IEJIMAI = [
    # (vardas, grupe, kontrole, parametrai, is kur)
    ("A1 uzdaryme: kritimas>=1.0 ATR, uzd<=0.40", "A", "K uzdaryme",
     dict(kritimas=1.0, uzd=0.40), "dabartinio 1 scen. kontekstas"),
    ("A2 uzdaryme: kritimas>=1.6 ATR", "A", "K uzdaryme",
     dict(kritimas=1.6, uzd=1.01), "Guo 2024: -1.96 sigma"),
    ("B1 atidaryme, be patvirtinimo", "B", "K atidaryme",
     dict(baras="atidarymas"), "pirma galima kaina"),
    ("B2 po 30 min, be patvirtinimo", "B", "K po 30 min",
     dict(baras="orb"), "po atidarymo diapazono"),
    ("B3 DABARTINIS, su patvirtinimu", "B", "K atsitiktinis",
     dict(baras="scenarijus"), "atskaitos taskas"),
    ("C1 sokas >=4% ir >=8x", "C", "K atsitiktinis",
     dict(proc=0.04, kartai=8.0), "Zawadowski 2006 pagrindine"),
    ("C2 sokas >=3% ir >=6x", "C", "K atsitiktinis",
     dict(proc=0.03, kartai=6.0), "Zawadowski rezio vidurys"),
    ("C3 sokas >=2% ir >=6x", "C", "K atsitiktinis",
     dict(proc=0.02, kartai=6.0), "Zawadowski rezio apacia"),
]

# Isejimai. Laiko isejimai matuoja GRYNA drifta - be jokiu barjeru, nes
# simetriski barjerai patys pranasumo neprideda (optional stopping).
# "sim05" - simetriski 0.5 ATR barjerai per ta pacia D.baigtis().
ISEJIMAI = {
    "A": ["kitas rytas", "kitos d. uzdarymas", "sim05"],
    "B": ["dienos uzdarymas", "horizonto pabaiga", "sim05"],
    "C": ["+60 min", "+120 min", "dienos uzdarymas", "sim05"],
}

KONTROLES = {
    "K uzdaryme":    "A",
    "K atidaryme":   "B",
    "K po 30 min":   "B",
    "K atsitiktinis": None,       # abiems B3 ir C - skaiciuojami visi isejimai
}


# ============================================================ pagalbines

def isejimo_kaina(sesijos, si, i, iejimo_kaina, kur, atr, ieina_atidaryme=False):
    """Grazina pelna % arba None, jei to isejimo duomenyse nera.

    sesijos - [(data, DataFrame)], si - sesijos indeksas, i - iejimo baras.
    Iejimas vyksta baro i UZDARYMO kaina (arba atidarymo, jei ieina_atidaryme).
    """
    sd = sesijos[si][1]
    try:
        if kur == "+60 min":
            j = i + 12
            if j >= len(sd):
                return None
            x = float(sd["Close"].iloc[j])
        elif kur == "+120 min":
            j = i + 24
            if j >= len(sd):
                return None
            x = float(sd["Close"].iloc[j])
        elif kur == "dienos uzdarymas":
            if i >= len(sd) - 1 and not ieina_atidaryme:
                return None
            x = float(sd["Close"].iloc[-1])
        elif kur == "kitas rytas":
            if si + 1 >= len(sesijos):
                return None
            x = float(sesijos[si + 1][1]["Open"].iloc[0])
        elif kur == "kitos d. uzdarymas":
            if si + 1 >= len(sesijos):
                return None
            x = float(sesijos[si + 1][1]["Close"].iloc[-1])
        elif kur == "horizonto pabaiga":
            k = si + D.HORIZONTAS_SESIJU - 1
            if k >= len(sesijos):
                return None
            x = float(sesijos[k][1]["Close"].iloc[-1])
        elif kur == "sim05":
            pradzia = sd.iloc[i:] if ieina_atidaryme else sd.iloc[i + 1:]
            toliau = pd.concat(
                [pradzia] + [sesijos[si + j][1]
                             for j in range(1, D.HORIZONTAS_SESIJU)
                             if si + j < len(sesijos)])
            if not len(toliau):
                return None
            b = D.baigtis(toliau, dict(tipas=1, ieina=iejimo_kaina,
                                       tikslas=iejimo_kaina + 0.5 * atr,
                                       stop=iejimo_kaina - 0.5 * atr,
                                       atr_abs=atr))
            return float(b["pelnas_pct"])
        else:
            return None
    except Exception:
        return None
    return (x / iejimo_kaina - 1.0) * 100.0


def s1_kontekstas(kd, sd):
    """Dabartinio 1 scenarijaus DIENOS salygos - tik is praeities (kd) ir
    sios dienos atidarymo. Naudojama B iejimams, kad kontekstas butu tas pats,
    o skirtusi TIK iejimo momentas."""
    atr = kd["atr_abs"]
    if not (atr > 0):
        return False
    krit = kd["virsune_n"] - kd["uzdarymas"] - kd["div_lange"]
    atid = float(sd["ses_atidarymas"].iloc[0])
    return bool(krit >= D.S1_MIN_KRITIMAS_ATR * atr
                and kd["uzd_vieta"] <= D.S1_MAX_UZD_VIETA
                and atid - kd["uzdarymas"] < D.S2_MIN_TARPAS_ATR * atr)


def soko_stulpeliai(d):
    """r12 - 60 min graza; s12 - iprastas TO PACIO paros meto r12 svyravimas.

    s12 skaiciuojamas is ANKSTESNIU sesiju (shift(1) per minutes grupe),
    tad siandienos sokas pats saves "iprasto" svyravimo nepakelia. Tai tas
    pats principas kaip apyv_tipine.
    """
    d = d.copy()
    d["r12"] = d.groupby("sesija")["Close"].transform(
        lambda x: x / x.shift(SOKO_LANGAS) - 1.0)
    d["s12"] = d.groupby("minute")["r12"].transform(
        lambda x: x.shift(1).rolling(SOKO_ISTORIJA, min_periods=10).std())
    return d


def atsitiktinis_baras(t, ses, n):
    """Atkuriamas atsitiktinis baras: ta pati akcija ir diena -> tas pats baras."""
    lo = D.S2_ORB_BARU + 3
    if n - 1 <= lo:
        return None
    sekla = zlib.crc32(f"{t}|{ses}".encode())
    return int(np.random.default_rng(sekla).integers(lo, n - 1))


def sesija_pilna(sd, rinka):
    """Ar sesija baigesi (paskutinis baras arti uzdarymo)?"""
    return int(sd["minute"].iloc[-1]) >= D.RINKOS[rinka]["uzdarymas"] - 10


# ============================================================ surinkimas

def surinkti(rinka, dienos, duomenys=None):
    zyme = D.RINKOS[rinka]["zyme"]
    print(f"\n{'='*84}\n{zyme}\n{'='*84}")
    rod, barai = duomenys if duomenys else D.parsisiusti(rinka, dienos)
    eil = []

    def irasas(vardas, t, ses, si, sesijos, i, kaina, atr, ieina_atid, grupe):
        r = dict(iejimas=vardas, rinka=rinka, tickeris=t, sesija=str(ses))
        kurie = (sorted({x for g in ISEJIMAI.values() for x in g})
                 if grupe is None else ISEJIMAI[grupe])
        for kur in kurie:
            p = isejimo_kaina(sesijos, si, i, kaina, kur, atr, ieina_atid)
            r[kur] = (np.nan if p is None
                      else p / 100.0 * D.POZICIJA - D.sanaudos(rinka))
        eil.append(r)

    for t, d in barai.items():
        d = soko_stulpeliai(d)
        sesijos = list(d.groupby("sesija", sort=True))
        for si, (ses, sd) in enumerate(sesijos):
            if len(sd) < D.S2_ORB_BARU + 4:
                continue
            kd = D._kd(rod[t], ses)
            if kd is None:
                continue
            if kd["apyvarta"] < D.min_apyvarta(rinka):
                continue          # tai ne kandidatas - tai Manto pozicijos dydis
            atr = float(kd["atr_abs"])
            if not (atr > 0):
                continue
            pilna = sesija_pilna(sd, rinka)

            # ---------------- A: uzdaryme ----------------
            if pilna:
                c = float(sd["Close"].iloc[-1])
                i_pask = len(sd) - 1
                irasas("K uzdaryme", t, ses, si, sesijos, i_pask, c, atr,
                       False, "A")
                hi, lo = float(sd["High"].max()), float(sd["Low"].min())
                uzd = (c - lo) / (hi - lo) if hi > lo else 0.5
                krit = (float(kd["virsune_n"]) - c) / atr
                for vardas, grupe, _, prm, _ in IEJIMAI:
                    if grupe == "A" and krit >= prm["kritimas"] and uzd <= prm["uzd"]:
                        irasas(vardas, t, ses, si, sesijos, i_pask, c, atr,
                               False, "A")

            # ---------------- B: atidaryme / po 30 min ----------------
            o = float(sd["Open"].iloc[0])
            irasas("K atidaryme", t, ses, si, sesijos, 0, o, atr, True, "B")
            i_orb = D.S2_ORB_BARU - 1
            c_orb = float(sd["Close"].iloc[i_orb])
            irasas("K po 30 min", t, ses, si, sesijos, i_orb, c_orb, atr,
                   False, "B")
            if s1_kontekstas(kd, sd):
                irasas("B1 atidaryme, be patvirtinimo", t, ses, si, sesijos,
                       0, o, atr, True, "B")
                irasas("B2 po 30 min, be patvirtinimo", t, ses, si, sesijos,
                       i_orb, c_orb, atr, False, "B")
                for i in range(D.S2_ORB_BARU + 3, len(sd)):
                    s = D.scenarijus_1(sd.iloc[:i + 1], kd, rinka)
                    if s and not s["kliutys"]:
                        irasas("B3 DABARTINIS, su patvirtinimu", t, ses, si,
                               sesijos, i, float(s["ieina"]), atr, False, "B")
                        break

            # ---------------- atsitiktine kontrole ----------------
            ia = atsitiktinis_baras(t, ses, len(sd))
            if ia is not None:
                irasas("K atsitiktinis", t, ses, si, sesijos, ia,
                       float(sd["Close"].iloc[ia]), atr, False, None)

            # ---------------- C: sokas ----------------
            r12 = sd["r12"].values
            s12 = sd["s12"].values
            for vardas, grupe, _, prm, _ in IEJIMAI:
                if grupe != "C":
                    continue
                for i in range(SOKO_LANGAS, len(sd)):
                    if not (np.isfinite(r12[i]) and np.isfinite(s12[i])
                            and s12[i] > 0):
                        continue
                    if r12[i] <= -prm["proc"] and -r12[i] >= prm["kartai"] * s12[i]:
                        irasas(vardas, t, ses, si, sesijos, i,
                               float(sd["Close"].iloc[i]), atr, False, "C")
                        break     # PIRMAS sokas per sesija

    df = pd.DataFrame(eil)
    if len(df):
        n = df.groupby("iejimas").size()
        print("  ivykiu:")
        for k, v in n.items():
            print(f"    {k:<44}{v:>7}")
    return df


# ============================================================ verdiktas

TRIUKSMO_FAILAS = "iejimu_triuksmas.json"


def triuksmo_praejo(irasyti=None):
    if irasyti is not None:
        try:
            with open(TRIUKSMO_FAILAS, "w", encoding="utf-8") as f:
                json.dump(sorted(irasyti), f, ensure_ascii=False, indent=1)
        except Exception:
            pass
        return set(irasyti)
    try:
        if os.path.exists(TRIUKSMO_FAILAS):
            with open(TRIUKSMO_FAILAS, encoding="utf-8") as f:
                return set(json.load(f))
    except Exception:
        pass
    return set()


def vertinti(dalys, triuksmas=None):
    """Grazina praejusiu (iejimas|isejimas) sarasa ir spausdina lenteles."""
    matavimai = list(dalys)
    riba = max(D.SANAUDOS.values())
    praejo, testu = [], 0

    for vardas, grupe, kontrole, _, saltinis in IEJIMAI:
        print(f"\n{'-'*112}\n{vardas}   [{saltinis}]   kontrole: {kontrole}")
        print(f"{'-'*112}")
        print(f"  {'isejimas':<20}{'':<9}" + "".join(f"{m:>15}" for m in matavimai)
              + f"{'p (nemat.)':>12}  verdiktas")
        for kur in ISEJIMAI[grupe]:
            testu += 1
            gryn, skirt, n_ev, p_nemat = [], [], [], []
            for m in matavimai:
                df = dalys[m]
                if kur not in df:            # to isejimo stulpelio nera
                    a = b = df.iloc[:0]
                else:
                    a = df[(df["iejimas"] == vardas)].dropna(subset=[kur])
                    b = df[(df["iejimas"] == kontrole)].dropna(subset=[kur])
                n_ev.append(len(a))
                if len(a) < 30 or len(b) < 30:
                    gryn.append(None)
                    skirt.append(None)
                    continue
                ad = a.groupby("sesija")[kur].mean().values
                bd = b.groupby("sesija")[kur].mean().values
                gryn.append(ad.mean())
                skirt.append(ad.mean() - bd.mean())
                if "NEMATYTA" in m:
                    p_nemat.append((ad, bd))
            p = (R.bootstrap_p(np.concatenate([x[0] for x in p_nemat]),
                               np.concatenate([x[1] for x in p_nemat]))
                 if p_nemat else float("nan"))

            def eil(vals, pav):
                return f"  {kur if pav=='grynas' else '':<20}{pav:<9}" + "".join(
                    f"{v:>15.1f}" if v is not None else f"{f'N={n}':>15}"
                    for v, n in zip(vals, n_ev))

            if any(x is None for x in skirt):
                v = "per mazai ivykiu"
            else:
                vienodi = all(y > riba for y in skirt)
                teigiami = all(y > 0 for y in gryn)
                raktas = f"{vardas}|{kur}"
                if not vienodi:
                    v = "nelaikosi"
                elif not teigiami:
                    v = "geriau uz kontrole, bet grynas <= 0"
                elif not np.isfinite(p) or p >= 0.05:
                    v = f"kryptis sutampa, bet p={p:.2f}"
                elif triuksmas and raktas in triuksmas:
                    v = "ARTEFAKTAS (praeina ir ant triuksmo)"
                else:
                    v = "TINKA"
                    praejo.append(raktas)
            print(eil(gryn, "grynas"))
            print(eil(skirt, "- kontr.") + f"{p:>12.3f}  {v}")

    print(f"\n{'='*112}")
    print(f"  Patikrinta iejimo x isejimo poru: {testu}.  Praejo: {len(praejo)}"
          + (f"  ({'; '.join(praejo)})" if praejo else "") + ".")
    print(f"  Atsitiktinai ta pacia kryptimi visose keturiose pusese tiketina")
    print(f"  apie {testu * 0.125:.1f} poru (be p ir be grynojo rezultato ribu).")
    print(f"  Pastaba: slydimas ir spredas NEMODELIUOJAMI. Nagel (2012): apie 40%")
    print(f"  bruto apsisukimo pelno yra spredas. Praejusi pora turi atlaikyti ir ji.")
    print("=" * 112)
    return praejo


def paleisti(rinkos, dienos, triuksmas=False):
    if triuksmas:
        print("!" * 84)
        print("TRIUKSMO KONTROLE: atsitiktinis klaidziojimas, pranasumo NERA.")
        print("Jei cia kas nors gaus 'TINKA', masina per laisva.")
        print("!" * 84)
        poros = [("eu", R.triuksmo_duomenys(seed=11)),
                 ("us", R.triuksmo_duomenys(seed=22))]
    else:
        poros = [(r, None) for r in rinkos]
    dalys = {}
    for rinka, duom in poros:
        df = surinkti(rinka, dienos, duom)
        if df.empty:
            continue
        zyme = D.RINKOS[rinka]["zyme"]
        ses = sorted(df["sesija"].unique())
        riba = ses[len(ses) // 2]
        dalys[f"{zyme} matyta"] = df[df["sesija"] <= riba]
        dalys[f"{zyme} NEMATYTA"] = df[df["sesija"] > riba]
    if not dalys:
        sys.exit("ivykiu nerasta")
    tr = None if triuksmas else triuksmo_praejo()
    if tr:
        print(f"\n  Triuksmo kontrole anksciau praejo: {'; '.join(sorted(tr))}")
    praejo = vertinti(dalys, tr)
    if triuksmas:
        triuksmo_praejo(irasyti=praejo)
        print(f"  Irasyta i {TRIUKSMO_FAILAS}: {len(praejo)}")


# ============================================================ savitikra

def savitikra():
    ok = True

    def tikrinti(s, a, b):
        nonlocal ok
        print(f"  {'OK ' if a == b else 'BLOGAI'}  {s}"
              f"{'' if a == b else f'  (gauta {a}, laukta {b})'}")
        ok = ok and (a == b)

    D._KURSAS.clear()
    D._KURSAS["v"] = 1.10

    # ---- 1. isejimu kainos is zinomu baru ----
    def ses_df(data, kainos, atid=None):
        t0 = pd.Timestamp(data, tz="Europe/Berlin") + pd.Timedelta(hours=9)
        n = len(kainos)
        ix = [t0 + pd.Timedelta(minutes=5 * j) for j in range(n)]
        k = np.asarray(kainos, float)
        o = np.r_[atid if atid else k[0], k[:-1]]
        df = pd.DataFrame(dict(Open=o, High=np.maximum(o, k) * 1.0001,
                               Low=np.minimum(o, k) * 0.9999, Close=k,
                               Volume=1e5), index=pd.DatetimeIndex(ix))
        df["sesija"] = pd.Timestamp(data).date()
        df["minute"] = [x.hour * 60 + x.minute for x in ix]
        return df

    s0 = ses_df("2026-07-20", np.linspace(100, 110, 102))
    # atidarymas SKIRIASI nuo pirmo baro uzdarymo - kitaip "kitas rytas"
    # su Open ir su Close duotu ta pati ir testas butu tuscias
    s1 = ses_df("2026-07-21", np.linspace(121, 130, 102), atid=118.0)
    s2 = ses_df("2026-07-22", np.linspace(130, 90, 102), atid=131.0)
    sesijos = [(x["sesija"].iloc[0], x) for x in (s0, s1, s2)]
    ieina = float(s0["Close"].iloc[10])
    tikrinti("+60 min = 12 baru veliau",
             round(isejimo_kaina(sesijos, 0, 10, ieina, "+60 min", 2.0), 6),
             round((float(s0["Close"].iloc[22]) / ieina - 1) * 100, 6))
    tikrinti("dienos uzdarymas = paskutinis sesijos baras",
             round(isejimo_kaina(sesijos, 0, 10, ieina, "dienos uzdarymas", 2.0), 6),
             round((110.0 / ieina - 1) * 100, 6))
    def r6(x):
        return None if x is None else round(x, 6)
    tikrinti("kitas rytas = KITOS sesijos ATIDARYMAS (118), ne uzdarymas",
             r6(isejimo_kaina(sesijos, 0, 101, 110.0, "kitas rytas", 2.0)),
             r6((118.0 / 110.0 - 1) * 100))
    tikrinti("horizonto pabaiga = TRECIOS sesijos uzdarymas (90)",
             r6(isejimo_kaina(sesijos, 0, 10, ieina, "horizonto pabaiga", 2.0)),
             r6((90.0 / ieina - 1) * 100))
    tikrinti("kitos d. uzdarymas = ANTROS sesijos uzdarymas (130)",
             r6(isejimo_kaina(sesijos, 0, 101, 110.0, "kitos d. uzdarymas", 2.0)),
             r6((130.0 / 110.0 - 1) * 100))
    tikrinti("uzdaryme iejus - 'dienos uzdarymas' NEEGZISTUOJA",
             isejimo_kaina(sesijos, 0, 101, 110.0, "dienos uzdarymas", 2.0), None)
    tikrinti("paskutinei sesijai 'kitas rytas' nera (None, ne klaida)",
             isejimo_kaina(sesijos, 2, 10, 100.0, "kitas rytas", 2.0), None)

    # ---- 2. soko svyravimas be ateities ----
    rng = np.random.default_rng(5)
    visos = []
    for k in range(30):
        data = (pd.Timestamp("2026-06-01") + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        visos.append(ses_df(data, 100 * np.cumprod(1 + rng.normal(0, 0.002, 102))))
    d = pd.concat(visos)
    a = soko_stulpeliai(d)
    d2 = d.copy()
    pask = d2["sesija"] == d2["sesija"].max()
    d2.loc[pask, "Close"] = d2.loc[pask, "Close"] * np.linspace(1, 0.5, pask.sum())
    b = soko_stulpeliai(d2)
    pr = a["sesija"] < a["sesija"].max()
    tikrinti("s12: paskutines sesijos pakeitimas NEKEICIA ankstesniu s12",
             bool(np.allclose(a.loc[pr, "s12"].fillna(-1), b.loc[pr, "s12"].fillna(-1))),
             True)
    tikrinti("s12: ir siandienos s12 nepriklauso nuo siandienos judesio",
             bool(np.allclose(a.loc[pask, "s12"].fillna(-1), b.loc[pask, "s12"].fillna(-1))),
             True)
    tikrinti("r12: siandien sokas matomas (kitaip testas butu tuscias)",
             bool(b.loc[pask, "r12"].min() < a.loc[pask, "r12"].min() - 0.05), True)

    # ---- 3. B3 = tikrasis D.scenarijus_1 ----
    import inspect
    tikrinti("B3 naudoja TIKRA D.scenarijus_1, ne kopija",
             "D.scenarijus_1(" in inspect.getsource(surinkti), True)
    tikrinti("B kontekstas naudoja TAS PACIAS detektoriaus ribas",
             all(x in inspect.getsource(s1_kontekstas) for x in
                 ("S1_MIN_KRITIMAS_ATR", "S1_MAX_UZD_VIETA", "S2_MIN_TARPAS_ATR")),
             True)

    # ---- 4. atsitiktinis baras atkuriamas ir leistiname reze ----
    x1 = atsitiktinis_baras("SAP.DE", "2026-07-24", 102)
    x2 = atsitiktinis_baras("SAP.DE", "2026-07-24", 102)
    tikrinti("atsitiktinis baras atkuriamas", x1, x2)
    tikrinti("        ir leistiname reze",
             D.S2_ORB_BARU + 3 <= x1 < 101, True)
    tikrinti("        skiriasi tarp dienu",
             len({atsitiktinis_baras("SAP.DE", f"2026-07-{k:02d}", 102)
                  for k in range(1, 20)}) > 3, True)

    # ---- 5. kiekvienas iejimas turi savo kontrole ir isejimus ----
    tikrinti("kiekvieno iejimo kontrole egzistuoja",
             all(k in KONTROLES for _, _, k, _, _ in IEJIMAI), True)
    tikrinti("A kontrole ieina uzdaryme, B1 kontrole - atidaryme",
             (KONTROLES["K uzdaryme"], KONTROLES["K atidaryme"]), ("A", "B"))
    tikrinti("visi trys alternatyvus iejimai + dabartinis yra sarase",
             {g for _, g, _, _, _ in IEJIMAI}, {"A", "B", "C"})

    # ---- 6. verdiktas: reikia IR skirtumo, IR grynojo pelno ----
    import io, contextlib
    def dalis(skirt_eur, gryn_eur):
        eil = []
        for di in range(40):
            ses = f"2026-07-{1 + di % 28:02d}-{di}"
            eil.append(dict(iejimas="B1 atidaryme, be patvirtinimo", sesija=ses,
                            **{k: gryn_eur + np.sin(di) for k in ISEJIMAI["B"]}))
            eil.append(dict(iejimas="K atidaryme", sesija=ses,
                            **{k: gryn_eur - skirt_eur + np.cos(di) for k in ISEJIMAI["B"]}))
        return pd.DataFrame(eil)
    for pav, sk, gr, laukta in (("aiskiai geriau ir pelninga", 60, 50, True),
                                ("geriau, bet grynas neigiamas", 60, -20, False),
                                ("pelninga, bet ne geriau uz kontrole", 2, 50, False)):
        dal = {m: dalis(sk, gr) for m in ("EU matyta", "EU NEMATYTA",
                                          "JAV matyta", "JAV NEMATYTA")}
        with contextlib.redirect_stdout(io.StringIO()):
            pr = vertinti(dal)
        tikrinti(f"verdiktas: {pav}",
                 any(x.startswith("B1 atidaryme") for x in pr), laukta)

    print("-" * 60)
    print("SAVITIKRA: VISKAS GERAI" if ok else "SAVITIKRA: YRA KLAIDU")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rinka", choices=["eu", "us", "abi"], default="abi")
    ap.add_argument("--dienos", type=int, default=60)
    ap.add_argument("--savitikra", action="store_true")
    ap.add_argument("--triuksmas", action="store_true")
    a = ap.parse_args()
    if a.savitikra:
        sys.exit(savitikra())
    paleisti(["eu", "us"] if a.rinka == "abi" else [a.rinka], a.dienos,
             a.triuksmas)
