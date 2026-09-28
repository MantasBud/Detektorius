#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RIBU KALIBRACIJA  —  2026-09-28
===============================

Kam sitas failas egzistuoja
---------------------------
Detektoriuje yra desimt ranka parinktu skaiciu. Jie buvo paimti is dvieju
Manto atveju (CAP, SAP) ir kode uzrasyti kaip "HIPOTEZE, ne radinys ... privalo
buti patikrintos nematytoje puseje". Niekada nebuvo patikrintos. Kandidatams i
filtrus pastatyta griezta keturiu matavimu masina, o i paties scenarijaus
skaicius ji nebuvo nukreipta ne karto. Tai tiksliai reitinguotojo klaida.

Sitas failas ta ir daro. Jis NELIECIA detektorius.py - live kelias lieka
nepaliestas. Jis importuoja detektoriu ir naudoja TAS PACIAS funkcijas
(parsisiusti, _kd, baigtis), kad matuotu ta pati, ka matuoja modulis.

Ka literatura is tikruju sako (2026-09-28 tyrimas)
-------------------------------------------------
Pilnas saltiniu sarasas - claude/literatura-ir-ribos.md. Trys dalykai, kurie
tiesiogiai keicia SITO failo konstrukcija:

1. Collin-Dufresne & Daniel (2014), 100 didziausiu JAV bendroviu, 1972-2014,
   dienos duomenys: idiosinkratinio kainos soko **~90% yra nuolatinis, tik
   ~10% laikinas**, ir laikinoji dalis gesta eksponentiskai, **pusejimas 2.4
   dienos**. Patikrinta atskirai, skaitant pati darba.
   -> Laukiamas atsistatymas yra apie **10% kritimo**. Musu tikslas yra 0.5
      ATR prie minimalaus 1.0 ATR kritimo, t.y. **50% kritimo** - penkis
      kartus daugiau, nei literatura sako esant laikina. Todel testuojami
      tikslai, isreiksti KRITIMO dalimi, o ne fiksuotu ATR kartotiniu.

2. Optional stopping teorema (patikrinta simuliacija, 100 tuks. keliu:
   gauta 49.95% prie teoriniu 50.00%): driftui lygiam nuliui,
   P(pirma pasieks +b, o ne -a) = a/(a+b). Vadinasi **simetriski barjerai
   duoda TIKSLIAI nuline laukiama verte**, ir joks R:R santykis pats savaime
   pranasumo neprideda. Dabartine geometrija (tikslas 0.5 ATR, rizika 0.5 ATR)
   yra butent ten. Visas pranasumas privalo ateiti is drifto po iejimo.

3. Sullivan/Timmermann/White (1999) ir Marshall/Cahan/Cahan (2008): 7 846
   taisykliu visata, is ju 2 040 apyvartos taisykliu, su duomenu paieskos
   pataisa - **nematytame laikotarpyje nieko**, o intraday versijoje **visai
   nieko**. Ir tai atsitiko po to, kai Brock/Lakonishok/LeBaron rado stiprius
   matytus rezultatus. Todel cia:
     - kandidatai uzrasyti IS ANKSTO, sitame faile, ir paleidus nekeiciami;
     - spausdinama, kiek ju praeitu **grynai atsitiktinai**.

Statistine disciplina
---------------------
Kandidatas priimamas tik jei: (a) poveikio zenklas tas pats visuose keturiuose
matavimuose, (b) poveikis didesnis uz sanaudas visur, (c) sujungtose
NEMATYTOSE pusese bootstrap p < 0.05. Tikimybe, kad vienas nieko nevertas
kandidatas praeis (a), yra 2*(1/2)^4 = 0.125, tad is 30 kandidatu atsitiktinai
praeitu ~3.8. Sis skaicius spausdinamas salia rezultato.

Paleidimas:
    python ribos.py --rinka abi --dienos 60
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import detektorius as D


# ============================================================ IS ANKSTO
# ============================================================ UZRASYTI KANDIDATAI
#
# Sarasas uzrasytas PRIES paleidima. Jei po rezultatu ji keisime, tai jau bus
# paieska, o ne patikra, ir viska teks pradeti is naujo su nematytais duomenimis.

IEJIMO_KANDIDATAI = [
    # (vardas, laukas, kryptis, riba, is kur skaicius)
    ("kritimas >= 0.5 ATR",   "kritimas_atr",   ">=", 0.50, "tankesnis uz dabartini"),
    ("kritimas >= 1.0 ATR",   "kritimas_atr",   ">=", 1.00, "DABARTINIS"),
    ("kritimas >= 1.6 ATR",   "kritimas_atr",   ">=", 1.60, "Guo 2024: -1.96 sigma ~ 1.6 ATR"),
    ("kritimas >= 2.5 ATR",   "kritimas_atr",   ">=", 2.50, "Bremer&Sweeney -10% ~ 2.5 ATR"),
    ("uzd. vieta <= 0.25",    "uzd_vieta",      "<=", 0.25, "grieztesnis uz dabartini"),
    ("uzd. vieta <= 0.40",    "uzd_vieta",      "<=", 0.40, "DABARTINIS"),
    ("uzd. vieta <= 0.60",    "uzd_vieta",      "<=", 0.60, "laisvesnis"),
    ("uzd. vieta - isjungta", "uzd_vieta",      "<=", 1.01, "Marshall 2006: neigiamas rezultatas"),
    ("apyvarta >= 1.0x",      "apyv_santykis",  ">=", 1.00, "Zarattini 2024 grindys"),
    ("apyvarta >= 1.2x",      "apyv_santykis",  ">=", 1.20, "DABARTINIS"),
    ("apyvarta >= 2.0x",      "apyv_santykis",  ">=", 2.00, "prekybos konvencija"),
    ("apyvarta - isjungta",   "apyv_santykis",  ">=", 0.00, "GKM: nera intraday pagrindo"),
    ("virs VWAP",             "virs_vwap",      ">=", 0.50, "DABARTINIS (nera literaturos)"),
    ("VWAP - isjungtas",      "virs_vwap",      ">=", 0.00, "VWAP yra vykdymo matas, ne signalas"),
    ("virs ORB (30 min)",     "virs_orb",       ">=", 0.50, "DABARTINIS"),
    ("ORB - isjungtas",       "virs_orb",       ">=", 0.00, "STW 1999: proverziai neislieka"),
    ("atsiemimas >= 0.25 ATR", "atsiemimas_atr", ">=", 0.25, "peilio skirtukas is atveju tyrimo"),
    ("atsiemimas >= 0.50 ATR", "atsiemimas_atr", ">=", 0.50, "grieztesnis"),
    ("R:R >= 1.0",            "rr_bazinis",     ">=", 1.00, "DABARTINIS (konvencija)"),
    ("R:R >= 1.5",            "rr_bazinis",     ">=", 1.50, "grieztesnis"),
    ("R:R - isjungtas",       "rr_bazinis",     ">=", 0.00, "nera tyrimo, nustatancio optimalu R:R"),
]

# Isejimai. tikslas_atr - fiksuotas ATR kartotinis; tikslas_dalis - KRITIMO
# dalis (Collin-Dufresne & Daniel). rizika_atr - rizikos riba ATR vienetais.
ISEJIMO_KANDIDATAI = [
    # (vardas, tikslas_atr, tikslas_dalis, rizika_atr, trail_atr, is kur)
    ("T 0.5 ATR / R 0.5 ATR", 0.50, None, 0.50, None, "DABARTINIS"),
    ("T 0.5 ATR / R 0.25 ATR", 0.50, None, 0.25, None, "R:R 2.0"),
    ("T 1.0 ATR / R 0.5 ATR", 1.00, None, 0.50, None, "R:R 2.0, platesnis tikslas"),
    ("T 1.0 ATR / R 1.0 ATR", 1.00, None, 1.00, None, "simetriskas, platus"),
    ("T 0.25 ATR / R 0.5 ATR", 0.25, None, 0.50, None, "Zarattini: siauriau geriau"),
    ("T 10% kritimo / R 0.5 ATR", None, 0.10, 0.50, None, "CDD 2014: laikina dalis ~10%"),
    ("T 20% kritimo / R 0.5 ATR", None, 0.20, 0.50, None, "CDD 2014 x2"),
    ("T 33% kritimo / R 0.5 ATR", None, 0.33, 0.50, None, "tarpinis"),
    ("T 10% kritimo / R 0.25 ATR", None, 0.10, 0.25, None, "CDD + siauras stop'as"),
    ("Be tikslo, R 0.5 ATR", None, None, 0.50, None, "Wu 2021: tikslas kenkia"),
    ("Be tikslo, slenkantis 1 ATR", None, None, None, 1.00, "slenkantis"),
    ("Be tikslo, slenkantis 0.5 ATR", None, None, None, 0.50, "siauresnis slenkantis"),
]


# ============================================================ aptikimas
# Laisviausios salygos. Filtrai taikomi PO TO, kaip pjuviai - tad viena
# kalibracija patikrina visas ribas, ir signalu aibe visoms ribos ta pati.

def laisvas_scenarijus_1(langas, kd):
    """Tas pats, kas D.scenarijus_1, bet be filtruojamu ribu.

    Paliktas TIK tai, kas yra apibrezimas, ne riba:
      - kritimas apskritai buvo (> 0)
      - ne tarpo diena (tai 2 scenarijus)
      - kaina atsieme vakarykscio uzdarymo (tai TRIGERIS, ne filtras)
      - buves lygis dar virs kainos (aritmetika)
    Visa kita irasoma kaip skaicius ir tikrinama veliau.

    savitikra() tikrina, kad su BAZINEMIS ribomis sita funkcija duoda
    TA PATI signala, kaip D.scenarijus_1 - kitaip cia butu antras modulis,
    ir mes matuotume ne ta, kuo prekiaujame.
    """
    if len(langas) < D.S2_ORB_BARU + 2:
        return None
    atr = kd["atr_abs"]
    if not (atr > 0):
        return None
    vakar_uzd = kd["uzdarymas"]
    kaina = float(langas["Close"].iloc[-1])
    kritimas = kd["virsune_n"] - vakar_uzd - kd["div_lange"]
    if kritimas <= 0:
        return None
    atid = float(langas["ses_atidarymas"].iloc[-1])
    if atid - vakar_uzd >= D.S2_MIN_TARPAS_ATR * atr:
        return None
    if kaina <= vakar_uzd:
        return None
    R_pilnas = float(kd["virsune_n"])
    if R_pilnas <= kaina:
        return None

    orb = langas.iloc[:D.S2_ORB_BARU]
    sant = float(langas["apyv_santykis"].iloc[-1])
    vwap = float(langas["vwap"].iloc[-1])
    L = min(float(langas["Low"].min()), vakar_uzd)
    return dict(
        tipas=1, ieina=kaina, atr_abs=atr, L=L, R_pilnas=R_pilnas,
        vakar_uzd=vakar_uzd,
        kritimas_atr=kritimas / atr,
        kritimas_abs=kritimas,
        uzd_vieta=float(kd["uzd_vieta"]),
        apyv_santykis=sant if np.isfinite(sant) else 0.0,
        virs_vwap=1.0 if (np.isfinite(vwap) and kaina > vwap) else 0.0,
        virs_orb=1.0 if kaina > float(orb["High"].max()) else 0.0,
        atsiemimas_atr=(kaina - vakar_uzd) / atr,
        apyvarta=float(kd["apyvarta"]))


def isejimas(toliau, s, tikslas_atr, tikslas_dalis, rizika_atr, trail_atr):
    """Sukuria signala su duota geometrija ir paleidzia TA PACIA D.baigtis()."""
    kaina, atr = s["ieina"], s["atr_abs"]
    if trail_atr is not None:
        stop = max(s["L"] - D.S1_STOP_ATR * atr, kaina - trail_atr * atr)
        sig = dict(tipas=2, ieina=kaina, tikslas=None, stop=stop, atr_abs=atr)
        return D.baigtis(toliau, dict(sig, _trail=trail_atr))
    stop = max(s["L"] - D.S1_STOP_ATR * atr, kaina - rizika_atr * atr)
    if tikslas_atr is not None:
        t = min(s["R_pilnas"], kaina + tikslas_atr * atr)
    elif tikslas_dalis is not None:
        t = min(s["R_pilnas"], kaina + tikslas_dalis * s["kritimas_abs"])
    else:
        t = None
    if t is not None and t <= kaina:
        return None
    return D.baigtis(toliau, dict(tipas=1, ieina=kaina, tikslas=t, stop=stop,
                                  atr_abs=atr))


# ============================================================ matavimas

def eur(pct, rinka):
    return pct / 100.0 * D.POZICIJA - D.sanaudos(rinka)


def dienos_vidurkis(df, stulp="eur"):
    return df.groupby("sesija")[stulp].mean().values


def bootstrap_p(a, b, n=4000, seed=7):
    """Ar a vidurkis didesnis uz b? Persamplinama PAGAL DIENA."""
    if len(a) < 8 or len(b) < 8:
        return float("nan")
    r = np.random.default_rng(seed)
    sk = [r.choice(a, len(a), True).mean() - r.choice(b, len(b), True).mean()
          for _ in range(n)]
    sk = np.array(sk)
    # dvipusis
    return 2 * min((sk <= 0).mean(), (sk >= 0).mean())


def surinkti(rinka, dienos, duomenys=None):
    """Vienas perejimas: laisvi signalai + VISI isejimai kiekvienam."""
    zyme = D.RINKOS[rinka]["zyme"]
    print(f"\n{'='*84}\n{zyme}\n{'='*84}")
    rod, barai = duomenys if duomenys else D.parsisiusti(rinka, dienos)
    eil = []
    for t, d in barai.items():
        sesijos = list(d.groupby("sesija", sort=True))
        for si, (ses, sd) in enumerate(sesijos):
            if len(sd) < D.S2_ORB_BARU + 4:
                continue
            kd = D._kd(rod[t], ses)
            if kd is None:
                continue
            if kd["apyvarta"] < D.min_apyvarta(rinka):
                continue          # tai NE kandidatas - tai Manto pozicijos dydis
            for i in range(D.S2_ORB_BARU + 3, len(sd)):
                s = laisvas_scenarijus_1(sd.iloc[:i + 1], kd)
                if s is None:
                    continue
                toliau = pd.concat(
                    [sd.iloc[i + 1:]] +
                    [sesijos[si + j][1] for j in range(1, D.HORIZONTAS_SESIJU)
                     if si + j < len(sesijos)])
                # R:R pagal BAZINE geometrija - kad ir ji butu kandidatu
                # sarase, o ne tyliai galiojanti konvencija
                _t = min(s["R_pilnas"], s["ieina"] + D.S1_TIKSLAS_ATR * s["atr_abs"])
                _st = max(s["L"] - D.S1_STOP_ATR * s["atr_abs"],
                          s["ieina"] - D.S1_MAX_RIZIKA_ATR * s["atr_abs"])
                s["rr_bazinis"] = ((_t - s["ieina"]) /
                                   max(1e-9, s["ieina"] - _st))
                s.update(tickeris=t, sesija=str(ses), rinka=rinka,
                         minute=int(sd["minute"].iloc[i]))
                for pav, ta, td, ra, tr, _ in ISEJIMO_KANDIDATAI:
                    b = isejimas(toliau, s, ta, td, ra, tr)
                    s[f"iš|{pav}"] = (np.nan if b is None
                                      else eur(b["pelnas_pct"], rinka))
                eil.append(s)
                break             # PIRMAS sesijos signalas, kaip kalibracijoje
    print(f"  laisvu signalu: {len(eil)}")
    return pd.DataFrame(eil)


# ============================================================ verdiktas

def verdiktas(dalys, kandidatai, bazinis):
    """dalys: {matavimo vardas: DataFrame}. Grazina eilutes spausdinimui."""
    out = []
    for pav, f in kandidatai:
        eil, p_nemat = {}, []
        for mv, df in dalys.items():
            if df.empty:
                eil[mv] = None
                continue
            su = df[f(df)]
            if len(su) < 30:
                eil[mv] = ("N", len(su))
                continue
            a = dienos_vidurkis(su.assign(eur=su[bazinis]))
            b = dienos_vidurkis(df.assign(eur=df[bazinis]))
            eil[mv] = (a.mean() - b.mean(), len(su))
            if "NEMATYTA" in mv:
                p_nemat.append((a, b))
        p = float("nan")
        if p_nemat:
            p = bootstrap_p(np.concatenate([x[0] for x in p_nemat]),
                            np.concatenate([x[1] for x in p_nemat]))
        out.append((pav, eil, p))
    return out


TRIUKSMO_FAILAS = "triuksmo_praejo.json"


def triuksmo_praejo(irasyti=None):
    """Kandidatai, praeje ant ATSITIKTINIU duomenu.

    Toks kandidatas nera pranasumas - tai geometrinis artefaktas. Pvz.
    reikalavimas "virs VWAP" ir "virs ORB" ant atsitiktinio klaidziojimo
    praejo su p < 0.01 ir NEIGIAMU poveikiu: iejus po jau ivykusio kilimo,
    tikslas apkarpomas pagal buvusi lygi, o rizika lieka pilna. Tai
    aritmetika, ne radinys.
    """
    if irasyti is not None:
        try:
            with open(TRIUKSMO_FAILAS, "w", encoding="utf-8") as f:
                json.dump(sorted(irasyti), f, ensure_ascii=False, indent=1)
        except Exception:
            pass
        return irasyti
    try:
        if os.path.exists(TRIUKSMO_FAILAS):
            with open(TRIUKSMO_FAILAS, encoding="utf-8") as f:
                return set(json.load(f))
    except Exception:
        pass
    return set()


def spausdinti(pav_lentele, eilutes, matavimai, riba, triuksmas=None):
    print(f"\n{'='*104}\n{pav_lentele}\n{'='*104}")
    print(f"  {'kandidatas':<30}" + "".join(f"{m:>16}" for m in matavimai)
          + f"{'p (nemat.)':>12}  verdiktas")
    praejo = []
    for pav, eil, p in eilutes:
        t = ""
        for m in matavimai:
            v = eil.get(m)
            t += (f"{f'N={v[1]}':>16}" if (v and v[0] == "N")
                  else f"{v[0]:>16.1f}" if v else f"{'-':>16}")
        turim = [eil.get(m) for m in matavimai]
        if any(x is None or x[0] == "N" for x in turim):
            v = "per mazai duomenu"
        else:
            d = [x[0] for x in turim]
            if all(abs(y) < 1e-9 for y in d):
                v = "nieko neismeta (nera ko tikrinti)"
                print(f"  {pav:<30}{t}{p:>12.3f}  {v}")
                continue
            vienodi = all(y > riba for y in d) or all(y < -riba for y in d)
            if not vienodi:
                v = "nelaikosi"
            elif not np.isfinite(p) or p >= 0.05:
                v = f"zenklas sutampa, bet p={p:.2f}"
            elif triuksmas is not None and pav in triuksmas:
                v = "ARTEFAKTAS (praeina ir ant triuksmo)"
            else:
                v = "TINKA"
                praejo.append(pav)
        print(f"  {pav:<30}{t}{p:>12.3f}  {v}")
    return praejo


def paleisti(rinkos, dienos, triuksmas=False):
    dalys = {}
    if triuksmas:
        print("\n" + "!" * 84)
        print("TRIUKSMO KONTROLE: atsitiktinis klaidziojimas, pranasumo NERA.")
        print("Jei cia atsiras 'TINKA', masina per laisva ir jos radiniais")
        print("ant tikru duomenu pasitiketi negalima.")
        print("!" * 84)
        d1 = triuksmo_duomenys(seed=11)
        d2 = triuksmo_duomenys(seed=22)
        rinkos = [("eu", d1), ("us", d2)]
    else:
        rinkos = [(r, None) for r in rinkos]
    for rinka, duom in rinkos:
        df = surinkti(rinka, dienos, duom)
        if df.empty:
            continue
        zyme = D.RINKOS[rinka]["zyme"]
        ses = sorted(df["sesija"].unique())
        riba = ses[len(ses) // 2]
        dalys[f"{zyme} matyta"] = df[df["sesija"] <= riba]
        dalys[f"{zyme} NEMATYTA"] = df[df["sesija"] > riba]
    if not dalys:
        sys.exit("signalu nerasta")
    matavimai = list(dalys)
    bazinis = f"iš|{ISEJIMO_KANDIDATAI[0][0]}"

    # --- 1. IEJIMO ribos (bazinis isejimas visoms) ---
    kand = [(f"{pav}", (lambda df, l=laukas, k=kryptis, r=rb:
                        (df[l] >= r) if k == ">=" else (df[l] <= r)))
            for pav, laukas, kryptis, rb, _ in IEJIMO_KANDIDATAI]
    tr = None if triuksmas else triuksmo_praejo()
    if tr:
        print(f"\n  Triuksmo kontrole anksciau praejo: {', '.join(sorted(tr))}")
        print("  Sie kandidatai bus zymimi ARTEFAKTAIS - jie praeina ir ten,")
        print("  kur pranasumo nera pagal apibrezima.")
    eil = verdiktas(dalys, kand, bazinis)
    p1 = spausdinti("IEJIMO RIBOS (isejimas fiksuotas: dabartinis)",
                    eil, matavimai, max(D.SANAUDOS.values()), tr)

    # --- 2. ISEJIMO geometrija (bazinis iejimas visoms) ---
    def bazinis_iejimas(df):
        return ((df["kritimas_atr"] >= D.S1_MIN_KRITIMAS_ATR) &
                (df["uzd_vieta"] <= D.S1_MAX_UZD_VIETA) &
                (df["apyv_santykis"] >= D.S1_MIN_APYV_SANTYKIS) &
                (df["virs_vwap"] > 0.5) & (df["virs_orb"] > 0.5))
    p2 = []
    print(f"\n{'='*104}\nISEJIMO GEOMETRIJA (iejimas fiksuotas: dabartinis)\n{'='*104}")
    print(f"  {'variantas':<30}" + "".join(f"{m:>16}" for m in matavimai)
          + f"{'p (nemat.)':>12}  verdiktas")
    baz_stulp = bazinis
    for pav, *_ in ISEJIMO_KANDIDATAI:
        st = f"iš|{pav}"
        t, p_nemat, turim = "", [], []
        for m in matavimai:
            df = dalys[m]
            g = df[bazinis_iejimas(df)].dropna(subset=[st, baz_stulp])
            if len(g) < 30:
                t += f"{f'N={len(g)}':>16}"
                turim.append(None)
                continue
            a = dienos_vidurkis(g.assign(eur=g[st]))
            b = dienos_vidurkis(g.assign(eur=g[baz_stulp]))
            turim.append(a.mean() - b.mean())
            t += f"{a.mean() - b.mean():>16.1f}"
            if "NEMATYTA" in m:
                p_nemat.append((a, b))
        p = (bootstrap_p(np.concatenate([x[0] for x in p_nemat]),
                         np.concatenate([x[1] for x in p_nemat]))
             if p_nemat else float("nan"))
        if any(x is None for x in turim):
            v = "per mazai duomenu"
        elif st == baz_stulp:
            v = "bazinis (0 pagal apibrezima)"
        else:
            r = max(D.SANAUDOS.values())
            vienodi = all(y > r for y in turim) or all(y < -r for y in turim)
            v = ("nelaikosi" if not vienodi else
                 f"zenklas sutampa, bet p={p:.2f}" if not np.isfinite(p) or p >= 0.05
                 else "ARTEFAKTAS (praeina ir ant triuksmo)"
                 if (tr and pav in tr) else "TINKA")
            if v == "TINKA":
                p2.append(pav)
        print(f"  {pav:<30}{t}{p:>12.3f}  {v}")

    # --- 3. kiek praeitu ATSITIKTINAI ---
    n = len(IEJIMO_KANDIDATAI) + len(ISEJIMO_KANDIDATAI) - 1
    visi = list(p1) + list(p2)
    if triuksmas:
        triuksmo_praejo(irasyti=visi)
        print(f"\n  Irasyta i {TRIUKSMO_FAILAS}: {len(visi)} kandidatai.")
    print(f"\n{'='*104}")
    print(f"  Patikrinta kandidatu: {n}.  Praejo: {len(visi)}"
          + (f" ({', '.join(visi)})" if visi else "") + ".")
    print(f"  Tikimybe, kad NIEKO NEVERTAS kandidatas duos ta pati zenkla")
    print(f"  visuose keturiuose matavimuose: 2*(1/2)^4 = 0.125.")
    print(f"  Vien del atsitiktinumo tiketina praeitu apie "
          f"{n * 0.125:.1f} kandidatu (be p ribos).")
    print(f"  Todel VIENAS praejes kandidatas NERA iranda. Ieskom tokio,")
    print(f"  kuris praeina ir turi p < 0.05 sujungtose nematytose pusese.")
    print("=" * 104)


# ============================================================ triuksmo kontrole

def triuksmo_duomenys(n_tickeriu=120, dienu=60, seed=11):
    """Atsitiktinis klaidziojimas: pranasumo NERA pagal apibrezima.

    Jei masina ir cia ras "TINKA", vadinasi ji per laisva, ir bet koks
    radinys ant tikru duomenu nieko nereikstu. Butent sito reitinguotojas
    niekada neturejo.
    """
    r = np.random.default_rng(seed)
    rod, barai = {}, {}
    idx_d = pd.bdate_range(end="2026-09-25", periods=400)
    for i in range(n_tickeriu):
        t = f"N{i:03d}"
        ret = r.normal(0.0, 0.020, len(idx_d))
        c = 60 * np.exp(np.cumsum(ret))
        o = c * (1 + r.normal(0, 0.004, len(idx_d)))
        h = np.maximum(o, c) * (1 + abs(r.normal(0, 0.008, len(idx_d))))
        l = np.minimum(o, c) * (1 - abs(r.normal(0, 0.008, len(idx_d))))
        v = r.integers(2e6, 9e6, len(idx_d)).astype(float)
        d = pd.DataFrame(dict(Open=o, High=h, Low=l, Close=c, Volume=v),
                         index=idx_d)
        rod[t] = D.dienos_rodikliai(d)
        eil, ix = [], []
        for ts, row in d.tail(dienu).iterrows():
            t0 = pd.Timestamp(ts.date(), tz="Europe/Berlin") + pd.Timedelta(hours=9)
            w = np.cumsum(r.normal(0, 1, 102)); w -= w.min()
            w = w / max(w.max(), 1e-9)
            k = row["Low"] + (row["High"] - row["Low"]) * w
            k = k + (row["Close"] - k[-1]) * np.linspace(0, 1, 102)
            k[0] = row["Open"]
            for j in range(102):
                ix.append(t0 + pd.Timedelta(minutes=5 * j))
                eil.append((k[j - 1] if j else row["Open"], k[j] * 1.0015,
                            k[j] * 0.9985, k[j],
                            row["Volume"] / 102 * r.uniform(0.4, 2.5)))
        raw = pd.DataFrame(eil, columns=["Open", "High", "Low", "Close",
                                         "Volume"], index=pd.DatetimeIndex(ix))
        barai[t] = D.sesijos_rodikliai(raw, "eu")
    return rod, barai


# ============================================================ savitikra

def savitikra():
    ok = True

    def tikrinti(s, a, b):
        nonlocal ok
        print(f"  {'OK ' if a == b else 'BLOGAI'}  {s}"
              f"{'' if a == b else f'  (gauta {a}, laukta {b})'}")
        ok = ok and (a == b)

    # 1) laisvas scenarijus su BAZINEMIS ribomis = tikrasis scenarijus
    D._KURSAS.clear()
    D._KURSAS["v"] = 1.10
    eiga = np.r_[np.full(6, -0.0008), np.full(24, 0.0012)]
    k = eiga.copy()
    atid = 130.88
    kain = atid * np.cumprod(1 + k)
    eil, pr = [], pd.Timestamp("2026-07-24 09:00", tz="Europe/Berlin")
    for dd in range(30, 0, -1):
        t0 = pr - pd.Timedelta(days=dd)
        for j in range(len(kain)):
            eil.append((t0 + pd.Timedelta(minutes=5 * j), atid, atid * 1.0008,
                        atid * 0.9992, atid, 5e5))
    for j in range(len(kain)):
        eil.append((pr + pd.Timedelta(minutes=5 * j),
                    atid if j == 0 else kain[j - 1], kain[j] * 1.0008,
                    kain[j] * 0.9992, kain[j], 5e5 * 3.0))
    raw = pd.DataFrame(eil, columns=["_t", "Open", "High", "Low", "Close",
                                     "Volume"]).set_index("_t")
    visos = D.sesijos_rodikliai(raw, "eu")
    sd = visos[visos["sesija"] == pr.date()]
    kd = pd.Series(dict(uzdarymas=128.32, atr_abs=3.0, atr_pct=2.34,
                        apyvarta=5e7, uzd_vieta=0.17, virsune_n=138.26,
                        dugnas_n=127.50, div_lange=0.0))

    tikras = laisvas = None
    for i in range(D.S2_ORB_BARU + 3, len(sd)):
        w = sd.iloc[:i + 1]
        if tikras is None:
            tikras = D.scenarijus_1(w, kd)
        if laisvas is None:
            l = laisvas_scenarijus_1(w, kd)
            if l is not None and l["kritimas_atr"] >= D.S1_MIN_KRITIMAS_ATR \
                    and l["uzd_vieta"] <= D.S1_MAX_UZD_VIETA \
                    and l["apyv_santykis"] >= D.S1_MIN_APYV_SANTYKIS \
                    and l["virs_vwap"] > 0.5 and l["virs_orb"] > 0.5:
                laisvas = l
        if tikras and laisvas:
            break
    tikrinti("laisvas scenarijus su bazinemis ribomis randa TA PATI iejima",
             bool(tikras) and bool(laisvas)
             and round(tikras["ieina"], 6) == round(laisvas["ieina"], 6), True)
    tikrinti("        ir ta pati ATR bei L",
             bool(tikras) and round(tikras["L"], 6) == round(laisvas["L"], 6),
             True)

    # 2) laisvas randa DAUGIAU signalu nei tikrasis (ribos tikrai atlaisvintos)
    n_t = sum(1 for i in range(D.S2_ORB_BARU + 3, len(sd))
              if D.scenarijus_1(sd.iloc[:i + 1], kd))
    n_l = sum(1 for i in range(D.S2_ORB_BARU + 3, len(sd))
              if laisvas_scenarijus_1(sd.iloc[:i + 1], kd))
    tikrinti("laisvas randa NE MAZIAU signalu nei tikrasis", n_l >= n_t, True)

    # 3) isejimo variantai tikrai skiriasi
    s = laisvas_scenarijus_1(sd.iloc[:D.S2_ORB_BARU + 8], kd)
    bar = sd.iloc[D.S2_ORB_BARU + 9:]
    rez = {}
    for pav, ta, td, ra, tr, _ in ISEJIMO_KANDIDATAI:
        b = isejimas(bar, s, ta, td, ra, tr)
        rez[pav] = None if b is None else round(b["pelnas_pct"], 4)
    tikrinti("visi isejimo variantai suskaiciuoti", len(rez),
             len(ISEJIMO_KANDIDATAI))
    tikrinti("bent trys skirtingi rezultatai (variantai tikrai skiriasi)",
             len({v for v in rez.values() if v is not None}) >= 3, True)

    # 4) tikslas is KRITIMO dalies tikrai priklauso nuo kritimo
    s_maz = dict(s, kritimas_abs=1.0)
    s_did = dict(s, kritimas_abs=10.0)
    b1 = isejimas(bar, s_maz, None, 0.20, 0.50, None)
    b2 = isejimas(bar, s_did, None, 0.20, 0.50, None)
    tikrinti("tikslas KRITIMO dalimi keiciasi su kritimu",
             (b1 is None) or (b2 is None) or
             (round(b1["pelnas_pct"], 4) != round(b2["pelnas_pct"], 4)), True)

    # 5) bootstrap p: vienodos imtys -> p aukstas; skirtingos -> zemas
    r = np.random.default_rng(3)
    a = r.normal(0, 50, 60)
    tikrinti("bootstrap: vienodos imtys duoda auksta p",
             bootstrap_p(a, r.normal(0, 50, 60)) > 0.2, True)
    tikrinti("bootstrap: aiskiai skirtingos duoda zema p",
             bootstrap_p(r.normal(120, 50, 60), r.normal(0, 50, 60)) < 0.05,
             True)
    tikrinti("bootstrap: maza imtis duoda NaN, o ne netikra tikruma",
             bool(np.isnan(bootstrap_p(a[:5], a[:5]))), True)

    # 6) kandidatu sarasas turi dabartines reiksmes - kitaip nebutu su kuo lyginti
    # sesios dabartines iejimo ribos: kritimas, uzd_vieta, apyvarta, VWAP,
    # ORB ir R:R
    tikrinti("iejimo kandidatuose yra visos SESIOS dabartines ribos",
             sum(1 for x in IEJIMO_KANDIDATAI
                 if x[4].startswith("DABARTINIS")), 6)
    tikrinti("        ir kiekvienai yra bent viena ALTERNATYVA",
             all(sum(1 for x in IEJIMO_KANDIDATAI if x[1] == l) >= 2
                 for l in {x[1] for x in IEJIMO_KANDIDATAI}), True)
    tikrinti("isejimo kandidatuose pirmas yra DABARTINIS",
             ISEJIMO_KANDIDATAI[0][5], "DABARTINIS")

    # 7) triuksmo failo kelias: irasom ir perskaitom
    import tempfile as _tf, os as _os
    global TRIUKSMO_FAILAS
    _tikras = TRIUKSMO_FAILAS
    TRIUKSMO_FAILAS = _os.path.join(_tf.gettempdir(), "savitikra_triuksmas.json")
    try:
        triuksmo_praejo(irasyti=["virs VWAP", "virs ORB (30 min)"])
        tikrinti("triuksmo rezultatas irasomas ir perskaitomas",
                 triuksmo_praejo(), {"virs VWAP", "virs ORB (30 min)"})
        _os.remove(TRIUKSMO_FAILAS)
        tikrinti("be failo grazinama tuscia aibe (o ne isimtis)",
                 triuksmo_praejo(), set())
    finally:
        try:
            _os.remove(TRIUKSMO_FAILAS)
        except OSError:
            pass
        TRIUKSMO_FAILAS = _tikras

    # 8) R:R laukas tikrai skaiciuojamas
    tikrinti("R:R yra kandidatu sarase",
             any(x[1] == "rr_bazinis" for x in IEJIMO_KANDIDATAI), True)

    print("-" * 60)
    print("SAVITIKRA: VISKAS GERAI" if ok else "SAVITIKRA: YRA KLAIDU")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rinka", choices=["eu", "us", "abi"], default="abi")
    ap.add_argument("--dienos", type=int, default=60)
    ap.add_argument("--savitikra", action="store_true")
    ap.add_argument("--triuksmas", action="store_true",
                    help="paleisti ta pacia masina ant atsitiktiniu duomenu")
    a = ap.parse_args()
    if a.savitikra:
        sys.exit(savitikra())
    paleisti(["eu", "us"] if a.rinka == "abi" else [a.rinka], a.dienos,
             a.triuksmas)
