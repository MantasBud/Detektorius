#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A IEJIMAS SU 2 METU DUOMENIMIS  —  VIENKARTINIS, GALUTINIS TESTAS
=================================================================

Mantas 2026-09-28: "paruosk patikrinima su 2 metu duomenimis, taciau tik si
karta". Todel sis failas paleidziamas VIENA karta, ir jo rezultatas yra
galutinis. Jokiu ribu keitimo po rezultato, jokio antro paleidimo su "siek
tiek kitu" slenksciu - tai butu butent ta paieska, nuo kurios saugomes.

KAS TIKRINAMA (uzfiksuota PRIES paleidima)
-----------------------------------------
Iejimas: pirkti kritimo dienos UZDARYME. Kritimas = 5 d. uzdarymo virsune
(iki vakar) minus siandienos uzdarymas, ATR vienetais, be dividendu.

  PAGRINDINE HIPOTEZE (sprendzia tik ji):
    A2: kritimas >= 1.6 ATR (Guo, Dong & Patterson 2024: -1.96 sigma),
        isejimas "sim05" - simetriski 0.5 ATR barjerai per 2 sesijas.
    Tai tas vienintelis derinys, kuris 60 d. patikroje buvo teigiamas IR
    grynai, IR pries kontrole visose keturiose pusese (p = 0.28).

  ANTRINIAI (spausdinami, bet NIEKO nesprendzia):
    A1: kritimas >= 1.0 ATR ir uzdare apatineje 40% diapazono dalyje
        (dabartinio 1 scenarijaus kontekstas).
    Isejimai: kitas rytas, kitos dienos uzdarymas, sim05.

KONTROLE: pirkti BET KURIA likvidzia akcija tos dienos uzdaryme. Vertinamas
skirtumas nuo jos - kitaip dvieju metu rinkos kilimas atrodytu kaip pranasumas.

SPRENDIMO TAISYKLE (ta pati, kaip visose ankstesnese patikrose):
  "TINKA" tik jei VISOS trys:
    1. skirtumas nuo kontroles > 10 EUR visose keturiose pusese
       (EU matyta, EU NEMATYTA, JAV matyta, JAV NEMATYTA);
    2. grynas rezultatas po sanaudu > 0 visose keturiose;
    3. sujungtose nematytose pusese bootstrap p < 0.05 (pagal diena).

ZINOMI TRUKUMAI (uzrasyti is anksto, kad veliau nebutu "atradimas"):
  - Universas yra SIANDIENOS sarasas: akcijos, kurios isgyveno. Tai kelia
    ir kandidato, ir kontroles rezultata; skirtumas nuo kontroles tam
    atsparesnis, bet ne visiskai.
  - Slydimas ir spredas nemodeliuojami (Nagel 2012: ~40% bruto).
  - Dienos barai: jei tą pacia diena paliestas IR tikslas, IR stop'as,
    uzskaitomas stop'as (konservatyvu, ta pati D.baigtis()).
  - Iejimas tiksliai uzdarymo kaina. Realiai - uzdarymo aukcione arba
    kelios minutes pries.
  - TRIUKSMO KONTROLE (paleista 2026-09-28, PRIES tikrus duomenis): ant
    grynai atsitiktinio klaidziojimo A2 prie kontroles turi nedideli, bet
    nuosekliai teigiama poslinki +0.5...+7.9 EUR (sim05 p = 0.07). Tai
    mechanika: po kritimo absoliutus ATR procentais tampa platesnis, o
    paprastu grazu isgaubtumas duoda sistemingą pliusą. Verdikto nepraeina
    (riba 10 EUR). Vadinasi, TIKRUOSE duomenyse skirtumas maždaug iki
    8 EUR yra paaiskinamas vien mechanika ir pranasumu nelaikomas.

Paleidimas:
    python a_iejimas.py --savitikra
    python a_iejimas.py --triuksmas
    python a_iejimas.py
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import detektorius as D
import ribos as R

try:
    import yfinance as yf
except ImportError:
    yf = None

LAIKOTARPIS = "2y"
HORIZONTAS = 2            # sim05: tiek sesiju PO iejimo (kaip 60 d. patikroje)
PAGRINDINE = ("A2", "sim05")
ISEJIMAI = ["kitas rytas", "kitos d. uzdarymas", "sim05"]
# Fondai ir zaliavu ETC nera bendroves - kritimo ir atsigavimo ju kainoje
# nera ko ieskoti, o Manto IBKR analize buvo apie akcijas.
NE_AKCIJOS = {"4GLD.DE", "XAD6.DE", "8PSB.DE", "IGD.MI"}


# ============================================================ skaiciavimas

def eilutes_akcijai(d, rinka, tickeris):
    """Visos akcijos dienos kaip kontroles eilutes + A1/A2 zymos.

    d - dienos OHLC su Dividends. Sprendimas dienai t naudoja tik:
      - rod eilute t (paslinkta: informacija iki t-1),
      - tos dienos t Open/High/Low/Close ir dividendus.
    Isejimai naudoja dienas t+1, t+2 - bet tai JAU rezultatas, ne sprendimas.
    """
    if d is None or len(d) < 80:
        return []
    dv = (d["Dividends"] if "Dividends" in d else
          pd.Series(0.0, index=d.index)).fillna(0.0)
    rod = D.dienos_rodikliai(d, dv)
    if rod is None:
        return []
    o, h, l, c = (d[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    div = dv.values.astype(float)
    div5 = dv.rolling(5, min_periods=1).sum().values     # dienos t-4..t
    riba_apyv = D.min_apyvarta(rinka)
    sanaud = D.sanaudos(rinka)
    out = []
    for k in range(len(d) - 1):
        r = rod.iloc[k]
        atr, virs, apyv = r["atr_abs"], r["virsune_n"], r["apyvarta"]
        if not (np.isfinite(atr) and atr > 0 and np.isfinite(virs)
                and np.isfinite(apyv)):
            continue
        if apyv < riba_apyv:
            continue
        krit = (virs - c[k] - div5[k]) / atr
        uzd = (c[k] - l[k]) / (h[k] - l[k]) if h[k] > l[k] else 0.5
        e = dict(rinka=rinka, tickeris=tickeris, sesija=str(d.index[k].date()),
                 A1=bool(krit >= 1.0 and uzd <= 0.40), A2=bool(krit >= 1.6))
        ieina = c[k]
        # kitas rytas / kitos dienos uzdarymas: dividendas, jei t+1 yra
        # ex-diena, priklauso pirkusiam t uzdaryme
        e["kitas rytas"] = ((o[k + 1] + div[k + 1]) / ieina - 1) * 100
        e["kitos d. uzdarymas"] = ((c[k + 1] + div[k + 1]) / ieina - 1) * 100
        # sim05: tos pacios D.baigtis(), kainos pakoreguotos dividendais,
        # kad ex-dienos tarpas nesuveiktu kaip netikras stop'as
        gal = min(len(d), k + 1 + HORIZONTAS)
        tol = d.iloc[k + 1:gal][["Open", "High", "Low", "Close"]].copy()
        kum = np.cumsum(div[k + 1:gal])
        for kol in ("Open", "High", "Low", "Close"):
            tol[kol] = tol[kol].values + kum
        b = D.baigtis(tol, dict(tipas=1, ieina=ieina, tikslas=ieina + 0.5 * atr,
                                stop=ieina - 0.5 * atr, atr_abs=atr))
        e["sim05"] = float(b["pelnas_pct"])
        for x in ISEJIMAI:
            e[x] = e[x] / 100.0 * D.POZICIJA - sanaud
        out.append(e)
    return out


def surinkti(rinka, duomenys=None):
    zyme = D.RINKOS[rinka]["zyme"]
    print(f"\n{'='*84}\n{zyme}\n{'='*84}")
    if duomenys is None:
        tick = [t for t in D.universas(rinka) if t not in NE_AKCIJOS]
        raw = yf.download(tick, period=LAIKOTARPIS, interval="1d",
                          auto_adjust=False, progress=False, group_by="ticker",
                          threads=True, actions=True)
        duomenys = {t: D._vienas(raw, t) for t in tick}
    eil = []
    for t, d in duomenys.items():
        eil.extend(eilutes_akcijai(d, rinka, t))
    df = pd.DataFrame(eil)
    if df.empty:
        print("  eiluciu nera")
        return df
    print(f"  akciju: {df['tickeris'].nunique()}   dienu: {df['sesija'].nunique()}"
          f"   ({df['sesija'].min()} .. {df['sesija'].max()})")
    print(f"  kontroles eiluciu: {len(df)}   A1: {int(df['A1'].sum())}"
          f"   A2: {int(df['A2'].sum())}")
    return df


# ============================================================ verdiktas

def vertinti(dalys):
    matavimai = list(dalys)
    riba = max(D.SANAUDOS.values())
    rez = {}
    for kand in ("A2", "A1"):
        pagr = kand == PAGRINDINE[0]
        print(f"\n{'-'*112}\n{kand}"
              + ("   <<< PAGRINDINE HIPOTEZE (sprendzia tik isejimas "
                 f"'{PAGRINDINE[1]}')" if pagr else "   (antrinis - nieko nesprendzia)"))
        print(f"{'-'*112}")
        print(f"  {'isejimas':<20}{'':<9}" + "".join(f"{m:>15}" for m in matavimai)
              + f"{'p (nemat.)':>12}  verdiktas")
        for kur in ISEJIMAI:
            gryn, skirt, nn, p_nem = [], [], [], []
            for m in matavimai:
                df = dalys[m]
                a = df[df[kand]].dropna(subset=[kur])
                b = df.dropna(subset=[kur])
                nn.append(len(a))
                if len(a) < 30:
                    gryn.append(None)
                    skirt.append(None)
                    continue
                ad = a.groupby("sesija")[kur].mean()
                bd = b.groupby("sesija")[kur].mean()
                gryn.append(ad.mean())
                skirt.append(ad.mean() - bd.mean())
                if "NEMATYTA" in m:
                    p_nem.append((ad.values, bd.values))
            p = (R.bootstrap_p(np.concatenate([x[0] for x in p_nem]),
                               np.concatenate([x[1] for x in p_nem]))
                 if p_nem else float("nan"))
            if any(x is None for x in skirt):
                v = "per mazai ivykiu"
            elif not all(y > riba for y in skirt):
                v = "nelaikosi"
            elif not all(y > 0 for y in gryn):
                v = "geriau uz kontrole, bet grynas <= 0"
            elif not np.isfinite(p) or p >= 0.05:
                v = f"kryptis sutampa, bet p={p:.2f}"
            else:
                v = "TINKA"
            rez[(kand, kur)] = v

            def fmt(vals):
                return "".join(f"{x:>15.1f}" if x is not None else f"{f'N={n}':>15}"
                               for x, n in zip(vals, nn))
            print(f"  {kur:<20}{'grynas':<9}{fmt(gryn)}")
            print(f"  {'':<20}{'- kontr.':<9}{fmt(skirt)}{p:>12.3f}  {v}")
            print(f"  {'':<20}{'N':<9}" + "".join(f"{n:>15}" for n in nn))
    print(f"\n{'='*112}")
    v = rez.get(PAGRINDINE, "nera")
    print(f"  GALUTINIS ATSAKYMAS - {PAGRINDINE[0]} x {PAGRINDINE[1]}:  {v}")
    print("  Kiti derinai spausdinti tik informacijai. Pagal is anksto uzrasyta")
    print("  taisykle jie nieko nesprendzia, kad ir kaip atrodytu.")
    print("=" * 112)
    return v


def paleisti(triuksmas=False):
    if triuksmas:
        print("!" * 84)
        print("TRIUKSMO KONTROLE: dienos atsitiktinis klaidziojimas. Cia neturi")
        print("buti 'TINKA' - kitaip masina per laisva.")
        print("!" * 84)
        poros = [("eu", triuksmo_dienos(seed=31)), ("us", triuksmo_dienos(seed=32))]
    else:
        poros = [("eu", None), ("us", None)]
    dalys = {}
    for rinka, duom in poros:
        df = surinkti(rinka, duom)
        if df.empty:
            continue
        zyme = D.RINKOS[rinka]["zyme"]
        ses = sorted(df["sesija"].unique())
        riba = ses[len(ses) // 2]
        print(f"  pusiu riba: {riba}")
        dalys[f"{zyme} matyta"] = df[df["sesija"] <= riba]
        dalys[f"{zyme} NEMATYTA"] = df[df["sesija"] > riba]
    if len(dalys) < 4:
        sys.exit("truksta rinkos duomenu - keturiu matavimu nera")
    return vertinti(dalys)


# ============================================================ triuksmas

def triuksmo_dienos(n=150, dienu=520, seed=31):
    """Dienos atsitiktinis klaidziojimas: uzdarymas -> naktis -> diena.
    Naktis ir diena nepriklausomi, tad jokio apsisukimo nera."""
    r = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-09-25", periods=dienu)
    out = {}
    for i in range(n):
        naktis = r.normal(0, 0.008, dienu)
        diena = r.normal(0, 0.018, dienu)
        c = np.empty(dienu)
        o = np.empty(dienu)
        prev = 50.0
        for k in range(dienu):
            o[k] = prev * np.exp(naktis[k])
            c[k] = o[k] * np.exp(diena[k])
            prev = c[k]
        h = np.maximum(o, c) * np.exp(abs(r.normal(0, 0.007, dienu)))
        l = np.minimum(o, c) / np.exp(abs(r.normal(0, 0.007, dienu)))
        out[f"N{i:03d}"] = pd.DataFrame(
            dict(Open=o, High=h, Low=l, Close=c,
                 Volume=r.integers(3e5, 3e6, dienu).astype(float),
                 Dividends=0.0), index=idx)
    return out


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

    # bazine: ramios 100 dienu, paskui kritimas ir zinomi isejimai
    idx = pd.bdate_range("2026-01-05", periods=106)
    c = np.full(106, 100.0)
    c[100], c[101], c[102] = 95.0, 97.0, 99.0      # t=100 kritimo diena
    o = c.copy()
    o[101] = 96.0                                  # kitas rytas
    h, l = c * 1.01, c * 0.99
    h[100], l[100] = 100.5, 94.8                   # uzdare prie dugno
    d = pd.DataFrame(dict(Open=o, High=h, Low=l, Close=c, Volume=5e6,
                          Dividends=0.0), index=idx)
    e = {x["sesija"]: x for x in eilutes_akcijai(d, "eu", "X")}
    t = str(idx[100].date())
    tikrinti("kritimo diena pazymeta A1 ir A2", (e[t]["A1"], e[t]["A2"]),
             (True, True))
    tikrinti("rami diena nepazymeta", e[str(idx[90].date())]["A2"], False)
    tikrinti("kitas rytas = KITOS dienos ATIDARYMAS (96), ne uzdarymas",
             round(e[t]["kitas rytas"], 2),
             round((96 / 95 - 1) * D.POZICIJA - 10, 2))
    tikrinti("kitos d. uzdarymas = 97",
             round(e[t]["kitos d. uzdarymas"], 2),
             round((97 / 95 - 1) * D.POZICIJA - 10, 2))

    # ateitis: pakeitus t+1..t+5, dienos t zyma neturi keistis
    d2 = d.copy()
    d2.iloc[101:, d2.columns.get_loc("Close")] *= 0.5
    d2.iloc[101:, d2.columns.get_loc("Open")] *= 0.5
    e2 = {x["sesija"]: x for x in eilutes_akcijai(d2, "eu", "X")}
    tikrinti("dienos t zyma nepriklauso nuo t+1 ir veliau",
             (e2[t]["A1"], e2[t]["A2"]), (e[t]["A1"], e[t]["A2"]))
    tikrinti("        bet isejimas nuo ju priklauso (testas ne tuscias)",
             e2[t]["kitas rytas"] != e[t]["kitas rytas"], True)
    # grieztesne ateities patikra: RAMI diena t0 (A2=False). Jei kritimas
    # zvelgtu i t0+1, pakelta ryto kaina ja paverstu "kritimo diena".
    t0 = 90
    d7 = d.copy()
    d7.iloc[t0 + 1, d7.columns.get_loc("Close")] = 200.0
    d7.iloc[t0 + 1, d7.columns.get_loc("High")] = 201.0
    e7 = {x["sesija"]: x for x in eilutes_akcijai(d7, "eu", "X")}
    tikrinti("rami diena lieka rami, nors RYTOJ kaina iskyla (be ateities)",
             (e7[str(idx[t0].date())]["A1"], e7[str(idx[t0].date())]["A2"]),
             (False, False))

    # dividendas: ex-diena t+1 nesuveikia kaip nuostolis
    d3 = d.copy()
    d3.iloc[101, d3.columns.get_loc("Open")] = 95.0 - 2.0
    d3.iloc[101, d3.columns.get_loc("Close")] = 95.0 - 2.0
    d3.iloc[101, d3.columns.get_loc("Dividends")] = 2.0
    e3 = {x["sesija"]: x for x in eilutes_akcijai(d3, "eu", "X")}
    tikrinti("ex-dienos kritimas grazinamas dividendu (grynas ~ -10 EUR)",
             round(e3[t]["kitas rytas"], 1), -10.0)
    # ir sim05: ex-dienos tarpas (2.0 > 0.5 ATR) be pataisos suveiktu kaip stop
    atr3 = float(D.dienos_rodikliai(d3, d3["Dividends"])["atr_abs"].iloc[100])
    tikrinti("        dividendo tarpas NEsuveikia kaip sim05 stop'as",
             e3[t]["sim05"] > -0.5 * atr3 / 95 * D.POZICIJA - 10 + 1, True)
    tikrinti("        (dividendas tikrai didesnis uz 0.5 ATR - testas ne tuscias)",
             2.0 > 0.5 * atr3, True)

    # sim05 ta pati D.baigtis: diena su abiem barjerais = stop
    d4 = d.copy()
    # atidarymas = iejimo kaina, kad NEBUTU tarpo pro barjera - kitaip testas
    # tikrintu atidarymo taisykle, o ne "abu barjerai tą pacia diena"
    d4.iloc[101, d4.columns.get_loc("Open")] = 95.0
    atr = float(D.dienos_rodikliai(d4, d4["Dividends"])["atr_abs"].iloc[100])
    d4.iloc[101, d4.columns.get_loc("High")] = 95 + 5 * atr
    d4.iloc[101, d4.columns.get_loc("Low")] = 95 - 5 * atr
    e4 = {x["sesija"]: x for x in eilutes_akcijai(d4, "eu", "X")}
    tikrinti("sim05: tiksla ir stop'a palietusi diena = stop (konservatyvu)",
             round(e4[t]["sim05"], 2),
             round(-0.5 * atr / 95 * D.POZICIJA - 10, 2))

    # ir priesingai: tarpas virs tikslo vykdomas ties atidarymu (ne ties tikslu)
    d6 = d4.copy()
    d6.iloc[101, d6.columns.get_loc("Open")] = 95 + 2 * atr
    e6 = {x["sesija"]: x for x in eilutes_akcijai(d6, "eu", "X")}
    tikrinti("sim05: tarpas virs tikslo vykdomas ATIDARYMO kaina",
             round(e6[t]["sim05"], 2),
             round(2 * atr / 95 * D.POZICIJA - 10, 2))

    # likvidumas
    d5 = d.copy()
    d5["Volume"] = 10.0
    tikrinti("nelikvidi akcija neduoda eiluciu", len(eilutes_akcijai(d5, "eu", "X")), 0)

    # fondai isimti
    tikrinti("fondai ne universe", "4GLD.DE" in NE_AKCIJOS, True)

    # triuksmas be apsisukimo
    tr = triuksmo_dienos(n=60, dienu=400, seed=5)
    po, vis = [], []
    for x in tr.values():
        cc, oo = x["Close"].values, x["Open"].values
        for k in range(5, len(cc) - 1):
            f = oo[k + 1] / cc[k] - 1
            vis.append(f)
            if cc[k] / cc[k - 5:k].max() - 1 < -0.05:
                po.append(f)
    tikrinti("triuksme po kritimo naktis nesiskiria nuo vidurkio (<0.05 p.p.)",
             abs(np.mean(po) - np.mean(vis)) * 100 < 0.05, True)

    # sprendimo taisykle
    import io, contextlib
    def dal(sk, gr):
        eil = []
        for i in range(60):
            s = f"2025-{1 + i % 12:02d}-{1 + i % 27:02d}-{i}"
            eil.append(dict(sesija=s, A1=False, A2=True,
                            **{k: gr + np.sin(i) for k in ISEJIMAI}))
            eil.append(dict(sesija=s, A1=False, A2=False,
                            **{k: 2 * gr - (gr - sk) + np.cos(i) - sk * 2 for k in ISEJIMAI}))
        return pd.DataFrame(eil)
    for pav, sk, gr, laukta in (("aiskiai geriau ir pelninga", 60, 50, "TINKA"),
                                ("geriau, bet grynas neigiamas", 60, -20, None)):
        dd = {m: dal(sk, gr) for m in ("EU matyta", "EU NEMATYTA",
                                       "JAV matyta", "JAV NEMATYTA")}
        with contextlib.redirect_stdout(io.StringIO()):
            v = vertinti(dd)
        tikrinti(f"taisykle: {pav}", v == "TINKA", laukta == "TINKA")

    print("-" * 60)
    print("SAVITIKRA: VISKAS GERAI" if ok else "SAVITIKRA: YRA KLAIDU")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--savitikra", action="store_true")
    ap.add_argument("--triuksmas", action="store_true")
    a = ap.parse_args()
    if a.savitikra:
        sys.exit(savitikra())
    paleisti(a.triuksmas)
