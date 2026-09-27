#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETEKTORIUS v3  —  2026-09-27
=============================

Ne reitinguotojas. Neatsako "kuri akcija rytoj bus geresne" - to neradome ne
viename is ~60 testuotu signalu. Atsako "ar SI situacija JAU prasidejo", ir
rodo, kur ji yra savo kelyje.

TRYS SCENARIJAI
---------------
1. KRITIMAS + ATSISTATYMO PRADZIA   tikslas R (buves lygis), stop zemiau L
2. RALIO PRADZIA                    tikslo nera - slenkantis isejimas
3. ATIDARYMO TARPO UZPILDYMAS       tikslas: vakarykstis uzdarymas

1 ir 3 turi STRUKTURINI tiksla - isvedama is pacios situacijos, ne fiksuotas
procentas. Visi ankstesni testai naudojo fiksuota +1% ar +2%, ir butent del to
ATR ruozu rezultatai buvo tokie nevienodi.

KAS PAKEISTA v3
---------------
A. BAIGTIS MATUOJAMA TA PACIA SESIJA, ne per 3 sesijas kaip v2.
   Tavo scenarijai pagal apibrezima prasideda ir baigiasi ta pacia diena. Jei
   atsistatymas uztrunka tris dienas, tai NE laimejimas, o krintantis peilis -
   tiksliai tai, ka nori apeiti. Matuojant per 3 sesijas peiliai butu
   skaiciuojami kaip sekmes.
   Papildomai rodoma, kas buvo kito ryto atidarymu (nakties langas patvirtintas
   abiejose rinkose: EU +0.049%, JAV +0.077%, diena abiejose neigiama), ir
   NEISSPRESTU dalis - tiesioginis peiliu matas.

B. VIENU METU AKTYVIU signalu skaicius, ne tik suveikimai per diena.
   85 suveikimai per diena, kuriu kiekvienas gyvuoja 2 val., reiskia ~8
   korteles ekrane - visai kas kita nei 85.

C. Penkios pataisytos klaidos, rastos perziurint koda:
   1. vakar_uzdarymas live rezime imdavo SIANDIENOS nebaigta bara. 2026-09-07
      butent del to modulis skaiciavo dienos pokyti nuo uzvakar.
   2. Sektoriaus filtras buvo aprasytas, bet visada gaudavo None - t.y.
      niekada nesuveikdavo. Dabar sektoriaus mediana skaiciuojama realiai.
   3. Rinkos rezimas kalibracijoje buvo uzkoduotas "neutral", o live -
      skaiciuojamas. Matavome ne ta, ka rodom.
   4. Slenkantis stop'as keldavosi po KIEKVIENO baro dugnu ir issimusdavo is
      karto. Dabar po zemiausiu is 3 paskutiniu baru.
   5. Dividendu uzklausos - 214 atskiru kvietimu kas paleidima. Dabar
      kesuojama i faila, atnaujinama kas 7 dienas.

VIENAS KODAS LIVE IR KALIBRACIJAI
----------------------------------
Brangiausia sio projekto klaida buvo ta, kad modulis skaiciavo viena, o testas
kita (trys skirtingi Z-balo apibrezimai, 2026-09-07). Todel scenarijai cia yra
grynosios funkcijos, o rezimas ir sektorius skaiciuojami VIENODAI abiejuose.

Paleidimas:
    python detektorius.py --live
    python detektorius.py --kalibracija --dienos 60
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("KLAIDA: reikia yfinance (pip install yfinance)")

# ---------------------------------------------------------------- parametrai

POZICIJA = 18000.0
SANAUDOS_EUR = 10.0
MIN_APYVARTA = 5e6

BARAS_MIN = 5
BUSENOS_FAILAS = "busena.json"
DIVIDENDU_TALPYKLA = "dividendai.json"

# --- scenarijus 1 ---
S1_MIN_KRITIMAS_PCT = 0.40
S1_MIN_KRITIMAS_ATR = 0.80
S1_MAX_KRITIMAS_ATR = 2.50
S1_ATSOKIMAS = 0.15
S1_MAX_ATSOKIMAS = 0.40
S1_MIN_BARU_NUO_L = 2
S1_R_LANGAS_BARU = 78
S1_STOP_MARZA = 0.20
# R:R = (1-k)/(k+0.20), kur k - kelio dalis nuo L iki R:
#   k=0.15 -> 2.43    k=0.40 -> 1.00    k=0.60 -> 0.50
# Todel 0.40 riba nera nuomone: virs jos signala vis tiek blokuotu R:R filtras.

# --- scenarijus 2 ---
S2_BAZE_BARU = 12
S2_BAZE_MAX_ATR = 0.50
S2_APYVARTOS_SUOLIS = 2.0
S2_MAX_NUO_ATIDARYMO_ATR = 2.5
S2_NERODYTI_PASKUTINES_MIN = 15
S2_TRAIL_BARU = 3

# --- scenarijus 3 ---
S3_MIN_TARPAS_PCT = 0.30
S3_MIN_TARPAS_ATR = 0.50
S3_MAX_TARPAS_ATR = 2.00
S3_ATSOKIMAS = 0.15
S3_MAX_ATSOKIMAS = 0.40

# --- kietieji filtrai (visi is ISMATUOTU dalyku) ---
F_MAX_ATR = 3.5                  # >3.5% grupe blogiausia abiejose rinkose
F_SMA200_TURI_KILTI = True       # didziausias vienos salygos inasas: +0.198 p.p.
F_MAX_SEKTORIAUS_KRITIMAS = -1.0 # 81% giliu kritimu lydejo sektoriaus kritimas
F_MAX_KRITIMO_DIENU = 3
F_DIVIDENDU_LANGAS_D = 3         # ex-div kritimas yra mechaninis, ne dipas

RINKOS = {
    "eu": dict(zyme="EU", indeksas="EXSA.DE", tz="Europe/Berlin",
               uzdarymas=17 * 60 + 30),
    "us": dict(zyme="JAV", indeksas="SPY", tz="America/New_York",
               uzdarymas=16 * 60),
}


# ================================================================ universas

def universas(rinka):
    saltinis, t = None, []
    try:
        import universas as U
        vardai = (("visi_tikeriai", "visi_tickeriai", "UNIVERSAS")
                  if rinka == "eu" else ("UNIVERSAS_US", "JAV_UNIVERSAS"))
        for nm in vardai:
            if not hasattr(U, nm):
                continue
            o = getattr(U, nm)
            o = o() if callable(o) else o
            if isinstance(o, dict):
                t = [x for v in o.values()
                     for x in (v if isinstance(v, (list, tuple, set)) else [v])]
            elif isinstance(o, (list, tuple, set)):
                t = list(o)
            if t:
                saltinis = f"universas.{nm}"
                break
        if not t:
            print("  universas.py turimi vardai:",
                  [n for n in dir(U) if not n.startswith("_")][:40])
    except Exception as e:
        print(f"  universas.py neimportuojamas: {e}")
    t = sorted({str(x).strip() for x in t if x and isinstance(x, str)})
    if len(t) < 40:
        sys.exit(f"NUTRAUKTA: is universas.py gauta tik {len(t)} tikeriu (reikia >=40).")
    print(f"  {RINKOS[rinka]['zyme']}: {saltinis}, {len(t)} tikeriu")
    return t


def sektoriu_zemelapis():
    """tickeris -> sektorius. v2 sio zemelapio neturejo, tad sektoriaus
    filtras niekada nesuveikdavo."""
    try:
        import universas as U
        for nm in ("sektoriai", "SEKTORIAI", "UNIVERSAS"):
            if not hasattr(U, nm):
                continue
            o = getattr(U, nm)
            o = o() if callable(o) else o
            if isinstance(o, dict) and o:
                pirmas = next(iter(o.values()))
                if isinstance(pirmas, (list, tuple, set)):
                    return {t: s for s, ts in o.items() for t in ts}
                if isinstance(pirmas, str):
                    return dict(o)
    except Exception:
        pass
    return {}


# ================================================================ duomenys

def dienos_kontekstas(tickers, rinka, metai=2, gyvai=False):
    """gyvai=True: paskutine eilute yra SIANDIENOS nebaigtas baras - ismetam.

    2026-09-07 modulis del panasaus neapsizuirejimo skaiciavo dienos pokyti
    nuo uzvakar. Scenarijui 3 vakarykstis uzdarymas yra pats tikslas, tad
    klaida cia butu lemiama.
    """
    df = yf.download(tickers, period=f"{metai}y", interval="1d",
                     auto_adjust=False, progress=False, group_by="ticker",
                     threads=True)
    siandien = datetime.now().date()
    out = {}
    for t in tickers:
        try:
            d = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            d = d.dropna(subset=["Open", "High", "Low", "Close"])
            if gyvai and len(d) and d.index[-1].date() >= siandien:
                d = d.iloc[:-1]
            if len(d) < 220:
                continue
            c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
            prev = c.shift(1)
            tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
            atr = float((tr.rolling(20).mean() / c).iloc[-1] * 100)
            sma200 = c.rolling(200).mean()
            kyla = bool(sma200.iloc[-1] > sma200.iloc[-21]) if len(sma200.dropna()) > 21 else False
            apyv = float((c * v).rolling(20).median().iloc[-1])
            pok = (c / prev - 1.0).iloc[-6:]
            serija = 0
            for x in reversed(pok.tolist()):
                if x < 0:
                    serija += 1
                else:
                    break
            out[t] = dict(atr=atr, sma200_kyla=kyla, apyvarta=apyv,
                          vakar_uzdarymas=float(c.iloc[-1]),
                          kritimo_dienu=serija, uzdarymai=c)
        except Exception:
            continue
    return out


def vakarykstis_pagal_sesija(kont, ses):
    """Kalibracijai: vakarykstis uzdarymas TOS sesijos atzvilgiu, ne paskutinis.

    Be sito scenarijus 3 per visa 60 dienu istorija lygintu su siandienos
    uzdarymu - tai butu zvilgsnis i ateiti.
    """
    try:
        c = kont["uzdarymai"]
        pries = c[[x.date() < ses for x in c.index]]
        return float(pries.iloc[-1]) if len(pries) else None
    except Exception:
        return None


def dividendu_kalendorius(tickers):
    """Su talpykla: v2 kviesdavo 214 atskiru uzklausu kas paleidima."""
    if os.path.exists(DIVIDENDU_TALPYKLA):
        try:
            with open(DIVIDENDU_TALPYKLA, encoding="utf-8") as f:
                tal = json.load(f)
            kada = tal.get("_atnaujinta", "")
            if kada and (datetime.now() - datetime.fromisoformat(kada)).days < 7:
                return tal
        except Exception:
            pass
    rez = {"_atnaujinta": datetime.now().isoformat()}
    for t in tickers:
        try:
            div = yf.Ticker(t).dividends
            if div is None or len(div) < 2:
                continue
            div = div[-12:]
            datos = [d.tz_localize(None) if d.tz else d for d in div.index]
            tarpai = [(datos[i + 1] - datos[i]).days for i in range(len(datos) - 1)]
            tarpai = [x for x in tarpai if 20 <= x <= 400]
            if not tarpai:
                continue
            vid = float(np.median(tarpai))
            kita = datos[-1] + timedelta(days=vid)
            while kita < datetime.now():
                kita += timedelta(days=vid)
            rez[t] = dict(kita_ex=kita.date().isoformat(), suma=float(div.iloc[-1]),
                          dienu_iki=(kita.date() - datetime.now().date()).days)
        except Exception:
            continue
    try:
        with open(DIVIDENDU_TALPYKLA, "w", encoding="utf-8") as f:
            json.dump(rez, f, ensure_ascii=False, indent=1)
    except Exception:
        pass
    return rez


def intraday_barai(tickers, dienos=60):
    df = yf.download(tickers, period=f"{min(dienos, 60)}d", interval="5m",
                     auto_adjust=False, progress=False, group_by="ticker",
                     threads=True, prepost=False)
    out = {}
    for t in tickers:
        try:
            d = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            d = d.dropna(subset=["Open", "High", "Low", "Close"])
            if len(d) > 100:
                out[t] = d
        except Exception:
            continue
    return out


# ================================================================ rodikliai

def sesijos_rodikliai(barai, rinka):
    d = barai.copy()
    try:
        idx = d.index.tz_convert(RINKOS[rinka]["tz"])
    except Exception:
        idx = d.index
    d["sesija"] = [x.date() for x in idx]
    d["minute"] = [x.hour * 60 + x.minute for x in idx]
    d["_pv"] = ((d["High"] + d["Low"] + d["Close"]) / 3.0) * d["Volume"]
    g = d.groupby("sesija", sort=False)
    d["vwap"] = g["_pv"].cumsum() / g["Volume"].cumsum().replace(0, np.nan)
    d["sesijos_atidarymas"] = g["Open"].transform("first")
    d["nuo_atidarymo"] = (d["Close"] / d["sesijos_atidarymas"] - 1.0) * 100
    d["apyv_tipine"] = d.groupby("minute")["Volume"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).median())
    d["apyv_santykis"] = d["Volume"] / d["apyv_tipine"].replace(0, np.nan)
    d.drop(columns=["_pv"], inplace=True)
    return d


def sektoriaus_pokyciai(visi_barai, zemelapis):
    """(sektorius, laikas) -> mediana nuo sesijos atidarymo, procentais."""
    eil = []
    for t, d in visi_barai.items():
        s = zemelapis.get(t)
        if not s or "nuo_atidarymo" not in d:
            continue
        eil.append(pd.DataFrame({"laikas": d.index, "sekt": s,
                                 "pok": d["nuo_atidarymo"].values}))
    if not eil:
        return {}
    v = pd.concat(eil, ignore_index=True)
    return {k: float(x) for k, x in
            v.groupby(["sekt", "laikas"])["pok"].median().items()}


# ================================================ scenarijai (grynos funkcijos)

def _auksteja_dugnai(langas, n=2):
    lo = langas["Low"].tolist()
    if len(lo) < n + 1:
        return False
    return all(lo[-i] > lo[-i - 1] for i in range(1, n + 1))


def scenarijus_1(langas, kont):
    if len(langas) < S1_MIN_BARU_NUO_L + 3:
        return None
    atr = kont["atr"]
    kaina = float(langas["Close"].iloc[-1])

    lang = langas.iloc[-S1_R_LANGAS_BARU:] if len(langas) > S1_R_LANGAS_BARU else langas
    i_min = int(np.argmin(lang["Low"].values))
    if i_min < 2 or i_min >= len(lang) - S1_MIN_BARU_NUO_L:
        return None
    R = float(lang["Close"].iloc[:i_min].max())
    L = float(lang["Low"].iloc[i_min])
    if not (R > L > 0):
        return None
    if L > float(langas["Low"].min()) * 1.0005:
        return None                      # turi buti SESIJOS dugnas

    kritimas_pct = (R - L) / R * 100.0
    if kritimas_pct < max(S1_MIN_KRITIMAS_PCT, S1_MIN_KRITIMAS_ATR * atr):
        return None
    if kritimas_pct > S1_MAX_KRITIMAS_ATR * atr:
        return None
    kelias = (kaina - L) / (R - L)
    if not (S1_ATSOKIMAS <= kelias <= S1_MAX_ATSOKIMAS):
        return None

    po_dugno = lang.iloc[i_min + 1:]
    if len(po_dugno) < S1_MIN_BARU_NUO_L:
        return None
    if float(po_dugno["Low"].iloc[-3:].min()) <= L:
        return None
    if not _auksteja_dugnai(po_dugno, 2):
        return None
    if kaina <= float(langas["Close"].iloc[-3]):
        return None
    kb = lang.iloc[:i_min + 1]
    if len(kb) >= 2:
        did = float((kb["High"] - kb["Low"]).max())
        if R > 0 and did / R * 100.0 > 1.0 * atr:
            return None                  # vienas baras > dienos ATR -> naujiena

    return dict(scenarijus="Kritimas + atsistatymas", tipas=1,
                ieina=kaina, tikslas=R, stop=L - S1_STOP_MARZA * (R - L),
                R=R, L=L, progresas=kelias, progresas_tikslus=True,
                kritimas_pct=kritimas_pct)


def scenarijus_2(langas, kont, iki_uzdarymo):
    if len(langas) < S2_BAZE_BARU + 3 or iki_uzdarymo < S2_NERODYTI_PASKUTINES_MIN:
        return None
    atr = kont["atr"]
    dab = langas.iloc[-1]
    kaina = float(dab["Close"])

    baze = langas.iloc[-(S2_BAZE_BARU + 1):-1]
    b_max, b_min = float(baze["High"].max()), float(baze["Low"].min())
    if b_max <= 0 or (b_max - b_min) / b_max * 100.0 > S2_BAZE_MAX_ATR * atr:
        return None
    if kaina <= b_max:
        return None
    pries = langas.iloc[:-1]
    if len(pries) and kaina <= float(pries["High"].max()):
        return None
    sant = dab.get("apyv_santykis", np.nan)
    if not (np.isfinite(sant) and sant >= S2_APYVARTOS_SUOLIS):
        return None
    vwap = dab.get("vwap", np.nan)
    if not (np.isfinite(vwap) and kaina > vwap):
        return None
    if not _auksteja_dugnai(langas.iloc[-4:], 2):
        return None
    atid = float(dab["sesijos_atidarymas"])
    if atid > 0 and (kaina - atid) / atid * 100.0 > S2_MAX_NUO_ATIDARYMO_ATR * atr:
        return None
    vu = kont.get("vakar_uzdarymas") or 0
    if vu > 0 and (kaina / vu - 1.0) * 100.0 < 0:
        return None

    nuejo = (kaina - b_max) / b_max * 100.0
    tiketina = 1.0 * atr
    prog_j = min(1.0, nuejo / tiketina) if tiketina > 0 else 0.0
    liko = max(1.0, iki_uzdarymo)
    prog_l = 1.0 - liko / (liko + 30.0)
    return dict(scenarijus="Ralio pradzia", tipas=2,
                ieina=kaina, tikslas=None, stop=b_min, R=None, L=b_min,
                progresas=max(prog_j, prog_l), progresas_tikslus=False,
                virsune_vertinimas=b_max * (1 + tiketina / 100.0), baze=b_max)


def scenarijus_3(langas, kont, vakar_uzd):
    if len(langas) < 3 or not vakar_uzd or vakar_uzd <= 0:
        return None
    atr = kont["atr"]
    atid = float(langas["Open"].iloc[0])
    if atid >= vakar_uzd:
        return None
    tarpas_pct = (vakar_uzd - atid) / vakar_uzd * 100.0
    if tarpas_pct < max(S3_MIN_TARPAS_PCT, S3_MIN_TARPAS_ATR * atr):
        return None
    if tarpas_pct > S3_MAX_TARPAS_ATR * atr:
        return None
    L = float(langas["Low"].min())
    kaina = float(langas["Close"].iloc[-1])
    if vakar_uzd <= L:
        return None
    kelias = (kaina - L) / (vakar_uzd - L)
    if not (S3_ATSOKIMAS <= kelias <= S3_MAX_ATSOKIMAS):
        return None
    if float(langas["Low"].iloc[-3:].min()) <= L:
        return None
    if not _auksteja_dugnai(langas.iloc[-3:], 2):
        return None
    if kaina <= float(langas["Close"].iloc[-3]):
        return None
    return dict(scenarijus="Tarpo uzpildymas", tipas=3,
                ieina=kaina, tikslas=vakar_uzd,
                stop=L - S1_STOP_MARZA * (vakar_uzd - L),
                R=vakar_uzd, L=L, progresas=kelias, progresas_tikslus=True,
                tarpas_pct=tarpas_pct)


# ================================================================ filtrai

def kietieji_filtrai(kont, sig, rezimas, sekt_pok, div):
    kl = []
    if kont["atr"] > F_MAX_ATR:
        kl.append(f"ATR {kont['atr']:.1f}%")
    if F_SMA200_TURI_KILTI and not kont["sma200_kyla"]:
        kl.append("SMA200 nekyla")
    if kont["apyvarta"] < MIN_APYVARTA:
        kl.append("maza apyvarta")
    if kont["kritimo_dienu"] >= F_MAX_KRITIMO_DIENU:
        kl.append(f"{kont['kritimo_dienu']} kritimo dienos")
    if rezimas == "bear":
        kl.append("rinka krenta")
    if sekt_pok is not None and sekt_pok < F_MAX_SEKTORIAUS_KRITIMAS:
        kl.append(f"sektorius {sekt_pok:+.1f}%")
    if div and abs(div.get("dienu_iki", 999)) <= F_DIVIDENDU_LANGAS_D:
        kl.append(f"ex-div po {div['dienu_iki']} d.")
    if sig.get("tikslas"):
        rr = (sig["tikslas"] - sig["ieina"]) / max(1e-9, sig["ieina"] - sig["stop"])
        if rr < 1.0:
            kl.append(f"R:R {rr:.2f}")
    return kl


def rezimu_serija(indeksas_d):
    """data -> rezimas. Skaiciuojama VIENODAI live ir kalibracijai.
    v2 kalibracijoje buvo uzkoduota "neutral" - matavome ne ta, ka rodom."""
    out = {}
    try:
        c = indeksas_d["Close"].dropna()
        sma20 = c.rolling(20).mean()
        for i in range(21, len(c)):
            r5 = float(c.iloc[i] / c.iloc[i - 5] - 1.0) * 100
            virs = bool(c.iloc[i] > sma20.iloc[i])
            siandien = float(c.iloc[i] / c.iloc[i - 1] - 1.0) * 100
            if r5 < -1 and siandien < 0:
                r = "bear"
            elif r5 < -1 or not virs:
                r = "bear_soft"
            elif r5 > 1 and virs:
                r = "bull"
            else:
                r = "neutral"
            out[c.index[i].date()] = r
    except Exception:
        pass
    return out


# ================================================================ aptikimas

def aptikti(langas, kont, rezimas, sekt_pok, div, iki_uzdarymo, vakar_uzd):
    rez = []
    for f, arg in ((scenarijus_1, (langas, kont)),
                   (scenarijus_2, (langas, kont, iki_uzdarymo)),
                   (scenarijus_3, (langas, kont, vakar_uzd))):
        try:
            s = f(*arg)
        except Exception:
            s = None
        if not s:
            continue
        s["kliutys"] = kietieji_filtrai(kont, s, rezimas, sekt_pok, div)
        s["tinkamas"] = len(s["kliutys"]) == 0
        rez.append(s)
    return rez


# ================================================================ baigtis

def baigtis(sesijos_likutis, kitas_rytas, sig):
    """PAGRINDINIS matas - ta pati sesija.

    Jei scenarijus neissisprendzia iki uzdarymo, tai pagal apibrezima nebuvo
    tas, ko ieskojom. Todel "neissprestas" yra atskira baigtis, o ne dingsta
    i vidurki.
    """
    ieina, tikslas, stop = sig["ieina"], sig.get("tikslas"), sig["stop"]
    rytas = np.nan
    if len(kitas_rytas):
        rytas = (float(kitas_rytas["Open"].iloc[0]) / ieina - 1.0) * 100
    if len(sesijos_likutis) == 0:
        return dict(baigtis="nespejo", pelnas_pct=0.0, minuciu=0, blogiausia=0.0,
                    neissprestas=True, rytas_pct=rytas, isejo_bare=1)

    blog = 0.0
    if sig["tipas"] == 2:
        auksciausia, maks, zem = ieina, [], []
        for j, (_, b) in enumerate(sesijos_likutis.iterrows()):
            lo, hi = float(b["Low"]), float(b["High"])
            zem.append(lo)
            blog = min(blog, (lo / ieina - 1.0) * 100)
            if lo <= stop:
                return dict(baigtis="slenkantis stop",
                            pelnas_pct=(stop / ieina - 1) * 100,
                            minuciu=(j + 1) * BARAS_MIN, blogiausia=blog,
                            neissprestas=False, rytas_pct=rytas, isejo_bare=j + 1)
            if hi > auksciausia:
                auksciausia = hi
                if len(zem) >= S2_TRAIL_BARU:
                    # po ZEMIAUSIU is 3 paskutiniu baru, ne po sio baro dugnu
                    stop = max(stop, min(zem[-S2_TRAIL_BARU:]))
            maks.append(hi)
            if len(maks) >= 3 and maks[-1] < maks[-2] < maks[-3]:
                sant = b.get("apyv_santykis", np.nan)
                if not np.isfinite(sant) or sant < 1.0:
                    return dict(baigtis="momentas dingo",
                                pelnas_pct=(float(b["Close"]) / ieina - 1) * 100,
                                minuciu=(j + 1) * BARAS_MIN, blogiausia=blog,
                                neissprestas=False, rytas_pct=rytas, isejo_bare=j + 1)
    else:
        for j, (_, b) in enumerate(sesijos_likutis.iterrows()):
            blog = min(blog, (float(b["Low"]) / ieina - 1.0) * 100)
            if float(b["Low"]) <= stop:
                return dict(baigtis="stop", pelnas_pct=(stop / ieina - 1) * 100,
                            minuciu=(j + 1) * BARAS_MIN, blogiausia=blog,
                            neissprestas=False, rytas_pct=rytas, isejo_bare=j + 1)
            if tikslas and float(b["High"]) >= tikslas:
                return dict(baigtis="tikslas", pelnas_pct=(tikslas / ieina - 1) * 100,
                            minuciu=(j + 1) * BARAS_MIN, blogiausia=blog,
                            neissprestas=False, rytas_pct=rytas, isejo_bare=j + 1)

    gal = float(sesijos_likutis["Close"].iloc[-1])
    return dict(baigtis="neissprestas", pelnas_pct=(gal / ieina - 1) * 100,
                minuciu=len(sesijos_likutis) * BARAS_MIN, blogiausia=blog,
                neissprestas=True, rytas_pct=rytas,
                isejo_bare=len(sesijos_likutis))


# ================================================================ busena

def ikelti_busena():
    if os.path.exists(BUSENOS_FAILAS):
        try:
            with open(BUSENOS_FAILAS, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def issaugoti_busena(b):
    try:
        with open(BUSENOS_FAILAS, "w", encoding="utf-8") as f:
            json.dump(b, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


# ================================================================ live

def paleisti_live(rinkos):
    """Busena saugo PIRMINI trigeri: progreso juosta matuojama nuo tikrojo L
    ir R, o ne perskaiciuojama kas 5 min."""
    busena = ikelti_busena()
    eilutes = []
    for rinka in rinkos:
        zyme = RINKOS[rinka]["zyme"]
        tick = universas(rinka)
        zem = sektoriu_zemelapis()
        kont_visi = dienos_kontekstas(tick, rinka, gyvai=True)
        div_visi = dividendu_kalendorius(list(kont_visi))
        ind = yf.download(RINKOS[rinka]["indeksas"], period="6mo", interval="1d",
                          progress=False, auto_adjust=False)
        rs = rezimu_serija(ind)
        rezimas = list(rs.values())[-1] if rs else "neutral"
        print(f"  {zyme}: rezimas {rezimas}, konteksto {len(kont_visi)}")

        barai = {t: sesijos_rodikliai(b, rinka)
                 for t, b in intraday_barai(list(kont_visi), dienos=2).items()}
        sekt = sektoriaus_pokyciai(barai, zem)

        for t, d in barai.items():
            kont = kont_visi.get(t)
            if not kont or len(d) < 8:
                continue
            ses = d["sesija"].iloc[-1]
            sesija = d[d["sesija"] == ses]
            if len(sesija) < 6:
                continue
            iki = RINKOS[rinka]["uzdarymas"] - int(sesija["minute"].iloc[-1])
            sp = sekt.get((zem.get(t), d.index[-1]))
            for s in aptikti(sesija, kont, rezimas, sp, div_visi.get(t), iki,
                             kont["vakar_uzdarymas"]):
                raktas = f"{zyme}|{t}|{s['tipas']}|{ses}"
                sena = busena.get(raktas)
                dabar = str(d.index[-1])
                if sena:
                    L, R = sena["L"], sena.get("R")
                    if R and R > L:
                        s["progresas"] = min(1.0, max(0.0, (float(sesija["Close"].iloc[-1]) - L) / (R - L)))
                    s["ieina"] = sena["ieina"]
                    s["pirmas_kartas"] = sena["laikas"]
                else:
                    busena[raktas] = dict(L=s["L"], R=s.get("R"),
                                          ieina=s["ieina"], laikas=dabar)
                    s["pirmas_kartas"] = dabar
                try:
                    amz = int((pd.Timestamp(dabar) -
                               pd.Timestamp(s["pirmas_kartas"])).total_seconds() // 60)
                except Exception:
                    amz = 0
                s.update(rinka=zyme, tickeris=t, laikas=dabar, atr=kont["atr"],
                         rezimas=rezimas, sektorius=zem.get(t), amzius_min=amz,
                         dividendas=div_visi.get(t))
                eilutes.append(s)

    siandien = datetime.now().date().isoformat()
    issaugoti_busena({k: v for k, v in busena.items() if k.endswith(siandien)})
    os.makedirs("docs", exist_ok=True)
    with open("docs/detektorius.json", "w", encoding="utf-8") as f:
        json.dump(dict(atnaujinta=datetime.now(timezone.utc).isoformat(),
                       signalai=eilutes), f, ensure_ascii=False, indent=1)
    tinkami = [e for e in eilutes if e["tinkamas"]]
    print(f"\n  aktyvus: {len(eilutes)}  (tinkami: {len(tinkami)})")
    for e in sorted(tinkami, key=lambda x: -x["progresas"]):
        print(f"   [{e['rinka']:>3}] {e['tickeris']:<10} {e['scenarijus']:<24} "
              f"{e['progresas']*100:>5.0f}%  {e['amzius_min']:>4} min")
    return eilutes


# ================================================================ kalibracija

def paleisti_kalibracija(rinkos, dienos):
    for rinka in rinkos:
        zyme = RINKOS[rinka]["zyme"]
        print(f"\n{'='*78}\n{zyme}\n{'='*78}")
        tick = universas(rinka)
        zem = sektoriu_zemelapis()
        kont_visi = dienos_kontekstas(tick, rinka)
        div_visi = dividendu_kalendorius(list(kont_visi))
        ind = yf.download(RINKOS[rinka]["indeksas"], period=f"{dienos+150}d",
                          interval="1d", progress=False, auto_adjust=False)
        rs = rezimu_serija(ind)

        barai = {t: sesijos_rodikliai(b, rinka)
                 for t, b in intraday_barai(list(kont_visi), dienos=dienos).items()}
        print(f"  akciju su 5 min duomenimis: {len(barai)}")
        if zem:
            print(f"  sektoriu zemelapis: {len(set(zem.values()))} sektoriai")
        else:
            print("  DEMESIO: sektoriu zemelapio nera - sektoriaus filtras neveiks")
        sekt = sektoriaus_pokyciai(barai, zem)

        ivykiai = []
        for t, d in barai.items():
            kont = kont_visi.get(t)
            if not kont:
                continue
            sesijos = list(d.groupby("sesija", sort=True))
            for si, (ses, sd) in enumerate(sesijos):
                if len(sd) < 12:
                    continue
                rytas = sesijos[si + 1][1].iloc[:1] if si + 1 < len(sesijos) else sd.iloc[:0]
                rezimas = rs.get(ses, "neutral")
                vakar = vakarykstis_pagal_sesija(kont, ses)
                suveike = set()
                for i in range(6, len(sd)):
                    langas = sd.iloc[:i + 1]
                    iki = RINKOS[rinka]["uzdarymas"] - int(langas["minute"].iloc[-1])
                    sp = sekt.get((zem.get(t), sd.index[i]))
                    for s in aptikti(langas, kont, rezimas, sp, div_visi.get(t),
                                     iki, vakar):
                        if s["tipas"] in suveike or not s["tinkamas"]:
                            continue
                        suveike.add(s["tipas"])
                        s.update(baigtis(sd.iloc[i + 1:], rytas, s))
                        s.update(tickeris=t, sesija=str(ses), atr=kont["atr"],
                                 rezimas=rezimas, pradzia_bare=i,
                                 baru_sesijoje=len(sd))
                        ivykiai.append(s)

        ataskaita(ivykiai, zyme)
        if ivykiai:
            ses = sorted({e["sesija"] for e in ivykiai})
            riba = ses[len(ses) // 2]
            print(f"\n  ---- {zyme}: 1-A PUSE (iki {riba}) ----")
            ataskaita([e for e in ivykiai if e["sesija"] <= riba], zyme, True)
            print(f"\n  ---- {zyme}: 2-A PUSE (NEMATYTA) ----")
            ataskaita([e for e in ivykiai if e["sesija"] > riba], zyme, True)


def vienalaikiskumas(df):
    """Kiek signalu vienu metu ekrane - svarbiau uz suveikimus per diena."""
    if df.empty:
        return 0.0, 0
    vid, mx = [], 0
    for _, g in df.groupby("sesija"):
        n = int(g["baru_sesijoje"].max())
        sk = np.zeros(n + 2)
        for _, r in g.iterrows():
            a = int(r["pradzia_bare"])
            b = min(n, a + int(r.get("isejo_bare", 1)))
            sk[a:b] += 1
        vid.append(sk.mean())
        mx = max(mx, int(sk.max()))
    return float(np.mean(vid)), mx


def ataskaita(ivykiai, zyme, trumpai=False):
    if not ivykiai:
        print("  suveikimu nebuvo")
        return
    df = pd.DataFrame(ivykiai)
    sesiju = df["sesija"].nunique()
    vid_k, max_k = vienalaikiskumas(df)
    print(f"\n  suveikimu: {len(df)}   sesiju: {sesiju}   "
          f"per diena: {len(df)/max(1,sesiju):.1f}")
    print(f"  VIENU METU ekrane: vidutiniskai {vid_k:.1f}, daugiausia {max_k}")
    print(f"\n{'SCENARIJUS':<24}{'N':>6}{'per d.':>7}{'kartu':>7}{'tiksl':>7}"
          f"{'stop':>7}{'NEISPR':>8}{'vid %':>8}{'EUR':>8}{'min.':>6}"
          f"{'rytas%':>8}{'95% EUR':>19}")
    print("-" * 117)
    rng = np.random.default_rng(42)
    for nm, g in df.groupby("scenarijus"):
        vk, _ = vienalaikiskumas(g)
        eur = g["pelnas_pct"] / 100 * POZICIJA - SANAUDOS_EUR
        pdd = g.assign(eur=eur).groupby("sesija")["eur"].mean().values
        if len(pdd) >= 10:
            bs = [rng.choice(pdd, len(pdd), replace=True).mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        print(f"{nm:<24}{len(g):>6}{len(g)/max(1,sesiju):>7.1f}{vk:>7.1f}"
              f"{(g['baigtis']=='tikslas').mean()*100:>6.1f}%"
              f"{g['baigtis'].isin(['stop','slenkantis stop']).mean()*100:>6.1f}%"
              f"{g['neissprestas'].mean()*100:>7.1f}%{g['pelnas_pct'].mean():>8.2f}"
              f"{eur.mean():>8.2f}{g['minuciu'].median():>6.0f}"
              f"{g['rytas_pct'].mean():>8.2f}  [{lo:>6.2f},{hi:>6.2f}]"
              f"{'  <<<' if lo > 0 else ''}")
    if trumpai:
        return
    print("\n  NEISPR = neissprestas iki uzdarymo. Tai krintanciu peiliu matas:")
    print("  scenarijus, kuris iki uzdarymo nepasieke nei tikslo, nei stop'o,")
    print("  pagal apibrezima nebuvo tas, ko ieskojom.")
    print("  'rytas%' - kas buvo kito ryto atidarymu (nakties langas patvirtintas")
    print("  abiejose rinkose: EU +0.049%, JAV +0.077%).")
    print("  Bootstrap pagal DIENAS: 5000 ivykiu yra ~60 stebejimu, ne 5000.")


# ================================================================ main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--kalibracija", action="store_true")
    ap.add_argument("--rinka", choices=["eu", "us", "abi"], default="abi")
    ap.add_argument("--dienos", type=int, default=60)
    a = ap.parse_args()
    rinkos = ["eu", "us"] if a.rinka == "abi" else [a.rinka]
    if a.kalibracija:
        paleisti_kalibracija(rinkos, a.dienos)
    else:
        paleisti_live(rinkos)
