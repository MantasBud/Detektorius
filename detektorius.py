#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETEKTORIUS v1  —  2026-09-27
=============================

KAS TAI
-------
Ne reitinguotojas. Detektorius neatsako "kuri akcija rytoj bus geresne" - to
neradome ne viename is ~60 testuotu signalu. Jis atsako "ar SI situacija JAU
prasidejo", ir rodo, kur ji yra savo kelyje.

TRYS SCENARIJAI
---------------
1. KRITIMAS + ATSISTATYMO PRADZIA
   Akcija nukrito nuo R (buves lygis), padare dugna L, ir pradeda grizti.
   Tikslas: R.  Stop'as: zemiau L.
2. RALIO PRADZIA
   Po siauros bazes kaina pramusa virsu su apyvarta, virs VWAP.
   Tikslo NERA - slenkantis isejimas.
3. ATIDARYMO TARPO UZPILDYMAS
   Atsidare zemiau vakarykscio uzdarymo ir pradeda grizti.
   Tikslas: vakarykstis uzdarymas.  Stop'as: zemiau tarpo dugno.

Scenarijai 1 ir 3 turi STRUKTURINI tiksla - jis isvedamas is pacios situacijos,
ne fiksuotas procentas. Tai svarbu: visi ankstesni testai naudojo fiksuota +1%
ar +2%, ir butent del to ATR ruozu rezultatai buvo tokie nevienodi (ramiai
akcijai +1% per daug, judriai per mazai). Struktūrinis tikslas sios ydos neturi.

KODEL BUSENU MASINA, O NE BALAS
--------------------------------
Balas neturi atminties. Senasis modulis kas 5 min perskaiciuodavo viska is
naujo ir nezinodavo, kad pries 20 min ta pati akcija dare dugna. Scenarijus be
atminties neaprasomas.

    RAMYBE -> SEKAMA -> PARUOSTA -> SUVEIKE -> (ATSAUKTA)

Busena issaugoma tarp paleidimu (busena.json).

VIENAS KODAS LIVE IR KALIBRACIJAI
----------------------------------
Sio projekto brangiausia klaida buvo ta, kad modulis skaiciavo viena, o testas
kita (trys skirtingi Z-balo apibrezimai, 2026-09-07). Todel cia scenarijai yra
GRYNOSIOS FUNKCIJOS, kurioms paduodamas baru langas, ir jas kviecia ir live, ir
kalibracija. Kitaip matuoti negalima is principo.

Paleidimas:
    python detektorius.py --live
    python detektorius.py --kalibracija --dienos 60
    python detektorius.py --kalibracija --rinka us
"""

import argparse
import json
import math
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
MIN_APYVARTA = 5e6           # tavo pozicijai pakanka; 20-30 mln buvo per grieztai

BARAS_MIN = 5                # 5 min barai
BUSENOS_FAILAS = "busena.json"
ZURNALAS = "docs/zurnalas_detektorius.csv"

# --- scenarijus 1: kritimas + atsistatymas ---
S1_MIN_KRITIMAS_PCT = 0.40   # min kritimas nuo R, procentais
S1_MIN_KRITIMAS_ATR = 0.50   # ...ir bent tiek ATR daliu
S1_MAX_KRITIMAS_ATR = 2.50   # giliau - jau ne triuksmas (>3% nuslavimas: EU -0.095%, JAV -0.407%)
S1_ATSOKIMAS = 0.15          # kiek pakilti nuo L, kad laikytume pradzia: 15% kelio iki R
S1_MAX_ATSOKIMAS = 0.60      # jei jau nuejo >60% kelio, veluojam - nerodom
S1_MIN_BARU_NUO_L = 2        # ne akimirksninis atsokimas
S1_R_LANGAS_BARU = 78        # ~1 sesija atgal ieskom R (5 min x 78 = 6.5 val)
S1_STOP_MARZA = 0.20         # stop'as: L - 0.20 x (R - L)

# --- scenarijus 2: ralio pradzia ---
S2_BAZE_BARU = 9             # ~45 min siauros bazes
S2_BAZE_MAX_ATR = 0.50       # bazes plotis < 0.5 x ATR
S2_APYVARTOS_SUOLIS = 1.5    # vs tos PACIOS paros minutes tipine apyvarta
S2_MAX_NUO_ATIDARYMO_ATR = 2.5   # jei jau nuejo toliau - didzioji dalis praeityje
S2_MIN_DIENOS_POKYTIS = 0.0  # diena turi buti teigiama nuo vakar uzdarymo
S2_NERODYTI_PASKUTINES_MIN = 15

# --- scenarijus 3: tarpo uzpildymas ---
S3_MIN_TARPAS_PCT = 0.30
S3_MIN_TARPAS_ATR = 0.35
S3_MAX_TARPAS_ATR = 2.00
S3_ATSOKIMAS = 0.15
S3_MAX_ATSOKIMAS = 0.70

# --- kietieji filtrai (visi is ISMATUOTU dalyku) ---
F_MAX_ATR = 3.5              # >3.5% grupe blogiausia abiejose rinkose
F_SMA200_TURI_KILTI = True   # didziausias vienos salygos inasas: +0.198 p.p.
F_MAX_SEKTORIAUS_KRITIMAS = -1.0   # jei sektorius krenta stipriau - ne triuksmas
F_MAX_KRITIMO_DIENU = 3      # 3+ dienos is eiles zemyn - trendas, ne dipas
F_ATASKAITA_DIENU = 2
F_DIVIDENDU_LANGAS_D = 3     # +-3 d. nuo prognozuojamos ex-datos

RINKOS = {
    "eu": dict(zyme="EU", indeksas="EXSA.DE", tz="Europe/Berlin",
               atidarymas="09:00", uzdarymas="17:30", valiuta="EUR"),
    "us": dict(zyme="JAV", indeksas="SPY", tz="America/New_York",
               atidarymas="09:30", uzdarymas="16:00", valiuta="USD"),
}


# ================================================================ universas

def universas(rinka):
    """Tas pats sarasas, kuri naudoja ir kalibracija, ir live."""
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


# ================================================================ duomenys

def dienos_kontekstas(tickers, rinka, metai=2):
    """Dienos lygio kontekstas: ATR, SMA200, vakarykstis uzdarymas, apyvarta,
    kritimo dienu serija, dividendu prognoze. Skaiciuojama KARTA per diena."""
    df = yf.download(tickers, period=f"{metai}y", interval="1d",
                     auto_adjust=False, progress=False, group_by="ticker",
                     threads=True)
    out = {}
    for t in tickers:
        try:
            d = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            d = d.dropna(subset=["Open", "High", "Low", "Close"])
            if len(d) < 220:
                continue
            c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
            prev = c.shift(1)
            tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
            atr = float((tr.rolling(20).mean() / c).iloc[-1] * 100)
            sma200 = c.rolling(200).mean()
            kyla = bool(sma200.iloc[-1] > sma200.iloc[-21]) if len(sma200.dropna()) > 21 else False
            apyv = float((c * v).rolling(20).median().iloc[-1])

            # kritimo dienu serija
            pok = (c / prev - 1.0).iloc[-6:]
            serija = 0
            for x in reversed(pok.tolist()):
                if x < 0:
                    serija += 1
                else:
                    break

            out[t] = dict(
                atr=atr, sma200_kyla=kyla, apyvarta=apyv,
                vakar_uzdarymas=float(c.iloc[-1]),
                vakar_max=float(h.iloc[-1]), vakar_min=float(l.iloc[-1]),
                kritimo_dienu=serija,
                dividendas=dividendu_prognoze(t),
            )
        except Exception:
            continue
    return out


def dividendu_prognoze(tickeris):
    """Kito ex-dividendo prognoze is istorinio ritmo.

    yfinance busimu datu Europos akcijoms nepateikia patikimai (calendar ir
    info["exDividendDate"] daznai grazina PASKUTINE, ne kita). Todel
    prognozuojam is ritmo: Europos bendroves moka reguliariai, daznai po
    metinio susirinkimo.
    """
    try:
        div = yf.Ticker(tickeris).dividends
        if div is None or len(div) < 2:
            return None
        div = div[-12:]
        datos = [d.tz_localize(None) if d.tz else d for d in div.index]
        if len(datos) < 2:
            return None
        tarpai = [(datos[i + 1] - datos[i]).days for i in range(len(datos) - 1)]
        tarpai = [x for x in tarpai if 20 <= x <= 400]
        if not tarpai:
            return None
        vid = float(np.median(tarpai))
        kita = datos[-1] + timedelta(days=vid)
        while kita < datetime.now():
            kita += timedelta(days=vid)
        return dict(kita_ex=kita.date().isoformat(),
                    suma=float(div.iloc[-1]),
                    dienu_iki=(kita.date() - datetime.now().date()).days,
                    patikima=False)
    except Exception:
        return None


def intraday_barai(tickers, dienos=60):
    """5 min barai. Ta pati funkcija naudojama ir live, ir kalibracijai."""
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
    """Prideda sesijos lygio rodiklius. Viskas skaiciuojama TIK is praeities."""
    d = barai.copy()
    tz = RINKOS[rinka]["tz"]
    try:
        idx = d.index.tz_convert(tz)
    except Exception:
        idx = d.index
    d["sesija"] = [x.date() for x in idx]
    d["minute"] = [x.hour * 60 + x.minute for x in idx]

    d["_tipine"] = (d["High"] + d["Low"] + d["Close"]) / 3.0
    d["_pv"] = d["_tipine"] * d["Volume"]
    g = d.groupby("sesija", sort=False)
    d["vwap"] = g["_pv"].cumsum() / g["Volume"].cumsum().replace(0, np.nan)
    d["sesijos_max"] = g["High"].cummax()
    d["sesijos_min"] = g["Low"].cummin()
    d["sesijos_atidarymas"] = g["Open"].transform("first")
    d.drop(columns=["_tipine", "_pv"], inplace=True)

    # tipine tos PACIOS paros minutes apyvarta (ne dienos vidurkis -
    # apyvarta per diena labai nevienoda)
    d["apyv_tipine"] = d.groupby("minute")["Volume"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).median())
    d["apyv_santykis"] = d["Volume"] / d["apyv_tipine"].replace(0, np.nan)
    return d


# ================================================ scenarijai (grynos funkcijos)

def _ibs(b):
    r = b["High"] - b["Low"]
    return 0.5 if r <= 0 else float((b["Close"] - b["Low"]) / r)


def _auksteja_dugnai(langas, n=2):
    lo = langas["Low"].tolist()
    if len(lo) < n + 1:
        return False
    return all(lo[-i] > lo[-i - 1] for i in range(1, n + 1))


def scenarijus_1(langas, kont):
    """Kritimas + atsistatymo pradzia. Grazina signala arba None.

    langas — sesijos barai iki dabartinio IMTINAI. Jokio zvilgsnio i ateiti.
    """
    if len(langas) < S1_MIN_BARU_NUO_L + 3:
        return None
    atr = kont["atr"]
    dab = langas.iloc[-1]
    kaina = float(dab["Close"])

    # R — auksciausias uzdarymas lange PRIES dugna
    lang = langas.iloc[-S1_R_LANGAS_BARU:] if len(langas) > S1_R_LANGAS_BARU else langas
    i_min = int(np.argmin(lang["Low"].values))
    if i_min == 0 or i_min >= len(lang) - S1_MIN_BARU_NUO_L:
        return None                      # dugnas per anksti arba per velai lange
    R = float(lang["Close"].iloc[:i_min].max())
    L = float(lang["Low"].iloc[i_min])
    if not (R > L > 0):
        return None

    kritimas_pct = (R - L) / R * 100.0
    if kritimas_pct < max(S1_MIN_KRITIMAS_PCT, S1_MIN_KRITIMAS_ATR * atr):
        return None                      # per seklu - nepadengtu sanaudu
    if kritimas_pct > S1_MAX_KRITIMAS_ATR * atr:
        return None                      # per gilu - ne triuksmas

    kelias = (kaina - L) / (R - L)
    if not (S1_ATSOKIMAS <= kelias <= S1_MAX_ATSOKIMAS):
        return None

    # --- krintancio peilio apsauga ---
    po_dugno = lang.iloc[i_min + 1:]
    if len(po_dugno) < S1_MIN_BARU_NUO_L:
        return None
    if float(po_dugno["Low"].iloc[-3:].min()) <= L:
        return None                      # naujas minimumas per paskutinius 3 barus
    if not _auksteja_dugnai(po_dugno, 2):
        return None
    if len(langas) >= 3 and float(dab["Close"]) <= float(langas["Close"].iloc[-3]):
        return None                      # per paskutines 10 min nekyla
    kritimo_barai = lang.iloc[:i_min + 1]
    if len(kritimo_barai) >= 2:
        did = float((kritimo_barai["High"] - kritimo_barai["Low"]).max())
        if R > 0 and did / R * 100.0 > 1.0 * atr:
            return None                  # vienas baras didesnis uz dienos ATR
                                         # -> naujiena, ne triuksmas

    stop = L - S1_STOP_MARZA * (R - L)
    return dict(scenarijus="Kritimas + atsistatymas", tipas=1,
                ieina=kaina, tikslas=R, stop=stop, R=R, L=L,
                progresas=kelias, progresas_tikslus=True,
                kritimas_pct=kritimas_pct)


def scenarijus_2(langas, kont, minute_iki_uzdarymo):
    """Ralio pradzia. Tikslo nera - slenkantis isejimas."""
    if len(langas) < S2_BAZE_BARU + 3:
        return None
    if minute_iki_uzdarymo < S2_NERODYTI_PASKUTINES_MIN:
        return None
    atr = kont["atr"]
    dab = langas.iloc[-1]
    kaina = float(dab["Close"])

    baze = langas.iloc[-(S2_BAZE_BARU + 1):-1]
    b_max, b_min = float(baze["High"].max()), float(baze["Low"].min())
    if b_max <= 0:
        return None
    plotis = (b_max - b_min) / b_max * 100.0
    if plotis > S2_BAZE_MAX_ATR * atr:
        return None                      # bazes nebuvo
    if kaina <= b_max:
        return None                      # nera pramusimo

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
        return None                      # didzioji judesio dalis praeityje
    if kont["vakar_uzdarymas"] > 0:
        if (kaina / kont["vakar_uzdarymas"] - 1.0) * 100.0 < S2_MIN_DIENOS_POKYTIS:
            return None

    # konservatyvus virsunes vertinimas ir progresas
    nuejo = (kaina - b_max) / b_max * 100.0
    tiketina = 1.0 * atr
    prog_judesio = min(1.0, nuejo / tiketina) if tiketina > 0 else 0.0
    liko = max(1.0, minute_iki_uzdarymo)
    prog_laiko = 1.0 - liko / (liko + 30.0)
    progresas = max(prog_judesio, prog_laiko)   # konservatyviai - i "veliau" puse

    return dict(scenarijus="Ralio pradzia", tipas=2,
                ieina=kaina, tikslas=None, stop=b_min, R=None, L=b_min,
                progresas=progresas, progresas_tikslus=False,
                virsune_vertinimas=b_max * (1 + tiketina / 100.0),
                baze=b_max)


def scenarijus_3(langas, kont):
    """Atidarymo tarpo uzpildymas. Tikslas: vakarykstis uzdarymas."""
    if len(langas) < 3:
        return None
    atr = kont["atr"]
    vu = kont["vakar_uzdarymas"]
    if vu <= 0:
        return None
    atid = float(langas["Open"].iloc[0])
    if atid >= vu:
        return None                      # tarpo zemyn nebuvo

    tarpas_pct = (vu - atid) / vu * 100.0
    if tarpas_pct < max(S3_MIN_TARPAS_PCT, S3_MIN_TARPAS_ATR * atr):
        return None
    if tarpas_pct > S3_MAX_TARPAS_ATR * atr:
        return None                      # per didelis - greiciausiai naujiena

    L = float(langas["Low"].min())
    kaina = float(langas["Close"].iloc[-1])
    if vu <= L:
        return None
    kelias = (kaina - L) / (vu - L)
    if not (S3_ATSOKIMAS <= kelias <= S3_MAX_ATSOKIMAS):
        return None
    if float(langas["Low"].iloc[-3:].min()) <= L:
        return None
    if not _auksteja_dugnai(langas.iloc[-3:], 2):
        return None
    if len(langas) >= 3 and float(langas["Close"].iloc[-1]) <= float(langas["Close"].iloc[-3]):
        return None

    stop = L - S1_STOP_MARZA * (vu - L)
    return dict(scenarijus="Tarpo uzpildymas", tipas=3,
                ieina=kaina, tikslas=vu, stop=stop, R=vu, L=L,
                progresas=kelias, progresas_tikslus=True,
                tarpas_pct=tarpas_pct)


# ================================================================ filtrai

def kietieji_filtrai(kont, sig, rinkos_rezimas, sektoriaus_pokytis):
    """Grazina saraso priezasciu, KODEL signalas atmetamas. Tuscias = praeina.

    Kiekviena salyga remiasi ismatuotu skaiciumi - saltiniai komentaruose.
    """
    kl = []
    if kont["atr"] > F_MAX_ATR:
        kl.append(f"ATR {kont['atr']:.1f}% > {F_MAX_ATR}%")      # blogiausia grupe abiejose rinkose
    if F_SMA200_TURI_KILTI and not kont["sma200_kyla"]:
        kl.append("SMA200 nekyla")                                # +0.198 p.p. inasas
    if kont["apyvarta"] < MIN_APYVARTA:
        kl.append("per maza apyvarta")
    if kont["kritimo_dienu"] >= F_MAX_KRITIMO_DIENU:
        kl.append(f"{kont['kritimo_dienu']} kritimo dienos is eiles")
    if rinkos_rezimas == "bear":
        kl.append("rinka krenta")                                 # bear dienomis vidurkis -0.45%
    if sektoriaus_pokytis is not None and sektoriaus_pokytis < F_MAX_SEKTORIAUS_KRITIMAS:
        kl.append(f"sektorius {sektoriaus_pokytis:+.1f}%")        # 81% giliu kritimu lydejo sektorius
    d = kont.get("dividendas")
    if d and abs(d.get("dienu_iki", 999)) <= F_DIVIDENDU_LANGAS_D:
        kl.append(f"ex-dividendas po {d['dienu_iki']} d.")        # kritimas butu mechaninis
    if sig.get("tikslas"):
        rr = (sig["tikslas"] - sig["ieina"]) / max(1e-9, sig["ieina"] - sig["stop"])
        if rr < 1.0:
            kl.append(f"R:R {rr:.2f} < 1.0")
    return kl


def rinkos_rezimas(indeksas_d):
    """5 dienu kryptis + padetis pries SMA20. Vienos dienos pokytis per triuksmingas."""
    try:
        c = indeksas_d["Close"]
        r5 = float(c.iloc[-1] / c.iloc[-6] - 1.0) * 100
        virs = bool(c.iloc[-1] > c.rolling(20).mean().iloc[-1])
        siandien = float(c.iloc[-1] / c.iloc[-2] - 1.0) * 100
        if r5 < -1 and siandien < 0:
            return "bear"
        if r5 < -1 or not virs:
            return "bear_soft"
        if r5 > 1 and virs:
            return "bull"
        return "neutral"
    except Exception:
        return "neutral"


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
    with open(BUSENOS_FAILAS, "w", encoding="utf-8") as f:
        json.dump(b, f, ensure_ascii=False, indent=1)


# ================================================================ aptikimas

def aptikti_sesijoje(barai_sesijos, kont, rinka, rezimas, sekt_pok, iki_uzdarymo):
    """Viena vieta, kur kvieciami visi scenarijai.

    Live kviecia su paskutiniu baru; kalibracija - su kiekvienu baru is eiles.
    Taip garantuojama, kad matuojam TA PATI, ka rodom.
    """
    rez = []
    for f, arg in ((scenarijus_1, (barai_sesijos, kont)),
                   (scenarijus_2, (barai_sesijos, kont, iki_uzdarymo)),
                   (scenarijus_3, (barai_sesijos, kont))):
        try:
            s = f(*arg)
        except Exception:
            s = None
        if not s:
            continue
        s["kliutys"] = kietieji_filtrai(kont, s, rezimas, sekt_pok)
        s["tinkamas"] = len(s["kliutys"]) == 0
        rez.append(s)
    return rez


# ================================================================ rezimai

def paleisti_live(rinkos):
    busena = ikelti_busena()
    eilutes = []
    for rinka in rinkos:
        zyme = RINKOS[rinka]["zyme"]
        tick = universas(rinka)
        kont_visi = dienos_kontekstas(tick, rinka)
        ind = yf.download(RINKOS[rinka]["indeksas"], period="3mo", interval="1d",
                          progress=False, auto_adjust=False)
        rezimas = rinkos_rezimas(ind)
        print(f"  {zyme}: rezimas {rezimas}, konteksto {len(kont_visi)}")

        barai = intraday_barai([t for t in tick if t in kont_visi], dienos=2)
        for t, b in barai.items():
            kont = kont_visi.get(t)
            if not kont:
                continue
            d = sesijos_rodikliai(b, rinka)
            pask = d["sesija"].iloc[-1]
            sesija = d[d["sesija"] == pask]
            if len(sesija) < 4:
                continue
            uzd_min = int(RINKOS[rinka]["uzdarymas"].split(":")[0]) * 60 + \
                int(RINKOS[rinka]["uzdarymas"].split(":")[1])
            iki = uzd_min - int(sesija["minute"].iloc[-1])
            for s in aptikti_sesijoje(sesija, kont, rinka, rezimas, None, iki):
                s.update(rinka=zyme, tickeris=t, laikas=str(d.index[-1]),
                         atr=kont["atr"], rezimas=rezimas)
                eilutes.append(s)

    issaugoti_busena(busena)
    os.makedirs("docs", exist_ok=True)
    with open("docs/detektorius.json", "w", encoding="utf-8") as f:
        json.dump(dict(atnaujinta=datetime.now(timezone.utc).isoformat(),
                       signalai=eilutes), f, ensure_ascii=False, indent=1)
    print(f"\n  suveikimu: {len(eilutes)} "
          f"(tinkamu: {sum(1 for x in eilutes if x['tinkamas'])})")
    for e in sorted(eilutes, key=lambda x: -x["progresas"]):
        zyma = "" if e["tinkamas"] else "  BLOKUOTA: " + "; ".join(e["kliutys"])
        print(f"   [{e['rinka']:>3}] {e['tickeris']:<10} {e['scenarijus']:<24} "
              f"{e['progresas']*100:>5.0f}%{zyma}")
    return eilutes


def paleisti_kalibracija(rinkos, dienos):
    """Praleidzia 60 dienu 5 min istorijos per TAS PACIAS funkcijas.

    Matuoja: daznis, P(tikslas pries stop'a), laikas, blogiausias nuosmukis, EUR.
    """
    for rinka in rinkos:
        zyme = RINKOS[rinka]["zyme"]
        print(f"\n{'='*70}\n{zyme}\n{'='*70}")
        tick = universas(rinka)
        kont_visi = dienos_kontekstas(tick, rinka)
        barai = intraday_barai([t for t in tick if t in kont_visi], dienos=dienos)
        print(f"  akciju su 5 min duomenimis: {len(barai)}")

        ivykiai = []
        for t, b in barai.items():
            kont = kont_visi.get(t)
            if not kont:
                continue
            d = sesijos_rodikliai(b, rinka)
            uzd_min = int(RINKOS[rinka]["uzdarymas"].split(":")[0]) * 60 + \
                int(RINKOS[rinka]["uzdarymas"].split(":")[1])
            for ses, sd in d.groupby("sesija", sort=True):
                if len(sd) < 12:
                    continue
                suveike = set()
                for i in range(6, len(sd)):
                    langas = sd.iloc[:i + 1]
                    iki = uzd_min - int(langas["minute"].iloc[-1])
                    for s in aptikti_sesijoje(langas, kont, rinka, "neutral", None, iki):
                        if s["tipas"] in suveike or not s["tinkamas"]:
                            continue
                        suveike.add(s["tipas"])
                        s.update(baigtis(sd.iloc[i + 1:], s, kont))
                        s.update(tickeris=t, sesija=str(ses), atr=kont["atr"])
                        ivykiai.append(s)
        ataskaita(ivykiai, zyme, len(barai), dienos)


def baigtis(toliau, sig, kont):
    """Kas nutiko PO suveikimo. Tik ateities barai - cia zvilgsnis i ateiti
    leistinas, nes matuojam rezultata, ne priimam sprendima."""
    if len(toliau) == 0:
        return dict(baigtis="nespejo", pelnas_pct=0.0, minuciu=0, blogiausia=0.0)
    ieina = sig["ieina"]
    tikslas = sig.get("tikslas")
    stop = sig["stop"]
    blog = 0.0
    for j, (_, b) in enumerate(toliau.iterrows()):
        blog = min(blog, (float(b["Low"]) / ieina - 1.0) * 100)
        if float(b["Low"]) <= stop:
            return dict(baigtis="stop", pelnas_pct=(stop / ieina - 1) * 100,
                        minuciu=(j + 1) * BARAS_MIN, blogiausia=blog)
        if tikslas and float(b["High"]) >= tikslas:
            return dict(baigtis="tikslas", pelnas_pct=(tikslas / ieina - 1) * 100,
                        minuciu=(j + 1) * BARAS_MIN, blogiausia=blog)
    gal = float(toliau["Close"].iloc[-1])
    return dict(baigtis="sesijos pabaiga", pelnas_pct=(gal / ieina - 1) * 100,
                minuciu=len(toliau) * BARAS_MIN, blogiausia=blog)


def ataskaita(ivykiai, zyme, akciju, dienos):
    if not ivykiai:
        print("  suveikimu nebuvo")
        return
    df = pd.DataFrame(ivykiai)
    sesiju = df["sesija"].nunique()
    print(f"\n  suveikimu is viso: {len(df)}   sesiju: {sesiju}   "
          f"vidutiniskai per diena: {len(df)/max(1,sesiju):.1f}")
    print(f"\n{'SCENARIJUS':<26}{'N':>6}{'per d.':>8}{'tikslas':>9}"
          f"{'stop':>8}{'vid %':>8}{'vid EUR':>9}{'min.':>7}{'blog.%':>8}")
    print("-" * 90)
    for nm, g in df.groupby("scenarijus"):
        tiks = (g["baigtis"] == "tikslas").mean() * 100
        st = (g["baigtis"] == "stop").mean() * 100
        eur = g["pelnas_pct"] / 100 * POZICIJA - SANAUDOS_EUR
        print(f"{nm:<26}{len(g):>6}{len(g)/max(1,sesiju):>8.1f}{tiks:>8.1f}%"
              f"{st:>7.1f}%{g['pelnas_pct'].mean():>8.2f}{eur.mean():>9.2f}"
              f"{g['minuciu'].median():>7.0f}{g['blogiausia'].median():>8.2f}")
    print("\n  PASTABA: su ~60 nepriklausomu dienu galima atskirti 50% nuo 80%,")
    print("  bet NEGALIMA atskirti 65% nuo 72%. Skaiciai rodo dydi, ne tikslu lygi.")
    print("  Tikslas: priverzti filtrus iki 5-15 suveikimu per diena.")


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
