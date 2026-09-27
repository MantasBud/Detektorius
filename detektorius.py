#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETEKTORIUS v4  —  2026-09-27
=============================

Perrasyta nuo balto lapo. Is v3 NEPERKELTA NE VIENA taisykle, kuri buvo
isvesta is senojo reitinguotojo tyrimu.

KODEL
-----
2026-09-27 atveju tyrimas (penki Manto nurodyti atvejai) parode, kad
detekcija veike, o ja nuzude butent paveldeti kietieji filtrai:

    AMD  08-14  suveike 15:45  -> BLOKUOTA: ATR 7.0% > 3.5%
    AMD  08-19  suveike 14:45  -> BLOKUOTA: ATR 6.9%
    AMD  08-21  suveike 13:30  -> BLOKUOTA: ATR 7.1% + 3 kritimo dienos
    PTX  08-03  suveike 10:25  -> BLOKUOTA: ATR 5.1%, SMA200, apyvarta

Tos ribos buvo ismatuotos nuvidurkinta graza per visa universa - prietaisu,
kuri sis projektas jau pripazino netinkamu pirmo prisilietimo uzdaviniui.
Mantas perka JUDRIAS akcijas PO TO, kai jos nukrito. Filtrai aprase tiksliai
ta akcija, kuria jis perka, ir ja ismete. `kritimo_dienu >= 3` yra jo IEJIMO
SALYGA, ne pavojaus zenklas.

TRYS FILTRU PAKOPOS
-------------------
Mantas patikslino: bloga ne tai, kad filtrai yra, o tai, kad jie KOPIJUOJAMI
is reitinguotojo. Tai teisinga, ir pirma sio failo versija perlenke i kita
puse - isimti visus filtrus reikstu palikti detektoriu be jokios apsaugos.
Todel filtrai skirstomi pagal KILME, ne pagal buvima:

  1 PAKOPA (yra dabar): isvedama is Manto mechanikos arba aritmetikos.
     Apyvarta pagal pozicijos dydi, minimalus judesys pagal sanaudas, R:R,
     tikslo pasiekiamumas pagal ATR. Sitoms nereikia jokio tyrimo - jos
     teisingos pagal apibrezima.

  2 PAKOPA (uzsidirbama): bet kokia riba, kuria SI detektoriaus kalibracija
     parodo esant naudinga IR nematytoje puseje, IR abiejose rinkose.
     Todel ataskaita() spausdina pjuvius pagal ATR, kritimo gyli ir paros
     laika - ne tam, kad ju laikytumemes dabar, o kad matytume, kuri riba
     uzsidirbo teise egzistuoti. Filtras ateina kaip kalibracijos ISVADA,
     ne kaip jos PRIELAIDA.

  3 PAKOPA (niekada): perkelta is reitinguotojo tyrimu. F_MAX_ATR = 3.5,
     SMA200, kritimo dienos, sektorius, rezimas. Jos buvo ismatuotos kitu
     prietaisu ir kitam klausimui.

DU SCENARIJAI (tiek Mantas ir apibreze)
---------------------------------------
1. KRITIMAS IR APSISUKIMAS
   Akcija krito 1-5 sesijas ir uzdare prie dienos dugno; kita sesija atsiima
   vakarykscio uzdarymo lygi. Tikslas - lygis pries kritima.
   Etalonas: CAP.PA 07-23 -> 07-24..07-28 (+18%), SAP.DE 07-23 -> (+24%).
   Svarbu: 07-23 signalo NERA (kaina taip ir neatsieme uzdarymo) - tai ir
   yra krintancio peilio apsauga, ir ji gaunama is pacios salygos, ne is
   atskiro filtro.

2. NAUJIENU TARPAS IR EIGA
   Atidarymo tarpas >= 1 dienos ATR su apyvarta; kaina laikosi virs VWAP;
   pozicija nesama slenkanciu stop'u ATR vienetais.
   Etalonas: ADYEN.AS 08-13 (tarpas +5.7%, diena +10.1%),
             PTX.DE 08-04 (tarpas +16.4%, diena +8.8%).

Scenarijai nepersidengia PAGAL KONSTRUKCIJA: lyginamas ATIDARYMAS, ne iejimas.
Jei atidarymas - vakar_uzd >= 1.0*ATR, diena valdo 2 scenarijus, ir 1 grazina
None; kitu atveju atvirksciai. Ribos yra tikslus vienas kito papildiniai, ir
tai patikrinta savitikroje skenuojant tarpa 0.00..2.00 ATR.

HORIZONTAS
----------
v3 viska uzdarydavo iki skambucio. Atveju tyrimas parode, kad 39% judesio
susidaro PER NAKTI ir kad tikrasis dydis yra 1-3 sesijos. Todel baigtis
matuojama iki HORIZONTAS_SESIJU pabaigos, o pozicija nesama per nakti.

ZVILGSNIS I ATEITI
------------------
Visi dienos rodikliai skaiciuojami viena kartu ir PASLENKIAMI per viena diena
(`.shift(1)`). Po to eilute, pazymeta data D, fiziskai negali tureti D
informacijos. v3 cia turejo klaida: atr/sma200/kritimo_dienu buvo paskutines
dienos reiksmes, taikomos visoms 60 kalibracijos sesiju (PTX atveju SMA200
filtras davė kita atsakyma 53 sesijas is 60).

Paleidimas:
    python detektorius.py --live
    python detektorius.py --kalibracija --dienos 60
    python detektorius.py --savitikra          # be tinklo, etalonai
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

# ---------------------------------------------------------------- mechanika
# Sitie keturi skaiciai ateina is Manto saskaitos, ne is jokio tyrimo.
POZICIJA = 18000.0          # visas portfelis vienai pozicijai
SANAUDOS_EUR = 10.0         # pirkimas + pardavimas
LUZIO_TASKAS = SANAUDOS_EUR / POZICIJA * 100.0      # 0.0556%
MAX_APYVARTOS_DALIS = 0.01  # pozicija ne daugiau 1% dienos apyvartos
MIN_APYVARTA = POZICIJA / MAX_APYVARTOS_DALIS       # -> 1.8 mln EUR

# Maziausias prasmingas judesys: kad sandoris butu vertas darymo, tikslas
# turi buti bent 10x uz luzio taska. Tai isvesta is sanaudu, ne is tyrimo.
MIN_JUDESYS_PCT = 10 * LUZIO_TASKAS                 # -> 0.556%
# SALUTINE PASEKME, uzrasyta samoningai (rasta perziurint 2026-09-27):
# kadangi 1 scenarijaus tikslas yra 0.5*ATR, o judesys turi buti >= 0.556%,
# akcijos su ATR < 1.11% 1 scenarijaus signalo NEDUODA NIEKADA. Tai de facto
# MIN_ATR riba. Ji teisėta pagal 1 pakopa (aritmetika is sanaudu), NE
# paveldėta is reitinguotojo, ir ji veikia PRIESINGA puse nei senasis
# F_MAX_ATR = 3.5: anksciau buvo ismetamos JUDRIOS akcijos, dabar - RAMIOS.
MIN_RR = 1.0                                        # aritmetika

BARAS_MIN = 5
HORIZONTAS_SESIJU = 3
# Live rezimui reikia tiek pat 5 min istorijos, kiek kalibracijai: apyv_tipine
# yra tos pacios minutes mediana per 20 ANKSTESNIU dienu (min_periods=5).
# Su 5 dienomis po shift(1) lieka 4 stebejimai -> apyv_santykis visada NaN ->
# abu scenarijai reikalauja np.isfinite(sant) -> live NIEKADA neduoda signalo.
# Patikrinta 2026-09-27: dienos=5 -> 390/390 NaN.
LIVE_DIENOS = 30
BUSENOS_FAILAS = "busena.json"
ZURNALAS = "docs/zurnalas.csv"
ZURNALO_STULPELIAI = [
    "raktas", "gimimas", "rinka", "tickeris", "scenarijus", "tipas", "sesija",
    "ieina", "tikslas", "stop", "rr", "rizika_eur", "atr_pct",
    "tinkamas", "kliutys", "atr_abs",
    "paskut_laikas", "paskut_kaina", "mfe_pct", "mae_pct",
    "baigtis", "pelnas_pct", "eur", "minuciu",
]
DIVIDENDU_TALPYKLA = "dividendai.json"

# --- scenarijus 1: kritimas ir apsisukimas ---
# Ribos paimtos is dvieju etalonu (CAP, SAP) ir yra HIPOTEZE, ne radinys.
# Jos privalo buti patikrintos nematytoje puseje - zr. ataskaita().
S1_LANGAS_D = 5             # per kiek sesiju vertinamas kritimas
S1_MIN_KRITIMAS_ATR = 1.0   # kritimas bent vienas dienos ATR
S1_MAX_UZD_VIETA = 0.40     # vakar uzdare apatiniame diapazono ketvirtyje
S1_STOP_ATR = 0.30          # stop'as zemiau atsiimto lygio, ATR vienetais
# KONSERVATYVUS TIKSLAS (Manto sprendimas 2026-09-27). Jis iseina PATS ir nori
# tikslo, kuris dar yra pelningoje zonoje, o ne absoliutaus maksimumo:
# "kartais paimu 100-500 euru, kol dar pozicija nepradejo kristi".
# 0.5 ATR prie EU medianos ~2.6% yra ~1.3% = ~234 EUR nuo 18 000 - tiksliai
# tame ruoze. Buvusio lygio (virsune_n) tikslas niekada nevirsija.
S1_TIKSLAS_ATR = 0.50
# Rizikos riba - to paties principo antra puse. Konservatyvus tikslas su
# placiu stop'u duotu beviltiska geometrija: SAP etalone stop'as buvo 1.15 ATR
# nuo iejimo, tad prie 0.5 ATR tikslo R:R butu 0.43 ir signalas uzsiblokuotu.
S1_MAX_RIZIKA_ATR = 0.50
MIN_ATR_PCT = MIN_JUDESYS_PCT / S1_TIKSLAS_ATR      # -> 1.11%, zr. komentara virs
# ISEJIMO VARIANTAI. Skaiciuojami VIENU perejimu ant TU PACIU signalu, tad
# palyginimas svarus - skiriasi tik isejimas, ne aptikimas. Is anksto uzrasyti
# trys, ne tinklelis. Priimamas tik tas, kuris teigiamas EU 1-oje, EU 2-oje,
# JAV 1-oje ir JAV 2-oje pusese.
ISEJIMO_VARIANTAI = {
    "T05": dict(tikslas_atr=0.50, trail=False),   # kortelėje rodomas
    "T10": dict(tikslas_atr=1.00, trail=False),
    "SL":  dict(tikslas_atr=None, trail=True),    # slenkantis, be tikslo
}
# Cia BUVO S1_MAX_NUO_UZD_ATR = 0.50 ("nesivyti nubegusios"). Isimta: savitikra
# parode, kad ji blokuoja SAP 07-24 - viena is dvieju etalonu - nes SAP atidare
# 0.85 ATR virs uzdarymo. Ta pati darba jau daro R:R patikra, kuri isvedama is
# aritmetikos. Dvi ribos tam paciam dalykui yra butent tai, kas nuzude v3.
S1_MIN_APYV_SANTYKIS = 1.2  # atsiemimas turi vykti su apyvarta

# --- scenarijus 2: naujienu tarpas ir eiga ---
S2_MIN_TARPAS_ATR = 1.0
S2_MIN_APYVARTA_X = 2.0     # pirmu 30 min apyvarta
S2_ORB_BARU = 6             # atidarymo diapazonas = 30 min
S2_TRAIL_ATR = 1.0          # slenkantis stop'as ATR vienetais (ne 3 barai!)

RINKOS = {
    "eu": dict(zyme="EU", indeksas="EXSA.DE", tz="Europe/Berlin",
               atidarymas=9 * 60, uzdarymas=17 * 60 + 30),
    "us": dict(zyme="JAV", indeksas="SPY", tz="America/New_York",
               atidarymas=9 * 60 + 30, uzdarymas=16 * 60),
}

try:
    import yfinance as yf
except ImportError:
    yf = None


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
        sys.exit(f"NUTRAUKTA: is universas.py gauta tik {len(t)} tikeriu.")
    print(f"  {RINKOS[rinka]['zyme']}: {saltinis}, {len(t)} tikeriu")
    return t


# ================================================================ duomenys

def _vienas(df, t):
    """yfinance MultiIndex gali buti (laukas, tikeris) arba (tikeris, laukas)."""
    if df is None or len(df) == 0:
        return None
    d = df
    if isinstance(df.columns, pd.MultiIndex):
        l0 = set(df.columns.get_level_values(0))
        l1 = set(df.columns.get_level_values(1))
        if t in l0:
            d = df[t]
        elif t in l1:
            d = df.xs(t, axis=1, level=1)
        elif len(l1) == 1:
            d = df.droplevel(1, axis=1)
        elif len(l0) == 1:
            d = df.droplevel(0, axis=1)
        else:
            return None
    if not {"Open", "High", "Low", "Close"} <= set(d.columns):
        return None
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    return d if len(d) else None


def dienos_rodikliai(d, dividendai=None):
    """Dienos rodikliai, PASLINKTI per viena diena.

    Grazinamoje lenteleje eilute su data D turi tik informacija IKI D imtinai
    ATEMUS ta pacia diena. Tai vienas veiksmas (`shift(1)`) vietoj desimties
    atskiru "ar cia neziuriu i ateiti" patikrinimu, ir butent to v3 trūko.
    """
    if len(d) < 60:
        return None
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr_abs = tr.rolling(20).mean()
    r = pd.DataFrame(index=d.index)
    r["uzdarymas"] = c
    r["atr_abs"] = atr_abs
    r["atr_pct"] = atr_abs / c * 100.0
    r["apyvarta"] = (c * v).rolling(20).median()
    # kur uzdare dienos diapazone: 0 = ties dugnu, 1 = ties virsune
    diap = (h - l).replace(0, np.nan)
    r["uzd_vieta"] = ((c - l) / diap).clip(0, 1)
    # virsune ir dugnas per S1 langa (ir ta pacia diena imtinai)
    r["virsune_n"] = c.rolling(S1_LANGAS_D).max()
    r["dugnas_n"] = l.rolling(S1_LANGAS_D).min()
    # kiek dividendu iskrito per langa - kritimas tiek mechaninis
    if dividendai is not None and len(dividendai):
        dv = dividendai.reindex(d.index).fillna(0.0)
        r["div_lange"] = dv.rolling(S1_LANGAS_D).sum()
    else:
        r["div_lange"] = 0.0
    return r.shift(1)          # <<< viena eilute, panaikinanti visa klase klaidu


def dividendu_serija(t):
    try:
        s = yf.Ticker(t).dividends
        if s is None or not len(s):
            return None
        s.index = [x.tz_localize(None) if getattr(x, "tz", None) else x
                   for x in s.index]
        return s
    except Exception:
        return None


def dividendu_kalendorius(tickers):
    """Tik INFORMACIJAI kortelėje. Ne filtras."""
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
        s = dividendu_serija(t)
        if s is None or len(s) < 2:
            continue
        try:
            datos = list(s.index)[-12:]
            tarpai = [(datos[i + 1] - datos[i]).days for i in range(len(datos) - 1)]
            tarpai = [x for x in tarpai if 20 <= x <= 400]
            if not tarpai:
                continue
            kita = datos[-1] + timedelta(days=float(np.median(tarpai)))
            while kita < datetime.now():
                kita += timedelta(days=float(np.median(tarpai)))
            rez[t] = dict(kita_ex=kita.date().isoformat(), suma=float(s.iloc[-1]),
                          dienu_iki=(kita.date() - datetime.now().date()).days)
        except Exception:
            continue
    try:
        with open(DIVIDENDU_TALPYKLA, "w", encoding="utf-8") as f:
            json.dump(rez, f, ensure_ascii=False, indent=1)
    except Exception:
        pass
    return rez


def sesijos_rodikliai(barai, rinka):
    d = barai.copy()
    try:
        idx = d.index.tz_convert(RINKOS[rinka]["tz"])
    except Exception as e:
        # Anksciau cia buvo tylus nukritimas i UTC: "minute" tapdavo UTC
        # minutemis, ir pjuvis "pagal iejimo valanda" rodydavo ne ta laika.
        print(f"  DEMESIO: laiko juostos konversija nepavyko ({e}) - "
              f"minutes bus UTC, ne {RINKOS[rinka]['tz']}")
        idx = d.index
    d["sesija"] = [x.date() for x in idx]
    d["minute"] = [x.hour * 60 + x.minute for x in idx]
    d["_pv"] = ((d["High"] + d["Low"] + d["Close"]) / 3.0) * d["Volume"]
    g = d.groupby("sesija", sort=False)
    d["vwap"] = g["_pv"].cumsum() / g["Volume"].cumsum().replace(0, np.nan)
    d["ses_atidarymas"] = g["Open"].transform("first")
    d["apyv_tipine"] = d.groupby("minute")["Volume"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).median())
    d["apyv_santykis"] = d["Volume"] / d["apyv_tipine"].replace(0, np.nan)
    d.drop(columns=["_pv"], inplace=True)
    return d


# ================================================ scenarijai (grynos funkcijos)

def _bendra(sig, kd):
    """1 PAKOPOS patikros: isvestos is mechanikos ir aritmetikos.

    2 pakopos (kalibracijos uzsidirbtu) ribu cia kol kas nera ne vienos -
    jos atsiras tik tada, kai ataskaitos pjuviai parodys, kuri riba laikosi
    ir nematytoje puseje, ir abiejose rinkose.
    """
    kl = []
    if kd["apyvarta"] < MIN_APYVARTA:
        kl.append(f"apyvarta {kd['apyvarta']/1e6:.1f} mln < "
                  f"{MIN_APYVARTA/1e6:.1f}")
    # rizika eurais rodoma kortelėje - sprendzia Mantas, kodas neblokuoja
    sig["rizika_pct"] = (sig["ieina"] - sig["stop"]) / sig["ieina"] * 100.0
    sig["rizika_eur"] = sig["rizika_pct"] / 100.0 * POZICIJA
    if sig.get("tikslas"):
        judesys = (sig["tikslas"] - sig["ieina"]) / sig["ieina"] * 100.0
        if judesys < MIN_JUDESYS_PCT:
            kl.append(f"judesys {judesys:.2f}% < {MIN_JUDESYS_PCT:.2f}%")
        rr = (sig["tikslas"] - sig["ieina"]) / max(1e-9, sig["ieina"] - sig["stop"])
        sig["rr"] = rr
        if rr < MIN_RR - 1e-9:
            kl.append(f"R:R {rr:.2f} < {MIN_RR}")
    return kl


def scenarijus_1(langas, kd):
    """Kritimas 1-5 sesijas + vakarykscio uzdarymo atsiemimas.

    Krintancio peilio apsauga yra pacioje salygoje: kol kaina neatsieme
    vakarykscio uzdarymo, signalo nera. SAP 07-23 (kritimo diena) signalo
    NEDUODA; SAP 07-24 (apsisukimo diena) duoda.
    """
    if len(langas) < S2_ORB_BARU + 2:
        return None
    atr = kd["atr_abs"]
    if not (atr > 0):
        return None
    vakar_uzd = kd["uzdarymas"]
    kaina = float(langas["Close"].iloc[-1])

    # --- vakarykste busena (tik is kd, t.y. tik is praeities) ---
    kritimas = kd["virsune_n"] - vakar_uzd - kd["div_lange"]   # be dividendu
    if kritimas < S1_MIN_KRITIMAS_ATR * atr:
        return None
    if kd["uzd_vieta"] > S1_MAX_UZD_VIETA:
        return None

    # --- siandienos apsisukimas ---
    # Tarpo dienos priklauso 2 scenarijui. Viena riba, uzrasyta viena karta,
    # uztikrina, kad scenarijai nepersidengia (v3 ju persidengimas buvo
    # patikrintas skaiciais: tas pats atsokimas duodavo dvi korteles).
    atid = float(langas["ses_atidarymas"].iloc[-1])
    if atid - vakar_uzd >= S2_MIN_TARPAS_ATR * atr:
        return None
    if kaina <= vakar_uzd:
        return None                       # dar neatsieme - peilis
    orb = langas.iloc[:S2_ORB_BARU]
    if kaina <= float(orb["High"].max()):
        return None                       # turi virsyti atidarymo diapazona
    sant = float(langas["apyv_santykis"].iloc[-1])
    if not (np.isfinite(sant) and sant >= S1_MIN_APYV_SANTYKIS):
        return None
    vwap = float(langas["vwap"].iloc[-1])
    if not (np.isfinite(vwap) and kaina > vwap):
        return None

    # STOP'AS turi dvi dedamasias, ir imama AUKSTESNE:
    #   a) struktūrinė: po atsiimtu lygiu (L - 0.3 ATR). Jei kaina krenta
    #      atgal po vakarykscio uzdarymo, apsisukimas neivyko.
    #   b) rizikos limitas: ne daugiau 0.5 ATR nuo iejimo.
    # Pusėje atveju laimi (b), t.y. stop'as atsiduria VIRS atsiimto lygio -
    # pozicija uzdaroma dar galiojant apsisukimo prielaidai. Tai samoningas
    # mainas: be jo konservatyvus 0.5 ATR tikslas duotu R:R ~0.43 (SAP
    # etalone) ir signalas uzsiblokuotu pats.
    # 5 dienu dugnas (kd["dugnas_n"]) butu 3-5 ATR zemiau ir netiktu niekaip.
    L = min(float(langas["Low"].min()), vakar_uzd)
    stop = max(L - S1_STOP_ATR * atr, kaina - S1_MAX_RIZIKA_ATR * atr)

    # TIKSLAS PAGAL PASIEKIAMUMA (Manto 2 pastaba 09-27):
    # "Reikia vertinti koks kritimas buvo, ar akcija gali tiek atsistatyti ...
    #  ir ar turi pakankamai judesio bei apyvartos tokiam atsigavimui."
    # ATR yra tipinis DIENOS diapazonas, tad per HORIZONTAS_SESIJU tipiskai
    # nueinama apie tiek ATR. Jei buves lygis toliau - tikslas ribojamas iki
    # to, kas telpa. Nauju parametru tai neprideda: naudojamas tas pats ATR
    # ir tas pats horizontas. Jei ir po ribojimo R:R < 1 - signalo nera,
    # ir tai padaro _bendra().
    R_pilnas = float(kd["virsune_n"])
    if R_pilnas <= kaina:
        return None      # buves lygis jau pasiektas - tikslo nebera. Tai ne
                         # filtras, o aritmetika: kitaip kortele rodytu tiksla
                         # ZEMIAU iejimo ir neigiama R:R.
    R = min(R_pilnas, kaina + S1_TIKSLAS_ATR * atr)
    sig = dict(scenarijus="Atsistatymas", tipas=1, ieina=kaina,
               tikslas=R, stop=stop, R=R, L=L, atr_abs=atr,
               progresas=(kaina - L) / (R - L) if R > L else 0.0,
               progresas_tikslus=True, kritimas_atr=kritimas / atr,
               atsiemimas_atr=(kaina - vakar_uzd) / atr,
               # TIK puslapiui. I "kliutys" NEdedama tycia: kitaip pasikeistu
               # "tinkamas", o su juo - kalibracijos imtis, ir nebegaletume
               # patikrinti, ar slepti buvo teisinga.
               silpnas_atsiemimas=bool((kaina - vakar_uzd) / atr < 0.25),
               R_pilnas=R_pilnas, tikslas_ribotas=bool(R < R_pilnas - 1e-9),
               kelias_atr=(R_pilnas - kaina) / atr)
    sig["kliutys"] = _bendra(sig, kd)
    return sig


def scenarijus_2(langas, kd, iki_uzdarymo=None):
    """Naujienu tarpas + eiga. Tikslo nera, nesama slenkanciu stop'u."""
    if len(langas) < S2_ORB_BARU + 2:
        return None
    atr = kd["atr_abs"]
    if not (atr > 0):
        return None
    vakar_uzd = kd["uzdarymas"]
    atid = float(langas["ses_atidarymas"].iloc[-1])
    kaina = float(langas["Close"].iloc[-1])

    tarpas = atid - vakar_uzd
    if tarpas < S2_MIN_TARPAS_ATR * atr:
        return None
    orb = langas.iloc[:S2_ORB_BARU]
    sant = float(pd.to_numeric(orb["apyv_santykis"], errors="coerce").median())
    if not (np.isfinite(sant) and sant >= S2_MIN_APYVARTA_X):
        return None
    orb_max, orb_min = float(orb["High"].max()), float(orb["Low"].min())
    if kaina <= orb_max:
        return None                       # laukiam atidarymo diapazono proverzio
    vwap = float(langas["vwap"].iloc[-1])
    if not (np.isfinite(vwap) and kaina > vwap):
        return None
    if float(langas["Low"].min()) < vakar_uzd:
        return None                       # tarpas jau buvo uzpildytas

    stop = max(orb_min, kaina - S2_TRAIL_ATR * atr)
    # PROGRESAS. Tikslo nera, tad tai vertinimas is dvieju dedamuju, imant
    # DIDESNE - kad juosta klystu i "veliau, nei manai" puse.
    #   a) judesys nuo PROVERZIO lygio (orb_max), ne nuo sesijos atidarymo.
    #      Nuo atidarymo buvo klaida: tarpo dienos atidarymas jau yra 1+ ATR
    #      zemiau kainos, tad juosta is karto rodydavo 100% ir visos kortelės
    #      kristu i "velyva stadija".
    #   b) paros laikas: 10:00 prasidejes ralis turi visa diena, 16:00 - nebe.
    prog_j = (kaina - orb_max) / max(1e-9, atr)
    prog_l = 0.0
    if iki_uzdarymo is not None:
        viso = RINKOS["eu"]["uzdarymas"] - RINKOS["eu"]["atidarymas"]
        prog_l = max(0.0, 1.0 - float(iki_uzdarymo) / max(1.0, viso))
    sig = dict(scenarijus="Ralis", tipas=2, ieina=kaina,
               tikslas=None, stop=stop, R=None, L=orb_min, atr_abs=atr,
               progresas=min(1.0, max(prog_j, prog_l)),
               progresas_tikslus=False, tarpas_atr=tarpas / atr,
               virsune_vertinimas=kaina + S2_TRAIL_ATR * atr)
    sig["kliutys"] = _bendra(sig, kd)
    return sig


_KLAIDOS = {}


def aptikti(langas, kd, iki_uzdarymo=None):
    """Klaidos nebetylimos: anksciau `except: s = None` prarydavo VISKA, tad
    toks gedimas kaip visur-NaN apyv_santykis atrodydavo kaip "signalu nera"."""
    rez = []
    for f in (scenarijus_1, scenarijus_2):
        try:
            s = (f(langas, kd) if f is scenarijus_1
                 else f(langas, kd, iki_uzdarymo))
        except Exception as e:
            raktas = f"{f.__name__}: {type(e).__name__}: {e}"
            _KLAIDOS[raktas] = _KLAIDOS.get(raktas, 0) + 1
            s = None
        if s:
            s["tinkamas"] = len(s["kliutys"]) == 0
            rez.append(s)
    return rez


# ================================================================ baigtis

def baigtis(toliau, sig):
    """Baigtis per HORIZONTAS_SESIJU, nesant pozicija per nakti.

    v3 viska uzdarydavo iki skambucio. Atveju tyrimas parode, kad 39%
    judesio susidaro per nakti (PTX atveju 63%), tad uzdarymas ties
    skambuciu matavo ne ta sandori.

    Grazina ir MFE - kiek daugiausia buvo naudai. v3 turejo tik MAE, todel
    lenteleje matesi desimtys euru ten, kur realus sandoris duoda simtus.
    """
    ieina, tikslas, stop = sig["ieina"], sig.get("tikslas"), sig["stop"]
    atr = sig["atr_abs"]
    if len(toliau) == 0:
        # Buvo atskira baigtis "nespejo", del kurios trys ataskaitos procentai
        # nebesusidedavo i 100. Tai ta pati "laikas baigesi" baigtis.
        return dict(baigtis="laikas", pelnas_pct=0.0, minuciu=0,
                    mfe_pct=0.0, mae_pct=0.0, mfe_min=0, laikas_baige=True,
                    isejo_bare=1)

    auksciausia, mfe, mae, mfe_i = ieina, 0.0, 0.0, 0
    for j, (_, b) in enumerate(toliau.iterrows()):
        lo, hi, op = float(b["Low"]), float(b["High"]), float(b["Open"])

        # 1) ATIDARYMAS ivyksta pirmas - pries viska kita bare. Pozicija
        # nesama per nakti, tad baras gali atidaryti jau anapus stop'o arba
        # anapus tikslo, ir tada vykdymas yra ties atidarymu. Be sios tvarkos
        # baras O=115 (virs tikslo 110) su veliau L=95 buvo uzskaitomas kaip
        # stop -2%, nors pozicijos tuo metu jau seniai nebuvo.
        if op <= stop:
            return dict(baigtis="stop", pelnas_pct=(op / ieina - 1) * 100,
                        minuciu=(j + 1) * BARAS_MIN, mfe_pct=mfe, mae_pct=mae,
                        mfe_min=mfe_i * BARAS_MIN, laikas_baige=False,
                        isejo_bare=j + 1)
        if tikslas and op >= tikslas:
            return dict(baigtis="tikslas", pelnas_pct=(op / ieina - 1) * 100,
                        minuciu=(j + 1) * BARAS_MIN, mfe_pct=mfe, mae_pct=mae,
                        mfe_min=mfe_i * BARAS_MIN, laikas_baige=False,
                        isejo_bare=j + 1)

        if (lo / ieina - 1) * 100 < mae:
            mae = (lo / ieina - 1) * 100
        if (hi / ieina - 1) * 100 > mfe:
            mfe, mfe_i = (hi / ieina - 1) * 100, j + 1

        # stop'as tikrinamas PIRMAS: kai baro diapazonas apima ir stop'a, ir
        # tiksla, laikom, kad issimuse. Konservatyvu ir tycia.
        #
        # TARPAS: pozicija nesama per nakti, tad baras gali ATIDARYTI gerokai
        # zemiau stop'o. Tada vykdymas yra ties atidarymu, ne ties stop'u.
        # Be sito kodas blogiausius sandorius vertino per gerai: O=90 prie
        # stop'o 98 buvo uzskaitoma kaip -2.00%, nors realiai -10.00%
        # (18 000 EUR pozicijai tai 1 440 EUR paklaida vienam sandoriui).
        if lo <= stop:
            return dict(baigtis="stop", pelnas_pct=(stop / ieina - 1) * 100,
                        minuciu=(j + 1) * BARAS_MIN, mfe_pct=mfe, mae_pct=mae,
                        mfe_min=mfe_i * BARAS_MIN, laikas_baige=False,
                        isejo_bare=j + 1)
        if tikslas and hi >= tikslas:
            return dict(baigtis="tikslas", pelnas_pct=(tikslas / ieina - 1) * 100,
                        minuciu=(j + 1) * BARAS_MIN, mfe_pct=mfe, mae_pct=mae,
                        mfe_min=mfe_i * BARAS_MIN, laikas_baige=False,
                        isejo_bare=j + 1)
        # Slenkantis stop'as TIK 2 scenarijui: jis neturi tikslo, ir kortele
        # ji deklaruoja. 1 scenarijus kortelėje rodo FIKSUOTA stop'a, ir pagal
        # ji skaiciuojamas R:R bei rizika eurais - tad jo slinkimas reikstu,
        # kad matuojam ne ta taisykle, kuria rodom. (Patikrinta: su slinkimu
        # 13% baigciu skirdavosi, ir jos patekdavo i "stop" stulpeli.)
        if sig.get("tipas") == 2 and hi > auksciausia:
            auksciausia = hi
            stop = max(stop, auksciausia - S2_TRAIL_ATR * atr)

    gal = float(toliau["Close"].iloc[-1])
    return dict(baigtis="laikas", pelnas_pct=(gal / ieina - 1) * 100,
                minuciu=len(toliau) * BARAS_MIN, mfe_pct=mfe, mae_pct=mae,
                mfe_min=mfe_i * BARAS_MIN, laikas_baige=True,
                isejo_bare=len(toliau))


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


# ================================================================ duomenu siuntimas

def parsisiusti(rinka, dienos):
    tick = universas(rinka)
    # actions=True atsiunčia ir Dividends stulpeli TOJE PACIOJE uzklausoje -
    # be jo dividendu pataisa buvo negyvas kodas (div_lange visada 0.0), nes
    # parsisiusti() kviete dienos_rodikliai(d) be dividendu argumento.
    dien = yf.download(tick, period="2y", interval="1d", auto_adjust=False,
                       progress=False, group_by="ticker", threads=True,
                       actions=True)
    intr = yf.download(tick, period=f"{min(dienos, 60)}d", interval="5m",
                       auto_adjust=False, progress=False, group_by="ticker",
                       threads=True, prepost=False)
    rod, barai = {}, {}
    for t in tick:
        d = _vienas(dien, t)
        if d is None or len(d) < 60:
            continue
        r = dienos_rodikliai(d, d["Dividends"] if "Dividends" in d else None)
        if r is None:
            continue
        b = _vienas(intr, t)
        if b is None or len(b) < 100:
            continue
        rod[t] = r
        barai[t] = sesijos_rodikliai(b, rinka)
    print(f"  akciju su duomenimis: {len(barai)}")

    # Garsi patikra: jei apyv_santykis visur NaN, scenarijai negali suveikti
    # is principo, ir tyliai gautume tuscia puslapi. 2026-09-27 butent taip
    # ir buvo live rezime.
    if barai:
        nan = [b["apyv_santykis"].isna().all() for b in barai.values()]
        if all(nan):
            sys.exit(f"NUTRAUKTA: apyv_santykis visur NaN ({dienos} d. per mazai "
                     f"apyv_tipine skaiciavimui - reikia bent 6, realiai 30).")
        if sum(nan) > len(nan) * 0.5:
            print(f"  DEMESIO: {sum(nan)}/{len(nan)} akciju apyv_santykis visur NaN")
    return rod, barai


def _kd(rod_t, ses):
    """Tos sesijos dienos kontekstas. Lentele jau paslinkta, tad ateities nera."""
    try:
        eil = rod_t.loc[[x for x in rod_t.index if x.date() == ses]]
        if not len(eil):
            return None
        r = eil.iloc[0]
        # Visi laukai, kuriais remiasi scenarijai ir 1 pakopos filtrai, turi
        # buti skaiciai. Anksciau buvo tikrinami tik atr_abs ir virsune_n, tad
        # NaN apyvarta TYLIAI praeidavo pro likvidumo riba (vienintele tikra
        # 1 pakopos salyga), o NaN uzd_vieta - pro kapituliacijos salyga.
        for f in ("atr_abs", "virsune_n", "dugnas_n", "uzdarymas",
                  "apyvarta", "uzd_vieta"):
            if not np.isfinite(r[f]):
                return None
        return r
    except Exception:
        return None



# ================================================================ puslapis

PUSLAPIO_SABLONAS = r'''<!doctype html>
<html lang="lt">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>Detektorius</title>
<style>
:root{
  --bg:#f6f7f9; --card:#fff; --line:#e3e6ea; --txt:#14171a; --dim:#697583;
  --ok:#1f9d55; --ok-bg:#eaf7ef; --blok:#b7791f; --blok-bg:#fdf6e7;
  --velyva:#8a93a0; --juosta:#e8ebef; --acc:#2d6cdf;
  --eu:#3b5bdb; --us:#0b7285;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0f1216; --card:#171b21; --line:#262c35; --txt:#e6e9ed; --dim:#95a0ae;
  --ok:#3ddc84; --ok-bg:#12281c; --blok:#e0b341; --blok-bg:#2a2313;
  --velyva:#6b7482; --juosta:#232932; --acc:#6ea8fe;
  --eu:#7a90f0; --us:#4db8c9;
}}
:root[data-theme="dark"]{
  --bg:#0f1216; --card:#171b21; --line:#262c35; --txt:#e6e9ed; --dim:#95a0ae;
  --ok:#3ddc84; --ok-bg:#12281c; --blok:#e0b341; --blok-bg:#2a2313;
  --velyva:#6b7482; --juosta:#232932; --acc:#6ea8fe;
  --eu:#7a90f0; --us:#4db8c9;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
  font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px 64px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;
  justify-content:space-between;margin-bottom:6px}
h1{font-size:20px;margin:0;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13px}
.zurnalas[hidden]{display:none}
.zurnalas{display:flex;flex-wrap:wrap;gap:18px;margin:14px 0 18px;padding:12px 14px;
  background:var(--card);border:1px solid var(--line);border-radius:10px}
.z div{font-size:12px;color:var(--dim)}
.z b{display:block;font-size:17px;color:var(--txt);font-variant-numeric:tabular-nums}
.valdymas{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
button.f{background:var(--card);border:1px solid var(--line);color:var(--dim);
  padding:6px 12px;border-radius:999px;font-size:13px;cursor:pointer}
button.f[aria-pressed="true"]{border-color:var(--acc);color:var(--acc);font-weight:600}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);
  margin:26px 0 10px;font-weight:600}
.tinkl{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(330px,1fr))}
.k{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--ok);
  border-radius:10px;padding:13px 14px}
.k.blok{border-left-color:var(--blok)}
.k.velyva{opacity:.72;border-left-color:var(--blok)}
.vir{display:flex;align-items:baseline;gap:8px;margin-bottom:2px}
.tick{font-weight:700;font-size:16px;letter-spacing:-.01em}
.zenk{font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px;
  border:1px solid currentColor;letter-spacing:.04em}
.zenk.EU{color:var(--eu)} .zenk.JAV{color:var(--us)}
.scen{color:var(--dim);font-size:12.5px;margin-bottom:10px}
.juosta{height:8px;background:var(--juosta);border-radius:5px;overflow:hidden;margin:2px 0 4px}
.uzp{height:100%;background:var(--ok);border-radius:5px}
.k.blok .uzp{background:var(--blok)}
.k.velyva .uzp{background:var(--blok)}
.uzp.vert{background:repeating-linear-gradient(90deg,var(--ok) 0 6px,transparent 6px 10px)}
.k.blok .uzp.vert{background:repeating-linear-gradient(90deg,var(--blok) 0 6px,transparent 6px 10px)}
.proc{display:flex;justify-content:space-between;font-size:11.5px;color:var(--dim);
  margin-bottom:10px;font-variant-numeric:tabular-nums}
.kainos{display:flex;justify-content:space-between;gap:6px;font-size:12.5px;
  font-variant-numeric:tabular-nums;padding:8px 0;border-top:1px solid var(--line)}
.kainos span{color:var(--dim);display:block;font-size:10.5px;text-transform:uppercase;
  letter-spacing:.04em}
.kainos b{font-weight:600}
.meta{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.z2{font-size:11px;padding:2px 7px;border-radius:999px;background:var(--juosta);
  color:var(--dim);font-variant-numeric:tabular-nums}
.z2.geras{background:var(--ok-bg);color:var(--ok);font-weight:600}
.z2.kliutis{background:var(--blok-bg);color:var(--blok);font-weight:600}
.tuscia{color:var(--dim);padding:28px 4px;font-size:14px}
footer{margin-top:34px;color:var(--dim);font-size:12px;line-height:1.6;
  border-top:1px solid var(--line);padding-top:14px}
@media (max-width:520px){.wrap{padding:16px 16px 48px}.tinkl{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>Detektorius</h1>
    <div class="sub" id="antraste">kraunama…</div>
  </div>
  <div class="sub" id="laikmatis"></div>
</header>

<div class="zurnalas z" id="zurnalas" hidden></div>

<div class="valdymas">
  <button class="f" id="f-visi" aria-pressed="false">Visi</button>
  <button class="f" id="f-tinkami" aria-pressed="true">Tik be kliūčių</button>
  <button class="f" id="f-eu" aria-pressed="true">EU</button>
  <button class="f" id="f-us" aria-pressed="true">JAV</button>
</div>

<div id="turinys"></div>

<footer>
  Pozicija 18&nbsp;000&nbsp;€, sąnaudos 10&nbsp;€ už ciklą (lūžio taškas 0,0556&nbsp;%).
  Horizontas 3 sesijos, pozicija nešama per naktį.<br>
  Žurnalas: <a href="zurnalas.csv">zurnalas.csv</a> · duomenys:
  <a href="detektorius.json">detektorius.json</a>
</footer>
</div>

<script>
const DUOM = __DUOM__;
const B = {visi:false, tinkami:true, EU:true, JAV:true};
let duom = DUOM;

const PAV = {
  "Atsistatymas":"Atsistatymas",
  "Ralis":"Ralis"
};
const silpnas = s => s.silpnas_atsiemimas === true;
const nr = (x,n=2)=> (x===null||x===undefined||x==='')?'–':Number(x).toFixed(n);
const esc = s => String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function amzius(m){
  m = Math.max(0, Math.round(m||0));
  if (m < 60) return m + ' min';
  return Math.floor(m/60) + ' val ' + (m%60) + ' min';
}

function kortele(s){
  const blok = !s.tinkamas;
  const p = Math.max(0, Math.min(1, Number(s.progresas)||0));
  const velyva = p >= 0.8;
  const tikslus = s.progresas_tikslus !== false;
  const cls = ['k', (blok || silpnas(s))?'blok':'', velyva?'velyva':''
              ].filter(Boolean).join(' ');
  const zenkl = [];
  if (Number.isFinite(Number(s.rr)) && s.rr)
    zenkl.push(`<span class="z2 ${Number(s.rr) >= 1 ? 'geras' : ''}">R:R ${nr(s.rr)}</span>`);
  if (s.rizika_eur) zenkl.push(`<span class="z2">rizika ${Math.round(s.rizika_eur)} €</span>`);
  if (s.atr_pct) zenkl.push(`<span class="z2">ATR ${nr(s.atr_pct,1)}%</span>`);
  if (s.kritimas_atr) zenkl.push(`<span class="z2">krito ${nr(s.kritimas_atr,1)} ATR</span>`);
  if (s.kelias_atr) zenkl.push(`<span class="z2 ${Number(s.kelias_atr) < 1 ? 'kliutis' : ''}">iki lygio ${nr(s.kelias_atr,1)} ATR</span>`);
  if (s.atsiemimas_atr !== undefined && s.atsiemimas_atr !== null)
    zenkl.push(`<span class="z2 ${Number(s.atsiemimas_atr) < 0.25 ? 'kliutis' : ''}">atsiėmė ${nr(s.atsiemimas_atr,2)} ATR</span>`);
  if (s.tarpas_atr) zenkl.push(`<span class="z2">tarpas ${nr(s.tarpas_atr,1)} ATR</span>`);
  zenkl.push(`<span class="z2">${amzius(s.amzius_min)}</span>`);
  if (s.tikslas_ribotas) zenkl.push(`<span class="z2">tikslas ribotas</span>`);
  if (s.dividendas && s.dividendas.dienu_iki !== undefined && Math.abs(s.dividendas.dienu_iki) <= 7)
    zenkl.push(`<span class="z2 kliutis">ex-div po ${s.dividendas.dienu_iki} d.</span>`);
  (s.kliutys||[]).forEach(k => zenkl.push(`<span class="z2 kliutis">${esc(k)}</span>`));
  return `<article class="${cls}">
    <div class="vir"><span class="tick">${esc(s.tickeris)}</span>
      <span class="zenk ${esc(s.rinka)}">${esc(s.rinka)}</span></div>
    <div class="scen">${esc(PAV[s.scenarijus] || s.scenarijus)}</div>
    <div class="juosta"><div class="uzp ${tikslus?'':'vert'}" style="width:${(p*100).toFixed(0)}%"></div></div>
    <div class="proc"><span>${(p*100).toFixed(0)}%${tikslus?'':' · vertinimas'}</span>
      <span>${velyva?'vėlyva':''}</span></div>
    <div class="kainos">
      <div><span>stop</span><b>${nr(s.stop)}</b></div>
      <div><span>įeina</span><b>${nr(s.ieina)}</b></div>
      <div><span>tikslas</span><b>${s.tikslas ? nr(s.tikslas) : '—'}</b></div>
    </div>
    <div class="meta">${zenkl.join('')}</div>
  </article>`;
}

let rodomi = [], sig = [];
function piesti(){
  const el = document.getElementById('turinys');
  sig = (duom.signalai||[]).filter(s => B[s.rinka] !== false);
  sig.sort((a,b)=> (Number(b.progresas)||0) - (Number(a.progresas)||0));
  const velyva  = s => (Number(s.progresas)||0) >= 0.8;
  // Paslepiami: krintancio peilio kandidatai ir tie, kuriuos blokuoja kliutys.
  // Zurnalas juos vis tiek raso, tad veliau matysime, ar slepti buvo teisinga.
  const slepti   = sig.filter(s => silpnas(s) || !s.tinkamas);
  rodomi = sig.filter(s => !silpnas(s) && s.tinkamas);
  const ankstyvi = rodomi.filter(s => !velyva(s));
  const velyvi   = rodomi.filter(velyva);
  let h = '';
  if (!sig.length){
    h = '<div class="tuscia">Šiuo metu nė vieno scenarijaus, atitinkančio pasirinkimą.' +
        '<br>Puslapis persikrauna kas 5 min.</div>';
  } else {
    if (ankstyvi.length) h += `<h2>Aktyvūs · ${ankstyvi.length}</h2>
      <div class="tinkl">${ankstyvi.map(kortele).join('')}</div>`;
    if (velyvi.length) h += `<h2>Išsikvėpę · per vėlu šokti · ${velyvi.length}</h2>
      <div class="tinkl">${velyvi.map(kortele).join('')}</div>`;
    if (slepti.length && B.visi) h += `<h2>Silpnas atsiėmimas ir blokuoti · ${slepti.length}</h2>
      <div class="tinkl">${slepti.map(kortele).join('')}</div>`;
    if (slepti.length && !B.visi) h += `<div class="tuscia">Paslėpta ${slepti.length}: `
      + `krintančio peilio kandidatai (atsiėmė &lt; 0,25 ATR) ir blokuoti. `
      + `Spausk „Visi", jei nori juos matyti.</div>`;
    if (!ankstyvi.length && !velyvi.length)
      h = '<div class="tuscia">Nė vieno švaraus scenarijaus.</div>' + h;
  }
  el.innerHTML = h;
  const t = duom.atnaujinta ? new Date(duom.atnaujinta) : null;
  document.getElementById('antraste').textContent =
    (t ? 'atnaujinta ' + t.toLocaleTimeString('lt-LT',{hour:'2-digit',minute:'2-digit'}) : '') +
    ` · rodoma ${rodomi.length} iš ${sig.length}`;
}

function zurnalas(z){
  const el = document.getElementById('zurnalas');
  if (!z || !z.signalu){ el.hidden = true; return; }
  el.innerHTML = `
    <div><span>signalų</span><b>${z.signalu}</b></div>
    <div><span>atvirų</span><b>${z.atviru}</b></div>
    <div><span>baigtų</span><b>${z.baigtu}</b></div>
    <div><span>tikslas</span><b>${z.baigtu ? Math.round(z.tikslo_dalis)+'%' : '–'}</b></div>
    <div><span>vid. rezultatas</span><b>${z.vid_eur===null?'–':(z.vid_eur>=0?'+':'')+Math.round(z.vid_eur)+' €'}</b></div>`;
  el.hidden = false;
}

for (const [id,key] of [['f-visi','visi'],['f-tinkami','tinkami'],['f-eu','EU'],['f-us','JAV']]){
  const b = document.getElementById(id);
  b.addEventListener('click', ()=>{
    if (key === 'visi' || key === 'tinkami'){
      B.visi = (key === 'visi');
      document.getElementById('f-visi').setAttribute('aria-pressed', B.visi);
      document.getElementById('f-tinkami').setAttribute('aria-pressed', !B.visi);
    } else { B[key] = !B[key]; b.setAttribute('aria-pressed', B[key]); }
    piesti();
  });
}

let liko = 300;
setInterval(()=>{
  liko--;
  document.getElementById('laikmatis').textContent =
    'atnaujinimas po ' + Math.floor(liko/60) + ':' + String(Math.max(0,liko)%60).padStart(2,'0');
}, 1000);

piesti();
zurnalas(DUOM.zurnalas);
</script>
</body>
</html>
'''


def zurnalo_santrauka(z):
    uzd = [r for r in z.values() if str(r.get("baigtis") or "")]
    eur = []
    for r in uzd:
        try:
            eur.append(float(r["eur"]))
        except Exception:
            pass
    return dict(signalu=len(z), atviru=len(z) - len(uzd), baigtu=len(uzd),
                tikslo_dalis=(100.0 * sum(1 for r in uzd
                                          if r.get("baigtis") == "tikslas") / len(uzd))
                if uzd else 0.0,
                vid_eur=(sum(eur) / len(eur)) if eur else None)


def puslapis_html(eilutes, z):
    """Puslapi generuoja PATS detektorius, kaip ir senasis dip_reitingas.py.

    Duomenys ikepami i faila, ne siunciami fetch'u: GitHub Pages atiduoda
    viena statini faila, be papildomu uzklausu ir be talpyklos netikrumo.
    Atnaujinimas - meta refresh kas 5 min, t. y. tiksliai tuo ritmu, kuriuo
    workflow perrasO faila.
    """
    duom = dict(atnaujinta=datetime.now(timezone.utc).isoformat(),
                signalai=eilutes, zurnalas=zurnalo_santrauka(z))
    return PUSLAPIO_SABLONAS.replace(
        "__DUOM__", json.dumps(duom, ensure_ascii=False, default=str))

# ================================================================ live

def isejimo_variantai(toliau, sig):
    """Visi ISEJIMO_VARIANTAI ant to paties signalo ir tu paciu baru."""
    out = {}
    atr = sig["atr_abs"]
    for vardas, v in ISEJIMO_VARIANTAI.items():
        s2 = dict(sig)
        if v["trail"]:
            s2["tipas"] = 2          # tipas 2 ijungia slenkanti stop'a
            s2["tikslas"] = None
            s2["stop"] = sig["L"] - S1_STOP_ATR * atr   # platus, struktūrinis
        else:
            s2["tipas"] = 1
            s2["tikslas"] = min(float(sig["R_pilnas"]),
                                sig["ieina"] + v["tikslas_atr"] * atr)
        b = baigtis(toliau, s2)
        out[vardas] = dict(baigtis=b["baigtis"], pelnas_pct=b["pelnas_pct"],
                           minuciu=b["minuciu"])
    return out


def zurnalas_ikelti():
    if not os.path.exists(ZURNALAS):
        return {}
    try:
        df = pd.read_csv(ZURNALAS, dtype=str, keep_default_na=False)
        return {str(r["raktas"]): dict(r) for _, r in df.iterrows()}
    except Exception as e:
        print(f"  zurnalo nuskaityti nepavyko ({e}) - pradedamas naujas")
        return {}


def barai_signalams(z, eilutes, visi_barai):
    """Kiekvienam ATVIRAM zurnalo signalui - barai NUO jo gimimo iki dabar.

    Tai ta pati lentele, kuria kalibracijoje gauna baigtis(): bareliai po
    iejimo baro. Todel zurnalas ir kalibracija matuoja identiskai.
    """
    gimimai = {f"{e['rinka']}|{e['tickeris']}|{e['tipas']}|{e['sesija_data']}":
               e.get("pirmas_kartas", e["laikas"]) for e in eilutes}
    out = {}
    for raktas, r in list(z.items()) + [(k, dict(gimimas=v, baigtis=""))
                                        for k, v in gimimai.items()]:
        if raktas in out or str(r.get("baigtis") or ""):
            continue
        dalys = raktas.split("|")
        if len(dalys) != 4:
            continue
        d = visi_barai.get((dalys[0], dalys[1]))
        if d is None:
            continue
        try:
            nuo = pd.Timestamp(str(r["gimimas"]))
            po = d[d.index > nuo]
        except Exception:
            continue
        if len(po):
            out[raktas] = po
    return out


def zurnalas_atnaujinti(eilutes, barai_pagal_rakta):
    """Pirmyneiginis testas: viena eilute vienam signalui, atnaujinama kas 5 min.

    Baigtis skaiciuojama TA PACIA baigtis() funkcija, kaip ir kalibracijoje.
    Jei zurnalas turetu savo isejimo logika, gautume tiksliai ta klaida, kuri
    sitame projekte kartojosi keturis kartus: modulis skaiciuoja viena, testas
    kita. Todel cia nera NE VIENOS savos taisykles.
    """
    z = zurnalas_ikelti()
    dabar = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for e in eilutes:
        raktas = f"{e['rinka']}|{e['tickeris']}|{e['tipas']}|{e['sesija_data']}"
        if raktas not in z:
            z[raktas] = dict(
                raktas=raktas, gimimas=e["laikas"], rinka=e["rinka"],
                tickeris=e["tickeris"], scenarijus=e["scenarijus"],
                tipas=e["tipas"], sesija=e["sesija_data"],
                ieina=round(e["ieina"], 4), tikslas=e.get("tikslas"),
                stop=round(e["stop"], 4), rr=(round(e["rr"], 2) if np.isfinite(e.get("rr", np.nan)) else ""),
                rizika_eur=round(e.get("rizika_eur", float("nan")), 1),
                atr_pct=round(e.get("atr_pct", float("nan")), 2),
                atr_abs=round(float(e["atr_abs"]), 6),
                tinkamas=bool(e["tinkamas"]),
                kliutys="; ".join(e["kliutys"]),
                paskut_laikas="", paskut_kaina="", mfe_pct="", mae_pct="",
                baigtis="", pelnas_pct="", eur="", minuciu="")

    # atviros eilutes: perskaiciuojam ta pacia baigtis() funkcija
    for raktas, r in z.items():
        if str(r.get("baigtis") or ""):
            continue
        toliau = barai_pagal_rakta.get(raktas)
        if toliau is None or not len(toliau):
            continue
        sig = dict(tipas=int(r["tipas"]), ieina=float(r["ieina"]),
                   tikslas=(float(r["tikslas"])
                            if str(r.get("tikslas") or "") not in ("", "nan")
                            else None),
                   stop=float(r["stop"]),
                   atr_abs=float(r["atr_abs"]))
        b = baigtis(toliau, sig)
        r["paskut_laikas"] = dabar
        r["paskut_kaina"] = round(float(toliau["Close"].iloc[-1]), 4)
        r["mfe_pct"] = round(b["mfe_pct"], 3)
        r["mae_pct"] = round(b["mae_pct"], 3)
        # uzdarom TIK tada, kai tikrai issisprende. "laikas" gyvai reiskia
        # tik tai, kad kol kas neissisprende - horizontas dar nesibaige.
        if b["baigtis"] in ("stop", "tikslas"):
            r["baigtis"] = b["baigtis"]
            r["pelnas_pct"] = round(b["pelnas_pct"], 3)
            r["eur"] = round(b["pelnas_pct"] / 100 * POZICIJA - SANAUDOS_EUR, 1)
            r["minuciu"] = b["minuciu"]
        else:
            try:
                nuo = datetime.fromisoformat(str(r["sesija"])).date()
                praejo = len(pd.bdate_range(nuo, datetime.now().date())) - 1
            except Exception:
                praejo = 0
            if praejo >= HORIZONTAS_SESIJU:
                r["baigtis"] = "laikas"
                r["pelnas_pct"] = round(b["pelnas_pct"], 3)
                r["eur"] = round(b["pelnas_pct"] / 100 * POZICIJA - SANAUDOS_EUR, 1)
                r["minuciu"] = b["minuciu"]

    os.makedirs(os.path.dirname(ZURNALAS), exist_ok=True)
    pd.DataFrame(list(z.values()), columns=ZURNALO_STULPELIAI).to_csv(
        ZURNALAS, index=False)
    atviros = sum(1 for r in z.values() if not str(r.get("baigtis") or ""))
    print(f"  zurnalas: {len(z)} eiluciu ({atviros} atviros)")
    return z


def atkurti(s, sena, kd, kaina):
    """Atstato signala is busenos ir PERSKAICIUOJA viska, kas nuo jos priklauso.

    Kortele turi rodyti viena nuosekliai suderinta trijule: ieina, stop,
    tikslas. Anksciau buvo atstatomi tik ieina ir stop, o tikslas likdavo
    perskaiciuotas nuo DABARTINES kainos (tikslas = min(R_pilnas, kaina+3*ATR)),
    todel rodomas R:R augdavo 2.60 -> 4.70 vien del to, kad kaina kyla, nors
    nei iejimas, nei stop'as nepasikeite. Ir dar anksciau _bendra() apskritai
    nebuvo perskaiciuojama, tad rizika eurais rodyta 1396 vietoj 476.
    """
    s["ieina"] = sena["ieina"]
    s["stop"] = sena.get("stop", s["stop"])
    if sena.get("R"):
        s["R"] = s["tikslas"] = sena["R"]
    s["pirmas_kartas"] = sena["laikas"]
    s["kliutys"] = _bendra(s, kd)
    s["tinkamas"] = len(s["kliutys"]) == 0
    L = sena["L"]
    if s.get("R") and s["R"] > L:
        s["progresas"] = min(1.0, max(0.0, (kaina - L) / (s["R"] - L)))
    return s


def paleisti_live(rinkos):
    busena = ikelti_busena()
    eilutes = []
    visi_barai = {}          # (zyme, tickeris) -> visi turimi 5 min barai
    for rinka in rinkos:
        zyme = RINKOS[rinka]["zyme"]
        rod, barai = parsisiusti(rinka, LIVE_DIENOS)
        div = dividendu_kalendorius(list(barai))
        for t, d in barai.items():
            visi_barai[(zyme, t)] = d
            ses = d["sesija"].iloc[-1]
            sesija = d[d["sesija"] == ses]
            if len(sesija) < S2_ORB_BARU + 4:
                continue
            kd = _kd(rod[t], ses)
            if kd is None:
                continue
            iki = (RINKOS[rinka]["uzdarymas"] - int(sesija["minute"].iloc[-1]))
            for s in aptikti(sesija, kd, iki):
                raktas = f"{zyme}|{t}|{s['tipas']}|{ses}"
                dabar = str(d.index[-1])
                sena = busena.get(raktas)
                if sena:
                    atkurti(s, sena, kd, float(sesija["Close"].iloc[-1]))
                else:
                    busena[raktas] = dict(L=s["L"], R=s.get("R"), stop=s["stop"],
                                          ieina=s["ieina"], laikas=dabar)
                    s["pirmas_kartas"] = dabar
                try:
                    amz = int((pd.Timestamp(dabar) -
                               pd.Timestamp(s["pirmas_kartas"])).total_seconds() // 60)
                except Exception:
                    amz = 0
                s.update(rinka=zyme, tickeris=t, laikas=dabar,
                         sesija_data=str(ses),
                         atr_pct=float(kd["atr_pct"]), amzius_min=amz,
                         dividendas=div.get(t))
                eilutes.append(s)

    siandien = datetime.now().date().isoformat()
    issaugoti_busena({k: v for k, v in busena.items() if k.endswith(siandien)})
    os.makedirs("docs", exist_ok=True)
    with open("docs/detektorius.json", "w", encoding="utf-8") as f:
        json.dump(dict(atnaujinta=datetime.now(timezone.utc).isoformat(),
                       signalai=eilutes), f, ensure_ascii=False, indent=1,
                  default=str)
    if _KLAIDOS:
        print("\n  SCENARIJU KLAIDOS:")
        for k, n in sorted(_KLAIDOS.items(), key=lambda x: -x[1])[:5]:
            print(f"    {n:>6}x  {k}")
        _KLAIDOS.clear()
    z = zurnalas_atnaujinti(eilutes, barai_signalams(zurnalas_ikelti(), eilutes,
                                                     visi_barai))
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(puslapis_html(eilutes, z))
    print(f"  puslapis: docs/index.html ({len(eilutes)} korteliu)")
    tinkami = [e for e in eilutes if e["tinkamas"]]
    print(f"\n  aktyvus: {len(eilutes)}  (tinkami: {len(tinkami)})")
    for e in sorted(tinkami, key=lambda x: -x["progresas"]):
        print(f"   [{e['rinka']:>3}] {e['tickeris']:<10} {e['scenarijus']:<26} "
              f"{e['progresas']*100:>5.0f}%  {e['amzius_min']:>4} min")
    return eilutes


# ================================================================ kalibracija

def paleisti_kalibracija(rinkos, dienos):
    for rinka in rinkos:
        zyme = RINKOS[rinka]["zyme"]
        print(f"\n{'='*84}\n{zyme}\n{'='*84}")
        rod, barai = parsisiusti(rinka, dienos)

        ivykiai = []
        for t, d in barai.items():
            sesijos = list(d.groupby("sesija", sort=True))
            for si, (ses, sd) in enumerate(sesijos):
                if len(sd) < S2_ORB_BARU + 4:
                    continue
                kd = _kd(rod[t], ses)
                if kd is None:
                    continue
                suveike = set()
                for i in range(S2_ORB_BARU + 1, len(sd)):
                    iki = (RINKOS[rinka]["uzdarymas"] - int(sd["minute"].iloc[i]))
                    for s in aptikti(sd.iloc[:i + 1], kd, iki):
                        if s["tipas"] in suveike or not s["tinkamas"]:
                            continue
                        suveike.add(s["tipas"])
                        toliau = pd.concat(
                            [sd.iloc[i + 1:]] +
                            [sesijos[si + j][1] for j in range(1, HORIZONTAS_SESIJU)
                             if si + j < len(sesijos)])
                        s.update(baigtis(toliau, s))
                        if s["tipas"] == 1 and s.get("R_pilnas"):
                            for vn, vb in isejimo_variantai(toliau, s).items():
                                s[f"v_{vn}_baigtis"] = vb["baigtis"]
                                s[f"v_{vn}_pct"] = vb["pelnas_pct"]
                        s.update(tickeris=t, sesija=str(ses),
                                 atr_pct=float(kd["atr_pct"]),
                                 minute=int(sd["minute"].iloc[i]),
                                 pradzia_bare=i, baru_sesijoje=len(sd))
                        ivykiai.append(s)

        if _KLAIDOS:
            print("\n  SCENARIJU KLAIDOS (anksciau buvo tyliai prarytos):")
            for k, n in sorted(_KLAIDOS.items(), key=lambda x: -x[1])[:5]:
                print(f"    {n:>6}x  {k}")
            _KLAIDOS.clear()
        ataskaita(ivykiai, zyme)
        if ivykiai:
            ses = sorted({e["sesija"] for e in ivykiai})
            riba = ses[len(ses) // 2]
            print(f"\n  ---- {zyme}: 1-A PUSE (iki {riba}) ----")
            ataskaita([e for e in ivykiai if e["sesija"] <= riba], zyme, True)
            print(f"\n  ---- {zyme}: 2-A PUSE (NEMATYTA) ----")
            ataskaita([e for e in ivykiai if e["sesija"] > riba], zyme, True)


def vienalaikiskumas(df):
    if df.empty:
        return 0.0, 0
    vid, mx = [], 0
    for _, g in df.groupby("sesija"):
        n = int(g["baru_sesijoje"].max())
        sk = np.zeros(n + 2)
        for _, r in g.iterrows():
            a = int(r["pradzia_bare"])
            sk[a:min(n, a + int(r.get("isejo_bare", 1)))] += 1
        vid.append(sk[:n].mean())      # buvo sk.mean() per n+2 -> ~20% per mazai
        mx = max(mx, int(sk.max()))
    return float(np.mean(vid)), mx


def variantu_lentele(df):
    """Trys isejimo variantai ant TU PACIU signalu.

    Skiriasi tik isejimas. Todel skirtumas tarp eiluciu yra grynas isejimo
    taisykles indelis - ne kitokie signalai, ne kitos dienos.
    """
    g = df[df["tipas"] == 1] if "tipas" in df else df
    if g.empty or "v_T05_pct" not in g:
        return
    print("\n  ISEJIMO VARIANTAI (1 scenarijus, tie patys signalai)")
    print(f"    {'variantas':<12}{'N':>6}{'tiksl':>7}{'stop':>7}{'laikas':>8}"
          f"{'vid %':>8}{'EUR':>9}{'95% EUR':>19}")
    rng = np.random.default_rng(42)
    for vn in ISEJIMO_VARIANTAI:
        pc, bg = f"v_{vn}_pct", f"v_{vn}_baigtis"
        if pc not in g:
            continue
        gg = g.dropna(subset=[pc])
        if gg.empty:
            continue
        eur = gg[pc] / 100 * POZICIJA - SANAUDOS_EUR
        pdd = gg.assign(eur=eur).groupby("sesija")["eur"].mean().values
        if len(pdd) >= 10:
            bs = [rng.choice(pdd, len(pdd), replace=True).mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        print(f"    {vn:<12}{len(gg):>6}"
              f"{(gg[bg]=='tikslas').mean()*100:>6.1f}%"
              f"{(gg[bg]=='stop').mean()*100:>6.1f}%"
              f"{(gg[bg]=='laikas').mean()*100:>7.1f}%"
              f"{gg[pc].mean():>8.2f}{eur.mean():>9.2f}"
              f"  [{lo:>6.2f},{hi:>6.2f}]{'  <<<' if lo > 0 else ''}")


def kandidato_testas(df):
    """Vienas is anksto uzrasytas binarinis pjuvis, tikrinamas VISOSE pusese.

    Is 2026-09-27 kalibracijos vienintelis kandidatas, rodantis ta pacia
    krypti ABIEJOSE rinkose, yra kelias iki buvusio lygio: zemiausias
    ketvirtis (0.5-1.0 ATR) neigiamas ir EU (-35.0 EUR), ir JAV (-36.6 EUR).
    Kiti kandidatai (ATR, rizika) rinkose priestarauja vienas kitam, tad
    netikrinami.

    Riba 1.0 ATR nera ieskota - tai ketvirciu riba, kuria pasiule patys
    duomenys, ir ji cia uzrasoma, kad kita kalibracija galetu ja PRIIMTI
    arba ATMESTI keturiuose matavimuose, o ne likti itarimu.
    """
    if "kelias_atr" not in df or df["kelias_atr"].isna().all():
        return
    g = df.dropna(subset=["kelias_atr"]).copy()
    g["eur"] = g["pelnas_pct"] / 100 * POZICIJA - SANAUDOS_EUR
    print("\n  KANDIDATAS: kelias iki buvusio lygio >= 1.0 ATR")
    print(f"    {'grupe':<16}{'N':>6}{'vid EUR':>10}{'tiksl':>7}{'stop':>7}"
          f"{'95% EUR':>19}")
    rng = np.random.default_rng(42)
    for nm, mask in (("< 1.0 ATR", g["kelias_atr"] < 1.0),
                     (">= 1.0 ATR", g["kelias_atr"] >= 1.0)):
        gg = g[mask]
        if gg.empty:
            continue
        pdd = gg.groupby("sesija")["eur"].mean().values
        if len(pdd) >= 10:
            bs = [rng.choice(pdd, len(pdd), replace=True).mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        print(f"    {nm:<16}{len(gg):>6}{gg['eur'].mean():>10.2f}"
              f"{(gg['baigtis']=='tikslas').mean()*100:>6.1f}%"
              f"{(gg['baigtis']=='stop').mean()*100:>6.1f}%"
              f"  [{lo:>6.2f},{hi:>6.2f}]{'  <<<' if lo > 0 else ''}")


def pjuviai(df):
    """KANDIDATAI I FILTRUS (2 pakopa).

    Cia NIEKAS neblokuojama. Rodoma tik tam, kad matytusi, kuri riba galetu
    uzsidirbti teise egzistuoti. Riba priimama tik jeigu tas pats pjuvis
    laikosi IR nematytoje puseje, IR kitoje rinkoje. Butent sio zingsnio
    truko reitinguotojui: ten ribos buvo prielaidos, ne isvados.
    """
    print("\n  KANDIDATAI I FILTRUS (nieko neblokuoja - tik matoma)")
    df = df.copy()
    df["eur"] = df["pelnas_pct"] / 100 * POZICIJA - SANAUDOS_EUR
    df["valanda"] = (df["minute"] // 60) if "minute" in df else np.nan

    def pjuvis(pav, stulp, kvantiliai=True):
        if stulp not in df or df[stulp].isna().all():
            return
        g = df.dropna(subset=[stulp]).copy()
        if kvantiliai:
            try:
                g["_gr"] = pd.qcut(g[stulp], 4, duplicates="drop")
            except Exception:
                return
        else:
            g["_gr"] = g[stulp].astype(int)
        print(f"\n  {pav}")
        print(f"    {'grupe':<22}{'N':>6}{'vid EUR':>10}{'MFE %':>8}"
              f"{'tiksl':>7}{'stop':>7}")
        for k, gg in g.groupby("_gr", observed=True):
            et = (f"{k.left:,.1f} .. {k.right:,.1f}"
                  if hasattr(k, "left") else str(k))
            print(f"    {et:<22}{len(gg):>6}{gg['eur'].mean():>10.1f}"
                  f"{gg['mfe_pct'].mean():>8.2f}"
                  f"{(gg['baigtis']=='tikslas').mean()*100:>6.0f}%"
                  f"{(gg['baigtis']=='stop').mean()*100:>6.0f}%")

    pjuvis("pagal ATR (ar didesnis judrumas kenkia, ar padeda?)", "atr_pct")
    pjuvis("pagal ATSIEMIMO dydi, ATR vienetais (peilio kandidatas)",
           "atsiemimas_atr")
    pjuvis("pagal kritimo gyli, ATR vienetais", "kritimas_atr")
    pjuvis("pagal kelia iki buvusio lygio, ATR vienetais", "kelias_atr")
    pjuvis("pagal rizika eurais", "rizika_eur")
    pjuvis("pagal iejimo valanda", "valanda", kvantiliai=False)


def ataskaita(ivykiai, zyme, trumpai=False):
    if not ivykiai:
        print("  suveikimu nebuvo")
        return
    df = pd.DataFrame(ivykiai)
    sesiju = df["sesija"].nunique()
    vid_k, max_k = vienalaikiskumas(df)
    print(f"\n  suveikimu: {len(df)}   sesiju: {sesiju}   "
          f"per diena: {len(df)/max(1,sesiju):.1f}   "
          f"vienu metu ekrane: {vid_k:.1f} (daugiausia {max_k})")
    print(f"\n{'SCENARIJUS':<26}{'N':>6}{'per d.':>7}{'tiksl':>7}{'stop':>7}"
          f"{'laikas':>8}{'vid %':>8}{'EUR':>8}{'MFE %':>8}{'MAE %':>8}"
          f"{'val.':>6}{'95% EUR':>19}")
    print("-" * 124)
    rng = np.random.default_rng(42)
    for nm, g in df.groupby("scenarijus"):
        eur = g["pelnas_pct"] / 100 * POZICIJA - SANAUDOS_EUR
        pdd = g.assign(eur=eur).groupby("sesija")["eur"].mean().values
        if len(pdd) >= 10:
            bs = [rng.choice(pdd, len(pdd), replace=True).mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        print(f"{nm:<26}{len(g):>6}{len(g)/max(1,sesiju):>7.1f}"
              f"{(g['baigtis']=='tikslas').mean()*100:>6.1f}%"
              f"{(g['baigtis']=='stop').mean()*100:>6.1f}%"
              f"{(g['baigtis']=='laikas').mean()*100:>7.1f}%"
              f"{g['pelnas_pct'].mean():>8.2f}{eur.mean():>8.2f}"
              f"{g['mfe_pct'].mean():>8.2f}{g['mae_pct'].mean():>8.2f}"
              f"{g['minuciu'].median()/60:>6.1f}  [{lo:>6.2f},{hi:>6.2f}]"
              f"{'  <<<' if lo > 0 else ''}")
    variantu_lentele(df)
    kandidato_testas(df)
    if trumpai:
        return
    pjuviai(df)
    print(f"\n  Pozicija {POZICIJA:.0f} EUR, sanaudos {SANAUDOS_EUR:.0f} EUR "
          f"(luzio taskas {LUZIO_TASKAS:.4f}%). Horizontas {HORIZONTAS_SESIJU} sesijos,")
    print("  pozicija nesama per nakti. MFE - kiek daugiausia buvo naudai;")
    print("  jei MFE dideles, o 'vid %' mazas, klaida yra isejimo taisykleje.")
    print("  Bootstrap pagal DIENAS, ne pagal eilutes.")


# ================================================================ savitikra

def savitikra():
    """Etalonai is Manto nurodytu atveju. Veikia BE tinklo.

    Tikrinama tai, kas kainavo brangiausiai: kad kodas darytu ta, kas
    aprasyta, ir kad scenarijai nepersidengtu.
    """
    ok = True

    def tikrinti(s, salyga, ar):
        nonlocal ok
        print(f"  {'OK ' if ar == salyga else 'BLOGAI'}  {s}"
              f"{'' if ar == salyga else f'  (gauta {salyga}, laukta {ar})'}")
        ok = ok and (salyga == ar)

    def sesija(atid, eiga, apyv_x=3.0, istorijos_d=LIVE_DIENOS):
        """Barai per TIKRA sesijos_rodikliai() - ne rankomis suklijuoti.

        Pirma savitikros versija pati susikurdavo vwap, ses_atidarymas ir
        apyv_santykis stulpelius, tad sesijos_rodikliai() apskritai nebuvo
        tikrinamas. Butent del to liko nepastebeta, kad live rezime
        apyv_santykis visada NaN ir signalu neduoda is principo.
        Dabar kuriami ZALI barai, o visi rodikliai skaiciuojami tikruoju keliu.
        """
        eiga = np.asarray(eiga)
        k = atid * np.cumprod(1 + eiga)
        eil, pr = [], pd.Timestamp("2026-07-24 09:00", tz="Europe/Berlin")
        # istorija: ramios dienos su TIPINE apyvarta (jos formuoja apyv_tipine)
        for d in range(istorijos_d, 0, -1):
            t0 = pr - pd.Timedelta(days=d)
            for j in range(len(k)):
                eil.append((t0 + pd.Timedelta(minutes=5 * j), atid, atid * 1.0008,
                            atid * 0.9992, atid, 5e5))
        # tiriamoji diena: apyvarta apyv_x kartu didesne
        for j in range(len(k)):
            eil.append((pr + pd.Timedelta(minutes=5 * j),
                        atid if j == 0 else k[j - 1], k[j] * 1.0008,
                        k[j] * 0.9992, k[j], 5e5 * apyv_x))
        raw = pd.DataFrame(eil, columns=["_t", "Open", "High", "Low", "Close",
                                         "Volume"]).set_index("_t")
        visos = sesijos_rodikliai(raw, "eu")
        return visos[visos["sesija"] == pr.date()]

    def kd(uzdarymas, atr_abs, virsune_n, uzd_vieta, dugnas_n, apyvarta=5e7):
        return pd.Series(dict(uzdarymas=uzdarymas, atr_abs=atr_abs,
                              atr_pct=atr_abs / uzdarymas * 100,
                              apyvarta=apyvarta, uzd_vieta=uzd_vieta,
                              virsune_n=virsune_n, dugnas_n=dugnas_n,
                              div_lange=0.0))

    def pirmas_signalas(f, sd, k):
        """Kaip kalibracijoje: augantis langas, imamas PIRMAS suveikimas.

        Pirma savitikros versija paduodavo visa sesija vienu kartu, t.y.
        vertindavo tik PASKUTINI bara - iejimas gaudavosi 3% aukstesnis nei
        tikrasis, ir R:R atrodydavo blogesnis nei yra.
        """
        for i in range(S2_ORB_BARU + 1, len(sd)):
            s = f(sd.iloc[:i + 1], k)
            if s:
                return s
        return None

    print("\nSAVITIKRA\n" + "-" * 60)

    # --- ar LIVE_DIENOS pakanka, kad apyv_santykis apskritai butu skaicius --
    # Sita patikra egzistuoja todel, kad 2026-09-27 live rezimas su 5 dienomis
    # negalejo duoti NE VIENO signalo, ir nei viena savitikra to nematė.
    # sesija(istorijos_d=N) duoda N+1 sesiju; parsisiusti(rinka, N) duoda N.
    # Todel live atitikmuo yra istorijos_d = LIVE_DIENOS - 1.
    sd = sesija(100.0, np.full(30, 0.0005), istorijos_d=LIVE_DIENOS - 1)
    tikrinti(f"LIVE_DIENOS={LIVE_DIENOS}: live gauna apyv_santykis (ne NaN)",
             bool(np.isfinite(sd["apyv_santykis"]).any()), True)
    sd4 = sesija(100.0, np.full(30, 0.0005), istorijos_d=4)
    tikrinti("su 5 sesijomis apyv_santykis dar NeRA skaicius (riba ten pat)",
             bool(np.isfinite(sd4["apyv_santykis"]).any()), False)

    # --- SAP.DE etalonas ---------------------------------------------------
    # 07-22 uzdare 132.04 ties dienos dugnu (uzd_vieta 0.00), 5 d. virsune
    # 138.26, ATR ~3.0. 07-23 kaina taip ir NEATSIEME 132.04 -> signalo nera.
    # 2026-09-27 atveju patikra parode, kad si rekonstrukcija buvo NETIKRA:
    # tikruose baruose SAP 07-23 kaina 10:05 buvo 132.20, t.y. TRUMPAM atsieme
    # 132.04, ir signalas suveike (po to diena nukrito iki 127.50). Todel testas
    # perrasytas: jis nebeteigia, kad peilio nera, o fiksuoja TIKRA elgsena -
    # menkas atsiemimas (0.03 ATR) signala DUODA, ir tai matoma kortelėje.
    k_sap = kd(132.04, 4.60, 138.26, 0.00, 127.50)
    # ramus atidarymo diapazonas, paskui trumpas kilstelejimas vos virs
    # 132.04 (kaip 10:05 tikroveje), ir apsivertimas - kaip 07-23.
    knife = sesija(131.00, np.r_[np.full(6, 0.0005), np.full(3, 0.0025),
                                 np.full(21, -0.0012)])
    s_knife = pirmas_signalas(scenarijus_1, knife, k_sap)
    tikrinti("SAP 07-23: menkas atsiemimas signala DUODA (peilis praeina)",
             s_knife is not None, True)
    if s_knife:
        print(f"         atsieme tik {s_knife['atsiemimas_atr']:.2f} ATR "
              f"-> kortelėje gintaro zenklas (riba 0.25)")
        tikrinti("         atsiemimas mazesnis uz 0.25 ATR",
                 s_knife["atsiemimas_atr"] < 0.25, True)

    # 07-24: atidare 130.88, kilo ir atsieme 128.32 (vakarykscio uzdarymo)
    k_sap2 = kd(128.32, 3.0, 138.26, 0.17, 127.50)
    atsok = sesija(130.88, np.r_[np.full(6, -0.001), np.full(24, 0.0012)])
    s = pirmas_signalas(scenarijus_1, atsok, k_sap2)
    tikrinti("SAP 07-24 (apsisukimas) -> signalas YRA", s is not None, True)
    if s:
        print(f"         ieina {s['ieina']:.2f}  tikslas {s['tikslas']:.2f}  "
              f"stop {s['stop']:.2f}  R:R {s.get('rr', float('nan')):.2f}  "
              f"kliutys: {s['kliutys'] or 'nera'}")
        tikrinti("         tikslas = iejimas + 0.5 ATR (konservatyvus)",
                 round(s["tikslas"], 2), round(s["ieina"] + 0.5 * 3.0, 2))
        tikrinti("         rizika neviršija 0.5 ATR",
                 round(s["ieina"] - s["stop"], 2) <= round(0.5 * 3.0, 2) + 1e-9, True)
        tikrinti("         R:R = 1.00 (tikslas 0.5 ATR / rizika 0.5 ATR)",
                 round(s["rr"], 2), 1.00)
        tikrinti("         tikras apsisukimas atsiima DAUG (>0.5 ATR)",
                 s["atsiemimas_atr"] > 0.5, True)
        tikrinti("         ir todel NEZYMIMAS kaip peilio kandidatas",
                 s["silpnas_atsiemimas"], False)

    # --- tolimas buves lygis NEBEDIDINA tikslo -----------------------------
    # Konservatyvus tikslas nuo buvusio lygio nepriklauso, kol tas lygis toli.
    k_toli = kd(128.32, 3.0, 152.00, 0.17, 127.50)
    s_toli = pirmas_signalas(scenarijus_1, atsok, k_toli)
    tikrinti("buves lygis 8 ATR toliau - tikslas vis tiek 0.5 ATR",
             bool(s_toli) and round(s_toli["tikslas"], 2) ==
             round(s_toli["ieina"] + 0.5 * 3.0, 2), True)

    # --- artimas buves lygis APKARPO tiksla ir signalas krinta per R:R ----
    k_arti = kd(128.32, 3.0, 131.60, 0.17, 127.50)   # kritimas 1.09 ATR,
                                                 # bet lygis tik 0.24 ATR virs iejimo
    s_arti = pirmas_signalas(scenarijus_1, atsok, k_arti)
    if s_arti:
        print(f"         artimas lygis {s_arti['R_pilnas']:.2f} -> tikslas "
              f"{s_arti['tikslas']:.2f}  R:R {s_arti['rr']:.2f}  "
              f"kliutys: {s_arti['kliutys'] or 'nera'}")
    tikrinti("artimas buves lygis: tikslas apkarpomas ir R:R blokuoja",
             bool(s_arti) and len(s_arti["kliutys"]) > 0, True)

    # --- ADYEN etalonas: 08-12 uzdare 910.00, 08-13 atidare 962.00 --------
    # (pirma savitikros versija cia turejo 930.30 - tai 08-07 uzdarymas, ne
    #  vakaryksis. Testo duomenys buvo blogi, ne kodas.)
    k_ady = kd(910.00, 31.6, 934.70, 0.504, 898.20, apyvarta=1.1e8)
    tarpas = sesija(962.00, np.r_[np.full(6, 0.0008), np.full(24, 0.0015)])
    s2 = pirmas_signalas(scenarijus_2, tarpas, k_ady)
    tikrinti("ADYEN 08-13 (tarpas +5.7% su apyvarta) -> signalas YRA",
             s2 is not None, True)
    if s2:
        print(f"         ieina {s2['ieina']:.2f}  stop {s2['stop']:.2f}  "
              f"tarpas {s2['tarpas_atr']:.1f} ATR  kliutys: "
              f"{s2['kliutys'] or 'nera'}")

    # --- scenarijai NEGALI suveikti kartu ---------------------------------
    tikrinti("tarpo dienos scenarijus 1 NEIMA (per toli nuo uzdarymo)",
             scenarijus_1(tarpas, k_ady) is None, True)
    tikrinti("apsisukimo dienos scenarijus 2 NEIMA (tarpo nera)",
             scenarijus_2(atsok, k_sap2) is None, True)

    # --- maza apyvarta: PTX 3.0 mln prie 1.8 mln ribos praeina ------------
    k_ptx = kd(108.58, 5.5, 110.08, 0.37, 104.52, apyvarta=3.0e6)
    ptx = sesija(126.40, np.r_[np.full(6, -0.002), np.full(24, 0.0018)])
    sp = pirmas_signalas(scenarijus_2, ptx, k_ptx)
    tikrinti("PTX (apyvarta 3.0 mln) NEBEBLOKUOJAMAS",
             bool(sp) and not sp["kliutys"], True)

    # --- shift(1): dienos rodikliai negali tureti tos dienos --------------
    n = 300
    idx = pd.bdate_range(end="2026-09-25", periods=n)
    c = pd.Series(np.linspace(100, 140, n), index=idx)
    d = pd.DataFrame(dict(Open=c, High=c * 1.01, Low=c * 0.99, Close=c,
                          Volume=np.full(n, 1e6)), index=idx)
    r = dienos_rodikliai(d)
    tikrinti("dienos_rodikliai: eilutes D 'uzdarymas' yra D-1 uzdarymas",
             float(r["uzdarymas"].iloc[-1]) == float(c.iloc[-2]), True)

    # --- baigtis: MFE matuojamas ------------------------------------------
    kilo = sesija(100.0, np.r_[np.full(10, 0.002), np.full(20, -0.001)])
    b = baigtis(kilo, dict(tipas=2, ieina=100.0, tikslas=None,
                           stop=98.0, atr_abs=2.0))
    tikrinti("baigtis grazina MFE > 0 kylanciam ruozui", b["mfe_pct"] > 1.0, True)

    # --- 2026-09-27 nepriklausomos perziuros radiniai: uzrakinami testais ---
    def bb(rows):
        return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"]
                            ).assign(apyv_santykis=1.5)

    s1f = dict(tipas=1, ieina=100.0, tikslas=110.0, stop=98.0, atr_abs=2.0)
    r = baigtis(bb([[90.0, 90.5, 88.0, 89.0]]), s1f)
    tikrinti("tarpas PRO stop'a vykdomas ties atidarymu, ne ties stop'u",
             round(r["pelnas_pct"], 2), -10.00)
    r = baigtis(bb([[99.0, 99.5, 97.0, 97.5]]), s1f)
    tikrinti("normalus issimusimas vis dar ties stop'u",
             round(r["pelnas_pct"], 2), -2.00)
    r = baigtis(bb([[115.0, 116.0, 114.0, 115.5]]), s1f)
    tikrinti("palankus tarpas virs tikslo vykdomas ties atidarymu",
             round(r["pelnas_pct"], 2), 15.00)

    kyla = bb([[100, 103, 99.8, 102], [102, 106, 101, 105],
               [105, 105.5, 103, 103.5], [103.5, 104, 102.5, 103]])
    tikrinti("1 scenarijui slenkancio stop'o NEBEtaikom (kortele zada fiksuota)",
             baigtis(kyla, s1f)["baigtis"], "laikas")
    tikrinti("2 scenarijui slenkantis stop'as veikia",
             baigtis(kyla, dict(tipas=2, ieina=100.0, tikslas=None, stop=96.0,
                                atr_abs=2.0))["baigtis"], "stop")

    idx = pd.bdate_range(end="2026-09-25", periods=300)
    c = pd.Series(np.linspace(100, 140, 300), index=idx)
    dd = pd.DataFrame(dict(Open=c, High=c * 1.01, Low=c * 0.99, Close=c,
                           Volume=np.full(300, 1e6)), index=idx)
    rr = dienos_rodikliai(dd)
    sesd = idx[-1].date()
    tikrinti("_kd praleidzia tvarkinga konteksta", _kd(rr, sesd) is not None, True)
    bloga = rr.copy()
    bloga.loc[bloga.index[-1], "apyvarta"] = np.nan
    tikrinti("_kd ATMETA NaN apyvarta (anksciau tyliai praeidavo)",
             _kd(bloga, sesd) is None, True)

    vid, _ = vienalaikiskumas(pd.DataFrame([
        dict(sesija="A", baru_sesijoje=10, pradzia_bare=0, isejo_bare=10),
        dict(sesija="A", baru_sesijoje=10, pradzia_bare=0, isejo_bare=10)]))
    tikrinti("vienalaikiskumas dalija is baru sesijoje, ne is n+2",
             round(vid, 2), 2.00)

    r = baigtis(bb([[115.0, 116.0, 95.0, 97.0]]), s1f)
    tikrinti("atidarymas virs tikslo skaitomas PIRMIAU uz kritima bare",
             (r["baigtis"], round(r["pelnas_pct"], 2)), ("tikslas", 15.00))

    # --- busenos atkurimas: viena nuosekli trijule ------------------------
    k_a = kd(128.32, 3.0, 138.26, 0.17, 127.50)
    sig_a = dict(tipas=1, ieina=134.00, stop=127.42, tikslas=137.00,
                 R=137.00, L=127.42, atr_abs=3.0)
    _bendra(sig_a, k_a)
    atkurti(sig_a, dict(ieina=130.88, stop=127.42, R=133.88, L=127.42,
                        laikas="x"), k_a, 134.00)
    laukiamas_rr = (133.88 - 130.88) / (130.88 - 127.42)
    tikrinti("atkurus busena R:R skaiciuojamas is ATKURTU ieina/stop/tikslas",
             round(sig_a["rr"], 2), round(laukiamas_rr, 2))
    tikrinti("atkurus busena rizika eurais atitinka atkurta iejima",
             round(sig_a["rizika_eur"]), round((130.88 - 127.42) / 130.88 * POZICIJA))

    # --- dividendu pataisa tikrai prijungta prie parsisiusti() ------------
    import inspect
    tikrinti("parsisiusti() paduoda dividendus i dienos_rodikliai",
             "Dividends" in inspect.getsource(parsisiusti), True)
    dv = pd.Series(0.0, index=idx)
    dv.iloc[-3] = 5.0
    tikrinti("dividendai patenka i div_lange",
             float(dienos_rodikliai(dd, dv)["div_lange"].iloc[-1]) > 0, True)

    tikrinti("tuscias horizontas duoda 'laikas', ne atskira baigti",
             baigtis(bb([]).iloc[:0], s1f)["baigtis"], "laikas")

    # --- tikslas negali buti zemiau iejimo -------------------------------
    # Atidarymas BE tarpo (128.60 vs vakar 128.32 = 0.09 ATR), kritimas 1.09 ATR,
    # bet kaina nubega VIRS buvusio lygio 131.60. Pirma sio testo versija turejo
    # atidaryma 133.00 = 1.56 ATR tarpas, tad scenarijus_1 nukrisdavo dar ties
    # tarpo filtru ir testas praeidavo NET ISJUNGUS tikrinama patikra.
    k_zem = kd(128.32, 3.0, 131.60, 0.17, 127.50)
    aukstai = sesija(128.60, np.r_[np.full(6, 0.0005), np.full(30, 0.0025)])
    # tikrinam PASKUTINI bara (kaina jau virs lygio), ne pirma: ankstyvuose
    # baruose kaina dar zemiau lygio ir signalas ten visiskai teisetas.
    pask = float(aukstai["Close"].iloc[-1])
    tikrinti("         bandomoji kaina tikrai virs buvusio lygio",
             pask > 131.60, True)
    tikrinti("kai kaina nubega virs buvusio lygio - signalo NERA",
             scenarijus_1(aukstai, k_zem) is None, True)

    # --- zurnalas: tuscia reiksme is CSV neturi reiksti "uzdaryta" -------
    import io
    tst = pd.DataFrame([dict(raktas="EU|X|1|2026-09-25", baigtis="",
                             tikslas="", rr="")])
    buf = io.StringIO()
    tst.to_csv(buf, index=False)
    atgal = pd.read_csv(io.StringIO(buf.getvalue()), dtype=str,
                        keep_default_na=False)
    tikrinti("zurnalo tuscia 'baigtis' nuskaitoma kaip tuscia, ne kaip nan",
             str(atgal["baigtis"].iloc[0] or ""), "")

    # --- isejimo variantai: nuo ju priklauso isejimo sprendimas ----------
    kyl = bb([[100.0, 100.8, 99.9, 100.6], [100.6, 101.4, 100.4, 101.2],
              [101.2, 102.4, 101.0, 102.2], [102.2, 102.6, 100.2, 100.4],
              [100.4, 100.6,  99.0,  99.2]])
    sbaze = dict(tipas=1, ieina=100.0, tikslas=101.0, stop=99.0, atr_abs=2.0,
                 L=99.6, R=101.0, R_pilnas=140.0)
    pries = dict(sbaze)
    v = isejimo_variantai(kyl, sbaze)
    tikrinti("isejimo_variantai NEKEICIA originalaus signalo", sbaze == pries, True)
    tikrinti("T05 sutampa su pagrindiniu baigtis() (tas pats tikslas)",
             round(v["T05"]["pelnas_pct"], 4),
             round(baigtis(kyl, sbaze)["pelnas_pct"], 4))
    tikrinti("T10 tikslas tolimesnis -> kitas rezultatas nei T05",
             v["T10"]["pelnas_pct"] != v["T05"]["pelnas_pct"], True)
    tikrinti("SL neturi tikslo (niekada 'tikslas')",
             v["SL"]["baigtis"] != "tikslas", True)
    tikrinti("SL naudoja PLATU struktūrini stop'a, ne kortelės",
             isejimo_variantai(kyl, sbaze) is not None and
             baigtis(kyl, dict(sbaze, tipas=2, tikslas=None,
                               stop=sbaze["L"] - S1_STOP_ATR * 2.0))["pelnas_pct"]
             == v["SL"]["pelnas_pct"], True)
    tikrinti("visi trys variantai grazinami",
             sorted(v) == sorted(ISEJIMO_VARIANTAI), True)

    # SL platus stop'as: rizikos riba prisisotinusi, tad kortelės stop'as (99.0)
    # yra AUKSCIAU uz struktūrini (97.4). Barai nukrenta iki 98.5 - kortele
    # issimuse, SL turi islikti.
    plat = bb([[100.0, 100.4, 98.5, 99.0], [99.0, 101.0, 98.9, 100.8],
               [100.8, 101.6, 100.4, 101.4]])
    sp = dict(tipas=1, ieina=100.0, tikslas=101.0, stop=99.0, atr_abs=2.0,
              L=98.0, R=101.0, R_pilnas=140.0)
    vp = isejimo_variantai(plat, sp)
    tikrinti("kortelės stop'as (99.0) issimusa ties 98.5",
             baigtis(plat, sp)["baigtis"], "stop")
    tikrinti("SL platus stop'as (97.4) tame paciame bare ISLIEKA",
             vp["SL"]["baigtis"] != "stop", True)

    # R:R riba: tikros kainos, kur (t-k)/(k-s) dvigubame tikslume < 1.0
    for kk, aa in ((261.50849255278456, 11.657260123286084),
                   (126.41480575695486, 9.938104626893272)):
        sr = dict(ieina=kk, tikslas=kk + 0.5*aa, stop=kk - 0.5*aa)
        kl = _bendra(sr, kd(kk, aa, kk + 5*aa, 0.1, kk - 2*aa))
        tikrinti(f"R:R riba nebetrapi ties kaina {kk:.2f}",
                 any("R:R" in x for x in kl), False)

    tikrinti("zurnalas raso atr_abs tiesiogiai, ne atkurineja is atr_pct",
             'float(r["atr_abs"])' in inspect.getsource(zurnalas_atnaujinti)
             and "atr_abs" in ZURNALO_STULPELIAI, True)

    tuscias = pd.DataFrame()
    variantu_lentele(tuscias)                      # neturi luzti
    tikrinti("variantu_lentele nelūžta su tusciu df", True, True)

    print("-" * 60)
    print("SAVITIKRA: " + ("VISKAS GERAI" if ok else "YRA KLAIDU"))
    return 0 if ok else 1


# ================================================================ main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--kalibracija", action="store_true")
    ap.add_argument("--savitikra", action="store_true")
    ap.add_argument("--rinka", choices=["eu", "us", "abi"], default="abi")
    ap.add_argument("--dienos", type=int, default=60)
    a = ap.parse_args()
    if a.savitikra:
        sys.exit(savitikra())
    if yf is None:
        sys.exit("KLAIDA: reikia yfinance (pip install yfinance)")
    rinkos = ["eu", "us"] if a.rinka == "abi" else [a.rinka]
    if a.kalibracija:
        paleisti_kalibracija(rinkos, a.dienos)
    else:
        paleisti_live(rinkos)
