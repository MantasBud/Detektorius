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
   Krintancio peilio apsauga DALINE: salyga atmeta dienas, kuriomis kaina
   VISAI neatsieme lygio, bet ne tas, kuriomis ji atsieme trumpam ir
   apsiverte. SAP 07-23 tikruose baruose 10:05 buvo 132.20 vs vakar 132.04,
   tad signalas SUVEIKE, o diena nukrito iki 127.50. Tokius atvejus zymi
   mazas `atsiemimas_atr` (0.03 ATR pries 0.91-1.75 tikruose apsisukimuose),
   ir puslapis juos slepia, taciau i `kliutys` tai NEdedama - kitaip
   pasikeistu kalibracijos imtis ir nebegaletume patikrinti, ar slepti
   teisinga.

2. NAUJIENU TARPAS IR EIGA
   Atidarymo tarpas >= 1 dienos ATR su apyvarta; kaina laikosi virs VWAP;
   pozicija nesama slenkanciu stop'u ATR vienetais.
   Etalonas: ADYEN.AS 08-13 (tarpas +5.7%, diena +10.1%),
             PTX.DE 08-04 (tarpas +16.4%, diena +8.8%).

Scenarijai nepersidengia PAGAL KONSTRUKCIJA: lyginamas ATIDARYMAS, ne iejimas.
Jei atidarymas - vakar_uzd >= 1.0*ATR, diena valdo 2 scenarijus, ir 1 grazina
None; kitu atveju atvirksciai. Ribos yra tikslus vienas kito papildiniai
(savitikroje - du taskiniai testai i abi puses nuo ribos).

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
import math
import os
import shutil
import sys
import tempfile
import warnings
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- mechanika
# Sitie keturi skaiciai ateina is Manto saskaitos, ne is jokio tyrimo.
POZICIJA = 18000.0          # visas portfelis vienai pozicijai, EUR verte
# Sanaudos SKIRIASI pagal rinka (Manto atsakymas 2026-09-28): JAV rinkoms
# perpus maziau. Tai ne smulkmena - is sanaudu isvesta VISA kita aritmetika
# (luzio taskas, minimalus judesys, de facto ATR riba), tad JAV ribos
# automatiskai perpus zemesnes, ir tai teisinga, o ne "svelnesnis filtras".
SANAUDOS = {"eu": 10.0, "us": 5.0}      # pirkimas + pardavimas, EUR
SANAUDOS_EUR = SANAUDOS["eu"]           # istorinis vardas = EU reiksme
MAX_APYVARTOS_DALIS = 0.01  # pozicija ne daugiau 1% dienos apyvartos
# Kai Mantas prekiaus JAV, portfelis bus doleriais - nuolatines konversijos
# nebus. Procentine grazа valiutai abejinga, tad i eurus versti reikia tik
# APYVARTOS riba: 1.8 mln EUR verte doleriais yra kitas skaicius.
EURUSD_ATSARGINIS = 1.08
KURSO_TALPYKLA = "kursas.json"
_KURSAS = {}


def eur_usd():
    """USD uz 1 EUR. Kesuojama parai; nepavykus - atsargine reiksme, garsiai."""
    if "v" in _KURSAS:
        return _KURSAS["v"]
    try:
        if os.path.exists(KURSO_TALPYKLA):
            with open(KURSO_TALPYKLA, encoding="utf-8") as f:
                t = json.load(f)
            if (datetime.now() - datetime.fromisoformat(t["kada"])).days < 1:
                _KURSAS["v"] = float(t["kursas"])
                return _KURSAS["v"]
    except Exception:
        pass
    v = None
    try:
        d = yf.download("EURUSD=X", period="5d", interval="1d",
                        progress=False, auto_adjust=False)
        c = _vienas(d, "EURUSD=X")
        if c is not None and len(c):
            v = float(c["Close"].dropna().iloc[-1])
    except Exception:
        v = None
    if not v or not (0.5 < v < 2.0):
        print(f"  DEMESIO: EUR/USD kurso negauta, imama atsargine "
              f"{EURUSD_ATSARGINIS}")
        v = EURUSD_ATSARGINIS
    else:
        try:
            with open(KURSO_TALPYKLA, "w", encoding="utf-8") as f:
                json.dump(dict(kursas=v, kada=datetime.now().isoformat()), f)
        except Exception:
            pass
    _KURSAS["v"] = v
    return v


def sanaudos(rinka):
    return SANAUDOS.get(rinka, SANAUDOS["eu"])


ZYME_I_RINKA = {"EU": "eu", "JAV": "us"}


def sanaudos_zymei(zyme):
    """Sanaudos pagal puslapio/zurnalo zyme ('EU' / 'JAV')."""
    return sanaudos(ZYME_I_RINKA.get(str(zyme), "eu"))


def luzio_taskas(rinka):
    return sanaudos(rinka) / POZICIJA * 100.0       # EU 0.0556%, JAV 0.0278%


def min_judesys(rinka):
    """Maziausias prasmingas judesys: tikslas bent 10x uz luzio taska.

    Isvesta is sanaudu, ne is tyrimo.
    """
    return 10 * luzio_taskas(rinka)                 # EU 0.556%, JAV 0.278%


def min_atr_pct(rinka):
    """De facto ATR riba: tikslas = 0.5 ATR, tad ATR < sito niekada nepraeis."""
    return min_judesys(rinka) / S1_TIKSLAS_ATR      # EU 1.11%, JAV 0.556%


def min_apyvarta(rinka):
    """Apyvartos riba KOTIRAVIMO valiuta.

    Iki 2026-09-28 cia buvo viena 1.8 mln riba, taikoma ir doleriniu akciju
    apyvartai - t.y. eurine riba lyginama su doleriniu skaiciumi. Paklaida
    nedidele, bet tai aritmetikos klaida, ir pleciant JAV dali ji auga.
    """
    v = POZICIJA / MAX_APYVARTOS_DALIS
    return v * (eur_usd() if rinka == "us" else 1.0)


LUZIO_TASKAS = SANAUDOS_EUR / POZICIJA * 100.0      # EU, ataskaitoms
MIN_JUDESYS_PCT = 10 * LUZIO_TASKAS                 # EU, ataskaitoms
# SALUTINE PASEKME, uzrasyta samoningai (rasta perziurint 2026-09-27):
# kadangi 1 scenarijaus tikslas yra 0.5*ATR, o judesys turi buti >= 0.556%,
# akcijos su ATR < 1.11% 1 scenarijaus signalo NEDUODA NIEKADA. Tai de facto
# MIN_ATR riba. Ji teisėta pagal 1 pakopa (aritmetika is sanaudu), NE
# paveldėta is reitinguotojo, ir ji veikia PRIESINGA puse nei senasis
# F_MAX_ATR = 3.5: anksciau buvo ismetamos JUDRIOS akcijos, dabar - RAMIOS.
MIN_RR = 1.0                                        # aritmetika

BARAS_MIN = 5
HORIZONTAS_SESIJU = 3
# Kiek darbo dienu PO horizonto laukiama, kol eilute uzdaroma "is bedos"
# (Yahoo sutrikimas, trukstamos sesijos, isbrauktas tickeris). Iki tol eilute
# lieka atvira ir kiekvienas paleidimas bando ja ismatuoti is naujo.
HORIZONTO_ATSARGA_D = 5
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
    "ieina", "tikslas", "stop", "rr", "rizika_eur", "atr_pct", "sanaudos",
    "tinkamas", "kliutys", "atr_abs",
    "paskut_laikas", "paskut_kaina", "mfe_pct", "mae_pct",
    "baigtis", "pelnas_pct", "eur", "minuciu", "saltinis",
]
# saltinis: "kortele" - signalas, kuri gyvas paleidimas parode kortelėje
# (senos eilutes be reiksmes - irgi kortele); "baru perziura" - tas pats
# scenarijus, bet nuo PIRMO baro, kuriame jis atsirado (kaip kalibracijoje).
# Paleidimai vyksta kas ~12-20 min, todel kortele signala pamato veliau arba
# visai nepamato (SOI 2026-09-28). Puslapio skaiciai - TIK is korteliu.
PERZIUROS_ZYME = "B"
# Puslapio statistika skaiciuojama nuo sios sesijos. Iki 2026-09-29 korteles
# vėluodavo 15-20 min (GitHub planuoklis), todel ju rezultatai neatspindi
# dabartinio detektoriaus (ciklas kas 5 min). Senos eilutes zurnale lieka.
# 2026-10-06: pradzia pastumta po 6 AI auditu pataisymu (S2 slenkancio stop'o
# tvarka bare, pirmas tinkamas signalas gyvai). Zurnalas nekeiciamas.
PUSLAPIO_PRADZIA = "2026-10-06"
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
# Nenaudojama kode - tai IsVESTINE, rodanti, kur de facto atsiduria riba.
# Tikroji riba pagal rinka - min_atr_pct(rinka).
MIN_ATR_PCT = MIN_JUDESYS_PCT / S1_TIKSLAS_ATR      # -> EU 1.11%
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
# PROGRESO LUBOS (Manto sprendimas 2026-09-28). Kiek ATR virs proverzio
# lygio laikome "judesys jau nueitas". Tai VIENINTELIS puslapio skaicius,
# neisvestas is mechanikos - 2 scenarijus tikslo neturi, tad nera ir dydzio,
# i kuri butu galima matuoti progresa.
#
# Kodel ne 1.0: slenkantis stop'as yra 1.0 ATR. Vadinasi, nuejus 1 ATR virs
# proverzio, sandoris tik ka pasieke ta atstuma, kuri stop'as ATIDUOS
# apsisukus - t.y. grynasis rezultatas ten dar nulis. Vadinti ta taska
# "100% nueita" yra atvirkscia.
#
# Riba veikia TIK puslapio spalva ir rikiavima. Aptikimo, iejimo, stop'o ir
# isejimo ji nekeicia. Tikroji verte ateis is matavimo: irasomas laukas
# virs_proverzio_atr ir jis yra kandidatu sarase.
S2_PROGRESO_LUBOS = 1.5

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


ATASKAITU_TALPYKLA = "ataskaitos.json"
_ATASK = None           # {tickeris: {"datos": [...], "kada": iso}}


def _ataskaitu_talpykla():
    global _ATASK
    if _ATASK is None:
        _ATASK = {}
        if os.path.exists(ATASKAITU_TALPYKLA):
            try:
                with open(ATASKAITU_TALPYKLA, encoding="utf-8") as f:
                    t = json.load(f)
                if isinstance(t, dict):
                    _ATASK = {k: v for k, v in t.items()
                              if isinstance(v, dict) and "datos" in v}
            except Exception:
                _ATASK = {}
    return _ATASK


def ataskaitu_kalendorius(tickers):
    """Istorines ir busimos ataskaitu datos, po viena uzklausa tickeriui.

    Tai VIENINTELE naujienu rusis, kuria galima patikrinti atgal: antrasciu
    su laiko zymemis ir sentimentu istorijos musu saltinis neturi, tad joks
    sentimento filtras nebutu patikrinamas - o nepatikrinamu filtru sitas
    projektas jau turejo per daug.

    Talpykla PER TICKERI (ne per visa faila): live rezimas kviecia ja tik
    toms akcijoms, kurios turi signala, tad 357 uzklausos kas 5 minutes
    niekada nedaromos. Galiojimas - savaite, kaip ir dividendams.
    """
    tal = _ataskaitu_talpykla()
    dabar = datetime.now()
    truksta = []
    for t in tickers:
        v = tal.get(t)
        if not v:
            truksta.append(t)
            continue
        try:
            if (dabar - datetime.fromisoformat(v["kada"])).days >= 7:
                truksta.append(t)
        except Exception:
            truksta.append(t)
    if not truksta:
        return {k: tal[k]["datos"] for k in tickers if k in tal}
    print(f"  ataskaitu kalendorius: {len(truksta)} naujos uzklausos "
          f"({len(tickers) - len(truksta)} is talpyklos)")
    klaidu = 0
    for t in truksta:
        datos = []
        try:
            e = yf.Ticker(t).earnings_dates
            if e is not None and len(e):
                datos = sorted({str(x.date()) for x in e.index})[-12:]
        except Exception:
            klaidu += 1
        tal[t] = dict(datos=datos, kada=dabar.isoformat())
    if klaidu:
        print(f"    ({klaidu} tickeriu ataskaitu datu negauta - laukas liks NaN)")
    try:
        with open(ATASKAITU_TALPYKLA, "w", encoding="utf-8") as f:
            json.dump(tal, f, ensure_ascii=False, indent=1)
    except Exception:
        pass
    return {k: tal[k]["datos"] for k in tickers if k in tal}


def dienu_iki_ataskaitos(datos, ses):
    """Arciausios ataskaitos atstumas dienomis (neigiamas - jau buvo)."""
    if not datos:
        return np.nan
    try:
        d = [date.fromisoformat(x) for x in datos]
        v = min(((x - ses).days for x in d), key=abs)
    except Exception:
        return np.nan
    # Dalis tikeriu turi tik kelis metus senas datas - tada "arciausia
    # ataskaita" yra -1465 dienu, ir tai ne informacija, o duomenu skyle.
    # Kalibracijoje tokios eilutes buvo sumestos i zemiausia ketvirti.
    return v if abs(v) <= 200 else np.nan


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


def _fono_laukai(fonas, laikas, sd, i):
    """Rinkos fono laukai signalui. Tik matavimui - niekas neblokuojama."""
    out = dict(rinkos_pokytis=np.nan, platumas=np.nan, plat_d30=np.nan,
               santykinis=np.nan)
    try:
        if fonas is None or fonas.empty or laikas not in fonas.index:
            return out
        r = fonas.loc[laikas]
        out["rinkos_pokytis"] = float(r.get("mediana", np.nan))
        out["platumas"] = float(r.get("platumas", np.nan))
        out["plat_d30"] = float(r.get("plat_d30", np.nan))
        atid = float(sd["ses_atidarymas"].iloc[i])
        if atid > 0:
            savo = (float(sd["Close"].iloc[i]) / atid - 1.0) * 100
            out["santykinis"] = savo - out["rinkos_pokytis"]
    except Exception:
        pass
    return out


def rinkos_fonas(barai, rod):
    """Rinkos fonas kiekvienam 5 min laikui - is TU PACIU baru, be papildomu
    uzklausu.

    Projekto tvirciausias radinys per visa istorija: rinkos rezimas svarbesnis
    uz akciju atranka apie 10 kartu (1.05 p.p. skirtumas). Senasis rezimo
    filtras buvo isimtas, nes jis (a) atejo is reitinguotojo ir (b) naudojo
    TOS DIENOS uzdaryma, t.y. ateiti. Sitas - kitoks: jis skaiciuojamas tik
    is baru IKI to laiko, skerspjuviu per visas akcijas.

    Grazina DataFrame su stulpeliais:
      mediana   - universo mediana nuo sesijos atidarymo, %
      platumas  - kiek % universo siuo metu VIRS vakarykscio uzdarymo
      plat_d30  - platumo pokytis per paskutines 30 min
    """
    pok, virs = [], []
    for t, d in barai.items():
        if not len(d):
            continue
        atid = d["ses_atidarymas"].replace(0, np.nan)
        pok.append(pd.Series((d["Close"] / atid - 1.0).values * 100, index=d.index))
        r = rod.get(t)
        if r is None:
            continue
        vu = {}
        for ses in d["sesija"].unique():
            k = _kd(r, ses)
            if k is not None:
                vu[ses] = float(k["uzdarymas"])
        if not vu:
            continue
        v = d["sesija"].map(vu).values.astype(float)
        virs.append(pd.Series((d["Close"].values > v).astype(float), index=d.index))
    if not pok:
        return pd.DataFrame()
    f = pd.DataFrame({"mediana": pd.concat(pok, axis=1).median(axis=1)}).sort_index()
    if virs:
        f["platumas"] = pd.concat(virs, axis=1).mean(axis=1).sort_index() * 100
        # 30 min = 6 barai, bet TIK toje pacioje sesijoje. Pirmoji versija
        # skaiciavo .diff(6) per visa stulpeli: pirmi 6 dienos barai buvo
        # lyginami su PRAEJUSIOS dienos pabaiga, t.y. su nakties tarpu.
        # Be to diff buvo skaiciuojamas PRIES sort_index().
        ses = pd.Index([x.date() for x in f.index], name="ses")
        f["plat_d30"] = f.groupby(ses)["platumas"].diff(6).values
    return f


# ================================================ scenarijai (grynos funkcijos)

def _bendra(sig, kd, rinka="eu"):
    """1 PAKOPOS patikros: isvestos is mechanikos ir aritmetikos.

    2 pakopos (kalibracijos uzsidirbtu) ribu cia kol kas nera ne vienos -
    jos atsiras tik tada, kai ataskaitos pjuviai parodys, kuri riba laikosi
    ir nematytoje puseje, ir abiejose rinkose.
    """
    kl = []
    sig["sanaudos"] = sanaudos(rinka)
    riba_apyv = min_apyvarta(rinka)
    if kd["apyvarta"] < riba_apyv:
        kl.append(f"apyvarta {kd['apyvarta']/1e6:.1f} mln < "
                  f"{riba_apyv/1e6:.1f}")
    # rizika eurais rodoma kortelėje - sprendzia Mantas, kodas neblokuoja
    sig["rizika_pct"] = (sig["ieina"] - sig["stop"]) / sig["ieina"] * 100.0
    sig["rizika_eur"] = sig["rizika_pct"] / 100.0 * POZICIJA
    if sig.get("tikslas"):
        judesys = (sig["tikslas"] - sig["ieina"]) / sig["ieina"] * 100.0
        riba_jud = min_judesys(rinka)
        if judesys < riba_jud:
            kl.append(f"judesys {judesys:.2f}% < {riba_jud:.2f}%")
        rr = (sig["tikslas"] - sig["ieina"]) / max(1e-9, sig["ieina"] - sig["stop"])
        sig["rr"] = rr
        if rr < MIN_RR - 1e-9:
            kl.append(f"R:R {rr:.3f} < {MIN_RR}")
    return kl


def scenarijus_1(langas, kd, rinka="eu"):
    """Kritimas 1-5 sesijas + vakarykscio uzdarymo atsiemimas.

    Kol kaina neatsieme vakarykscio uzdarymo, signalo nera. Bet trumpas
    lygio palietimas is apacios salyga PRAEINA - zr. modulio dokumentacija
    apie `atsiemimas_atr`.
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
    # Praktikoje (b) laimi 97% atveju, tad struktūrinė dedamoji beveik
    # niekada neveikia. Kai atsiemimas > 0.5 ATR, stop'as atsiduria VIRS
    # atsiimto lygio, t.y. pozicija uzdaroma dar galiojant prielaidai.
    # Tai samoningas
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
               progresas=(kaina - L) / (R_pilnas - L) if R_pilnas > L else 0.0,
               progresas_tikslus=True, kritimas_atr=kritimas / atr,
               atsiemimas_atr=(kaina - vakar_uzd) / atr,
               # kiek baru praejo nuo sesijos dugno: sviezias apsisukimas
               # ar jau senas
               baru_nuo_dugno=int(len(langas) - 1
                                  - int(np.argmin(langas["Low"].values))),
               # kiek kartu SIANDIEN kaina jau buvo atsiemusi lygi ir vel ji
               # prarado - kiekviena nesekme silpnina kita bandyma
               nesekmes=int(((langas["Close"].values[:-1] > vakar_uzd)
                             & (langas["Close"].values[1:] <= vakar_uzd)).sum()),
               # TIK puslapiui. I "kliutys" NEdedama tycia: kitaip pasikeistu
               # "tinkamas", o su juo - kalibracijos imtis, ir nebegaletume
               # patikrinti, ar slepti buvo teisinga.
               silpnas_atsiemimas=bool((kaina - vakar_uzd) / atr < 0.25),
               R_pilnas=R_pilnas, tikslas_ribotas=bool(R < R_pilnas - 1e-9),
               kelias_atr=(R_pilnas - kaina) / atr)
    sig["kliutys"] = _bendra(sig, kd, rinka)
    return sig


def scenarijus_2(langas, kd, iki_uzdarymo=None, rinka="eu"):
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
    virs = (kaina - orb_max) / max(1e-9, atr)
    prog_j = virs / S2_PROGRESO_LUBOS
    prog_l = 0.0
    if iki_uzdarymo is not None:
        r = RINKOS.get(rinka, RINKOS["eu"])
        viso = r["uzdarymas"] - r["atidarymas"]
        prog_l = max(0.0, 1.0 - float(iki_uzdarymo) / max(1.0, viso))
    sig = dict(scenarijus="Ralis", tipas=2, ieina=kaina,
               tikslas=None, stop=stop, R=None, L=orb_min, atr_abs=atr,
               progresas=min(1.0, max(prog_j, prog_l)),
               progresas_tikslus=False, tarpas_atr=tarpas / atr,
               virs_proverzio_atr=virs,
               virsune_vertinimas=kaina + S2_TRAIL_ATR * atr)
    sig["kliutys"] = _bendra(sig, kd, rinka)
    return sig


_KLAIDOS = {}


def aptikti(langas, kd, iki_uzdarymo=None, rinka="eu"):
    """Klaidos nebetylimos: anksciau `except: s = None` prarydavo VISKA, tad
    toks gedimas kaip visur-NaN apyv_santykis atrodydavo kaip "signalu nera"."""
    rez = []
    for f in (scenarijus_1, scenarijus_2):
        try:
            s = (f(langas, kd, rinka) if f is scenarijus_1
                 else f(langas, kd, iki_uzdarymo, rinka))
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
        # MFE/MAE tik IKI isejimo. Anksciau jie buvo atnaujinami is viso
        # baro diapazono PRIES tarpo patikras, tad tarpo bare i MFE patekdavo
        # judesys, ivykes jau po to, kai pozicijos nebebuvo: O=115 prie
        # tikslo 110 duodavo mfe +16%, nors realus isejimas +15%.
        iseina = None
        if op <= stop:
            iseina = op
        elif tikslas and op >= tikslas:
            iseina = op
        elif lo <= stop:
            iseina = stop
        elif tikslas and hi >= tikslas:
            iseina = tikslas
        if iseina is not None:
            # Isejimo bare nezinome judesiu eiles bare, tad i MFE/MAE
            # iskaitom tik pacia isejimo kaina. Anksciau buvo imamas visas
            # baro diapazonas, todel i MFE patekdavo judesys, ivykes jau po
            # to, kai pozicijos nebebuvo (baras [99, 108, 97, 98.5] prie
            # stop'o 98 duodavo mfe +8%, nors isejimas buvo -2%).
            v = (iseina / ieina - 1) * 100
            mae, mfe = min(mae, v), max(mfe, v)
        else:
            mae = min(mae, (lo / ieina - 1) * 100)
            if (hi / ieina - 1) * 100 > mfe:
                mfe, mfe_i = (hi / ieina - 1) * 100, j + 1
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
            naujas = max(stop, auksciausia - S2_TRAIL_ATR * atr)
            # KONSERVATYVU (2026-10-05, 6 AI auditai): 5 min bare nezinome,
            # ar High buvo pries Low. Jei baras pakele stop'a ir to paties
            # baro Low ji pasieke - laikom, kad isejo ties nauju stop'u.
            # Anksciau naujas stop'as galiojo tik nuo kito baro (6 m.: 32
            # sandoriai is 1915 buvo vertinami per palankiai).
            if naujas > stop and lo <= naujas:
                return dict(baigtis="stop", pelnas_pct=(naujas / ieina - 1) * 100,
                            minuciu=(j + 1) * BARAS_MIN, mfe_pct=mfe, mae_pct=mae,
                            mfe_min=mfe_i * BARAS_MIN, laikas_baige=False,
                            isejo_bare=j + 1)
            stop = naujas

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
.skirt{display:flex;gap:4px;margin:12px 0 4px;border-bottom:1px solid var(--line)}
.skirt button{background:none;border:0;border-bottom:2px solid transparent;color:var(--dim);
  padding:8px 12px;font-size:14px;cursor:pointer;margin-bottom:-1px}
.skirt button[aria-selected="true"]{color:var(--txt);border-bottom-color:var(--acc);font-weight:600}
section[hidden]{display:none}
.lent{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;margin-top:14px}
.lent table{width:100%;border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums}
.lent th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--dim);
  font-weight:600;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
.lent td{padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
.lent tr:last-child td{border-bottom:0}
.lent td.sk,.lent th.sk{text-align:right}
.apyt{color:var(--dim)}
.pastaba{color:var(--dim);font-size:12px;margin-top:10px;line-height:1.5}
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

<nav class="skirt" role="tablist">
  <button id="sk-signalai-b" role="tab" aria-selected="true">Signalai</button>
  <button id="sk-dividendai-b" role="tab" aria-selected="false">Dividendai</button>
</nav>

<section id="sk-signalai">
<div class="zurnalas z" id="zurnalas" hidden></div>

<div class="valdymas">
  <button class="f" id="f-visi" aria-pressed="false">Visi</button>
  <button class="f" id="f-tinkami" aria-pressed="true">Tik be kliūčių</button>
  <button class="f" id="f-eu" aria-pressed="true">EU</button>
  <button class="f" id="f-us" aria-pressed="true">JAV</button>
</div>

<div id="turinys"></div>
</section>

<section id="sk-dividendai" hidden>
<div id="div-turinys"></div>
</section>

<footer>
  Pozicija 18&nbsp;000&nbsp;€. Sąnaudos: EU 10&nbsp;€ (lūžio taškas 0,0556&nbsp;%), JAV 5&nbsp;€ (0,0278&nbsp;%).
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
  if (s.atr_pct) zenkl.push(`<span class="z2">ATR ${nr(s.atr_pct,1)}%</span>`);
  if (s.kritimas_atr) zenkl.push(`<span class="z2">krito ${nr(s.kritimas_atr,1)} ATR</span>`);
  if (s.tarpas_atr) zenkl.push(`<span class="z2">tarpas ${nr(s.tarpas_atr,1)} ATR</span>`);
  zenkl.push(`<span class="z2">signalo amžius ${amzius(s.amzius_min)}</span>`);
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
      <div><span>įėjimas</span><b>${nr(s.ieina)}</b></div>
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
  const svarus   = sig.filter(s => !silpnas(s) && s.tinkamas);
  const ankstyvi = svarus.filter(s => !velyva(s));
  // Isskvepe (vėlyvi) rodomi TIK pasirinkus "Visi" - "Tik be kliūčių"
  // rodo tik tai, i ka dar verta sokti.
  const velyvi   = svarus.filter(velyva);
  rodomi = B.visi ? sig : ankstyvi;
  let h = '';
  if (!sig.length){
    h = '<div class="tuscia">Šiuo metu nė vieno scenarijaus, atitinkančio pasirinkimą.' +
        '<br>Puslapis persikrauna kas 5 min.</div>';
  } else {
    if (ankstyvi.length) h += `<h2>Aktyvūs · ${ankstyvi.length}</h2>
      <div class="tinkl">${ankstyvi.map(kortele).join('')}</div>`;
    if (velyvi.length && B.visi) h += `<h2>Išsikvėpę · per vėlu šokti · ${velyvi.length}</h2>
      <div class="tinkl">${velyvi.map(kortele).join('')}</div>`;
    if (slepti.length && B.visi) h += `<h2>Silpnas atsiėmimas ir blokuoti · ${slepti.length}</h2>
      <div class="tinkl">${slepti.map(kortele).join('')}</div>`;
    const kas = [];
    if (velyvi.length) kas.push('išsikvėpę');
    if (slepti.length) kas.push('krintančio peilio kandidatai (atsiėmė &lt; 0,25 ATR) ir blokuoti');
    const paslepta = velyvi.length + slepti.length;
    if (paslepta && !B.visi) h += `<div class="tuscia">Paslėpta ${paslepta}: `
      + kas.join('; ') + `. Spausk „Visi", jei nori juos matyti.</div>`;
    if (!ankstyvi.length && !(B.visi && velyvi.length))
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
  if (!z){ el.hidden = true; return; }
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

function data_md(x){ return x ? String(x).slice(5) : '–'; }   // "2026-10-15" -> "10-15"

function dividendai(){
  const el = document.getElementById('div-turinys');
  const d = DUOM.dividendai;
  if (!d || !d.eilutes){
    el.innerHTML = '<div class="tuscia">Dividendų duomenų dar nėra (atnaujinami kartą per dieną).</div>';
    return;
  }
  const e = d.eilutes;
  const t = d.atnaujinta ? new Date(d.atnaujinta) : null;
  let h = `<div class="sub" style="margin-top:12px">Paskelbti dividendai su ex-data per ${d.langas_d||30} d. · ${e.length} akcijų`
        + (t ? ' · atnaujinta ' + t.toLocaleDateString('lt-LT') : '') + '</div>';
  if (!e.length){
    el.innerHTML = h + '<div class="tuscia">Per artimiausias dienas paskelbtų dividendų nėra.</div>';
    return;
  }
  h += `<div class="lent"><table><thead><tr>
      <th>Akcija</th><th class="sk">Kaina €</th><th class="sk">Dividendas €/akc.</th>
      <th>Ex-data</th><th>Fiksavimo d.</th><th>Mokėjimo d.</th></tr></thead><tbody>`;
  for (const r of e){
    const pj = (r.pajamingumas!==null && r.pajamingumas!==undefined) ? ` <span class="apyt">(${nr(r.pajamingumas,2)}%)</span>` : '';
    h += `<tr>
      <td><span class="tick" style="font-size:14px">${esc(r.tikeris)}</span>
        <span class="zenk ${esc(r.rinka)}">${esc(r.rinka)}</span></td>
      <td class="sk">${nr(r.kaina_eur)}</td>
      <td class="sk">${r.suma_apytiksle?'<span class="apyt">~</span>':''}${nr(r.div_eur,3)}${pj}</td>
      <td>${data_md(r.ex)}</td>
      <td>${r.irasymo_apytiksle?'<span class="apyt">~</span>':''}${data_md(r.irasymo)}</td>
      <td>${data_md(r.mokejimo)}</td></tr>`;
  }
  h += '</tbody></table></div>';
  h += `<div class="pastaba">Dividendą gauna tas, kas akciją turi <b>prieš ex-datą</b> (pirkti vėliausiai dieną prieš ją).
    „~" – apytiksliai: fiksavimo d. apskaičiuota (JAV = ex-data, EU = ex-data + 1 d. d.), arba suma – paskutinio išmokėto dividendo (Milano akcijos, Yahoo).
    JAV sumos ir kainos perskaičiuotos į EUR. Šaltinis: EODHD.</div>`;
  el.innerHTML = h;
}

function skirtukas(kuris){
  for (const k of ['signalai','dividendai']){
    document.getElementById('sk-'+k).hidden = (k !== kuris);
    document.getElementById('sk-'+k+'-b').setAttribute('aria-selected', k === kuris);
  }
  try { history.replaceState(null, '', kuris === 'dividendai' ? '#dividendai' : location.pathname); } catch(e){}
}
document.getElementById('sk-signalai-b').addEventListener('click', ()=>skirtukas('signalai'));
document.getElementById('sk-dividendai-b').addEventListener('click', ()=>skirtukas('dividendai'));

piesti();
zurnalas(DUOM.zurnalas);
dividendai();
if (location.hash === '#dividendai') skirtukas('dividendai');
</script>
</body>
</html>
'''


def _tinkama(r):
    """Ar zurnalo eilute yra SVARUS signalas (be kliuciu)."""
    v = r.get("tinkamas", True)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "")
    return bool(v)


def _pasiteisino(r):
    """1 scenarijus - tikslas; 2 scenarijus - pelnas po sanaudu (eur > 0)."""
    if str(r.get("tipas") or "").strip() == "2":
        try:
            return float(r.get("eur")) > 0
        except (TypeError, ValueError):
            return False
    return r.get("baigtis") == "tikslas"


def zurnalo_santrauka(z):
    """Santrauka skaiciuojama TIK is tinkamu signalu.

    Zurnale sedi ir blokuoti signalai - tycia, kad matytusi, ka kliutys
    ismete. Bet puslapio antraste buvo skaiciuojama is VISU eiluciu, o
    kalibracija matuoja tik tinkamus, tad du skaiciai apie ta pati dalyka
    niekada nesutapdavo (2026-09-28 nepriklausoma perziura).
    """
    # tik tai, ka rodė kortelės; "baru perziura" eilutes - tyrimui, ne puslapiui
    sv = {k: r for k, r in z.items() if _tinkama(r)
          and str(r.get("saltinis") or "") != "baru perziura"
          and not (str(r.get("sesija") or "") and str(r["sesija"]) < PUSLAPIO_PRADZIA)}
    # "duomenu nera" nera baigtis - tai eilute, kuriai pritruko duomenu.
    # I pataikymo dali jos iskaityti negalima nei i skaitikli, nei i vardikli.
    BAIGTYS = ("stop", "tikslas", "laikas", "slenkantis stop")
    uzd = [r for r in sv.values() if str(r.get("baigtis") or "") in BAIGTYS]
    NEMATUOJAMI = ("duomenu nera", "netvarkinga")
    be_duomenu = sum(1 for r in sv.values()
                     if str(r.get("baigtis") or "") in NEMATUOJAMI)
    eur = []
    for r in uzd:
        try:
            eur.append(float(r["eur"]))
        except Exception:
            pass
    return dict(signalu=len(sv), atviru=len(sv) - len(uzd) - be_duomenu,
                baigtu=len(uzd),
                # "tikslas" puslapyje = scenarijus PASITEISINO, kaip kalibracijoje:
                # 1 scenarijus - pasiektas tikslas; 2 scenarijus tikslo neturi,
                # jam pasiteisinimas = pelnas po sanaudu. Anksciau kiekvienas
                # 2 scenarijus (ir pelningas) cia buvo skaiciuojamas kaip
                # nepataikymas. Puslapio isvaizda nesikeicia.
                tikslo_dalis=(100.0 * sum(1 for r in uzd if _pasiteisino(r)) / len(uzd))
                if uzd else 0.0,
                vid_eur=(sum(eur) / len(eur)) if eur else None)


def _be_nan(o):
    """NaN/Inf -> None. JSON ju neturi, o musu laukai (fonas, ataskaitos)
    teisetai buna NaN."""
    if isinstance(o, dict):
        return {k: _be_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_be_nan(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        v = float(o)
        return v if math.isfinite(v) else None
    return o


DIVIDENDAI = "docs/dividendai.json"


def dividendai_ikelti(kelias=None):
    """Dividendu skirtuko duomenys (dividendai.py, kartą per dieną).
    Jokio poveikio signalams ir kortelems: jei failo nera ar jis sugadintas -
    skirtukas tiesiog rodo, kad duomenu nera."""
    try:
        with open(kelias or DIVIDENDAI, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) and isinstance(d.get("eilutes"), list) else None
    except Exception:
        return None


def puslapis_html(eilutes, z):
    """Puslapi generuoja PATS detektorius, kaip ir senasis dip_reitingas.py.

    Duomenys ikepami i faila, ne siunciami fetch'u: GitHub Pages atiduoda
    viena statini faila, be papildomu uzklausu ir be talpyklos netikrumo.
    Atnaujinimas - meta refresh kas 5 min, t. y. tiksliai tuo ritmu, kuriuo
    workflow perrasO faila.
    """
    duom = dict(atnaujinta=datetime.now(timezone.utc).isoformat(),
                signalai=eilutes, zurnalas=zurnalo_santrauka(z),
                dividendai=dividendai_ikelti())
    # < > & pabegami i \u00xx: kitaip laukas su "</script>" isardytu puslapi.
    # json.dumps ju NEekranuoja, ir pirmoji savitikros versija to nepagavo,
    # nes pati skaldydavo teksta ties tuo paciu "</script>".
    js = (json.dumps(_be_nan(duom), ensure_ascii=False, default=str)
          .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))
    return PUSLAPIO_SABLONAS.replace("__DUOM__", js)

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
        # NIEKADA negrazinam tuscio zodyno: _zurnalo_irasyti() tada perrasytu
        # faila viena paleidimo eilute, ir visas pirmyneiginis testas dingtu
        # del vienos sugadintos eilutes. Kopijuojam sugadinta faila ir
        # nutraukiam - geriau nutruke darbai, nei tyliai dingusi istorija.
        atsarga = ZURNALAS + ".sugadintas"
        try:
            shutil.copyfile(ZURNALAS, atsarga)
        except Exception:
            atsarga = "(kopijos padaryti nepavyko)"
        sys.exit(f"NUTRAUKTA: zurnalo nuskaityti nepavyko ({e}). "
                 f"Failas nepaliestas, kopija: {atsarga}")


def barai_signalams(z, eilutes, visi_barai, perziura=()):
    """Kiekvienam ATVIRAM zurnalo signalui - barai NUO jo gimimo iki dabar.

    Tai ta pati lentele, kuria kalibracijoje gauna baigtis(): bareliai po
    iejimo baro. Todel zurnalas ir kalibracija matuoja identiskai.
    """
    gimimai = {f"{e['rinka']}|{e['tickeris']}|{e['tipas']}|{e['sesija_data']}":
               e.get("pirmas_kartas", e["laikas"]) for e in eilutes}
    gimimai.update({e["_raktas"]: e["laikas"] for e in perziura})
    out = {}
    for raktas, r in list(z.items()) + [(k, dict(gimimas=v, baigtis=""))
                                        for k, v in gimimai.items()]:
        if raktas in out or str(r.get("baigtis") or ""):
            continue
        dalys = raktas.split("|")
        if len(dalys) not in (4, 5):      # 5 - baru perziuros eilute
            continue
        d = visi_barai.get((dalys[0], dalys[1]))
        if d is None:
            continue
        try:
            nuo = pd.Timestamp(str(r["gimimas"]))
            po = d[d.index > nuo]
            # TIK HORIZONTAS_SESIJU sesijos - lygiai tiek, kiek gauna
            # kalibracija. Be sito zurnalas ir kalibracija matuoja skirtingus
            # dalykus, ir zurnalas - pirmyneiginis testas - butu sistemingai
            # pesimistiskesnis uz ji.
            # Kalibracija duoda: likusius GIMIMO sesijos barus + dar
            # (HORIZONTAS_SESIJU - 1) sesijas. Jei signalas gime paskutiniame
            # sesijos bare, gimimo sesija duoda 0 baru, ir paprastas
            # "pirmos 3 sesijos" pjuvis duodavo 3 PILNAS sesijas vietoj 2 -
            # tas pats signalas zurnale +458.7 EUR, kalibracijoje +309.6.
            gim_ses = pd.Timestamp(str(r["gimimas"])).date()
            kiek = HORIZONTAS_SESIJU - (0 if gim_ses in set(po["sesija"]) else 1)
            ses = sorted(set(po["sesija"]))[:max(1, kiek)]
            po = po[po["sesija"].isin(ses)]
        except Exception:
            continue
        if len(po):
            out[raktas] = po
    return out


def zurnalas_atnaujinti(eilutes, barai_pagal_rakta, perziura=()):
    """Pirmyneiginis testas: viena eilute vienam signalui, atnaujinama kas 5 min.

    Baigtis skaiciuojama TA PACIA baigtis() funkcija, kaip ir kalibracijoje.
    Jei zurnalas turetu savo isejimo logika, gautume tiksliai ta klaida, kuri
    sitame projekte kartojosi keturis kartus: modulis skaiciuoja viena, testas
    kita. Todel cia nera NE VIENOS savos taisykles.
    """
    z = zurnalas_ikelti()
    dabar = datetime.now(timezone.utc).isoformat(timespec="seconds")

    nauji = ([(f"{e['rinka']}|{e['tickeris']}|{e['tipas']}|{e['sesija_data']}",
               e, "kortele") for e in eilutes]
             + [(e["_raktas"], e, "baru perziura") for e in perziura])
    for raktas, e, saltinis in nauji:
        if (raktas in z and saltinis == "kortele" and bool(e["tinkamas"])
                and not _tinkama(z[raktas]) and not str(z[raktas].get("baigtis") or "")
                and str(z[raktas].get("saltinis") or "") == "kortele"):
            # pirmas kandidatas buvo netinkamas, dabar - tinkamas: eilute
            # perrasoma nauju (pirmu tinkamu) signalu, kaip kalibracijoje
            del z[raktas]
        if raktas not in z:
            z[raktas] = dict(
                raktas=raktas, gimimas=e["laikas"], rinka=e["rinka"],
                tickeris=e["tickeris"], scenarijus=e["scenarijus"],
                tipas=e["tipas"], sesija=e["sesija_data"],
                ieina=round(e["ieina"], 4), tikslas=e.get("tikslas"),
                stop=round(e["stop"], 4), rr=(round(e["rr"], 2) if np.isfinite(e.get("rr", np.nan)) else ""),
                rizika_eur=round(e.get("rizika_eur", float("nan")), 1),
                sanaudos=e.get("sanaudos", sanaudos_zymei(e["rinka"])),
                atr_pct=round(e.get("atr_pct", float("nan")), 2),
                atr_abs=round(float(e["atr_abs"]), 6),
                tinkamas=bool(e["tinkamas"]),
                kliutys="; ".join(e["kliutys"]),
                paskut_laikas="", paskut_kaina="", mfe_pct="", mae_pct="",
                baigtis="", pelnas_pct="", eur="", minuciu="",
                saltinis=saltinis)

    # atviros eilutes: perskaiciuojam ta pacia baigtis() funkcija
    for raktas, r in z.items():
        if str(r.get("baigtis") or ""):
            continue
        toliau = barai_pagal_rakta.get(raktas)
        if toliau is None or not len(toliau):
            # Baru gali nebuti: skenuota tik viena rinka, tikeris iskrito is
            # universo, duomenu skyle. Anksciau tokia eilute likdavo ATVIRA
            # amzinai ir kaupdavosi puslapio "atvirų" skaitiklyje.
            _uzdaryti_pagal_laika(r, dabar)
            continue
        try:
            sig = _zurnalo_sig(r)
        except Exception as e:
            # Senojo formato eilute (pvz. be atr_abs) NIEKADA neissispres:
            # baigtis() jos suskaiciuoti negali. Anksciau tokia eilute
            # amzinai likdavo "atvira" ir gadino puslapio skaicius -
            # 2026-09-28 ju buvo 14 is 20.
            print(f"  zurnalo eilute {raktas} netvarkinga ({e}) - uzdarau")
            r["baigtis"] = "netvarkinga"
            r["paskut_laikas"] = dabar
            continue
        _zurnalo_eilute(r, toliau, sig, dabar)
    _zurnalo_irasyti(z)
    return z


def _zurnalo_sig(r):
    """Signalas is zurnalo eilutes.

    Sena eilute be atr_abs anksciau kelddavo KeyError, kuris nutraukdavo
    live PRIES docs/index.html irasyma - puslapis tiesiog nustodavo
    atsinaujinti be jokio pranesimo. Dabar tokia eilute praleidziama.
    """
    return dict(tipas=int(r["tipas"]), ieina=float(r["ieina"]),
                tikslas=(float(r["tikslas"])
                         if str(r.get("tikslas") or "") not in ("", "nan")
                         else None),
                stop=float(r["stop"]), atr_abs=float(r["atr_abs"]))


def _zurnalo_eilute(r, toliau, sig, dabar):
    """Atnaujina viena atvira eilute. Baigtis - ta pati baigtis() funkcija."""
    b = baigtis(toliau, sig)
    r["paskut_laikas"] = dabar
    r["paskut_kaina"] = round(float(toliau["Close"].iloc[-1]), 4)
    r["mfe_pct"] = round(b["mfe_pct"], 3)
    r["mae_pct"] = round(b["mae_pct"], 3)

    def uzdaryti(kuo):
        r["baigtis"] = kuo
        r["pelnas_pct"] = round(b["pelnas_pct"], 3)
        # sanaudos - is TOS eilutes rinkos. Senos eilutes be stulpelio
        # gauna ja pagal zyme, ne globalia EU reiksme.
        sn = r.get("sanaudos")
        try:
            sn = float(sn)
        except (TypeError, ValueError):
            sn = sanaudos_zymei(r.get("rinka", "EU"))
        r["eur"] = round(b["pelnas_pct"] / 100 * POZICIJA - sn, 1)
        r["minuciu"] = b["minuciu"]

    # uzdarom TIK tada, kai tikrai issisprende. "laikas" gyvai reiskia tik
    # tai, kad kol kas neissisprende - horizontas dar nesibaige.
    if b["baigtis"] in ("stop", "tikslas"):
        # 2 scenarijus neturi tikslo - jis baigiasi SLENKANCIU stop'u, kuris
        # dazniausiai jau buna virs iejimo (pelnas). Zyma "stop" zurnale
        # atrode kaip nuostolis, todel 2 scenarijaus stop'as vadinamas
        # atskirai. Rezultatas (pelnas_pct, eur) nesikeicia.
        uzdaryti("slenkantis stop" if (sig.get("tipas") == 2 and b["baigtis"] == "stop")
                 else b["baigtis"])
        return
    # "laikas" - TIK kai baruose yra VISOS horizonto sesijos ir paskutine jau
    # pasibaigusi. Anksciau buvo skaiciuojamos darbo dienos, tad per svente
    # (JAV Padekos diena, EU Kaledos) eilute uzsidarydavo viena sesija per
    # anksti - kitaip nei kalibracijoje.
    if _horizontas_baigtas(r, toliau, dabar):
        uzdaryti("laikas")
        return
    # Saugiklis: jei sesiju taip ir netrūksta (duomenu skyle, isbrauktas
    # tickeris), eilute neturi likti atvira amzinai.
    if _praejo_darbo_dienu(r, dabar) >= HORIZONTAS_SESIJU + HORIZONTO_ATSARGA_D:
        uzdaryti("laikas")


def _dabar_ts(dabar):
    """Paleidimo laikas is `dabar` (ISO eilute); jei neiskaitoma - dabar."""
    try:
        t = pd.Timestamp(str(dabar))
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    except Exception:
        return pd.Timestamp.now(tz="UTC")


def _praejo_darbo_dienu(r, dabar=None):
    """Darbo dienos nuo signalo sesijos. Nezinoma data -> labai daug (uzdaryti)."""
    try:
        nuo = datetime.fromisoformat(str(r["sesija"])).date()
        return len(pd.bdate_range(nuo, _dabar_ts(dabar).date())) - 1
    except Exception:
        return 10 ** 6


def _horizontas_baigtas(r, toliau, dabar=None):
    """Ar `toliau` turi VISAS horizonto sesijas, ir paskutine jau pasibaigusi.

    Tiek pat sesiju, kiek duoda kalibracija ir barai_signalams(): gimimo sesija
    (jei po gimimo joje dar buvo baru) + likusios iki HORIZONTAS_SESIJU.
    Sesijos skaiciuojamos pagal TIKRUS barus, tad svente ar pusdienis ju
    neišsaugo is skaiciaus.
    """
    if toliau is None or not len(toliau) or "sesija" not in toliau:
        return False
    rinka = ZYME_I_RINKA.get(str(r.get("rinka") or "EU"), "eu")
    tz = RINKOS[rinka]["tz"]
    try:
        g = pd.Timestamp(str(r.get("gimimas") or r["sesija"]))
        gim_ses = (g.tz_convert(tz) if g.tzinfo is not None else g).date()
    except Exception:
        return False
    sesijos = sorted(set(toliau["sesija"]))
    reikia = HORIZONTAS_SESIJU - (0 if gim_ses in sesijos else 1)
    if len(sesijos) < reikia:
        return False
    pask = sesijos[reikia - 1]
    dabar_v = _dabar_ts(dabar).tz_convert(tz)
    if pask < dabar_v.date():
        return True
    return (pask == dabar_v.date() and
            dabar_v.hour * 60 + dabar_v.minute >= RINKOS[rinka]["uzdarymas"])


def _uzdaryti_pagal_laika(r, dabar):
    """Uzdaro eilute, kuriai baru NEGAUNAME, tik praejus horizontui IR atsargai.

    Anksciau uztekdavo vieno Yahoo sutrikimo horizonto dieną - eilute buvo
    pazymima "duomenu nera" visam laikui, nors kitas paleidimas jau butu
    gaves barus ir ismataves baigti.
    """
    if _praejo_darbo_dienu(r, dabar) >= HORIZONTAS_SESIJU + HORIZONTO_ATSARGA_D:
        r["baigtis"] = "duomenu nera"
        r["paskut_laikas"] = dabar


def _zurnalo_irasyti(z):
    os.makedirs(os.path.dirname(ZURNALAS), exist_ok=True)
    pd.DataFrame(list(z.values()), columns=ZURNALO_STULPELIAI).to_csv(
        ZURNALAS, index=False)
    atviros = sum(1 for r in z.values() if not str(r.get("baigtis") or ""))
    print(f"  zurnalas: {len(z)} eiluciu ({atviros} atviros)")


def busenos_sena(busena, raktas, s):
    """Kokia issaugota busena taikoma siam kandidatui (None - naujas signalas).

    Kaip kalibracijoje: dienos signala fiksuoja PIRMAS TINKAMAS kandidatas.
    Iki 2026-10-05 busena fiksuodavo pirma bet koki, tad veliau atsirades
    tinkamas gaudavo sena (netinkama) iejima ir kortele netapdavo (6 m.:
    ~0.9 % sesiju). Senuose irasuose rakto "tinkamas" nera - jie laikomi
    tinkamais, t. y. ju elgsena nesikeicia.
    """
    sena = busena.get(raktas)
    if sena and not sena.get("tinkamas", True) and s.get("tinkamas"):
        return None
    return sena


def atkurti(s, sena, kd, kaina, rinka="eu"):
    """Atstato signala is busenos ir PERSKAICIUOJA viska, kas nuo jos priklauso.

    Kortele turi rodyti viena nuosekliai suderinta trijule: ieina, stop,
    tikslas. Anksciau buvo atstatomi tik ieina ir stop, o tikslas likdavo
    perskaiciuotas nuo DABARTINES kainos (tikslas = min(R_pilnas, kaina+3*ATR)),
    todel rodomas R:R augdavo 2.60 -> 4.70 vien del to, kad kaina kyla, nors
    nei iejimas, nei stop'as nepasikeite. Ir dar anksciau _bendra() apskritai
    nebuvo perskaiciuojama, tad rizika eurais rodyta 1396 vietoj 476.
    """
    # .get visur: nepilnas ar senas busena.json anksciau keldavo KeyError ir
    # nutraukdavo VISA skenavima.
    s["ieina"] = sena.get("ieina", s["ieina"])
    s["stop"] = sena.get("stop", s["stop"])
    if sena.get("R"):
        s["R"] = s["tikslas"] = sena["R"]
    if sena.get("R_pilnas"):
        s["R_pilnas"] = sena["R_pilnas"]
    # atsiemimas priklauso nuo IEJIMO, ne nuo dabartines kainos - kitaip
    # paslepta kortele veliau "atsirastu" su reiksme, kuri jos iejimui
    # niekada netiko, ir puslapis slėptu pagal viena, o kalibracija
    # matuotu kita.
    if sena.get("atsiemimas_atr") is not None:
        s["atsiemimas_atr"] = sena["atsiemimas_atr"]
        s["silpnas_atsiemimas"] = bool(sena["atsiemimas_atr"] < 0.25)
    s["pirmas_kartas"] = sena.get("laikas", s.get("laikas"))
    s["kliutys"] = _bendra(s, kd, rinka)
    s["tinkamas"] = len(s["kliutys"]) == 0
    L = sena.get("L", s.get("L"))
    rp = s.get("R_pilnas")
    if rp and L is not None and rp > L:
        s["progresas"] = min(1.0, max(0.0, (kaina - L) / (rp - L)))
    return s


def _amzius_min(pirmas_utc, pirmas_baras, dabar_baras):
    """Signalo amzius minutemis.

    Pirmenybe sieniniam laikrodziui (kiek laiko PRAEJO nuo aptikimo). Jei
    busenoje senas irasas be laikas_utc, grizt prie baru laiko skirtumo.
    """
    try:
        if pirmas_utc:
            d = (datetime.now(timezone.utc) -
                 datetime.fromisoformat(str(pirmas_utc)))
            return max(0, int(d.total_seconds() // 60))
    except Exception:
        pass
    try:
        return max(0, int((pd.Timestamp(dabar_baras) -
                           pd.Timestamp(pirmas_baras)).total_seconds() // 60))
    except Exception:
        return 0


SESIJOS_PABAIGOS_ATSARGA_MIN = 25   # Yahoo EU duomenys veluoja ~15-20 min


def _uzbaigti_barai(sesija, dabar_ts=None, rinka="eu"):
    """Tik UZBAIGTI barai: kol sesija vyksta, paskutinis grazintas baras
    visada laikomas nebaigtu.

    Ciklas sukasi 5 min zymuo + 60 s, tad JAV paskutinis baras turi tik ~1 is
    5 minuciu (auditas 2026-09-29: ORCL 10:15 vs 10:25). EU Yahoo duomenys
    veluoja ~15-20 min, ir paskutinis grazintas EU baras irgi buna dalinis,
    nors pagal laikrodi jis "seniai baigesi" - todel ankstesne laikrodzio
    taisykle EU nepadejo: savaites zurnale (09-30..10-02) EU kortele
    signala pamatydavo mediana 37 min veliau nei baru perziura, o 40% EU
    signalu kortele visai praleido; JAV - 0 min ir 2 is 37.
    Baras laikomas uzbaigtu, kai po jo jau yra kitas baras, arba kai sesija
    pasibaige (uzdarymas + SESIJOS_PABAIGOS_ATSARGA_MIN).
    """
    if not len(sesija):
        return sesija
    if dabar_ts is None:
        dabar_ts = pd.Timestamp.now(tz="UTC")
    tz = RINKOS[rinka]["tz"]
    v = dabar_ts.tz_convert(tz)
    pask = sesija.index[-1]
    ses_d = (pask.tz_convert(tz) if pask.tzinfo is not None else pask).date()
    if v.date() > ses_d or (v.hour * 60 + v.minute >=
                            RINKOS[rinka]["uzdarymas"] + SESIJOS_PABAIGOS_ATSARGA_MIN):
        return sesija
    return sesija.iloc[:-1]


def perziureti_dienos_barus(sesija, kd, rinka, zyme, t, ses, jau, dabar_ts=None):
    """Kiekvieno scenarijaus PIRMAS tinkamas signalas sios dienos baruose.

    Tas pats ciklas kaip kalibracijoje: baras po baro, aptikti() su baru
    istorija iki jo. Imami tik UZBAIGTI barai - paskutinis gyvas baras dar
    formuojasi. Scenarijus, jau irasytas zurnale (raktas `jau`), nebeieskomas.
    Korteliu tai nelieia: grazinami tik zurnalo irasai.
    """
    if dabar_ts is None:
        dabar_ts = pd.Timestamp.now(tz="UTC")
    reikia = {tp for tp in (1, 2)
              if f"{zyme}|{t}|{tp}|{ses}|{PERZIUROS_ZYME}" not in jau}
    out = []
    n_baigtu = len(_uzbaigti_barai(sesija, dabar_ts, rinka))
    for i in range(S2_ORB_BARU + 3, n_baigtu):
        if not reikia:
            break
        iki = RINKOS[rinka]["uzdarymas"] - int(sesija["minute"].iloc[i])
        for sg in aptikti(sesija.iloc[:i + 1], kd, iki, rinka):
            if sg["tipas"] not in reikia or not sg["tinkamas"]:
                continue
            reikia.discard(sg["tipas"])
            sg.update(rinka=zyme, tickeris=t, laikas=str(sesija.index[i]),
                      sesija_data=str(ses), atr_pct=float(kd["atr_pct"]),
                      _raktas=f"{zyme}|{t}|{sg['tipas']}|{ses}|{PERZIUROS_ZYME}")
            out.append(sg)
    return out


def paleisti_live(rinkos):
    busena = ikelti_busena()
    dabar_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    eilutes = []
    perziura = []            # zurnalui: pirmas scenarijaus baras (ne kortele)
    jau_zurnale = set(zurnalas_ikelti())
    visi_barai = {}          # (zyme, tickeris) -> visi turimi 5 min barai
    for rinka in rinkos:
        zyme = RINKOS[rinka]["zyme"]
        rod, barai = parsisiusti(rinka, LIVE_DIENOS)
        fonas = rinkos_fonas(barai, rod)
        div = dividendu_kalendorius(list(barai))
        # KURI DIENA yra TOJE rinkoje. Be sio patikrinimo pries JAV atidaryma
        # paskutine turima sesija yra praeitas penktadienis, ir detektorius
        # rodydavo penktadienio signalus kaip siandieninius - su "0 min"
        # amziumi, nes barai nebejuda. Rasta 2026-09-28 gyvai.
        siandien_rinkoje = datetime.now(ZoneInfo(RINKOS[rinka]["tz"])).date()
        pasenusiu = 0
        for t, d in barai.items():
            visi_barai[(zyme, t)] = d          # zurnalui reikia VISU baru
            ses = d["sesija"].iloc[-1]
            if ses != siandien_rinkoje:
                # Rinka uzdaryta arba duomenys pasene. Zurnala vis tiek
                # atnaujinam is siu baru - tik korteliu neberodom.
                pasenusiu += 1
                continue
            sesija = d[d["sesija"] == ses]
            if len(sesija) < S2_ORB_BARU + 4:
                continue
            kd = _kd(rod[t], ses)
            if kd is None:
                continue
            try:
                perziura += perziureti_dienos_barus(sesija, kd, rinka, zyme, t,
                                                    ses, jau_zurnale)
            except Exception as e:
                kl = f"baru perziura: {type(e).__name__}: {e}"
                _KLAIDOS[kl] = _KLAIDOS.get(kl, 0) + 1
            # Signalas tikrinamas tik UZBAIGTAIS barais (kaip kalibracijoje);
            # dabartine kaina korteles progresui - is paskutinio baro.
            det = _uzbaigti_barai(sesija, rinka=rinka)
            if len(det) < S2_ORB_BARU + 4:
                continue
            iki = (RINKOS[rinka]["uzdarymas"] - int(det["minute"].iloc[-1]))
            for s in aptikti(det, kd, iki, rinka):
                raktas = f"{zyme}|{t}|{s['tipas']}|{ses}"
                dabar = str(det.index[-1])
                sena = busenos_sena(busena, raktas, s)
                if sena:
                    atkurti(s, sena, kd, float(sesija["Close"].iloc[-1]), rinka)
                else:
                    busena[raktas] = dict(L=s["L"], R=s.get("R"), stop=s["stop"],
                                          R_pilnas=s.get("R_pilnas"),
                                          atsiemimas_atr=s.get("atsiemimas_atr"),
                                          ieina=s["ieina"], laikas=dabar,
                                          laikas_utc=dabar_utc,
                                          tinkamas=bool(s["tinkamas"]))
                    s["pirmas_kartas"] = dabar
                # AMZIUS skaiciuojamas sieniniu laikrodziu, ne baru laiku.
                # Baru laiku jis sustodavo, kai rinka uzsidarydavo: penktadieni
                # aptiktas signalas pirmadieni vis dar rode "0 min", nes
                # paskutinis baras nepajudejo.
                amz = _amzius_min(sena.get("laikas_utc") if sena else dabar_utc,
                                  sena.get("laikas") if sena else dabar, dabar)
                s.update(_fono_laukai(fonas, d.index[-1], sesija, len(sesija) - 1))
                s.update(rinka=zyme, tickeris=t, laikas=dabar,
                         sesija_data=str(ses),
                         atr_pct=float(kd["atr_pct"]), amzius_min=amz,
                         dividendas=div.get(t), _ses=ses)
                eilutes.append(s)

        if pasenusiu:
            print(f"  {zyme}: {pasenusiu} akciju paskutine sesija ne siandienos "
                  f"({siandien_rinkoje}) - rinka uzdaryta, korteliu nerodom")

    # Ataskaitu datos - TIK toms akcijoms, kurios turi signala. Bendram
    # universui tai butu 357 uzklausos kas 5 minutes, t.y. garantuotas
    # apribojimas is saltinio puses.
    if eilutes:
        ats = ataskaitu_kalendorius(sorted({e["tickeris"] for e in eilutes}))
        for e in eilutes:
            e["iki_ataskaitos"] = dienu_iki_ataskaitos(ats.get(e["tickeris"]),
                                                       e.pop("_ses"))

    # Valom pagal paskutine SKENUOTA sesija, ne pagal konteinerio data:
    # savaitgali ar pries atidaryma jos nesutampa, ir busena buvo trinama
    # kas paleidima - tada atkurti() niekada nesuveikdavo ir kortele kas
    # 5 min perkainuodavo iejima nuo dabartines kainos.
    ses_datos = {k.rsplit("|", 1)[-1] for k in busena}
    naujausia = max(ses_datos) if ses_datos else ""
    issaugoti_busena({k: v for k, v in busena.items()
                      if k.endswith(naujausia)})
    os.makedirs("docs", exist_ok=True)
    with open("docs/detektorius.json", "w", encoding="utf-8") as f:
        # allow_nan=False: NaN nera JSON. Puslapis islikdavo tik todel, kad
        # jame duomenys yra JS literalas, kur NaN legalus - bet paskelbtas
        # docs/detektorius.json buvo neisparsinamas (jq, JSON.parse).
        json.dump(dict(atnaujinta=datetime.now(timezone.utc).isoformat(),
                       signalai=_be_nan(eilutes)), f, ensure_ascii=False,
                  indent=1, default=str, allow_nan=False)
    if _KLAIDOS:
        print("\n  SCENARIJU KLAIDOS:")
        for k, n in sorted(_KLAIDOS.items(), key=lambda x: -x[1])[:5]:
            print(f"    {n:>6}x  {k}")
        _KLAIDOS.clear()
    z = zurnalas_atnaujinti(eilutes, barai_signalams(zurnalas_ikelti(), eilutes,
                                                     visi_barai, perziura),
                            perziura)
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
        _MATAVIMAI.extend([f"{zyme} matyta", f"{zyme} NEMATYTA"])
        rod, barai = parsisiusti(rinka, dienos)
        fonas = rinkos_fonas(barai, rod)
        print(f"  rinkos fonas: {len(fonas)} laiko tasku")
        atask = ataskaitu_kalendorius(list(barai))

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
                # PRADZIA turi sutapti su live: live reikalauja
                # len(sesija) >= S2_ORB_BARU + 4 ir vertina PASKUTINI bara,
                # t.y. anksciausias pasiekiamas indeksas yra S2_ORB_BARU + 3.
                # Kalibracija pradejo nuo +1, tad matavo du iejimus per
                # sesija, kuriu saskaita niekada negautu.
                for i in range(S2_ORB_BARU + 3, len(sd)):
                    iki = (RINKOS[rinka]["uzdarymas"] - int(sd["minute"].iloc[i]))
                    for s in aptikti(sd.iloc[:i + 1], kd, iki, rinka):
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
                        s.update(_fono_laukai(fonas, sd.index[i], sd, i))
                        s["iki_ataskaitos"] = dienu_iki_ataskaitos(
                            atask.get(t), ses)
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
            _DALIS[0] = f"{zyme} matyta"
            ataskaita([e for e in ivykiai if e["sesija"] <= riba], zyme, True)
            print(f"\n  ---- {zyme}: 2-A PUSE (NEMATYTA) ----")
            _DALIS[0] = f"{zyme} NEMATYTA"
            ataskaita([e for e in ivykiai if e["sesija"] > riba], zyme, True)
            _DALIS[0] = ""


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
        eur = _eur(gg, pc)
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
              f"{np.mean(gg.assign(p=gg[pc]).groupby('sesija')['p'].mean()):>8.2f}"
              f"{np.mean(pdd):>9.2f}"
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
    g["eur"] = _eur(g)
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
        print(f"    {nm:<16}{len(gg):>6}{np.mean(pdd):>10.2f}"
              f"{(gg['baigtis']=='tikslas').mean()*100:>6.1f}%"
              f"{(gg['baigtis']=='stop').mean()*100:>6.1f}%"
              f"  [{lo:>6.2f},{hi:>6.2f}]{'  <<<' if lo > 0 else ''}")


_KAND_REZ = {}          # (pjuvis, matavimas) -> (apacia EUR, virsus EUR, N)
_DALIS = [""]           # kurio matavimo dabar esame (nustato ataskaita())
_MATAVIMAI = []         # kurie matavimai TUREJO ivykti (nustato kalibracija)


def _eur(df, stulp="pelnas_pct"):
    """Rezultatas eurais su TOS eilutes rinkos sanaudomis.

    Iki 2026-09-28 visur buvo viena 10 EUR reiksme, tad JAV sandoriai buvo
    nubausti dvigubai uz sanaudas, kuriu nepatyre.
    """
    sn = (pd.to_numeric(df["sanaudos"], errors="coerce")
          if "sanaudos" in df else pd.Series(np.nan, index=df.index))
    if "rinka" in df:
        sn = sn.fillna(df["rinka"].map(sanaudos_zymei))
    return df[stulp] / 100 * POZICIJA - sn.fillna(SANAUDOS_EUR)


def pjuviai(df, tik_kandidatai=False):
    """KANDIDATAI I FILTRUS (2 pakopa).

    Cia NIEKAS neblokuojama. Rodoma tik tam, kad matytusi, kuri riba galetu
    uzsidirbti teise egzistuoti. Riba priimama tik jeigu tas pats pjuvis
    laikosi IR nematytoje puseje, IR kitoje rinkoje. Butent sio zingsnio
    truko reitinguotojui: ten ribos buvo prielaidos, ne isvados.
    """
    print("\n  KANDIDATAI I FILTRUS (nieko neblokuoja - tik matoma)")
    KAND = ("rinkos_pokytis", "platumas", "plat_d30", "santykinis",
            "atsiemimas_atr", "baru_nuo_dugno", "nesekmes",
            "virs_proverzio_atr")
    # iki_ataskaitos CIA NERA tycia: kalendorius paimamas SIANDIEN ir
    # taikomas visoms praeities sesijoms, o [-12:] dar ir nukerpa datas.
    # Vadinasi, praeityje jis "zinojo" tai, ko tuo metu nebuvo. Pjuvis
    # spausdinamas, bet filtru tapti negali, kol nebus datu su laiko zyme.
    df = df.copy()
    df["eur"] = _eur(df)
    df["valanda"] = (df["minute"] // 60) if "minute" in df else np.nan

    def pjuvis(pav, stulp, kvantiliai=True):
        if stulp not in df or df[stulp].isna().all():
            return
        if tik_kandidatai and stulp not in KAND:
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
        kraštai = []
        for k, gg in g.groupby("_gr", observed=True):
            et = (f"{k.left:,.1f} .. {k.right:,.1f}"
                  if hasattr(k, "left") else str(k))
            kraštai.append((float(gg["eur"].mean()), len(gg)))
            print(f"    {et:<22}{len(gg):>6}{gg['eur'].mean():>10.1f}"
                  f"{gg['mfe_pct'].mean():>8.2f}"
                  f"{(gg['baigtis']=='tikslas').mean()*100:>6.0f}%"
                  f"{(gg['baigtis']=='stop').mean()*100:>6.0f}%")
        # Isaugom KRASTUS verdiktui. Riba priimama tik tada, kai tas pats
        # skirtumas matomas visuose keturiuose matavimuose - be sito
        # pjuviai butu tiesiog skaiciai, is kuriu galima issirinkti patinkanti.
        if _DALIS[0] and stulp in KAND and len(kraštai) >= 2:
            # Kraštine grupe su 1 eilute duodavo +-200 EUR "skirtumus"
            # (nesekmes = 10, N = 1). Verdiktui paduodam MAZESNI kraštini
            # dydi, o ne bendra N.
            _KAND_REZ[(stulp, _DALIS[0])] = (
                kraštai[0][0], kraštai[-1][0],
                min(kraštai[0][1], kraštai[-1][1]))

    pjuvis("pagal ATR (ar didesnis judrumas kenkia, ar padeda?)", "atr_pct")
    pjuvis("pagal ATSIEMIMO dydi, ATR vienetais (peilio kandidatas)",
           "atsiemimas_atr")
    pjuvis("pagal RINKOS pokyti (universo mediana nuo atidarymo, %)",
           "rinkos_pokytis")
    pjuvis("pagal PLATUMA (% universo virs vakar uzdarymo)", "platumas")
    pjuvis("pagal PLATUMO pokyti per 30 min", "plat_d30")
    pjuvis("pagal SANTYKINI stipruma (savo - rinkos, %)", "santykinis")
    pjuvis("pagal atstuma VIRS PROVERZIO, ATR (2 scenarijus; is cia lubos)",
           "virs_proverzio_atr")
    pjuvis("pagal baru NUO DUGNO", "baru_nuo_dugno")
    pjuvis("pagal NESEKMIU skaiciu (kiek kartu jau prarado lygi)",
           "nesekmes", kvantiliai=False)
    pjuvis("pagal atstuma IKI ATASKAITOS (dienomis; <0 - jau buvo)",
           "iki_ataskaitos")
    pjuvis("pagal kritimo gyli, ATR vienetais", "kritimas_atr")
    pjuvis("pagal kelia iki buvusio lygio, ATR vienetais", "kelias_atr")
    pjuvis("pagal rizika eurais", "rizika_eur")
    pjuvis("pagal iejimo valanda", "valanda", kvantiliai=False)


def kandidatu_verdiktas():
    """Ar kandidatas laikosi VISUOSE keturiuose matavimuose?

    Keturi matavimai = (EU, JAV) x (matyta puse, NEMATYTA puse). Filtras
    priimamas tik tada, kai auksciausio ir zemiausio ketvircio skirtumas
    turi TA PACIA krypti visur ir niekur nera mazesnis uz sanaudas.
    Butent sito zingsnio truko reitinguotojui.
    """
    if not _KAND_REZ:
        return
    # Matavimu sarasas imamas is to, kas TUREJO ivykti, o ne is to, kas
    # pavyko. Kitaip rinka be signalu tiesiog dingsta is lenteles, ir
    # kandidatas gauna "TINKA" turedamas tris matavimus is keturiu -
    # butent taip ir atsitiko pirmame bandyme 2026-09-28.
    matavimai = _MATAVIMAI or sorted({m for _, m in _KAND_REZ})
    print("\n" + "=" * 84)
    print("KANDIDATU VERDIKTAS (skirtumas: virsutinis ketvirtis - apatinis, EUR)")
    print("=" * 84)
    print(f"  {'kandidatas':<18}" + "".join(f"{m:>15}" for m in matavimai)
          + f"{'  verdiktas':>14}")
    for k in sorted({k for k, _ in _KAND_REZ}):
        sk = [_KAND_REZ.get((k, m)) for m in matavimai]
        eil = "".join(f"{(x[1]-x[0]):>15.1f}" if x else f"{'-':>15}" for x in sk)
        turim = [x for x in sk if x]
        if len(turim) < len(matavimai) or any(x[2] < 40 for x in turim):
            v = "per mazai duomenu"
        else:
            d = [x[1] - x[0] for x in turim]
            riba = max(SANAUDOS.values())   # griezciausios sanaudos
            v = ("TINKA" if (all(y > riba for y in d)
                             or all(y < -riba for y in d))
                 else "nelaikosi")
        print(f"  {k:<18}{eil}  {v}")
    print("\n  TINKA reiskia: kryptis ta pati visur IR skirtumas didesnis uz")
    print("  sanaudas (10 EUR). Tik tokia riba gali tapti filtru. 'nelaikosi'")
    print("  reiskia, kad puseje matavimu zenklas kitoks - t.y. tai triuksmas.")


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
    # DU svoriai, nes jie reiskia skirtingus dalykus ir 2026-09-28 logas
    # parode, kad skirtumas nemazas: "% diena" yra tai, ka uzdirbtu
    # atsitiktinai paimtas vienas tos dienos signalas, o "% sand." - visu
    # sandoriu vidurkis, t.y. arciau to, kas gaunasi imant po kelis per
    # aktyvia diena. Anksciau abu buvo spausdinami, bet skirtingose
    # lentelese ir tuo paciu pavadinimu.
    print(f"\n{'SCENARIJUS':<26}{'N':>6}{'per d.':>7}{'tiksl':>7}{'stop':>7}"
          f"{'laikas':>8}{'% diena':>9}{'% sand.':>9}{'EUR':>8}{'MFE %':>8}"
          f"{'MAE %':>8}{'val.':>6}{'95% EUR':>19}")
    print("-" * 124)
    rng = np.random.default_rng(42)
    for nm, g in df.groupby("scenarijus"):
        eur = _eur(g)
        pdd = g.assign(eur=eur).groupby("sesija")["eur"].mean().values
        ppd = g.groupby("sesija")["pelnas_pct"].mean().values   # tas pats svoris
        if len(pdd) >= 10:
            bs = [rng.choice(pdd, len(pdd), replace=True).mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        print(f"{nm:<26}{len(g):>6}{len(g)/max(1,sesiju):>7.1f}"
              f"{(g['baigtis']=='tikslas').mean()*100:>6.1f}%"
              f"{(g['baigtis']=='stop').mean()*100:>6.1f}%"
              f"{(g['baigtis']=='laikas').mean()*100:>7.1f}%"
              f"{np.mean(ppd):>9.2f}{g['pelnas_pct'].mean():>9.2f}"
              f"{np.mean(pdd):>8.2f}"
              f"{g['mfe_pct'].mean():>8.2f}{g['mae_pct'].mean():>8.2f}"
              f"{g['minuciu'].median()/60:>6.1f}  [{lo:>6.2f},{hi:>6.2f}]"
              f"{'  <<<' if lo > 0 else ''}")
    variantu_lentele(df)
    kandidato_testas(df)
    if trumpai:
        pjuviai(df, tik_kandidatai=True)
        return
    pjuviai(df)
    print(f"\n  Pozicija {POZICIJA:.0f} EUR. Sanaudos: EU {SANAUDOS['eu']:.0f} EUR "
          f"(luzis {luzio_taskas('eu'):.4f}%), JAV {SANAUDOS['us']:.0f} EUR "
          f"(luzis {luzio_taskas('us'):.4f}%).")
    print(f"  Horizontas {HORIZONTAS_SESIJU} sesijos,")
    print("  pozicija nesama per nakti. MFE - kiek daugiausia buvo naudai;")
    print("  jei MFE dideles, o 'vid %' mazas, klaida yra isejimo taisykleje.")
    print("  EUR ir 95% intervalas - abu DIENOS vidurkiai (ne sandorio), tad\n"
          "  taskinis ivertis visada yra savo intervalo viduje.")


# ================================================================ savitikra

def _saugiai(f, *a):
    """Grazina isimti vietoj to, kad ja keltu - savitikros patogumui."""
    try:
        return f(*a)
    except Exception as e:
        return e


def savitikra():
    """Etalonai is Manto nurodytu atveju. Veikia BE tinklo.

    Tikrinama tai, kas kainavo brangiausiai: kad kodas darytu ta, kas
    aprasyta, ir kad scenarijai nepersidengtu.
    """
    ok = True
    # Kursas prisegamas: savitikra privalo veikti BE tinklo ir duoti ta pati
    # rezultata kiekviena karta. Kitaip JAV ribu testai priklausytu nuo to,
    # ar tuo metu pavyko atsiusti EUR/USD.
    _KURSAS.clear()
    _KURSAS["v"] = 1.10

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

    # --- besiformuojantis baras (auditas 2026-09-29) ----------------------
    _ix = pd.date_range("2026-07-24 10:00", periods=4, freq="5min", tz="Europe/Berlin")
    _sd = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0]}, index=_ix)
    _B = lambda h: pd.Timestamp(f"2026-07-24 {h}", tz="Europe/Berlin").tz_convert("UTC")
    tikrinti("uzbaigti barai: sesijai vykstant paskutinis (besiformuojantis) atmetamas",
             len(_uzbaigti_barai(_sd, _B("10:16"), "eu")), 3)
    tikrinti("uzbaigti barai: EU veluojantys duomenys - paskutinis atmetamas, "
             "nors pagal laikrodi 'seniai baigesi'",
             len(_uzbaigti_barai(_sd, _B("10:40"), "eu")), 3)
    tikrinti("uzbaigti barai: po uzdarymo + atsargos - visi",
             len(_uzbaigti_barai(_sd, _B("17:55"), "eu")), 4)
    tikrinti("uzbaigti barai: dar ne po atsargos (17:54) - paskutinis atmetamas",
             len(_uzbaigti_barai(_sd, _B("17:54"), "eu")), 3)
    tikrinti("uzbaigti barai: kita diena - visi",
             len(_uzbaigti_barai(_sd, _B("10:16") + pd.Timedelta(days=1), "eu")), 4)
    _ixu = pd.date_range("2026-07-24 15:40", periods=4, freq="5min", tz="America/New_York")
    _sdu = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0]}, index=_ixu)
    _N = lambda h: pd.Timestamp(f"2026-07-24 {h}", tz="America/New_York").tz_convert("UTC")
    tikrinti("uzbaigti barai: JAV iki 16:25 NY - paskutinis atmetamas, nuo 16:25 - visi",
             (len(_uzbaigti_barai(_sdu, _N("16:24"), "us")),
              len(_uzbaigti_barai(_sdu, _N("16:25"), "us"))), (3, 4))

    # --- KIEKVIENA salyga su riba iš ABIEJŲ pusių (auditas: isemus apyvartos
    # ar VWAP salyga, savitikra buvo zalia) ----------------------------------
    def _langas(f, sd, k):
        for i in range(S2_ORB_BARU + 1, len(sd)):
            if f(sd.iloc[:i + 1], k):
                return sd.iloc[:i + 1].copy()
        return None

    def _pk(w, stulp, reiksme, eil=-1):
        w2 = w.copy()
        if isinstance(eil, slice):
            w2.iloc[eil, w2.columns.get_loc(stulp)] = reiksme
        else:
            w2.iloc[eil, w2.columns.get_loc(stulp)] = reiksme
        return w2
    w1 = _langas(scenarijus_1, atsok, k_sap2)
    tikrinti("1 scen. bazinis langas turi signala", w1 is not None, True)
    if w1 is not None:
        k1 = float(w1["Close"].iloc[-1])
        tikrinti("1 scen. APYVARTA 1.19x -> nera, lygiai riba ir 1.21x -> yra",
                 (scenarijus_1(_pk(w1, "apyv_santykis", 1.19), k_sap2) is None,
                  scenarijus_1(_pk(w1, "apyv_santykis", S1_MIN_APYV_SANTYKIS), k_sap2) is not None,
                  scenarijus_1(_pk(w1, "apyv_santykis", 1.21), k_sap2) is not None),
                 (True, True, True))
        tikrinti("1 scen. VWAP virs kainos -> nera, po kaina -> yra",
                 (scenarijus_1(_pk(w1, "vwap", k1 * 1.001), k_sap2) is None,
                  scenarijus_1(_pk(w1, "vwap", k1 * 0.999), k_sap2) is not None),
                 (True, True))
    w2 = _langas(scenarijus_2, tarpas, k_ady)
    tikrinti("2 scen. bazinis langas turi signala", w2 is not None, True)
    if w2 is not None:
        k2 = float(w2["Close"].iloc[-1])
        o6 = slice(0, S2_ORB_BARU)
        tikrinti("2 scen. 30 min APYVARTA 1.9x -> nera, 2.1x -> yra",
                 (scenarijus_2(_pk(w2, "apyv_santykis", 1.9, o6), k_ady) is None,
                  scenarijus_2(_pk(w2, "apyv_santykis", 2.1, o6), k_ady) is not None),
                 (True, True))
        # MEDIANA, ne maksimumas: vienas didelis baras 30 min neatstoja
        _w2m = _pk(w2, "apyv_santykis", 1.9, o6)
        _w2m.iloc[0, _w2m.columns.get_loc("apyv_santykis")] = 9.0
        tikrinti("2 scen. 30 min apyvarta - MEDIANA (vienas 9x baras nepakanka)",
                 scenarijus_2(_w2m, k_ady) is None, True)
        tikrinti("2 scen. VWAP virs kainos -> nera",
                 scenarijus_2(_pk(w2, "vwap", k2 * 1.001), k_ady) is None, True)
        tikrinti("2 scen. ORB auksciau kainos -> nera",
                 scenarijus_2(_pk(w2, "High", k2 * 1.001, 0), k_ady) is None, True)
        tikrinti("2 scen. tarpas uzpildytas (Low < vakar uzd.) -> nera, vos virs -> yra",
                 (scenarijus_2(_pk(w2, "Low", 909.99, S2_ORB_BARU + 1), k_ady) is None,
                  scenarijus_2(_pk(w2, "Low", 910.01, S2_ORB_BARU + 1), k_ady) is not None),
                 (True, True))

    # --- zurnalo horizontas (auditas: be situ testu mutacijos praeidavo) ---
    _bi = []
    for _d in ("2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23"):
        for _m in (0, 5, 10):
            _bi.append(pd.Timestamp(f"{_d} 09:{_m:02d}", tz="Europe/Berlin"))
    _bd = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=_bi)
    _bd["sesija"] = [x.date() for x in _bd.index]
    _hz = barai_signalams({"EU|H|1|2026-07-20": dict(gimimas=str(_bi[0]), baigtis="")},
                          [], {("EU", "H"): _bd})
    tikrinti("horizontas: gimes sesijos viduryje -> 3 sesijos (gimimo + 2)",
             sorted({str(x) for x in _hz["EU|H|1|2026-07-20"]["sesija"]}),
             ["2026-07-20", "2026-07-21", "2026-07-22"])
    _hz2 = barai_signalams({"EU|H|1|2026-07-20": dict(gimimas=str(_bi[2]), baigtis="")},
                           [], {("EU", "H"): _bd})
    tikrinti("horizontas: gimes PASKUTINIAME bare -> 2 kitos sesijos",
             sorted({str(x) for x in _hz2["EU|H|1|2026-07-20"]["sesija"]}),
             ["2026-07-21", "2026-07-22"])

    # --- uzdarymas "laikas" pagal PAMATYTAS sesijas, ne darbo dienas -------
    def _sb(dienos, tz="America/New_York", h=10):
        ix = [pd.Timestamp(f"{d} {h}:{m:02d}", tz=tz) for d in dienos for m in (0, 5)]
        x = pd.DataFrame({"Open": 100.0, "High": 100.2, "Low": 99.9, "Close": 100.0}, index=ix)
        x["sesija"] = [i.date() for i in x.index]
        return x
    _sg = dict(tipas=1, ieina=100.0, tikslas=101.0, stop=99.0, atr_abs=2.0)
    # JAV: gimes antradieni 11-24 10:00, ketvirtadieni 11-26 - Padekos diena
    _jr = lambda: dict(rinka="JAV", sesija="2026-11-24", sanaudos="5",
                       gimimas="2026-11-24 10:00:00-05:00")
    _t1 = _sb(["2026-11-24", "2026-11-25"])
    _r = _jr(); _zurnalo_eilute(_r, _t1, _sg, "2026-11-27T07:00:00+00:00")
    tikrinti("svente: penktadieni ryta (2 is 3 sesiju) eilute LIEKA atvira",
             str(_r.get("baigtis") or ""), "")
    _t2 = _sb(["2026-11-24", "2026-11-25", "2026-11-27"])
    _r = _jr(); _zurnalo_eilute(_r, _t2, _sg, "2026-11-27T18:00:00+00:00")
    tikrinti("        penktadieni 13:00 NY (sesija dar vyksta) - atvira",
             str(_r.get("baigtis") or ""), "")
    _r = _jr(); _zurnalo_eilute(_r, _t2, _sg, "2026-11-27T21:00:00+00:00")
    tikrinti("        lygiai 16:00 NY uzdaryme (3 sesijos) - 'laikas'",
             _r.get("baigtis"), "laikas")
    # gimes PASKUTINIAME bare: reikia 2 kitu sesiju
    _rp = dict(_jr(), gimimas="2026-11-24 15:55:00-05:00")
    _r = dict(_rp); _zurnalo_eilute(_r, _sb(["2026-11-25"]), _sg, "2026-12-01T12:00:00+00:00")
    tikrinti("gimes paskutiniame bare: 1 is 2 sesiju - atvira",
             str(_r.get("baigtis") or ""), "")
    _r = dict(_rp); _zurnalo_eilute(_r, _sb(["2026-11-25", "2026-11-27"]), _sg,
                                    "2026-12-01T12:00:00+00:00")
    tikrinti("gimes paskutiniame bare: 2 is 2 sesiju - 'laikas'", _r.get("baigtis"), "laikas")
    # saugiklis: sesiju taip ir netrūksta -> uzdaroma tik po atsargos
    _r = _jr(); _zurnalo_eilute(_r, _t1, _sg, "2026-12-01T12:00:00+00:00")   # praejo 5 d.d.
    tikrinti("saugiklis: praejus horizontui be atsargos - dar atvira",
             str(_r.get("baigtis") or ""), "")
    _r = _jr(); _zurnalo_eilute(_r, _t1, _sg, "2026-12-04T12:00:00+00:00")   # praejo 8 d.d.
    tikrinti("saugiklis: praejus horizontui + atsargai - 'laikas'", _r.get("baigtis"), "laikas")

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
    plat = bb([[100.0, 100.4, 98.5, 99.0], [99.0, 101.0, 99.2, 100.8],
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

    eil = dict(tipas="2", ieina="110.0", tikslas="", stop="105.0",
               atr_pct="5.0", atr_abs="3.0")   # 5% nuo 110 butu 5.5, ne 3.0
    tikrinti("zurnalas ima atr_abs tiesiogiai, ne atkurineja is atr_pct",
             _zurnalo_sig(eil)["atr_abs"], 3.0)
    sena_eil = {k: v for k, v in eil.items() if k != "atr_abs"}
    tikrinti("sena zurnalo eilute be atr_abs kelia isimti (ja gaudo live)",
             isinstance(_saugiai(_zurnalo_sig, sena_eil), Exception), True)

    piktas = dict(rinka="EU", tickeris="</script><img src=x onerror=alert(1)>",
                  scenarijus="Atsistatymas", tipas=1, ieina=100.0, tikslas=101.0,
                  stop=99.0, progresas=0.3, progresas_tikslus=True, tinkamas=True,
                  kliutys=[], amzius_min=5, atr_pct=2.0, silpnas_atsiemimas=False)
    hp = puslapis_html([piktas], {})
    tikrinti("puslapis: JSON bloke nera neekranuoto '<' (neisardo <script>)",
             hp.count("<script>") == hp.count("</script>") == 1, True)

    # --- dividendu skirtukas: nekeicia signalu, ekranuojamas, be failo nelūžta
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        kel = os.path.join(td, "d.json")
        tikrinti("dividendai: failo nera -> None", dividendai_ikelti(kel), None)
        with open(kel, "w") as f:
            f.write("{sugadintas")
        tikrinti("dividendai: sugadintas failas -> None", dividendai_ikelti(kel), None)
        with open(kel, "w") as f:
            json.dump(dict(atnaujinta="2026-10-05T04:40:00+00:00", langas_d=30, eilutes=[
                dict(tikeris="</script><b>X", rinka="EU", kaina_eur=10.0, div_eur=0.5, ex="2026-10-15",
                     irasymo="2026-10-16", irasymo_apytiksle=True, mokejimo="2026-10-20")]), f)
        dv = dividendai_ikelti(kel)
        tikrinti("dividendai: geras failas ikeliamas", len(dv["eilutes"]), 1)
        sena = globals()["DIVIDENDAI"]
        globals()["DIVIDENDAI"] = kel
        try:
            hp2 = puslapis_html([piktas], {})
        finally:
            globals()["DIVIDENDAI"] = sena
        tikrinti("dividendai: puslapis su piktu tikeriu - vienas <script>",
                 hp2.count("<script>") == hp2.count("</script>") == 1, True)
        tikrinti("dividendai: signalu skirtukas numatytas (rodomas)",
                 '<section id="sk-signalai">' in hp2 and '<section id="sk-dividendai" hidden>' in hp2, True)
        tikrinti("dividendai: signalai puslapyje nepakito",
                 hp2.split("const DUOM = ")[1].count('"tickeris"'), 1)

    # --- S2 slenkantis stop'as: konservatyvi tvarka viename bare (2026-10-05)
    s2k = dict(tipas=2, ieina=100.0, tikslas=None, stop=97.0, atr_abs=2.0)
    vienas = bb([[100, 104, 101.5, 103.5], [103.5, 104, 103, 103.8]])
    b2k = baigtis(vienas, s2k)
    tikrinti("S2: baras pakele stop'a iki 102, to paties baro Low 101.5 -> isejimas 102",
             (b2k["baigtis"], round(b2k["pelnas_pct"], 4), b2k["isejo_bare"]), ("stop", 2.0, 1))
    nelieciant = bb([[100, 104, 102.1, 103.5], [103.5, 103.8, 102.5, 103]])
    tikrinti("S2: Low virs naujo stop'o -> pozicija islieka iki laiko",
             baigtis(nelieciant, s2k)["baigtis"], "laikas")
    tikrinti("S1 (fiksuotas stop'as) nuo pakeitimo nepriklauso",
             baigtis(vienas, dict(s2k, tipas=1, tikslas=110.0))["baigtis"], "laikas")

    # --- live busena: netinkamas pirmas kandidatas nebeuzrakina dienos
    bus = {"k": dict(ieina=100.0, tinkamas=False), "s": dict(ieina=99.0), "t": dict(ieina=98.0, tinkamas=True)}
    tikrinti("busena: netinkama sena + tinkamas dabar -> naujas signalas",
             busenos_sena(bus, "k", dict(tinkamas=True)), None)
    tikrinti("busena: netinkama sena + netinkamas dabar -> sena (kaip anksciau)",
             busenos_sena(bus, "k", dict(tinkamas=False))["ieina"], 100.0)
    tikrinti("busena: senas irasas be 'tinkamas' -> sena (elgsena nepakito)",
             busenos_sena(bus, "s", dict(tinkamas=True))["ieina"], 99.0)
    tikrinti("busena: tinkama sena -> sena", busenos_sena(bus, "t", dict(tinkamas=True))["ieina"], 98.0)
    tikrinti("busena: nera iraso -> None", busenos_sena(bus, "x", dict(tinkamas=True)), None)

    # --- zurnalas: atvira NETINKAMA kortele keiciama veliau atsiradusiu tinkamu
    import tempfile as _tf
    sena_z = globals()["ZURNALAS"]
    with _tf.TemporaryDirectory() as td:
        globals()["ZURNALAS"] = os.path.join(td, "z.csv")
        try:
            e0 = dict(rinka="EU", tickeris="X.DE", tipas=1, sesija_data="2026-10-05",
                      laikas="2026-10-05 10:00", scenarijus="Atsistatymas", ieina=100.0,
                      tikslas=100.4, stop=99.5, rr=0.8, rizika_eur=90.0, atr_pct=1.0,
                      atr_abs=1.0, tinkamas=False, kliutys=["R:R 0.8 < 1.0"])
            zurnalas_atnaujinti([e0], {})
            e1 = dict(e0, laikas="2026-10-05 10:30", ieina=101.0, tikslas=101.5, stop=100.5,
                      rr=1.0, tinkamas=True, kliutys=[])
            zurnalas_atnaujinti([e1], {})
            zz = zurnalas_ikelti()
            r_ = zz["EU|X.DE|1|2026-10-05"]
            tikrinti("zurnalas: netinkama atvira eilute pakeista pirmu tinkamu",
                     (_tinkama(r_), float(r_["ieina"]), r_["gimimas"]), (True, 101.0, "2026-10-05 10:30"))
            e2 = dict(e1, laikas="2026-10-05 11:00", ieina=105.0)
            zurnalas_atnaujinti([e2], {})
            tikrinti("zurnalas: tinkama eilute veliau NEperrasoma",
                     float(zurnalas_ikelti()["EU|X.DE|1|2026-10-05"]["ieina"]), 101.0)
        finally:
            globals()["ZURNALAS"] = sena_z

    tuscias = pd.DataFrame()
    variantu_lentele(tuscias)                      # neturi luzti
    tikrinti("variantu_lentele nelūžta su tusciu df", True, True)

    # --- KIEKVIENA 1 scenarijaus iejimo salyga atskirai --------------------
    # Nepriklausoma perziura 2026-09-28 parode, kad nė vienas ju nebuvo
    # tikrinamas: isemus bet kuri, savitikra vis tiek buvo zalia.
    def be(**kw):
        """SAP 07-24 kontekstas su pakeista viena salyga."""
        d = dict(uzdarymas=128.32, atr_abs=3.0, virsune_n=138.26,
                 uzd_vieta=0.17, dugnas_n=127.50, apyvarta=5e7)
        d.update(kw)
        return kd(d["uzdarymas"], d["atr_abs"], d["virsune_n"],
                  d["uzd_vieta"], d["dugnas_n"], d["apyvarta"])

    tikrinti("KRITIMAS: <1 ATR -> nera",
             pirmas_signalas(scenarijus_1, atsok, be(virsune_n=131.00)) is None, True)
    tikrinti("UZDARYMO VIETA: >0.40 -> nera",
             pirmas_signalas(scenarijus_1, atsok, be(uzd_vieta=0.60)) is None, True)
    tikrinti("APYVARTA: po ribos -> kliutis",
             bool(pirmas_signalas(scenarijus_1, atsok, be(apyvarta=5e5))["kliutys"]), True)
    # peilis: kaina taip ir neatsiema vakarykscio uzdarymo
    peilis = sesija(124.00, np.r_[np.full(6, 0.001), np.full(24, 0.0005)])
    tikrinti("ATSIEMIMAS: kaina neatsieme vakar uzd. -> nera",
             pirmas_signalas(scenarijus_1, peilis, k_sap2) is None, True)
    # tarpo diena priklauso 2 scenarijui
    tarpo_d = sesija(133.00, np.r_[np.full(6, 0.0005), np.full(24, 0.001)])
    tikrinti("TARPO diena 1 scenarijui -> nera",
             pirmas_signalas(scenarijus_1, tarpo_d, k_sap2) is None, True)
    # ORB: kaina virs vakar uzd., bet NEvirsija atidarymo diapazono
    orb_ne = sesija(130.00, np.r_[np.full(6, 0.004), np.full(24, -0.0008)])
    s_orb = pirmas_signalas(scenarijus_1, orb_ne, k_sap2)
    tikrinti("ORB: neprasilauze atidarymo diapazono -> nera", s_orb is None, True)

    # judesio dedamoji maza (kaina vos virs ORB), tad laimi LAIKO dedamoji,
    # ir tik tada matosi, kurios rinkos sesijos ilgis naudojamas
    let = sesija(962.00, np.r_[np.full(6, 0.0002), np.full(24, 0.00005)])
    p_eu = pirmas_signalas(lambda l, k: scenarijus_2(l, k, 350, "eu"), let, k_ady)
    p_us = pirmas_signalas(lambda l, k: scenarijus_2(l, k, 350, "us"), let, k_ady)
    # --- 2 scenarijaus progreso lubos ------------------------------------
    k_lub = kd(100.0, 2.0, 130.0, 0.5, 90.0)
    # tarpas 1.2 ATR, po to eiga - kad butu ka matuoti virs proverzio
    lub = sesija(102.4, np.r_[np.full(6, 0.0008), np.full(30, 0.0022)])
    s_lub = None
    for _i in range(S2_ORB_BARU + 1, len(lub)):
        x = scenarijus_2(lub.iloc[:_i + 1], k_lub, 300, "eu")
        if x and x["virs_proverzio_atr"] > 0.9:
            s_lub = x
            break
    tikrinti("2 scen.: irasomas atstumas VIRS PROVERZIO ATR vienetais",
             bool(s_lub) and s_lub["virs_proverzio_atr"] > 0.9, True)
    if s_lub:
        # TIKSLI lygybe: laisva patikra ("didesnis uz X, mazesnis uz 1")
        # praeidavo ir tada, kai lubos is viso nebuvo taikomos.
        laukt = min(1.0, s_lub["virs_proverzio_atr"] / S2_PROGRESO_LUBOS)
        tikrinti("         progresas = atstumas / LUBOS (ne / vienas ATR)",
                 round(s_lub["progresas"], 6), round(laukt, 6))
        tikrinti("         ir tai NESUTAMPA su dalyba is vieno ATR",
                 round(s_lub["progresas"], 6) !=
                 round(min(1.0, s_lub["virs_proverzio_atr"]), 6), True)
        # ne tautologija: imam TIKRA signala, nueijusi virs lubu
        s_pilnas = None
        for _i in range(S2_ORB_BARU + 1, len(lub)):
            x = scenarijus_2(lub.iloc[:_i + 1], k_lub, 300, "eu")
            if x and x["virs_proverzio_atr"] >= S2_PROGRESO_LUBOS:
                s_pilnas = x
                break
        tikrinti(f"         nuejus virs {S2_PROGRESO_LUBOS} ATR progresas = 100%",
                 bool(s_pilnas) and round(s_pilnas["progresas"], 4) == 1.0, True)
        tikrinti("         1.0 ATR virs proverzio NEBEZYMIMA kaip issikvepes",
                 (1.0 / S2_PROGRESO_LUBOS) < 0.8, True)
    tikrinti("lubos NEKEICIA iejimo, stop'o ir tikslo",
             bool(s_lub) and s_lub["tikslas"] is None
             and abs(s_lub["stop"] - max(float(lub.iloc[:S2_ORB_BARU]["Low"].min()),
                                         s_lub["ieina"] - S2_TRAIL_ATR * 2.0)) < 1e-9,
             True)

    tikrinti("2 scen. laiko progresas naudoja TOS rinkos sesijos ilgi",
             bool(p_eu) and bool(p_us) and
             round(p_eu["progresas"], 3) != round(p_us["progresas"], 3), True)

    # --- zurnalo horizontas = kalibracijos horizontas ---------------------
    eilh, kainos = [], []
    for di, d in enumerate(pd.bdate_range("2026-09-21", periods=6)):
        t0 = pd.Timestamp(d.date()).tz_localize("Europe/Berlin") + pd.Timedelta(hours=9)
        kd_ = 100.3 if di < 3 else 95.0          # po 3 sesiju - kritimas
        for j in range(78):
            eilh.append(t0 + pd.Timedelta(minutes=5 * j)); kainos.append(kd_)
    kk = np.array(kainos)
    dfh = pd.DataFrame(dict(Open=kk, High=kk * 1.001, Low=kk * 0.999, Close=kk,
                            apyv_santykis=1.5), index=pd.DatetimeIndex(eilh))
    dfh["sesija"] = [x.date() for x in dfh.index]
    sigh = dict(tipas=1, ieina=100.0, tikslas=101.0, stop=99.0, atr_abs=2.0, L=99.5)
    z0 = {"EU|X|1|2026-09-21": dict(gimimas=str(dfh.index[0]), baigtis="")}
    po = barai_signalams(z0, [], {("EU", "X"): dfh})["EU|X|1|2026-09-21"]
    tikrinti("zurnalas gauna TIEK PAT sesiju, kiek kalibracija",
             len(set(po["sesija"])), HORIZONTAS_SESIJU)
    tikrinti("ir todel ta pacia baigti",
             baigtis(po, sigh)["baigtis"],
             baigtis(dfh.iloc[1:3 * 78], sigh)["baigtis"])

    # R:R blokuoja VIENAS (judesys praeina): tikslas 1.00 = 0.76%, rizika 1.5
    s_rr = pirmas_signalas(scenarijus_1, atsok, be(virsune_n=131.88))
    tikrinti("R:R blokuoja, kai judesys dar praeina",
             bool(s_rr) and [k for k in s_rr["kliutys"] if "R:R" in k] != []
             and [k for k in s_rr["kliutys"] if "judesys" in k] == [], True)

    tikrinti("zurnalo horizontas TIKRAI 3 sesijos (ne savireferencija)",
             len(set(po["sesija"])), 3)

    # --- RINKOS FONAS ir ATASKAITOS (3-4 punktai) -------------------------
    # Fonas skaiciuojamas is TU PACIU baru, tad pagrindine rizika - ateitis.
    # Testuojama elgsena, ne buvimas: jeigu fonas imtu tos dienos uzdaryma
    # arba visos dienos vidurki, sie testai nukristu.
    def _kelias(atid, eiga):
        k = atid * np.cumprod(1 + np.asarray(eiga))
        pr = pd.Timestamp("2026-07-24 09:00", tz="Europe/Berlin")
        eil = [(pr + pd.Timedelta(minutes=5 * j),
                atid if j == 0 else k[j - 1], k[j] * 1.001, k[j] * 0.999,
                k[j], 5e5) for j in range(len(k))]
        raw = pd.DataFrame(eil, columns=["_t", "Open", "High", "Low", "Close",
                                         "Volume"]).set_index("_t")
        return sesijos_rodikliai(raw, "eu")

    # A kyla visa diena, B krenta visa diena -> mediana ~0, platumas 50%
    kylantis = _kelias(100.0, np.full(12, 0.002))
    krentantis = _kelias(100.0, np.full(12, -0.002))
    rodA = pd.DataFrame(dict(uzdarymas=100.0, atr_abs=2.0, atr_pct=2.0,
                             apyvarta=5e7, uzd_vieta=0.5, virsune_n=110.0,
                             dugnas_n=90.0, div_lange=0.0),
                        index=[pd.Timestamp("2026-07-24")])
    fon = rinkos_fonas({"A": kylantis, "B": krentantis}, {"A": rodA, "B": rodA})
    tikrinti("rinkos fonas turi po eilute kiekvienam 5 min bar'ui",
             len(fon), len(kylantis))
    # simetriski keliai: sudetinis augimas duoda ~0.03 p.p. asimetrija
    tikrinti("priesingi keliai duoda ~0 rinkos mediana",
             abs(float(fon["mediana"].iloc[-1])) < 0.05, True)
    tikrinti("platumas: viena akcija virs, kita zemiau -> 50%",
             round(float(fon["platumas"].iloc[-1])), 50)
    # ATEITIES PATIKRA. Pirma sio testo versija turejo "or", tad praeidavo
    # net ir tada, kai fonui priskirdavau visos dienos paskutine reiksme
    # (mutacijos testas 2026-09-28). Dabar tikrinama KIEKVIENO taško verte.
    vien = rinkos_fonas({"A": kylantis}, {"A": rodA})
    lauk = [round((float(kylantis["Close"].iloc[j]) /
                   float(kylantis["ses_atidarymas"].iloc[j]) - 1) * 100, 6)
            for j in range(len(kylantis))]
    gauta = [round(float(x), 6) for x in vien["mediana"].values]
    tikrinti("fonas KIEKVIENAME tashke = to meto skerspjuvis (ne ateitis)",
             gauta, lauk)
    tikrinti("         ir pirmas taskas skiriasi nuo paskutinio",
             gauta[0] != gauta[-1], True)
    # plat_d30 per SESIJU RIBA. Dvi sesijos: pirma visi virs, antra visi
    # zemiau vakar uzdarymo. Jei diff skaiciuojamas per visa stulpeli (taip
    # ir buvo iki 2026-09-28), antros sesijos pradzioje atsiranda -100 -
    # nakties tarpas, apsimetantis dienos platumo pokyciu.
    dvi = pd.concat([_kelias(100.0, np.full(12, 0.002)),
                     _kelias(100.0, np.full(12, -0.002)).set_index(
                         _kelias(100.0, np.full(12, -0.002)).index
                         + pd.Timedelta(days=1))])
    dvi["sesija"] = [x.date() for x in dvi.index]
    rod2 = pd.DataFrame(dict(uzdarymas=100.0, atr_abs=2.0, atr_pct=2.0,
                             apyvarta=5e7, uzd_vieta=0.5, virsune_n=110.0,
                             dugnas_n=90.0, div_lange=0.0),
                        index=pd.DatetimeIndex(["2026-07-24", "2026-07-25"]))
    f2 = rinkos_fonas({"A": dvi}, {"A": rod2})
    antros = f2[[x.date() == date(2026, 7, 25) for x in f2.index]]
    tikrinti("plat_d30 sesijos pradzioje yra NaN, ne nakties tarpas",
             bool(antros["plat_d30"].iloc[:6].isna().all()), True)
    tikrinti("         ir fonas grazinamas surikiuotas pagal laika",
             bool(f2.index.is_monotonic_increasing), True)

    fA = _fono_laukai(fon, kylantis.index[-1], kylantis, len(kylantis) - 1)
    fB = _fono_laukai(fon, krentantis.index[-1], krentantis, len(krentantis) - 1)
    tikrinti("santykinis stiprumas kylanciai akcijai TEIGIAMAS",
             fA["santykinis"] > 0, True)
    tikrinti("               ir krentanciai NEIGIAMAS", fB["santykinis"] < 0, True)
    tikrinti("santykinis = savo pokytis - rinkos pokytis",
             round(fA["santykinis"] + fB["santykinis"], 6), 0.0)
    tikrinti("nezinomas laikas duoda NaN, o ne nuli (ir nenulauzia)",
             bool(np.isnan(_fono_laukai(fon, pd.Timestamp("1999-01-01",
                   tz="Europe/Berlin"), kylantis, 0)["rinkos_pokytis"])), True)
    tikrinti("tuscias fonas nenulauzia",
             bool(np.isnan(_fono_laukai(pd.DataFrame(), kylantis.index[0],
                                        kylantis, 0)["platumas"])), True)

    # kraštiniu grupiu dydis: verdiktui svarbu MAZIAUSIA kraštine grupe
    import io as _io2, contextlib as _cl2
    _KAND_REZ.clear(); _DALIS[0] = "testas"
    kr = pd.DataFrame([dict(pelnas_pct=1.0, nesekmes=0, baigtis="tikslas",
                            mfe_pct=1.0, sanaudos=10.0, rinka="EU")] * 300 +
                      [dict(pelnas_pct=-1.0, nesekmes=9, baigtis="stop",
                            mfe_pct=0.1, sanaudos=10.0, rinka="EU")] * 2)
    with _cl2.redirect_stdout(_io2.StringIO()):
        pjuviai(kr, tik_kandidatai=True)
    tikrinti("verdiktui paduodamas MAZIAUSIOS kraštines grupes dydis",
             _KAND_REZ.get(("nesekmes", "testas"), (0, 0, -1))[2], 2)
    _DALIS[0] = ""; _KAND_REZ.clear()

    _KAND_REZ.clear()
    for m in ("EU matyta", "EU NEMATYTA", "JAV matyta", "JAV NEMATYTA"):
        _KAND_REZ[("santykinis", m)] = (-40.0, 30.0, 200)    # visur teigiamas
        _KAND_REZ[("platumas", m)] = (-40.0, 30.0, 200)
        _KAND_REZ[("nesekmes", m)] = (0.0, 3.0, 200)         # per mazas
    _KAND_REZ[("platumas", "JAV NEMATYTA")] = (30.0, -40.0, 200)  # zenklas kitoks
    _KAND_REZ[("plat_d30", "EU matyta")] = (0.0, 50.0, 200)  # truksta 3 matavimu
    _MATAVIMAI[:] = ["EU matyta", "EU NEMATYTA", "JAV matyta", "JAV NEMATYTA"]
    del _KAND_REZ[("santykinis", "JAV NEMATYTA")]   # visa rinka be signalu
    import io as _io, contextlib as _cl
    _b = _io.StringIO()
    with _cl.redirect_stdout(_b):
        kandidatu_verdiktas()
    _t = _b.getvalue()
    tikrinti("verdiktas: DINGES matavimas nepriimamas kaip TINKA",
             "TINKA" in [x for x in _t.splitlines() if "santykinis" in x][0],
             False)
    _KAND_REZ[("santykinis", "JAV NEMATYTA")] = (-40.0, 30.0, 200)
    _b = _io.StringIO()
    with _cl.redirect_stdout(_b):
        kandidatu_verdiktas()
    _t = _b.getvalue()
    tikrinti("verdiktas: visur ta pati kryptis + virs sanaudu -> TINKA",
             "TINKA" in [x for x in _t.splitlines() if "santykinis" in x][0], True)
    _KAND_REZ[("santykinis", "JAV NEMATYTA")] = (-40.0, 30.0, 12)
    _b = _io.StringIO()
    with _cl.redirect_stdout(_b):
        kandidatu_verdiktas()
    tikrinti("verdiktas: 12 ivykiu ketvirciuose - per mazai, ne TINKA",
             "per mazai" in [x for x in _b.getvalue().splitlines()
                             if "santykinis" in x][0], True)
    tikrinti("verdiktas: vienoje puseje kitas zenklas -> nelaikosi",
             "nelaikosi" in [x for x in _t.splitlines() if "platumas" in x][0], True)
    tikrinti("verdiktas: skirtumas mazesnis uz sanaudas -> NE TINKA",
             "TINKA" in [x for x in _t.splitlines() if "nesekmes" in x][0], False)
    tikrinti("verdiktas: truksta matavimo -> pasakoma, o ne nutylima",
             "per mazai duomenu" in _t, True)
    _KAND_REZ.clear()

    # talpykla: antras kvietimas NEBEEINA i tinkla
    global _ATASK
    _ATASK = {"X": dict(datos=["2026-07-26"], kada=datetime.now().isoformat()),
              "Y": dict(datos=[], kada=(datetime.now() -
                                        timedelta(days=9)).isoformat())}
    kviesta = []
    tikras_yf = globals().get("yf")
    global ATASKAITU_TALPYKLA
    tikras_kelias = ATASKAITU_TALPYKLA
    ATASKAITU_TALPYKLA = os.path.join(tempfile.gettempdir(),
                                      "savitikra_ataskaitos.json")

    class _FakeYF:
        @staticmethod
        def Ticker(t):
            kviesta.append(t)
            raise RuntimeError("tinklo nera")
    globals()["yf"] = _FakeYF
    try:
        r = ataskaitu_kalendorius(["X"])
        tikrinti("ataskaitos: sviezias irasas imamas is talpyklos (be tinklo)",
                 (kviesta, r), ([], {"X": ["2026-07-26"]}))
        ataskaitu_kalendorius(["Y"])
        tikrinti("ataskaitos: senesnis nei 7 d. irasas atnaujinamas",
                 kviesta, ["Y"])
        kviesta.clear()
        ataskaitu_kalendorius(["Z"])
        tikrinti("ataskaitos: nezinomas tickeris uzklausiamas", kviesta, ["Z"])
        kviesta.clear()
        ataskaitu_kalendorius(["Z"])
        tikrinti("ataskaitos: nepavykusi uzklausa NEkartojama kas karta",
                 kviesta, [])
    finally:
        globals()["yf"] = tikras_yf
        _ATASK = None
        try:
            os.remove(ATASKAITU_TALPYKLA)
        except OSError:
            pass
        ATASKAITU_TALPYKLA = tikras_kelias

    # --- SANAUDOS PAGAL RINKA ir VALIUTA (2026-09-28) --------------------
    tikrinti("JAV sanaudos perpus mazesnes uz EU",
             (sanaudos("eu"), sanaudos("us")), (10.0, 5.0))
    tikrinti("luzio taskas ir minimalus judesys seka sanaudas",
             (round(min_judesys("eu"), 4), round(min_judesys("us"), 4)),
             (0.5556, 0.2778))
    tikrinti("de facto ATR riba JAV perpus zemesne",
             round(min_atr_pct("us") / min_atr_pct("eu"), 4), 0.5)
    _KURSAS.clear(); _KURSAS["v"] = 1.10
    tikrinti("apyvartos riba JAV verciama i dolerius",
             (round(min_apyvarta("eu")), round(min_apyvarta("us"))),
             (1800000, 1980000))
    # elgsena: tas pats signalas, dvi rinkos
    k_maz = kd(100.0, 0.6, 130.0, 0.10, 95.0, apyvarta=1.9e6)
    sig_eu = dict(ieina=100.0, tikslas=100.30, stop=99.70)
    sig_us = dict(sig_eu)
    kl_eu = _bendra(sig_eu, k_maz, "eu")
    kl_us = _bendra(sig_us, k_maz, "us")
    tikrinti("0.30% judesys EU blokuojamas (riba 0.56%)",
             any("judesys" in x for x in kl_eu), True)
    tikrinti("               o JAV praeina (riba 0.28%)",
             any("judesys" in x for x in kl_us), False)
    tikrinti("signalas nesiojasi savo rinkos sanaudas",
             (sig_eu["sanaudos"], sig_us["sanaudos"]), (10.0, 5.0))
    # 1.9 mln: EU apyvarta praeina, JAV (riba 1.98 mln doleriu) - ne
    tikrinti("apyvartos kliutis skiriasi pagal valiuta",
             (any("apyvarta" in x for x in kl_eu),
              any("apyvarta" in x for x in kl_us)), (False, True))

    d_eur = pd.DataFrame([dict(pelnas_pct=1.0, sanaudos=10.0, rinka="EU"),
                          dict(pelnas_pct=1.0, sanaudos=5.0, rinka="JAV")])
    tikrinti("rezultatas eurais ima TOS eilutes sanaudas",
             [round(x, 1) for x in _eur(d_eur)], [170.0, 175.0])
    sena_z = pd.DataFrame([dict(pelnas_pct=1.0, rinka="JAV")])
    tikrinti("sena eilute be stulpelio gauna sanaudas pagal zyme",
             round(float(_eur(sena_z).iloc[0]), 1), 175.0)

    _KURSAS.clear()
    tikras_yf2 = globals().get("yf")

    class _BlogasYF:
        @staticmethod
        def download(*a, **k):
            raise RuntimeError("tinklo nera")
    globals()["yf"] = _BlogasYF
    tikras_kelias2 = KURSO_TALPYKLA
    try:
        globals()["KURSO_TALPYKLA"] = os.path.join(tempfile.gettempdir(),
                                                   "savitikra_kursas.json")
        k1 = eur_usd()
        tikrinti("kurso negavus imama atsargine reiksme, o ne nulis",
                 k1, EURUSD_ATSARGINIS)
    finally:
        globals()["yf"] = tikras_yf2
        globals()["KURSO_TALPYKLA"] = tikras_kelias2
        _KURSAS.clear()
        _KURSAS["v"] = 1.10

    # --- ZURNALAS: iki 2026-09-28 savitikra jo NEKVIETE is viso ----------
    # Nepriklausoma perziura parode, kad galima buvo sanaudas PRIDETI vietoj
    # atimti, arba laikyti pozicija 8 sesijas, ir savitikra liktu zalia.
    zbar = bb([[100.0, 100.4, 99.6, 100.2], [100.2, 101.6, 100.0, 101.4]])
    zsig = dict(tipas=1, ieina=100.0, tikslas=101.0, stop=99.0, atr_abs=2.0,
                L=99.5)
    zr = dict(raktas="EU|X|1|2026-07-24", sesija="2026-07-24", rinka="EU",
              sanaudos=10.0, baigtis="")
    _zurnalo_eilute(zr, zbar, zsig, "dabar")
    tikrinti("zurnalas: tikslas 101.0 nuo 100.0 = +1% -> 180 - 10 = 170 EUR",
             zr["eur"], 170.0)
    zr2 = dict(zr, sanaudos=5.0, baigtis="")
    _zurnalo_eilute(zr2, zbar, zsig, "dabar")
    tikrinti("        ir JAV sanaudomis (5 EUR) -> 175 EUR", zr2["eur"], 175.0)
    tikrinti("        sanaudos ATIMAMOS, ne pridedamos", zr2["eur"] < 180, True)

    # neissisprendusi eilute: uzdaroma TIK praejus horizontui
    zlaik = bb([[100.0, 100.4, 99.6, 100.1]])
    šian = datetime.now().date()
    zr3 = dict(raktas="EU|X|1", sesija=str(šian), rinka="EU", sanaudos=10.0,
               baigtis="")
    _zurnalo_eilute(zr3, zlaik, zsig, "dabar")
    tikrinti("zurnalas: ta pacia diena eilute LIEKA atvira",
             str(zr3.get("baigtis") or ""), "")
    sena = šian - timedelta(days=20)
    zr4 = dict(zr3, sesija=str(sena), baigtis="")
    _zurnalo_eilute(zr4, zlaik, zsig, "dabar")
    tikrinti("zurnalas: seniai praejus horizontui (be sesiju info) - 'laikas'",
             zr4.get("baigtis"), "laikas")

    # eilute be baru: vienas Yahoo sutrikimas jos NEUZDARO (auditas 2026-09-29)
    zr5 = dict(raktas="JAV|Y|1|x", sesija="2026-11-24", rinka="JAV", baigtis="")
    _uzdaryti_pagal_laika(zr5, "2026-11-30T12:00:00+00:00")     # 4 d.d. - horizontas praejo
    tikrinti("zurnalas: be baru horizonto pabaigoje - LIEKA atvira (gal kita karta bus)",
             str(zr5.get("baigtis") or ""), "")
    _uzdaryti_pagal_laika(zr5, "2026-12-04T12:00:00+00:00")     # 8 d.d. = 3 + atsarga 5
    tikrinti("zurnalas: be baru ir po atsargos - 'duomenu nera'",
             zr5.get("baigtis"), "duomenu nera")

    # santrauka: blokuoti signalai i skaicius NEPATENKA
    zz = {"a": dict(tinkamas=True, baigtis="tikslas", eur=170.0),
          "b": dict(tinkamas=False, baigtis="stop", eur=-370.0),
          "c": dict(tinkamas=False, baigtis="stop", eur=-370.0),
          "d": dict(tinkamas=True, baigtis="duomenu nera", eur="")}
    st = zurnalo_santrauka(zz)
    tikrinti("santrauka: blokuoti signalai neiskaitomi",
             (st["signalu"], st["baigtu"], round(st["tikslo_dalis"]),
              round(st["vid_eur"])), (2, 1, 100, 170))
    tikrinti("santrauka: 'duomenu nera' nei atvira, nei baigta",
             st["atviru"], 0)

    # baras, apimantis IR stop'a, IR tiksla -> stop (uzrasyta taisykle)
    abu = bb([[100.0, 111.0, 97.0, 105.0]])
    tikrinti("baras su stop'u IR tikslu uzskaitomas kaip stop",
             baigtis(abu, zsig)["baigtis"], "stop")
    ba = baigtis(abu, zsig)
    tikrinti("        MFE lieka 0 (pelno taip ir nebuvo), o ne +11%",
             (round(ba["mfe_pct"], 2), round(ba["mae_pct"], 2)), (0.0, -1.0))
    tarp_t = bb([[115.0, 116.0, 114.0, 115.5]])
    bt = baigtis(tarp_t, zsig)
    tikrinti("tarpas virs tikslo: MFE = isejimo kaina, ne baro virsune",
             (round(bt["pelnas_pct"], 2), round(bt["mfe_pct"], 2)),
             (15.0, 15.0))

    # kalibracijos pradzios baras = anksciausias, kuri pasiekia live
    tikrinti("kalibracija pradeda ten, kur live gali (S2_ORB_BARU + 3)",
             f"range(S2_ORB_BARU + 3, len(sd))" in
             inspect.getsource(paleisti_kalibracija), True)
    tikrinti("        ir live vartai yra S2_ORB_BARU + 4 baru",
             "S2_ORB_BARU + 4" in inspect.getsource(paleisti_live), True)

    # JSON be NaN
    _pl = inspect.getsource(paleisti_live)
    tikrinti("JSON: rasymo kelias naudoja allow_nan=False ir _be_nan",
             ("allow_nan=False" in _pl) and ("_be_nan(eilutes)" in _pl), True)
    tikrinti("JSON: NaN pakeiciamas i None",
             _be_nan(dict(a=float("nan"), b=[1.0, float("inf")], c=2.0)),
             dict(a=None, b=[1.0, None], c=2.0))

    # --- AMZIUS ir PASENUSIOS SESIJOS (2026-09-28, rasta gyvai) ----------
    pries5 = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    tikrinti("amzius: 5 min nuo aptikimo -> 5 min",
             _amzius_min(pries5, "x", "y"), 5)
    # baru laikas nebejuda, kai rinka uzdaryta - amzius vis tiek turi augti
    tikrinti("amzius: uzdarytoje rinkoje (barai nejuda) vis tiek auga",
             _amzius_min((datetime.now(timezone.utc) -
                          timedelta(minutes=4000)).isoformat(),
                         "2026-09-25 15:55:00-04:00",
                         "2026-09-25 15:55:00-04:00") > 3000, True)
    tikrinti("amzius: sena busena be laikas_utc grizta prie baru laiko",
             _amzius_min(None, "2026-09-28 10:00:00+02:00",
                         "2026-09-28 10:35:00+02:00"), 35)
    tikrinti("amzius: sugadintas laikas duoda 0, o ne isimti",
             _amzius_min("blogai", "blogai", "blogai"), 0)
    _pls = inspect.getsource(paleisti_live)
    tikrinti("live: signalai rodomi TIK is siandienos sesijos",
             "ses != siandien_rinkoje" in _pls, True)
    tikrinti("live: bet zurnalui barai paduodami ir tada",
             _pls.index("visi_barai[(zyme, t)] = d") <
             _pls.index("ses != siandien_rinkoje"), True)
    tikrinti("live: amzius imamas is laikas_utc, ne is baru",
             "laikas_utc=dabar_utc" in _pls, True)

    # netvarkinga eilute uzdaroma, o ne paliekama amzinai atvira
    znet = {"EU|X|1|2026-09-25": dict(raktas="EU|X|1|2026-09-25",
                                      sesija="2026-09-25", rinka="EU",
                                      tinkamas="True", baigtis="",
                                      ieina="10.0", tikslas="11.0",
                                      stop="9.0", tipas="1", atr_pct="2.0")}
    import io as _io3, contextlib as _cl3
    with _cl3.redirect_stdout(_io3.StringIO()):
        try:
            _zurnalo_sig(znet["EU|X|1|2026-09-25"])
            nera_isimties = True
        except Exception:
            nera_isimties = False
    tikrinti("sena eilute be atr_abs vis dar kelia isimti",
             nera_isimties, False)
    # ir TIKRAI uzdaroma - per zurnalas_atnaujinti(), ne tik teoriskai
    global ZURNALAS
    _tikras_z = ZURNALAS
    ZURNALAS = os.path.join(tempfile.gettempdir(), "savitikra_z", "z.csv")
    try:
        os.makedirs(os.path.dirname(ZURNALAS), exist_ok=True)
        pd.DataFrame([dict(raktas="EU|X|1|2026-09-25", gimimas="2026-09-25",
                           rinka="EU", tickeris="X", scenarijus="senas",
                           tipas="1", sesija="2026-09-25", ieina="10.0",
                           tikslas="11.0", stop="9.0", tinkamas="True",
                           kliutys="", baigtis="")],
                     columns=ZURNALO_STULPELIAI).to_csv(ZURNALAS, index=False)
        barai_x = {"EU|X|1|2026-09-25": bb([[10.0, 10.2, 9.9, 10.1]])}
        with _cl3.redirect_stdout(_io3.StringIO()):
            zpo = zurnalas_atnaujinti([], barai_x)
        tikrinti("netvarkinga eilute UZDAROMA, o ne paliekama atvira",
                 zpo["EU|X|1|2026-09-25"]["baigtis"], "netvarkinga")
    finally:
        try:
            os.remove(ZURNALAS)
        except OSError:
            pass
        ZURNALAS = _tikras_z

    st2 = zurnalo_santrauka({"a": dict(tinkamas=True, baigtis="netvarkinga",
                                       eur=""),
                             "b": dict(tinkamas=True, baigtis="tikslas",
                                       eur=170.0)})
    tikrinti("santrauka: 'netvarkinga' nei atvira, nei baigta",
             (st2["atviru"], st2["baigtu"], round(st2["tikslo_dalis"])),
             (0, 1, 100))

    # 2 scenarijus: pasiteisinimas = pelnas po sanaudu, ne "tikslas"
    st3 = zurnalo_santrauka({
        "a": dict(tinkamas=True, tipas="1", baigtis="tikslas", eur=170.0),
        "b": dict(tinkamas=True, tipas="1", baigtis="stop", eur=-190.0),
        "c": dict(tinkamas=True, tipas="2", baigtis="slenkantis stop", eur=420.0),
        "d": dict(tinkamas=True, tipas="2", baigtis="slenkantis stop", eur=-150.0),
        "e": dict(tinkamas=True, tipas="2", baigtis="stop", eur=90.0),     # sena zyma
        "f": dict(tinkamas=True, tipas="2", baigtis="laikas", eur=5.0)})
    tikrinti("santrauka: 2 scen. pelningas = pasiteisino (ir su sena 'stop' zyma)",
             (st3["baigtu"], round(st3["tikslo_dalis"])), (6, 67))
    tikrinti("santrauka: 1 scen. stop'as - ne pasiteisinimas; 2 scen. su nuostoliu - ne",
             (_pasiteisino(dict(tipas="1", baigtis="stop", eur=50.0)),
              _pasiteisino(dict(tipas="2", baigtis="slenkantis stop", eur=-1.0)),
              _pasiteisino(dict(tipas=2, baigtis="laikas", eur=1.0))),
             (False, False, True))
    tikrinti("santrauka: vid. rezultatas nesikeicia (visu baigtu vidurkis)",
             round(st3["vid_eur"], 2), round((170 - 190 + 420 - 150 + 90 + 5) / 6, 2))

    # zurnalo eilute: 2 scenarijaus stop'as vadinamas "slenkantis stop",
    # 1 scenarijaus - "stop"
    def _zb(bars):
        i = pd.date_range("2026-07-24 10:00", periods=len(bars), freq="5min")
        return pd.DataFrame(bars, columns=["Open", "High", "Low", "Close"], index=i)
    kyla_krenta = _zb([[100, 104, 102.5, 104], [104, 106, 104.5, 106], [106, 106, 103, 103]])
    r2 = dict(sesija="2026-07-24", sanaudos="10.0")
    _zurnalo_eilute(r2, kyla_krenta, dict(tipas=2, ieina=100.0, tikslas=None,
                                          stop=97.0, atr_abs=2.0), "dabar")
    tikrinti("zurnalas: 2 scen. slenkantis stop -> 'slenkantis stop', pelnas +4%",
             (r2.get("baigtis"), r2.get("pelnas_pct")), ("slenkantis stop", 4.0))
    r1 = dict(sesija="2026-07-24", sanaudos="10.0")
    _zurnalo_eilute(r1, _zb([[100, 100.5, 98, 98]]),
                    dict(tipas=1, ieina=100.0, tikslas=101.0, stop=98.5,
                         atr_abs=2.0), "dabar")
    tikrinti("zurnalas: 1 scen. stop -> 'stop'", r1.get("baigtis"), "stop")

    # --- baru perziura: pirmas TINKAMAS baras, tik uzbaigti barai ---
    ix = pd.date_range("2026-07-24 09:00", periods=30, freq="5min", tz="Europe/Berlin")
    ses_p = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
                          "minute": [9 * 60 + 5 * k for k in range(30)]}, index=ix)
    kd_p = pd.Series({"atr_pct": 2.0})

    def _netikras(langas, kd, iki=None, rinka="eu"):
        n = len(langas)
        if n < 12:
            return []
        return [dict(tipas=1, tinkamas=n >= 15, scenarijus="Atsistatymas",
                     ieina=float(n), stop=0.0, kliutys=[])]
    _tikras_apt = globals()["aptikti"]
    globals()["aptikti"] = _netikras
    try:
        pv = perziureti_dienos_barus(ses_p, kd_p, "eu", "EU", "X", ix[0].date(), set(),
                                     dabar_ts=pd.Timestamp("2026-07-24 20:00", tz="UTC"))
        tikrinti("baru perziura: pirmas TINKAMAS baras (15-as), ne pirmas netinkamas",
                 [(e["tipas"], e["laikas"], e["ieina"]) for e in pv],
                 [(1, str(ix[14]), 15.0)])
        tikrinti("baru perziura: raktas su perziuros zyme",
                 pv[0]["_raktas"], f"EU|X|1|{ix[0].date()}|{PERZIUROS_ZYME}")
        tikrinti("baru perziura: jau zurnale -> neieskoma",
                 perziureti_dienos_barus(ses_p, kd_p, "eu", "EU", "X", ix[0].date(),
                                         {pv[0]["_raktas"]},
                                         dabar_ts=pd.Timestamp("2026-07-24 20:00", tz="UTC")),
                 [])
        # 15-as baras (10:10) - PASKUTINIS grazintas, sesija vyksta -> dar nebaigtas
        tikrinti("baru perziura: paskutinis grazintas baras sesijos metu neimamas",
                 perziureti_dienos_barus(ses_p.iloc[:15], kd_p, "eu", "EU", "X",
                                         ix[0].date(), set(),
                                         dabar_ts=pd.Timestamp("2026-07-24 08:40", tz="UTC")),
                 [])
        tikrinti("baru perziura: atejus kitam barui jis jau imamas",
                 len(perziureti_dienos_barus(ses_p.iloc[:16], kd_p, "eu", "EU", "X",
                                             ix[0].date(), set(),
                                             dabar_ts=pd.Timestamp("2026-07-24 08:40", tz="UTC"))),
                 1)
    finally:
        globals()["aptikti"] = _tikras_apt

    # santrauka: baru perziuros eilutes i puslapio skaicius NEPATENKA
    st4 = zurnalo_santrauka({
        "a": dict(tinkamas=True, tipas="1", baigtis="tikslas", eur=170.0, saltinis="kortele"),
        "b": dict(tinkamas=True, tipas="1", baigtis="stop", eur=-190.0, saltinis="baru perziura"),
        "c": dict(tinkamas=True, tipas="1", baigtis="", eur="", saltinis="baru perziura"),
        "d": dict(tinkamas=True, tipas="1", baigtis="stop", eur=-190.0)})   # sena eilute
    st5 = zurnalo_santrauka({
        "a": dict(tinkamas=True, tipas="1", baigtis="stop", eur=-300.0, sesija="2026-09-29"),
        "b": dict(tinkamas=True, tipas="1", baigtis="tikslas", eur=170.0, sesija=PUSLAPIO_PRADZIA),
        "c": dict(tinkamas=True, tipas="1", baigtis="", eur="", sesija=PUSLAPIO_PRADZIA)})
    # juosta turi likti matoma ir tada, kai nuo PUSLAPIO_PRADZIA dar nieko nera
    # (2026-09-29 ji dingdavo, nes JS slepe juosta prie 0 signalu)
    tikrinti("puslapis: zurnalo juosta neslepiama prie 0 signalu",
             ("if (!z){ el.hidden = true; return; }" in PUSLAPIO_SABLONAS
              and "!z.signalu" not in PUSLAPIO_SABLONAS), True)
    tikrinti("santrauka: sesijos iki PUSLAPIO_PRADZIA neiskaitomos, nuo jos - taip",
             (st5["signalu"], st5["baigtu"], st5["atviru"], round(st5["vid_eur"])),
             (2, 1, 1, 170))
    tikrinti("santrauka: baru perziura neiskaitoma, sena eilute - iskaitoma",
             (st4["signalu"], st4["baigtu"], st4["atviru"], round(st4["tikslo_dalis"])),
             (2, 2, 0, 50))

    # barai_signalams: 5 daliu raktas gauna barus PO gimimo baro
    d_x = ses_p.assign(sesija=ix[0].date())
    e_p = dict(pv[0])
    bs = barai_signalams({}, [], {("EU", "X"): d_x}, [e_p])
    tikrinti("barai_signalams: perziuros eilute gauna barus po gimimo",
             ((len(bs[e_p["_raktas"]]), str(bs[e_p["_raktas"]].index[0]))
              if e_p["_raktas"] in bs else "eilute negavo baru"),
             (15, str(ix[15])))

    # zurnalas_atnaujinti: saltinis irasomas; kortele ir perziura - atskiros eilutes
    _tz = ZURNALAS
    ZURNALAS = os.path.join(tempfile.gettempdir(), "savitikra_z2", "z.csv")
    try:
        os.makedirs(os.path.dirname(ZURNALAS), exist_ok=True)
        if os.path.exists(ZURNALAS):
            os.remove(ZURNALAS)
        bazinis = dict(rinka="EU", tickeris="X", tipas=1, sesija_data=str(ix[0].date()),
                       laikas=str(ix[14]), scenarijus="Atsistatymas", ieina=100.0,
                       tikslas=101.0, stop=99.0, rr=1.0, rizika_eur=180.0,
                       sanaudos=10.0, atr_pct=2.0, atr_abs=2.0, tinkamas=True,
                       kliutys=[])
        e_z = dict(bazinis, _raktas=f"EU|X|1|{ix[0].date()}|{PERZIUROS_ZYME}")
        with _cl3.redirect_stdout(_io3.StringIO()):
            zs = zurnalas_atnaujinti([dict(bazinis)], {}, [e_z])
        tikrinti("zurnalas: kortele ir baru perziura - dvi eilutes su saltiniu",
                 sorted((k.count("|"), r["saltinis"]) for k, r in zs.items()),
                 [(3, "kortele"), (4, "baru perziura")])
    finally:
        try:
            os.remove(ZURNALAS)
        except OSError:
            pass
        ZURNALAS = _tz

    ses_t = date(2026, 7, 24)
    tikrinti("iki ataskaitos: rytojaus ataskaita -> +1",
             dienu_iki_ataskaitos(["2026-07-25"], ses_t), 1)
    tikrinti("                vakar buvusi -> -1",
             dienu_iki_ataskaitos(["2026-07-23"], ses_t), -1)
    tikrinti("                imama ARCIAUSIA is keliu",
             dienu_iki_ataskaitos(["2026-01-05", "2026-07-26", "2026-11-02"],
                                  ses_t), 2)
    tikrinti("                be kalendoriaus -> NaN",
             bool(np.isnan(dienu_iki_ataskaitos(None, ses_t))), True)
    tikrinti("                sugadinta data nenulauzia",
             bool(np.isnan(dienu_iki_ataskaitos(["blogai"], ses_t))), True)

    tikrinti("atkurti() su nepilna busena NEnulauzia",
             isinstance(_saugiai(atkurti, dict(s), {"laikas": "x"}, k_sap2, 131.0),
                        Exception), False)

    sv = pirmas_signalas(scenarijus_1, atsok, k_sap2)
    tikrinti("naujas signalas gimsta ANKSTYVAS, ne 'issikvepes'",
             sv["progresas"] < 0.8, True)
    tikrinti("         ir progresas matuojamas iki BUVUSIO lygio",
             round(sv["progresas"], 4),
             round((sv["ieina"] - sv["L"]) / (sv["R_pilnas"] - sv["L"]), 4))

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
        kandidatu_verdiktas()
    else:
        paleisti_live(rinkos)
