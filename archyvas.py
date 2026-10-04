#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EODHD ARCHYVAS: 5 MIN BARAI NUO 2020-10  -  2026-10-04
======================================================

EODHD plano 1 zingsnis. Parsisiuncia viso universo (EU + JAV + indeksai ir
sektoriu ETF) 5 min barus nuo 2020-10-01, dienos kainas (nekoreguotas ir
koreguotas) ir splitus. Rezultatas - failai aplanke archyvas/, kuriuos
workflow ikelia kaip GitHub Release prieda. detektorius.py nelieciamas.

Milanas (MI), Lisabona (LS) ir Viena (VI) praleidziami: EODHD patikra 2
parode, kad 5 min baru ten nera (Milano biržos EODHD sarase isvis nera).

Patikros (spausdinamos zurnale ir irasomos i archyvas/ataskaita.csv):
  A. Ar tikeris yra EODHD biržos simboliu sarase.
  B. Kiekvienam tikeriui: baru skaicius, pirmas/paskutinis baras, sesiju su
     5 min barais dalis tarp dienu, kai biržoje buvo prekiauta (pagal EODHD
     dienos duomenis), dublikatai, OHLC logika.
  C. Dienos sutikrinimas: paskutinio 5 min baro close pries dienos close
     (nekoreguota). Dideli nuokrypiai = splitas arba duomenu klaida.
  D. Yahoo palyginimas (imtis): tie patys 5 min barai per paskutines ~50
     dienu; kokia laiko poslinkis (0 / +-5 min) duoda geriausia sutapima.

Raktas imamas is aplinkos (GitHub Secret EODHD_TOKEN), niekada nespausdinamas.

Paleidimas:
    python archyvas.py --savitikra
    python archyvas.py               (reikia EODHD_TOKEN)
    python archyvas.py --tik 5       (bandomasis: tik 5 tikeriai is kiekvienos rinkos)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from eodhd_patikra import BIRZOS, _gauti, eodhd_simbolis

PRADZIA = datetime(2020, 10, 1)
EOD_PRADZIA = "2019-06-01"           # ATR ir kitu dienos rodikliu "isibegejimui"
LANGAS_D = 500                        # EODHD leidzia iki 600 d. 5 min uzklausai
BE_INTRADAY = {"MI", "LS", "VI"}
SRAUTAI = 4
BANDYMAI = 4
APLANKAS = "archyvas"
TZ = {"eu": "Europe/Berlin", "us": "America/New_York"}
US_VALANDOS = (9 * 60 + 30, 16 * 60)
DIENOS_NUOKRYPIS = 0.02               # >2 % tarp 5 min ir dienos close - zymima
YAHOO_IMTIS = 2                       # tikeriu is kiekvienos biržos


# ------------------------------------------------------------------ sarasas

def tikeriai(tik=None):
    """[(yahoo, eodhd, rinka)] be dublikatu; MI/LS/VI praleidziami atskirai."""
    import universas as U
    eu = list(U.visi_tikeriai()) + [U.INDEKSAS] + sorted(set(U.SEKTORIU_ETF.values()))
    us = [t for l in U.UNIVERSAS_US.values() for t in l] + [U.US_INDEKSAS] + sorted(set(U.US_ETF.values()))
    out, praleisti, matyti = [], [], set()
    for rinka, lst in (("eu", eu), ("us", us)):
        n = 0
        for t in lst:
            if t in matyti:
                continue
            matyti.add(t)
            g = t.rsplit(".", 1)[1].upper() if "." in t else "US"
            if g in BE_INTRADAY:
                praleisti.append(t)
                continue
            s = eodhd_simbolis(t)
            if s is None:
                praleisti.append(t)
                continue
            if tik is not None and n >= tik:
                continue
            out.append((t, s, rinka))
            n += 1
    return out, praleisti


def langai(nuo, iki, dienu=LANGAS_D):
    """Nepersidengiantys [nuo, iki) intervalai unix sekundemis, padengiantys visa laikotarpi."""
    a, b = int(nuo.replace(tzinfo=timezone.utc).timestamp()), int(iki.replace(tzinfo=timezone.utc).timestamp())
    zingsnis = dienu * 86400
    out = []
    while a < b:
        out.append((a, min(a + zingsnis, b)))
        a += zingsnis
    return out


# ------------------------------------------------------------------ tinklas

def gauti(kelias, token, **params):
    """_gauti su pakartojimais (429 / 5xx / rysio klaida). Raktas niekada nepatenka i teksta."""
    kl = None
    for k in range(BANDYMAI):
        d, kl = _gauti(kelias, token, **params)
        if kl is None:
            return d, None
        if kl.startswith("HTTP 4") and kl != "HTTP 429":
            return None, kl
        time.sleep(2 ** k * 3)
    return None, kl


def barai_5m(simb, token, nuo=PRADZIA, iki=None):
    iki = iki or datetime.now(timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
    dalys, klaidos = [], []
    for a, b in langai(nuo, iki):
        # EODHD "to" imtinai -> b-1, kad langai nepersidengtu
        d, kl = gauti(f"intraday/{simb}", token, interval="5m", **{"from": a, "to": b - 1})
        if kl:
            klaidos.append(kl)
            continue
        if d:
            dalys.append(pd.DataFrame(d))
    return sujungti(dalys), klaidos


def sujungti(dalys):
    if not dalys:
        return pd.DataFrame(columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df = pd.concat(dalys, ignore_index=True)
    df = df.rename(columns={"timestamp": "ts", "open": "Open", "high": "High",
                            "low": "Low", "close": "Close", "volume": "Volume"})
    df = df[["ts", "Open", "High", "Low", "Close", "Volume"]].dropna(subset=["ts", "Close"])
    df["ts"] = df["ts"].astype("int64")
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)


# ------------------------------------------------------------------ patikros

def sesijos(df, rinka):
    """ts -> sesijos data rinkos laiko zonoje; JAV - tik reguliarios valandos."""
    t = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(TZ[rinka])
    m = t.dt.hour * 60 + t.dt.minute
    keep = (m >= US_VALANDOS[0]) & (m < US_VALANDOS[1]) if rinka == "us" else pd.Series(True, index=df.index)
    d = df[keep].copy()
    d["sesija"] = t[keep].dt.strftime("%Y-%m-%d")
    return d


def patikrinti(df, eod, rinka):
    """Grazina patikros eilute vienam tikeriui (B ir C)."""
    r = dict(baru=len(df))
    if df.empty:
        return r
    r["pirmas"] = pd.to_datetime(df["ts"].iloc[0], unit="s", utc=True).strftime("%Y-%m-%d")
    r["paskutinis"] = pd.to_datetime(df["ts"].iloc[-1], unit="s", utc=True).strftime("%Y-%m-%d")
    r["dublikatu"] = int(df["ts"].duplicated().sum())
    blogi = (df["High"] < df[["Open", "Close"]].max(axis=1) - 1e-9) | (df["Low"] > df[["Open", "Close"]].min(axis=1) + 1e-9)
    r["ohlc_klaidu"] = int(blogi.sum())
    s = sesijos(df, rinka)
    if s.empty:
        return r
    g = s.groupby("sesija")
    r["sesiju"] = int(g.ngroups)
    r["baru_sesijoje_med"] = float(g.size().median())
    if eod is not None and len(eod):
        e = eod[(eod["date"] >= r["pirmas"]) & (eod["date"] <= r["paskutinis"]) & (eod["volume"] > 0)]
        if len(e):
            r["sesiju_dalis"] = round(len(set(e["date"]) & set(g.groups)) / len(e), 4)
            pask = g["Close"].last()
            j = pd.concat([pask.rename("c5"), e.set_index("date")["close"].rename("cd")], axis=1, join="inner")
            if len(j):
                nuok = (j["c5"] / j["cd"] - 1).abs()
                r["dienos_nuokr_med"] = round(float(nuok.median()), 5)
                r["dienu_virs_2proc"] = int((nuok > DIENOS_NUOKRYPIS).sum())
    return r


def poslinkis(eod5, yh5):
    """Palygina du 5 min close rinkinius (ts indeksas). Grazina (geriausias poslinkis min, sutapimas %)."""
    geriausias = (None, -1.0)
    for p in (-5, 0, 5):
        a = eod5.copy()
        a.index = a.index + p * 60
        j = pd.concat([a.rename("e"), yh5.rename("y")], axis=1, join="inner").dropna()
        if len(j) < 20:
            continue
        dalis = float(((j["e"] / j["y"] - 1).abs() < 0.001).mean()) * 100
        if dalis > geriausias[1]:
            geriausias = (p, round(dalis, 1))
    return geriausias


def yahoo_palyginimas(t, df):
    import yfinance as yf
    try:
        y = yf.download(t, period="55d", interval="5m", progress=False, auto_adjust=False)
    except Exception as e:
        return None, f"yahoo klaida ({type(e).__name__})"
    if y is None or y.empty:
        return None, "yahoo tuscia"
    yc = y["Close"]
    if isinstance(yc, pd.DataFrame):
        yc = yc.iloc[:, 0]
    yc.index = ((pd.to_datetime(yc.index, utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
                // pd.Timedelta(seconds=1)).astype("int64")
    e = df.set_index("ts")["Close"]
    e = e[e.index >= yc.index.min()]
    return poslinkis(e, yc), None


# ------------------------------------------------------------------ eiga

def vienas(args):
    t, s, rinka, token = args
    df, klaidos = barai_5m(s, token)
    eod, kl1 = gauti(f"eod/{s}", token, **{"from": EOD_PRADZIA})
    eod = pd.DataFrame(eod) if eod else pd.DataFrame(columns=["date", "open", "high", "low", "close", "adjusted_close", "volume"])
    spl, _ = gauti(f"splits/{s}", token, **{"from": EOD_PRADZIA})
    r = dict(tikeris=t, eodhd=s, rinka=rinka, klaidos=";".join(sorted(set(klaidos + ([kl1] if kl1 else [])))))
    r.update(patikrinti(df, eod, rinka))
    r["splitai"] = ";".join(f"{x.get('date')}:{x.get('split')}" for x in (spl or []))
    vardas = t.replace("/", "_")
    if len(df):
        irasyti_5m(df, os.path.join(APLANKAS, "5m", rinka, f"{vardas}.npz"))
    if len(eod):
        eod.to_csv(os.path.join(APLANKAS, "dienos", f"{vardas}.csv.gz"), index=False)
    return r, df


STULP = ["Open", "High", "Low", "Close", "Volume"]


def irasyti_5m(df, kelias):
    """Kompaktiskas formatas be papildomu bibliotekų: ts (int64) + OHLCV (float64)."""
    np.savez_compressed(kelias, ts=df["ts"].to_numpy("int64"),
                        ohlcv=df[STULP].to_numpy("float64"))


def skaityti_5m(kelias, tz="UTC"):
    """Grazina DataFrame kaip yfinance: indeksas - baro pradzia (tz), stulpeliai Open..Volume."""
    z = np.load(kelias)
    idx = pd.to_datetime(z["ts"], unit="s", utc=True).tz_convert(tz)
    return pd.DataFrame(z["ohlcv"], index=idx, columns=STULP)


def paleisti(tik=None):
    token = os.environ.get("EODHD_TOKEN", "").strip()
    if not token:
        sys.exit("KLAIDA: EODHD_TOKEN nenustatytas")
    for rinka in ("eu", "us"):
        os.makedirs(os.path.join(APLANKAS, "5m", rinka), exist_ok=True)
    os.makedirs(os.path.join(APLANKAS, "dienos"), exist_ok=True)
    sar, praleisti = tikeriai(tik)
    print(f"Tikeriu: {len(sar)} (EU {sum(r == 'eu' for *_, r in sar)}, JAV {sum(r == 'us' for *_, r in sar)}); "
          f"praleista be 5 min duomenu: {len(praleisti)} ({', '.join(praleisti[:8])}{'...' if len(praleisti) > 8 else ''})")

    print("\nA) SIMBOLIU SARASAI")
    nera = []
    for kod in sorted({s.rsplit(".", 1)[1] for _, s, _ in sar}):
        sl, kl = gauti(f"exchange-symbol-list/{kod}", token)
        if kl:
            print(f"   {kod}: sarasas negautas ({kl})")
            continue
        vardai = {x.get("Code") for x in sl}
        truksta = [s for _, s, _ in sar if s.endswith("." + kod) and s.rsplit(".", 1)[0] not in vardai]
        nera += truksta
        print(f"   {kod:<6} simboliu {len(sl):>6}; musu nerasta: {truksta if truksta else 'nera'}")

    print("\nB/C) PARSISIUNTIMAS")
    t0 = time.time()
    eil, duom = [], {}
    with ThreadPoolExecutor(SRAUTAI) as ex:
        for k, (r, df) in enumerate(ex.map(vienas, [(t, s, rn, token) for t, s, rn in sar]), 1):
            eil.append(r)
            duom[r["tikeris"]] = df
            if k % 25 == 0 or k == len(sar):
                print(f"   {k}/{len(sar)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    rep = pd.DataFrame(eil)

    print("\nD) YAHOO PALYGINIMAS (imtis)")
    imtis = rep.assign(g=rep["eodhd"].str.rsplit(".", n=1).str[1]).groupby("g").head(YAHOO_IMTIS)
    for _, x in imtis.iterrows():
        df = duom.get(x["tikeris"])
        if df is None or df.empty:
            continue
        res, kl = yahoo_palyginimas(x["tikeris"], df)
        txt = kl or f"geriausias poslinkis {res[0]:+d} min, sutampa {res[1]:.1f}% baru"
        rep.loc[rep["tikeris"] == x["tikeris"], "yahoo"] = txt
        print(f"   {x['tikeris']:<10} {txt}")

    rep.to_csv(os.path.join(APLANKAS, "ataskaita.csv"), index=False)
    santrauka(rep, nera, praleisti)


def santrauka(rep, nera, praleisti):
    print(f"\n{'=' * 78}\nSANTRAUKA\n{'=' * 78}")
    for rinka in ("eu", "us"):
        g = rep[rep["rinka"] == rinka]
        if g.empty:
            continue
        su = g[g["baru"] > 0]
        print(f"  {rinka.upper()}: tikeriu {len(g)}, su 5 min barais {len(su)}, baru is viso {int(g['baru'].sum()):,}")
        if len(su):
            print(f"      pirmo baro data (mediana) {sorted(su['pirmas'])[len(su) // 2]}; "
                  f"veliausiai prasideda: {su.sort_values('pirmas').iloc[-1]['tikeris']} {su['pirmas'].max()}")
            if "sesiju_dalis" in su:
                print(f"      sesiju su 5 min barais / prekybos dienu: mediana {su['sesiju_dalis'].median() * 100:.1f}%, "
                      f"blogiausias {su.loc[su['sesiju_dalis'].idxmin(), 'tikeris']} {su['sesiju_dalis'].min() * 100:.1f}%")
            if "dienos_nuokr_med" in su:
                print(f"      paskutinio 5 min close vs dienos close: nuokrypio mediana "
                      f"{su['dienos_nuokr_med'].median() * 100:.2f}%; tikeriu su dienomis >2%: "
                      f"{int((su['dienu_virs_2proc'] > 0).sum())}")
    tusti = rep[rep["baru"] == 0]["tikeris"].tolist()
    klaid = rep[rep["klaidos"].astype(str) != ""][["tikeris", "klaidos"]].values.tolist()
    print(f"  be baru: {tusti if tusti else 'nera'}")
    print(f"  su uzklausu klaidomis: {klaid if klaid else 'nera'}")
    print(f"  nerasta simboliu sarase: {nera if nera else 'nera'}")
    print(f"  praleista (MI/LS/VI): {len(praleisti)}")
    blogi = rep[rep.get("dienu_virs_2proc", pd.Series(0, index=rep.index)).fillna(0) > 0]
    if len(blogi):
        print("  dienos >2% nuokrypiai (dazniausiai splitai - ziurek 'splitai' stulpeli):")
        for _, x in blogi.sort_values("dienu_virs_2proc", ascending=False).head(15).iterrows():
            print(f"      {x['tikeris']:<10} dienu {int(x['dienu_virs_2proc']):>4}  splitai: {x.get('splitai') or '-'}")


# ------------------------------------------------------------------ savitikra

def savitikra():
    ok = True

    def tikrinti(s, a, b):
        nonlocal ok
        g = a == b
        print(f"  {'OK ' if g else 'BLOGAI'}  {s}{'' if g else f'  (gauta {a}, laukta {b})'}")
        ok = ok and g

    # langai: padengia visa intervala be tarpu ir persidengimu
    L = langai(datetime(2020, 10, 1), datetime(2026, 10, 4))
    tikrinti("langai prasideda nuo pradzios", L[0][0], int(datetime(2020, 10, 1, tzinfo=timezone.utc).timestamp()))
    tikrinti("langai baigiasi pabaigoje", L[-1][1], int(datetime(2026, 10, 4, tzinfo=timezone.utc).timestamp()))
    tikrinti("langai be tarpu", all(L[i][1] == L[i + 1][0] for i in range(len(L) - 1)), True)
    tikrinti("langas <= 600 d.", max(b - a for a, b in L) <= 600 * 86400, True)
    tikrinti("langu skaicius 5 (2194 d. / 500)", len(L), 5)

    # sarasas: MI/LS/VI praleisti, dublikatu nera, BRK-B -> BRK-B.US, ETF itraukti
    sar, pr = tikeriai()
    ys = [t for t, _, _ in sar]
    tikrinti("dublikatu nera", len(ys), len(set(ys)))
    tikrinti("MI/LS/VI nera sarase", any(t.endswith((".MI", ".LS", ".VI")) for t in ys), False)
    tikrinti("praleista 33 (MI 26 + LS 4 + VI 3)", len(pr), 33)
    tikrinti("BRK-B -> BRK-B.US", dict((t, s) for t, s, _ in sar).get("BRK-B"), "BRK-B.US")
    tikrinti("indeksai itraukti", all(x in ys for x in ("EXSA.DE", "SPY", "XLK", "EXV3.DE")), True)
    tikrinti("--tik 2: po 2 is rinkos", len(tikeriai(2)[0]), 4)

    # sujungimas: dublikatai pasalinami, rikiuojama
    d1 = [{"timestamp": 200, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
          {"timestamp": 100, "open": 1, "high": 2, "low": 0.5, "close": 1.4, "volume": 10}]
    d2 = [{"timestamp": 200, "open": 1, "high": 2, "low": 0.5, "close": 1.6, "volume": 10}]
    s = sujungti([pd.DataFrame(d1), pd.DataFrame(d2)])
    tikrinti("sujungta be dublikatu", list(s["ts"]), [100, 200])
    tikrinti("dublikate paliekamas velesnis", float(s["Close"].iloc[-1]), 1.6)
    tikrinti("tuscias sujungimas", len(sujungti([])), 0)

    # sesijos: JAV tik reguliarios valandos; EU data Berlyno laiku
    ts = [int(pd.Timestamp(x, tz="America/New_York").timestamp()) for x in
          ("2026-10-01 08:00", "2026-10-01 09:30", "2026-10-01 15:55", "2026-10-01 16:00")]
    tikrinti("JAV: lieka 09:30 ir 15:55", len(sesijos(pd.DataFrame({"ts": ts, "Close": 1.0}), "us")), 2)
    te = int(pd.Timestamp("2026-10-01 23:30", tz="UTC").timestamp())
    tikrinti("EU: 23:30 UTC = kita diena Berlyne", sesijos(pd.DataFrame({"ts": [te], "Close": 1.0}), "eu")["sesija"].iloc[0], "2026-10-02")

    # patikrinimas: sesiju dalis ir dienos nuokrypis
    dien = pd.date_range("2026-09-28", periods=4, freq="B", tz="Europe/Berlin")
    rows = []
    for i, d in enumerate(dien):
        if i == 2:
            continue                      # truksta vienos dienos
        for h in (10, 17):
            c = 100.0 if i != 3 else 110.0
            rows.append(dict(ts=int((d + pd.Timedelta(hours=h)).timestamp()), Open=c, High=c + 1, Low=c - 1, Close=c, Volume=1))
    df = pd.DataFrame(rows)
    eod = pd.DataFrame(dict(date=[x.strftime("%Y-%m-%d") for x in dien], close=[100.0, 100.0, 100.0, 100.0], volume=[1, 1, 1, 1]))
    r = patikrinti(df, eod, "eu")
    tikrinti("sesiju dalis 3/4", r["sesiju_dalis"], 0.75)
    tikrinti("diena su 10% nuokrypiu pazymeta", r["dienu_virs_2proc"], 1)
    tikrinti("OHLC klaidu nera", r["ohlc_klaidu"], 0)
    df2 = df.copy()
    df2.loc[0, "High"] = 50.0
    tikrinti("High < Close -> OHLC klaida", patikrinti(df2, eod, "eu")["ohlc_klaidu"], 1)

    # poslinkis: EODHD baras paslinktas 5 min -> aptinkamas
    idx = np.arange(100) * 300 + 1_700_000_000
    y = pd.Series(np.linspace(100, 120, 100) + np.sin(np.arange(100)), index=idx)
    tikrinti("tas pats laikas -> 0 min, 100%", poslinkis(y.copy(), y), (0, 100.0))
    e = pd.Series(y.values, index=idx - 300)
    tikrinti("EODHD 5 min anksciau -> +5", poslinkis(e, y)[0], 5)

    # pakartojimai: 429 kartojamas, 404 ne; raktas nepatenka i klaida
    import eodhd_patikra as P
    kvietimai = []
    sena, miegas = P._gauti, time.sleep
    globals()["time"].sleep = lambda s: None
    try:
        atsak = iter([(None, "HTTP 429"), (None, "HTTP 500"), ([1], None)])
        globals()["_gauti"] = lambda *a, **k: (kvietimai.append(1), next(atsak))[1]
        tikrinti("429/500 kartojami, tada pavyksta", gauti("x", "SLAPTAS"), ([1], None))
        tikrinti("trys kvietimai", len(kvietimai), 3)
        kvietimai.clear()
        globals()["_gauti"] = lambda *a, **k: (kvietimai.append(1), (None, "HTTP 404"))[1]
        tikrinti("404 nekartojamas", (gauti("x", "SLAPTAS")[1], len(kvietimai)), ("HTTP 404", 1))
        kvietimai.clear()
        globals()["_gauti"] = lambda *a, **k: (kvietimai.append(1), (None, "HTTP 503"))[1]
        tikrinti(f"503 kartojamas {BANDYMAI} k.", (gauti("x", "SLAPTAS")[1], len(kvietimai)), ("HTTP 503", BANDYMAI))
    finally:
        globals()["_gauti"] = sena
        globals()["time"].sleep = miegas

    # barai_5m: kiekvienam langui po uzklausa, "to" = b-1
    params = []

    def netikras(kelias, token, **p):
        params.append((p["from"], p["to"]))
        return [{"timestamp": p["from"], "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}], None
    globals()["gauti"], sena_g = netikras, gauti
    try:
        df, kl = barai_5m("X.US", "SLAPTAS", datetime(2020, 10, 1), datetime(2022, 1, 1))
    finally:
        globals()["gauti"] = sena_g
    tikrinti("barai_5m: 1 langas -> 1 uzklausa", len(params), 1)
    tikrinti("barai_5m: to = pabaiga - 1 s", params[0][1], int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp()) - 1)

    # irasymas / skaitymas: tas pats turinys, laikas teisingas
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        k = os.path.join(d, "x.npz")
        df = pd.DataFrame(dict(ts=[1790861400, 1790861700], Open=[1653.2, 1.5], High=[1660.0, 2.0],
                               Low=[1650.0, 1.0], Close=[1655.55, 1.25], Volume=[1e6, 3.0]))
        irasyti_5m(df, k)
        g = skaityti_5m(k, "America/New_York")
        tikrinti("skaitymas: kainos tikslios", g["Close"].tolist(), [1655.55, 1.25])
        tikrinti("skaitymas: laikas 09:30 NY", str(g.index[0]), "2026-10-01 09:30:00-04:00")

    print("-" * 60)
    print("SAVITIKRA: VISKAS GERAI" if ok else "SAVITIKRA: YRA KLAIDU")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--savitikra", action="store_true")
    ap.add_argument("--tik", type=int, default=None)
    a = ap.parse_args()
    sys.exit(savitikra() if a.savitikra else paleisti(a.tik))
