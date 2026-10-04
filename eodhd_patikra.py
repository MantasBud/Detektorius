#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EODHD RYSIO PATIKRA  -  2026-10-04
==================================

Vienkartine patikra pries archyvo parsisiuntima. Nieko nekeicia, nieko
nekomituoja. Raktas imamas IS APLINKOS (GitHub Secret EODHD_TOKEN) ir
NIEKADA nespausdinamas: klaidose rodomas tik klaidos tipas, ne URL.

Tikrina:
  1. Ar raktas veikia ir koks planas / kiek iskvietimu liko.
  2. Kiekvienai musu biržai (Yahoo galune -> EODHD kodas): ar gaunami
     vakarykscio 5 min barai, ir ar kaina sutampa su Yahoo.
  3. Kiek siekia 5 min istorija (2020-10 SAP ir AAPL).
  4. Ar gaunamos naujienos su tiksliu laiku (SAP, AAPL).

Paleidimas:
    python eodhd_patikra.py --savitikra     (be tinklo)
    python eodhd_patikra.py                 (reikia EODHD_TOKEN aplinkoje)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

# Yahoo galune -> EODHD biržos kodas (patikrinama paleidus)
BIRZOS = {"DE": "XETRA", "PA": "PA", "AS": "AS", "MI": "MI", "MC": "MC",
          "HE": "HE", "BR": "BR", "LS": "LS", "IR": "IR", "VI": "VI"}
BAZE = "https://eodhd.com/api"


def eodhd_simbolis(yahoo):
    """SAP.DE -> SAP.XETRA; AAPL -> AAPL.US. Nezinoma galune -> None."""
    if "." not in yahoo:
        return f"{yahoo}.US"
    t, g = yahoo.rsplit(".", 1)
    kod = BIRZOS.get(g.upper())
    return f"{t}.{kod}" if kod else None


def _gauti(kelias, token, **params):
    """GET be rakto atskleidimo: klaidoje grazinamas tik tipas ir statusas."""
    import requests
    params.update(api_token=token, fmt="json")
    try:
        r = requests.get(f"{BAZE}/{kelias}", params=params, timeout=60)
    except Exception as e:
        return None, f"rysio klaida ({type(e).__name__})"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    try:
        return r.json(), None
    except Exception:
        return None, "ne JSON atsakymas"


def _unix(d):
    return int(d.replace(tzinfo=timezone.utc).timestamp())


def paleisti():
    token = os.environ.get("EODHD_TOKEN", "").strip()
    if not token:
        sys.exit("KLAIDA: EODHD_TOKEN nenustatytas (GitHub -> Settings -> Secrets -> Actions)")
    import universas as U
    import yfinance as yf

    print("1) RAKTAS IR PLANAS")
    u, kl = _gauti("user", token)
    if kl:
        sys.exit(f"   raktas neveikia: {kl}")
    for k in ("subscriptionType", "paymentMethod", "apiRequests", "dailyRateLimit",
              "apiRequestsDate"):
        if k in u:
            print(f"   {k}: {u[k]}")

    print("\n2) BIRZOS: vakarykscio 5 min barai ir kaina pries Yahoo")
    visi = U.visi_tikeriai() + [t for l in U.UNIVERSAS_US.values() for t in l]
    pvz = {}
    for t in visi:
        g = t.rsplit(".", 1)[1].upper() if "." in t else "US"
        pvz.setdefault(g, t)
    iki = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    nuo = iki - timedelta(days=5)
    gerai = 0
    for g, t in sorted(pvz.items()):
        s = eodhd_simbolis(t)
        if s is None:
            print(f"   {g:<3} {t:<10} -> galune nezinoma")
            continue
        d, kl = _gauti(f"intraday/{s}", token, interval="5m", **{"from": _unix(nuo), "to": _unix(iki)})
        if kl or not d:
            print(f"   {g:<3} {t:<10} -> {s:<14} NEPAVYKO: {kl or 'tuscia'}")
            continue
        pask = d[-1]
        try:
            yh = yf.download(t, period="10d", interval="1d", progress=False, auto_adjust=False)
            yc = float(yh["Close"].dropna().iloc[-1].squeeze())
        except Exception:
            yc = float("nan")
        print(f"   {g:<3} {t:<10} -> {s:<14} baru: {len(d):>4}, paskutinis "
              f"{pask.get('datetime')} UTC close {pask.get('close')}  (Yahoo dienos close {yc:.4g})")
        gerai += 1
    print(f"   biržų su duomenimis: {gerai} is {len(pvz)}")

    print("\n3) 5 MIN ISTORIJOS GYLIS (2020-10-01 .. 2020-10-15)")
    for s in ("SAP.XETRA", "AAPL.US"):
        d, kl = _gauti(f"intraday/{s}", token, interval="5m",
                       **{"from": _unix(datetime(2020, 10, 1)), "to": _unix(datetime(2020, 10, 15))})
        print(f"   {s:<10} " + (f"NEPAVYKO: {kl}" if kl else
                                f"baru: {len(d)}" + (f", pirmas {d[0].get('datetime')}" if d else "")))

    print("\n4) NAUJIENOS (5 naujausios) ir istorija 2021-01")
    for s in ("SAP.XETRA", "AAPL.US"):
        d, kl = _gauti("news", token, s=s, limit=5)
        if kl:
            print(f"   {s}: NEPAVYKO {kl}")
            continue
        print(f"   {s}: {len(d)} naujienu")
        for n in d[:3]:
            print(f"      {n.get('date')}  {str(n.get('title'))[:80]}")
        d2, kl2 = _gauti("news", token, s=s, limit=100, **{"from": "2021-01-01", "to": "2021-01-31"})
        print(f"      2021-01: " + (f"NEPAVYKO {kl2}" if kl2 else f"{len(d2)} naujienu"))

    u2, _ = _gauti("user", token)
    if u2 and "apiRequests" in u2:
        print(f"\nIskvietimu panaudota is viso siandien: {u2['apiRequests']}")


def savitikra():
    ok = True

    def tikrinti(s, a, b):
        nonlocal ok
        g = a == b
        print(f"  {'OK ' if g else 'BLOGAI'}  {s}{'' if g else f'  (gauta {a}, laukta {b})'}")
        ok = ok and g
    tikrinti("SAP.DE -> SAP.XETRA", eodhd_simbolis("SAP.DE"), "SAP.XETRA")
    tikrinti("AAPL -> AAPL.US", eodhd_simbolis("AAPL"), "AAPL.US")
    tikrinti("NOKIA.HE -> NOKIA.HE", eodhd_simbolis("NOKIA.HE"), "NOKIA.HE")
    tikrinti("nezinoma galune -> None", eodhd_simbolis("X.ZZ"), None)

    # raktas niekada nepatenka i klaidos teksta
    import types
    sekretas = "SLAPTAS123"

    class R:
        status_code = 403

        def json(self):
            return {}
    fake = types.SimpleNamespace(get=lambda *a, **k: R())
    sys.modules["requests"] = fake
    _, kl = _gauti("user", sekretas)
    tikrinti("HTTP klaidoje rakto nera", sekretas in str(kl), False)

    def kelti(*a, **k):
        raise ConnectionError(f"https://eodhd.com/api/user?api_token={sekretas}")
    sys.modules["requests"] = types.SimpleNamespace(get=kelti)
    _, kl = _gauti("user", sekretas)
    tikrinti("rysio klaidoje (su URL) rakto nera", sekretas in str(kl), False)
    del sys.modules["requests"]

    print("-" * 60)
    print("SAVITIKRA: VISKAS GERAI" if ok else "SAVITIKRA: YRA KLAIDU")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--savitikra", action="store_true")
    a = ap.parse_args()
    sys.exit(savitikra() if a.savitikra else paleisti())
