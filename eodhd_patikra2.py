#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EODHD PATIKRA 2 - Milanas, Lisabona, Viena  -  2026-10-04
=========================================================
Pirmoje patikroje MI (404), LS ir VI (tuscia) 5 min baru negavo. Ar kaltas
biržos kodas, tikerio pavadinimas, ar tos biržos intraday duomenu EODHD tiesiog
neturi? Raktas imamas is aplinkos ir niekada nespausdinamas.
"""
import os, sys
from datetime import datetime, timedelta, timezone
from eodhd_patikra import _gauti, _unix

TIKRINTI = {"MI": ["PRY", "ENI", "UCG", "ISP"], "LS": ["EDP", "GALP"], "VI": ["OMV", "VER"]}


def main():
    token = os.environ.get("EODHD_TOKEN", "").strip()
    if not token:
        sys.exit("KLAIDA: EODHD_TOKEN nenustatytas")
    ex, kl = _gauti("exchanges-list/", token)
    if kl:
        sys.exit(f"exchanges-list nepavyko: {kl}")
    print("BIRZOS, kuriu pavadinime Milan/Italian/Lisbon/Vienna:")
    kodai = set()
    for e in ex:
        txt = f"{e.get('Name','')} {e.get('Country','')} {e.get('OperatingMIC','')}"
        if any(w in txt for w in ("Milan", "Ital", "Lisbon", "Portugal", "Vienna", "Austria")):
            print(f"   {e.get('Code'):<8} {e.get('Name')}  ({e.get('Country')}, {e.get('OperatingMIC')})")
            kodai.add(e.get("Code"))
    iki = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    nuo = iki - timedelta(days=5)
    for g, tk in TIKRINTI.items():
        for kod in sorted(kodai | {g}):
            sl, kl = _gauti(f"exchange-symbol-list/{kod}", token)
            if kl or not sl:
                continue
            vardai = {s.get("Code") for s in sl}
            yra = [t for t in tk if t in vardai]
            if not yra:
                continue
            print(f"\n{g}: biržos kodas {kod}: simboliu {len(sl)}; is musu rasta {yra}")
            for t in yra:
                s = f"{t}.{kod}"
                d, kl1 = _gauti(f"eod/{s}", token, **{"from": (iki - timedelta(days=7)).date().isoformat()})
                i, kl2 = _gauti(f"intraday/{s}", token, interval="5m",
                                **{"from": _unix(nuo.replace(tzinfo=None)), "to": _unix(iki.replace(tzinfo=None))})
                print(f"   {s:<12} dienos: {kl1 or len(d)}   5 min: {kl2 or len(i)}")
    u, _ = _gauti("user", token)
    if u:
        print(f"\nIskvietimu panaudota siandien: {u.get('apiRequests')}")


if __name__ == "__main__":
    main()
