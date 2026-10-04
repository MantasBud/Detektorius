#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIVIDENDU SKIRTUKO DUOMENYS  -  2026-10-04
==========================================

Kartą per dieną sudaro docs/dividendai.json: universo akcijos, kurių
PASKELBTA dividendų ex-data patenka į artimiausias 30 dienų.

Šaltinis - EODHD /div/{simbolis} (paskelbti dividendai su suma, ex-, įrašymo
ir mokėjimo datomis; 1 iškvietimas akcijai). Milano akcijoms EODHD duomenų
neturi (HTTP 404) - joms Yahoo (yfinance): ex-data ir mokėjimo data iš
kalendoriaus, suma - paskutinio išmokėto dividendo (pažymima „~").

Kaina - paskutinė dienos uždarymo kaina. Viskas perskaičiuojama į EUR
(USD ir kt. - pagal EODHD valiutų kursą).

Įrašymo (fiksavimo) data: jei EODHD jos neturi - apskaičiuojama ir
pažymima „~": JAV (nuo 2024-05-28, T+1) = ex-data; EU (T+2) = ex-data + 1 d. d.

detektorius.py šį failą tik įdeda į puslapį - kortelės ir signalai nekeičiami.

    python dividendai.py --savitikra
    python dividendai.py              (reikia EODHD_TOKEN)
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from eodhd_patikra import _gauti, eodhd_simbolis

LANGAS_D = 30
ISVESTIS = "docs/dividendai.json"
US_T1 = date(2024, 5, 28)


def _d(x):
    try:
        return date.fromisoformat(str(x)[:10])
    except (TypeError, ValueError):
        return None


def irasymo_data(ex, rinka, duota=None):
    """(data, apytiksle?)"""
    d = _d(duota)
    if d:
        return d, False
    if ex is None:
        return None, True
    if rinka == "us" and ex >= US_T1:
        return ex, True
    n = ex + timedelta(days=1)
    while n.weekday() >= 5:
        n += timedelta(days=1)
    return n, True


def artimiausi(irasai, siandien, langas=LANGAS_D):
    """EODHD /div irasai -> tie, kuriu ex-data [siandien, siandien+langas]."""
    out = []
    for r in irasai or []:
        ex = _d(r.get("date"))
        if ex and siandien <= ex <= siandien + timedelta(days=langas):
            out.append(r)
    return sorted(out, key=lambda r: r["date"])


class Kursai:
    """1 valiutos vnt. -> EUR. EODHD FOREX kodas EUR{CUR}.FOREX (kiek CUR uz 1 EUR)."""

    def __init__(self, token, gauti=None):
        self.token, self.gauti, self.k = token, gauti or _gauti, {"EUR": 1.0}

    def eur(self, cur):
        cur = (cur or "EUR").upper()
        if cur == "GBX":
            return self.eur("GBP") / 100.0
        if cur not in self.k:
            d, kl = self.gauti(f"eod/EUR{cur}.FOREX", self.token,
                               **{"from": (date.today() - timedelta(days=10)).isoformat()})
            v = None
            if not kl and d:
                try:
                    v = 1.0 / float(d[-1]["close"])
                except (KeyError, ValueError, ZeroDivisionError, TypeError):
                    v = None
            self.k[cur] = v
        return self.k[cur]


def kaina_eodhd(simb, token, gauti=None):
    gauti = gauti or _gauti
    d, kl = gauti(f"eod/{simb}", token, **{"from": (date.today() - timedelta(days=10)).isoformat()})
    if kl or not d:
        return None
    try:
        return float(d[-1]["close"])
    except (KeyError, ValueError, TypeError):
        return None


def yahoo_irasai(t, siandien):
    """Atsarginis saltinis (Milanas). Grazina EODHD formato irasus arba []."""
    try:
        import yfinance as yf
        tk = yf.Ticker(t)
        kal = tk.calendar or {}
        ex = _d(kal.get("Ex-Dividend Date"))
        if not ex or not (siandien <= ex <= siandien + timedelta(days=LANGAS_D)):
            return [], None
        div = tk.dividends
        suma = float(div.iloc[-1]) if div is not None and len(div) else None
        info = tk.fast_info
        try:
            kaina = float(info["last_price"])
        except Exception:
            kaina = None
        return [dict(date=ex.isoformat(), paymentDate=str(kal.get("Dividend Date") or "")[:10] or None,
                     recordDate=None, value=suma, currency=getattr(info, "currency", None) or "EUR",
                     apytiksle_suma=True)], kaina
    except Exception:
        return [], None


def sudaryti(tikeriai, token, siandien=None, gauti=None, yahoo=None):
    siandien = siandien or datetime.now(timezone.utc).date()
    gauti = gauti or _gauti
    yahoo = yahoo or yahoo_irasai
    kurs = Kursai(token, gauti)
    eil, klaidos = [], 0
    for t, rinka in tikeriai:
        s = eodhd_simbolis(t)
        irasai, kaina, saltinis = [], None, "EODHD"
        d, kl = gauti(f"div/{s}", token, **{"from": (siandien - timedelta(days=10)).isoformat()}) if s else (None, "nera")
        if kl:
            irasai, kaina = yahoo(t, siandien)
            saltinis = "Yahoo"
            if not irasai:
                klaidos += 1
                continue
        else:
            irasai = artimiausi(d, siandien)
        if not irasai:
            continue
        if kaina is None:
            kaina = kaina_eodhd(s, token, gauti) if s else None
        kaina_cur = "USD" if rinka == "us" else "EUR"
        k_kaina = kurs.eur(kaina_cur)
        for r in irasai:
            ex = _d(r.get("date"))
            cur = r.get("currency") or kaina_cur
            k = kurs.eur(cur)
            suma = r.get("value")
            div_eur = float(suma) * k if (suma is not None and k) else None
            ir, apyt = irasymo_data(ex, rinka, r.get("recordDate"))
            kaina_eur = kaina * k_kaina if (kaina and k_kaina) else None
            eil.append(dict(
                tikeris=t, rinka="EU" if rinka == "eu" else "JAV",
                kaina_eur=round(kaina_eur, 2) if kaina_eur else None,
                div_eur=round(div_eur, 4) if div_eur is not None else None,
                div_orig=suma, valiuta=cur,
                suma_apytiksle=bool(r.get("apytiksle_suma")),
                pajamingumas=round(div_eur / kaina_eur * 100, 2) if (div_eur and kaina_eur) else None,
                ex=ex.isoformat() if ex else None,
                irasymo=ir.isoformat() if ir else None, irasymo_apytiksle=apyt,
                mokejimo=(_d(r.get("paymentDate")).isoformat() if _d(r.get("paymentDate")) else None),
                saltinis=saltinis))
    eil.sort(key=lambda x: (x["ex"] or "9999", x["tikeris"]))
    return dict(atnaujinta=datetime.now(timezone.utc).isoformat(), langas_d=LANGAS_D,
                eilutes=eil, be_duomenu=klaidos)


def paleisti():
    token = os.environ.get("EODHD_TOKEN", "").strip()
    if not token:
        sys.exit("KLAIDA: EODHD_TOKEN nenustatytas")
    import universas as U
    tik = [(t, "eu") for t in U.visi_tikeriai()] + \
          [(t, "us") for l in U.UNIVERSAS_US.values() for t in l]
    tik = list(dict.fromkeys(tik))
    rez = sudaryti(tik, token)
    os.makedirs(os.path.dirname(ISVESTIS), exist_ok=True)
    with open(ISVESTIS, "w", encoding="utf-8") as f:
        json.dump(rez, f, ensure_ascii=False, indent=1)
    print(f"akciju {len(tik)}; su ex-data per {LANGAS_D} d.: {len(rez['eilutes'])}; "
          f"be duomenu: {rez['be_duomenu']}")
    for e in rez["eilutes"][:60]:
        print(f"  {e['ex']}  {e['tikeris']:<10} {e['div_eur']} EUR  kaina {e['kaina_eur']}  "
              f"irasymo {e['irasymo']}{'~' if e['irasymo_apytiksle'] else ''}  mokejimo {e['mokejimo']}  {e['saltinis']}")


def savitikra():
    ok = True

    def tikrinti(s, a, b):
        nonlocal ok
        g = a == b
        print(f"  {'OK ' if g else 'BLOGAI'}  {s}{'' if g else f'  (gauta {a}, laukta {b})'}")
        ok = ok and g
    s = date(2026, 10, 5)
    ir = [dict(date="2026-10-01", value=1), dict(date="2026-10-05", value=2),
          dict(date="2026-11-04", value=3), dict(date="2026-11-05", value=4)]
    tikrinti("langas [siandien, +30] imtinai", [r["value"] for r in artimiausi(ir, s)], [2, 3])
    tikrinti("JAV irasymo = ex (T+1)", irasymo_data(date(2026, 10, 9), "us"), (date(2026, 10, 9), True))
    tikrinti("EU penktadienis -> pirmadienis", irasymo_data(date(2026, 10, 9), "eu"), (date(2026, 10, 12), True))
    tikrinti("duota data nekeiciama", irasymo_data(date(2026, 10, 9), "eu", "2026-10-13"), (date(2026, 10, 13), False))

    def netikras(kelias, token, **p):
        if kelias == "div/KO.US":
            return [dict(date="2026-11-13", recordDate="2026-11-13", paymentDate="2026-12-01",
                         value=0.53, currency="USD")], None
        if kelias == "div/MC.PA":
            return [dict(date="2026-12-01", recordDate=None, paymentDate="2026-12-03", value=5.5, currency="EUR"),
                    dict(date="2026-04-28", value=7.5, currency="EUR")], None
        if kelias == "div/PRY.MI":
            return None, "HTTP 404"
        if kelias == "eod/KO.US":
            return [dict(close=70.0)], None
        if kelias == "eod/MC.PA":
            return [dict(close=550.0)], None
        if kelias == "eod/EURUSD.FOREX":
            return [dict(close=1.25)], None
        return None, "HTTP 404"

    def yahoo(t, siandien):
        return ([dict(date="2026-11-20", paymentDate="2026-11-25", value=0.8, currency="EUR",
                      apytiksle_suma=True)], 40.0) if t == "PRY.MI" else ([], None)
    r = sudaryti([("KO", "us"), ("MC.PA", "eu"), ("PRY.MI", "eu"), ("X.PA", "eu")], "SLAPTAS",
                 siandien=date(2026, 11, 5), gauti=netikras, yahoo=yahoo)
    e = {x["tikeris"]: x for x in r["eilutes"]}
    tikrinti("KO: 0.53 USD -> 0.424 EUR", e["KO"]["div_eur"], 0.424)
    tikrinti("KO: kaina 70 USD -> 56 EUR", e["KO"]["kaina_eur"], 56.0)
    tikrinti("MC: tik busimas (12-01), ne balandzio", e["MC.PA"]["ex"], "2026-12-01")
    tikrinti("MC: irasymo ~ ex+1 d.d.", (e["MC.PA"]["irasymo"], e["MC.PA"]["irasymo_apytiksle"]), ("2026-12-02", True))
    tikrinti("PRY.MI is Yahoo, suma apytiksle", (e["PRY.MI"]["saltinis"], e["PRY.MI"]["suma_apytiksle"]), ("Yahoo", True))
    tikrinti("X.PA be duomenu -> praleista ir suskaiciuota", ("X.PA" in e, r["be_duomenu"]), (False, 1))
    tikrinti("rikiuota pagal ex-data", [x["tikeris"] for x in r["eilutes"]], ["KO", "PRY.MI", "MC.PA"])
    tikrinti("pajamingumas MC 5.5/550 = 1%", e["MC.PA"]["pajamingumas"], 1.0)
    print("SAVITIKRA: VISKAS GERAI" if ok else "SAVITIKRA: YRA KLAIDU")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--savitikra", action="store_true")
    a = ap.parse_args()
    sys.exit(savitikra() if a.savitikra else paleisti())
