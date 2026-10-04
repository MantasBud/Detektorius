#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EODHD NAUJIENU ARCHYVAS IR PADENGIMO PATIKRA  -  2026-10-04
==========================================================

EODHD plano 3 zingsnis, 1 dalis. Pirmoji patikra (eodhd_patikra.py) EU
naujienas tikrino tik vienu simboliu (SAP.XETRA) ir rado labai mazai. Tai
galejo buti simbolio, o ne saltinio problema: EU straipsniai EODHD gali buti
pazymeti kita galune (pvz. Frankfurto .F). Todel kiekvienai EU akcijai
siunciami DU variantai - biržos simbolis ir .F - ir sujungiami (dublikatai
pagal nuoroda / pavadinima pasalinami). JAV akcijoms - .US.

Visas laikotarpis 2020-10-01 .. vakar, puslapiuojant (limit 1000, offset).
Saugoma: naujienos/<tikeris>.jsonl.gz - data (UTC), pavadinimas, simboliai,
zymos, sentimentas, nuoroda. Turinys (content) nesaugomas.

Ataskaita: straipsniu skaicius per metus kiekvienai biržai ir pavyzdinems
akcijoms (RHM, MC, SAP, ASML, SIE, AIR, AAPL); kuris variantas ka davė.

Raktas is aplinkos (EODHD_TOKEN), niekada nespausdinamas.
Saugiklis: jei dienos iskvietimu > SAUGIKLIS - sustojama (lieka kas gauta).

    python naujienos_archyvas.py --savitikra
    python naujienos_archyvas.py [--tik N]
"""

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

from archyvas import gauti
from eodhd_patikra import BIRZOS

PRADZIA = "2020-10-01"
LIMIT = 1000
LANGAS_MEN = 6                       # uzklausos langas (menesiai) - kad offset nebutu didziulis
APLANKAS = "naujienos"
SAUGIKLIS = 85000
PAVYZDZIAI = ["RHM.DE", "MC.PA", "SAP.DE", "ASML.AS", "SIE.DE", "AIR.PA", "AAPL", "NVDA"]
SRAUTAI = 4


def variantai(yahoo):
    """Kokiais EODHD simboliais ieskoti naujienu."""
    if "." not in yahoo:
        return [f"{yahoo}.US"]
    t, g = yahoo.rsplit(".", 1)
    kod = BIRZOS.get(g.upper(), g.upper())
    out = [f"{t}.{kod}"]
    if kod != "F":
        out.append(f"{t}.F")
    return out


def langai(nuo, iki, men=LANGAS_MEN):
    a = pd.Timestamp(nuo)
    b = pd.Timestamp(iki)
    out = []
    while a <= b:
        c = min(a + pd.DateOffset(months=men) - pd.Timedelta(days=1), b)
        out.append((a.strftime("%Y-%m-%d"), c.strftime("%Y-%m-%d")))
        a = c + pd.Timedelta(days=1)
    return out


def raktas(n):
    return n.get("link") or (n.get("date", "") + "|" + str(n.get("title", "")))


def sutraukti(n, variantas):
    s = n.get("sentiment") or {}
    return dict(date=n.get("date"), title=n.get("title"), symbols=n.get("symbols") or [],
                tags=n.get("tags") or [], polarity=s.get("polarity"), link=n.get("link"),
                variantas=variantas)


def simbolio_naujienos(simb, token, nuo, iki, gauti_f=None):
    gauti_f = gauti_f or gauti
    visi, klaidos = [], []
    for a, b in langai(nuo, iki):
        off = 0
        while True:
            d, kl = gauti_f("news", token, s=simb, limit=LIMIT, offset=off, **{"from": a, "to": b})
            if kl:
                klaidos.append(kl)
                break
            d = d or []
            visi += d
            if len(d) < LIMIT:
                break
            off += LIMIT
    return visi, klaidos


def tikeriui(args):
    try:
        return _tikeriui(args)
    except Exception as e:
        return dict(tikeris=args[0], straipsniu=0, variantai={}, metai={},
                    klaidos=[f"{type(e).__name__}"])


def _tikeriui(args):
    t, token, iki = args
    vienas, per_var, klaidos = {}, {}, []
    for v in variantai(t):
        d, kl = simbolio_naujienos(v, token, PRADZIA, iki)
        klaidos += kl
        per_var[v] = len(d)
        for n in d:
            k = raktas(n)
            if k not in vienas:
                vienas[k] = sutraukti(n, v)
    eil = sorted(vienas.values(), key=lambda x: x["date"] or "")
    with gzip.open(os.path.join(APLANKAS, f"{t}.jsonl.gz"), "wt", encoding="utf-8") as f:
        for e in eil:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    metai = pd.Series([str(e["date"])[:4] for e in eil]).value_counts().to_dict() if eil else {}
    return dict(tikeris=t, straipsniu=len(eil), variantai=per_var, metai=metai,
                klaidos=sorted(set(klaidos)))


def paleisti(tik=None):
    token = os.environ.get("EODHD_TOKEN", "").strip()
    if not token:
        sys.exit("KLAIDA: EODHD_TOKEN nenustatytas")
    import universas as U
    eu = list(U.visi_tikeriai())
    us = [t for l in U.UNIVERSAS_US.values() for t in l]
    if tik:
        eu = [t for t in PAVYZDZIAI if "." in t] + [t for t in eu if t not in PAVYZDZIAI][:tik]
        us = [t for t in PAVYZDZIAI if "." not in t] + [t for t in us if t not in PAVYZDZIAI][:tik]
    sar = list(dict.fromkeys(eu + us))
    os.makedirs(APLANKAS, exist_ok=True)
    iki = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    u, _ = gauti("user", token)
    pradzioj = int((u or {}).get("apiRequests", 0))
    print(f"Tikeriu: {len(sar)} (EU {len(eu)}, JAV {len(us)}); iskvietimu siandien pries: {pradzioj}", flush=True)

    rez, t0 = [], time.time()
    with ThreadPoolExecutor(SRAUTAI) as ex:
        ateitys = [ex.submit(tikeriui, (t, token, iki)) for t in sar]
        for k, a in enumerate(ateitys, 1):
            rez.append(a.result())
            if k % 25 == 0 or k == len(sar):
                u, _ = gauti("user", token)
                n = int((u or {}).get("apiRequests", 0))
                print(f"   {k}/{len(sar)}  ({(time.time() - t0) / 60:.1f} min, iskvietimu siandien {n})", flush=True)
                if n > SAUGIKLIS:
                    print("   SAUGIKLIS: sustojama, kad nebutu virsytas dienos limitas", flush=True)
                    for x in ateitys:
                        x.cancel()
                    break
    rep = pd.DataFrame(rez)
    rep["birza"] = rep["tikeris"].map(lambda t: t.rsplit(".", 1)[1] if "." in t else "US")
    rep.drop(columns=["metai"]).assign(variantai=rep["variantai"].astype(str)).to_csv(
        os.path.join(APLANKAS, "ataskaita.csv"), index=False)
    ataskaita(rep)


def ataskaita(rep):
    print(f"\n{'=' * 84}\nPADENGIMAS\n{'=' * 84}")
    metai = [str(m) for m in range(2020, datetime.now().year + 1)]
    print(f"  Straipsniu per metus (mediana per akcija) pagal birza:")
    print(f"  {'birza':<6}{'akciju':>7}" + "".join(f"{m:>7}" for m in metai) + f"{'be nieko':>10}")
    for g, x in rep.groupby("birza"):
        med = [int(pd.Series([r.get(m, 0) for r in x["metai"]]).median()) for m in metai]
        print(f"  {g:<6}{len(x):>7}" + "".join(f"{v:>7}" for v in med) + f"{int((x['straipsniu'] == 0).sum()):>10}")
    print("\n  Pavyzdines akcijos (straipsniu per metus; is kurio simbolio):")
    for _, x in rep[rep["tikeris"].isin(PAVYZDZIAI)].iterrows():
        print(f"    {x['tikeris']:<9}" + "".join(f"{x['metai'].get(m, 0):>7}" for m in metai)
              + f"   {x['variantai']}")
    kl = rep[rep["klaidos"].map(len) > 0]
    print(f"\n  su klaidomis: {len(kl)}" + (f"  pvz. {kl.iloc[0]['tikeris']}: {kl.iloc[0]['klaidos']}" if len(kl) else ""))


def savitikra():
    ok = True

    def tikrinti(s, a, b):
        nonlocal ok
        g = a == b
        print(f"  {'OK ' if g else 'BLOGAI'}  {s}{'' if g else f'  (gauta {a}, laukta {b})'}")
        ok = ok and g
    tikrinti("RHM.DE -> XETRA ir .F", variantai("RHM.DE"), ["RHM.XETRA", "RHM.F"])
    tikrinti("MC.PA -> PA ir .F", variantai("MC.PA"), ["MC.PA", "MC.F"])
    tikrinti("AAPL -> tik .US", variantai("AAPL"), ["AAPL.US"])
    tikrinti("BRK-B -> BRK-B.US", variantai("BRK-B"), ["BRK-B.US"])
    L = langai("2020-10-01", "2026-10-03")
    tikrinti("langai be tarpu", all((pd.Timestamp(L[i][1]) + pd.Timedelta(days=1)) == pd.Timestamp(L[i + 1][0])
                                     for i in range(len(L) - 1)), True)
    tikrinti("langai: pradzia ir pabaiga", (L[0][0], L[-1][1]), ("2020-10-01", "2026-10-03"))
    tikrinti("langu 13", len(L), 13)

    # puslapiavimas: 2500 straipsniu viename lange -> 3 uzklausos (1000+1000+500)
    kv = []

    def netikras(kelias, token, **p):
        kv.append((p["from"], p["offset"]))
        n = 2500 if p["from"] == "2026-04-01" else 3
        lik = max(0, min(LIMIT, n - p["offset"]))
        return [{"date": f"{p['from']}T10:00:00+00:00", "title": f"{p['offset']}-{i}", "link": f"{p['from']}/{p['offset']}/{i}"}
                for i in range(lik)], None
    d, kl = simbolio_naujienos("X.US", "SLAPTAS", "2026-04-01", "2026-12-31", netikras)
    tikrinti("puslapiavimas: visi 2500 + 3", len(d), 2503)
    tikrinti("uzklausu 3 + 1", len(kv), 4)
    # dublikatai tarp variantu pasalinami
    a = {"link": "x", "date": "2024-01-01", "title": "A"}
    tikrinti("raktas pagal nuoroda", raktas(a), "x")
    tikrinti("be nuorodos - data|pavadinimas", raktas({"date": "d", "title": "t"}), "d|t")
    print("-" * 60)
    print("SAVITIKRA: VISKAS GERAI" if ok else "SAVITIKRA: YRA KLAIDU")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--savitikra", action="store_true")
    ap.add_argument("--tik", type=int, default=None)
    a = ap.parse_args()
    sys.exit(savitikra() if a.savitikra else paleisti(a.tik))
