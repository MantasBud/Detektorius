#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZYMU PATIKRA — AR POZYMIS PADEDA ATPAZINTI TIKRA SCENARIJU  —  2026-09-28
=========================================================================

Tikslas (Manto): detektorius turi ATPAZINTI du apibreztus scenarijus. Sis
failas NEIESKO pranasumo pries atsitiktini pirkima ir NIEKO NEFILTRUOJA.

Klausimas vienas: ar pozymis padeda atskirti TIKRA scenariju nuo panasaus?

Kas yra "tikras scenarijus" (apibrezta PRIES paleidima):
  1 scenarijus (Atsistatymas): po signalo kaina pasiekia KORTELES tiksla
      (konservatyvu, 0.5 ATR) anksciau nei stop'a, per 3 sesijas.
  2 scenarijus (Ralis): isejus slenkanciu stop'u arba pasibaigus laikui,
      rezultatas teigiamas (tikslo nera pagal apibrezima).
Signalai - TIE PATYS, kuriuos rodo detektorius: D.aptikti(), pirmas tinkamas
kiekvieno scenarijaus signalas per akcijos sesija, kaip kalibracijoje.

Kas tikrinama - zymos is literaturos (claude/atpazinimas-amd-ir-skirtukai.md).
Visos uzrasytos cia PRIES paleidima; ribos naturalios (0, 2x, 95%), ne derintos:

  1 scenarijus:
    Z1 "kritimas per ataskaita"       ataskaita per [D-2, D]            Savor 2012, Chan 2003
    Z2 "rinka nekrito"                universo mediana 5 sesiju >= 0     Da, Liu & Schaumburg 2014
    Z3 "sektorius nekrito"            sektoriaus mediana 5 sesiju >= 0   Da, Liu & Schaumburg 2014
    Z4 "neramus rezimas"              rinkos 20 d. svyravimas > 1 m. medianos   Nagel 2012
    Z5 "apyvartos kulminacija"        kritimo dienos (D-1) apyvarta >= 2x       Cooper 1999 / CGW 1993
    Z6 "kulminacija be ataskaitos"    Z5 ir ne Z1                        Llorente ir kt. 2002
  2 scenarijus:
    Z7 "tarpas per ataskaita"         ataskaita per [D-1, D]             Brandt 2008, Jiang & Zhu 2017
    Z8 "prie 52 sav. aukstumos"       vakar uzdarymas >= 95% 52 sav. aukstumos  George & Hwang 2004

Visi pozymiai skaiciuojami tik is duomenu iki signalo (dienos lygmeniu - iki
D-1). Ataskaitu datos naudojamos TIK praeities ([D-2, D]) - tokia data signalo
metu jau buvo zinoma, tad ateities cia nera.

Kaip bus naudojama (Manto sprendimas: nesiaurinti): zyma, kuri praeina, bus
rodoma KORTELEJE kaip informacija. Signalu nesumazes. Zyma, kuri nepraeina,
nerodoma - kad korteles nebutu apkrautos nieko nereiskiancia informacija.

Verdiktas "PADEDA" - VISOS trys:
  1. "tikru" dalies skirtumas (su zyma - be zymos) ta pacia kryptimi visose
     keturiose pusese ir ne mazesnis nei 5 procentiniai punktai;
  2. sujungtose nematytose pusese bootstrap p < 0.05 (pagal diena);
  3. kiekvienoje puseje abiejose grupese >= 30 signalu.

Paleidimas:
    python zymos.py --savitikra
    python zymos.py
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import detektorius as D

try:
    import yfinance as yf
except ImportError:
    yf = None

MIN_PP = 5.0          # minimalus skirtumas procentiniais punktais
MIN_N = 30

ZYMOS = [
    # (kodas, scenarijus, pavadinimas, hipotezes kryptis: +1 - daugiau tikru, -1 - maziau)
    ("Z1", 1, "kritimas per ataskaita", -1),
    ("Z2", 1, "rinka nekrito", +1),
    ("Z3", 1, "sektorius nekrito", +1),
    ("Z4", 1, "neramus rezimas", +1),
    ("Z5", 1, "apyvartos kulminacija", 0),
    ("Z6", 1, "kulminacija be ataskaitos", +1),
    ("Z7", 2, "tarpas per ataskaita", +1),
    ("Z8", 2, "prie 52 sav. aukstumos", +1),
]


# ============================================================ dienos kontekstas

def sektoriu_zemelapis(rinka):
    """tickeris -> sektorius is universas.py. Jei neimanoma - tuscias."""
    try:
        import universas as U
        src = U.UNIVERSAS if rinka == "eu" else U.UNIVERSAS_US
        m = {}
        for sek, lst in src.items():
            for t in lst:
                m.setdefault(t, sek)
        return m
    except Exception:
        return {}


def dienos_kontekstas(dienos, sekt):
    """Visi dienos lygio pozymiai, PASLINKTI: eilute D turi tik info iki D-1.

    dienos: {tickeris: dienos OHLCV DataFrame}
    Grazina {tickeris: DataFrame(index=data)} su stulpeliais:
      ret5_rinka, ret5_sekt, rezimas_aukstas, relvol_vakar, iki_52s
    """
    closes = pd.DataFrame({t: d["Close"] for t, d in dienos.items()}).sort_index()
    ret1 = closes.pct_change()
    ret5 = closes / closes.shift(5) - 1.0
    # rinka
    rinka5 = ret5.median(axis=1)
    rinka1 = ret1.median(axis=1)
    vol20 = rinka1.rolling(20).std()
    rezimas = vol20 > vol20.rolling(250, min_periods=120).median()
    out = {}
    for t, d in dienos.items():
        sek = sekt.get(t)
        draugai = [x for x, s in sekt.items() if s == sek and x != t and x in ret5]
        if len(draugai) >= 4:
            sekt5 = ret5[draugai].median(axis=1)
        else:
            sekt5 = ret5.drop(columns=[t], errors="ignore").median(axis=1)
        v = d["Volume"]
        relvol = v / v.rolling(20).median().shift(1)
        h52 = d["High"].rolling(252, min_periods=120).max()
        k = pd.DataFrame(index=d.index)
        k["ret5_rinka"] = rinka5.reindex(d.index)
        k["ret5_sekt"] = sekt5.reindex(d.index)
        k["rezimas_aukstas"] = rezimas.reindex(d.index).astype(float)
        k["relvol"] = relvol
        k["iki_52s"] = d["Close"] / h52
        # <<< viena eilute: D mato tik iki D-1 (tas pats principas kaip detektoriuje)
        out[t] = k.shift(1)
        # relvol_vakar = kritimo dienos (D-1) apyvarta; po shift(1) tai jau ji
    return out


def ataskaita_lange(datos, ses, nuo, iki):
    """Ar yra ataskaitos data intervale [ses+nuo, ses+iki] dienomis (tik praeitis)."""
    if not datos:
        return np.nan
    try:
        for x in datos:
            dd = (pd.Timestamp(x).date() - ses).days
            if nuo <= dd <= iki:
                return 1.0
        return 0.0
    except Exception:
        return np.nan


def zymos_signalui(tipas, kontekstas, atask_datos, ses):
    """Grazina {Zx: 1.0/0.0/NaN} vienam signalui."""
    z = {}
    k = kontekstas
    if tipas == 1:
        z["Z1"] = ataskaita_lange(atask_datos, ses, -2, 0)
        z["Z2"] = np.nan if pd.isna(k["ret5_rinka"]) else float(k["ret5_rinka"] >= 0)
        z["Z3"] = np.nan if pd.isna(k["ret5_sekt"]) else float(k["ret5_sekt"] >= 0)
        z["Z4"] = np.nan if pd.isna(k["rezimas_aukstas"]) else float(k["rezimas_aukstas"] > 0.5)
        z["Z5"] = np.nan if pd.isna(k["relvol"]) else float(k["relvol"] >= 2.0)
        z["Z6"] = (np.nan if (pd.isna(z["Z5"]) or pd.isna(z["Z1"]))
                   else float(z["Z5"] == 1.0 and z["Z1"] == 0.0))
    else:
        z["Z7"] = ataskaita_lange(atask_datos, ses, -1, 0)
        z["Z8"] = np.nan if pd.isna(k["iki_52s"]) else float(k["iki_52s"] >= 0.95)
    return z


# ============================================================ surinkimas

def atsisiusti(rinka, dienos5):
    tick = D.universas(rinka)
    dien = yf.download(tick, period="2y", interval="1d", auto_adjust=False,
                       progress=False, group_by="ticker", threads=True, actions=True)
    intr = yf.download(tick, period=f"{min(dienos5, 60)}d", interval="5m",
                       auto_adjust=False, progress=False, group_by="ticker",
                       threads=True, prepost=False)
    dd, rod, barai = {}, {}, {}
    for t in tick:
        d = D._vienas(dien, t)
        if d is None or len(d) < 260:
            continue
        r = D.dienos_rodikliai(d, d["Dividends"] if "Dividends" in d else None)
        b = D._vienas(intr, t)
        if r is None or b is None or len(b) < 100:
            continue
        dd[t], rod[t] = d, r
        barai[t] = D.sesijos_rodikliai(b, rinka)
    return dd, rod, barai


def surinkti(rinka, dienos5, duomenys=None, atask=None):
    zyme = D.RINKOS[rinka]["zyme"]
    print(f"\n{'='*84}\n{zyme}\n{'='*84}")
    dd, rod, barai = duomenys if duomenys else atsisiusti(rinka, dienos5)
    print(f"  akciju: {len(barai)}")
    kont = dienos_kontekstas(dd, sektoriu_zemelapis(rinka))
    if atask is None:
        atask = D.ataskaitu_kalendorius(list(barai))
    su_atask = sum(1 for t in barai if atask.get(t))
    print(f"  ataskaitu datos turimos: {su_atask} is {len(barai)} akciju"
          + ("   <<< MAZAI - Z1 ir Z7 rezultatas nepatikimas" if su_atask < 0.6 * len(barai) else ""))
    eil = []
    for t, d in barai.items():
        sesijos = list(d.groupby("sesija", sort=True))
        for si, (ses, sd) in enumerate(sesijos):
            if len(sd) < D.S2_ORB_BARU + 4:
                continue
            kd = D._kd(rod[t], ses)
            if kd is None:
                continue
            kk = kont[t]
            kx = kk[[x.date() == ses for x in kk.index]]
            if not len(kx):
                continue
            kx = kx.iloc[0]
            suveike = set()
            for i in range(D.S2_ORB_BARU + 3, len(sd)):
                iki = D.RINKOS[rinka]["uzdarymas"] - int(sd["minute"].iloc[i])
                for s in D.aptikti(sd.iloc[:i + 1], kd, iki, rinka):
                    if s["tipas"] in suveike or not s["tinkamas"]:
                        continue
                    suveike.add(s["tipas"])
                    toliau = pd.concat(
                        [sd.iloc[i + 1:]] +
                        [sesijos[si + j][1] for j in range(1, D.HORIZONTAS_SESIJU)
                         if si + j < len(sesijos)])
                    b = D.baigtis(toliau, s)
                    tikras = (b["baigtis"] == "tikslas") if s["tipas"] == 1 \
                        else (b["pelnas_pct"] > 0)
                    e = dict(rinka=rinka, tickeris=t, sesija=str(ses),
                             tipas=s["tipas"], tikras=float(tikras))
                    e.update(zymos_signalui(s["tipas"], kx, atask.get(t), ses))
                    eil.append(e)
                if len(suveike) == 2:
                    break
    df = pd.DataFrame(eil)
    if len(df):
        for tp in (1, 2):
            g = df[df["tipas"] == tp]
            if len(g):
                print(f"  {tp} scenarijus: {len(g)} signalu, tikru "
                      f"{g['tikras'].mean()*100:.1f}%")
    return df


# ============================================================ verdiktas

def bootstrap_p(a_d, b_d, n=4000, seed=7):
    """Dienos vidurkiai a (su zyma) ir b (be zymos); dvipusis p."""
    if len(a_d) < 8 or len(b_d) < 8:
        return float("nan")
    r = np.random.default_rng(seed)
    sk = np.array([r.choice(a_d, len(a_d), True).mean()
                   - r.choice(b_d, len(b_d), True).mean() for _ in range(n)])
    return 2 * min((sk <= 0).mean(), (sk >= 0).mean())


def vertinti(dalys):
    matavimai = list(dalys)
    padeda = []
    print(f"\n{'='*112}")
    print("TIKRU SCENARIJU DALIS: SU ZYMA minus BE ZYMOS (procentiniai punktai)")
    print(f"{'='*112}")
    print(f"  {'zyma':<34}" + "".join(f"{m:>15}" for m in matavimai)
          + f"{'p (nemat.)':>12}  verdiktas")
    for kod, tp, pav, kryptis in ZYMOS:
        sk, nn, p_nem = [], [], []
        for m in matavimai:
            df = dalys[m]
            if kod not in df:
                nn.append((0, 0))
                sk.append(None)
                continue
            g = df[(df["tipas"] == tp)].dropna(subset=[kod])
            a, b = g[g[kod] == 1.0], g[g[kod] == 0.0]
            nn.append((len(a), len(b)))
            if len(a) < MIN_N or len(b) < MIN_N:
                sk.append(None)
                continue
            sk.append((a["tikras"].mean() - b["tikras"].mean()) * 100)
            if "NEMATYTA" in m:
                p_nem.append((a.groupby("sesija")["tikras"].mean().values,
                              b.groupby("sesija")["tikras"].mean().values))
        p = (bootstrap_p(np.concatenate([x[0] for x in p_nem]),
                         np.concatenate([x[1] for x in p_nem]))
             if p_nem else float("nan"))
        if any(x is None for x in sk):
            v = "per mazai signalu"
        elif not (all(x >= MIN_PP for x in sk) or all(x <= -MIN_PP for x in sk)):
            v = "nelaikosi"
        elif not np.isfinite(p) or p >= 0.05:
            v = f"kryptis sutampa, bet p={p:.2f}"
        else:
            v = "PADEDA"
            padeda.append(f"{kod} {pav}")
            if kryptis != 0 and np.sign(sk[0]) != kryptis:
                v += " (PRIESINGA hipotezei kryptimi)"
        t = "".join(f"{x:>+14.1f} " if x is not None else f"{'N=%d/%d' % n:>15}"
                    for x, n in zip(sk, nn))
        print(f"  {kod} {pav:<31}{t}{p:>12.3f}  {v}")
        print(f"  {'':<34}" + "".join(f"{'%d / %d' % n:>15}" for n in nn)
              + "   (signalu su zyma / be)")
    n = len(ZYMOS)
    print(f"\n  Patikrinta zymu: {n}.  Padeda: {len(padeda)}"
          + (f"  ({'; '.join(padeda)})" if padeda else "") + ".")
    print(f"  Atsitiktinai ta pacia kryptimi visose keturiose pusese tiketina ~{n*0.125:.1f}")
    print(f"  (be p ir be 5 p.p. ribu).")
    print("=" * 112)
    return padeda


def paleisti(dienos5):
    dalys = {}
    for rinka in ("eu", "us"):
        df = surinkti(rinka, dienos5)
        if df.empty:
            continue
        zyme = D.RINKOS[rinka]["zyme"]
        ses = sorted(df["sesija"].unique())
        riba = ses[len(ses) // 2]
        dalys[f"{zyme} matyta"] = df[df["sesija"] <= riba]
        dalys[f"{zyme} NEMATYTA"] = df[df["sesija"] > riba]
    if len(dalys) < 4:
        sys.exit("truksta duomenu - keturiu matavimu nera")
    vertinti(dalys)


# ============================================================ savitikra

def savitikra():
    ok = True

    def tikrinti(s, a, b):
        nonlocal ok
        print(f"  {'OK ' if a == b else 'BLOGAI'}  {s}"
              f"{'' if a == b else f'  (gauta {a}, laukta {b})'}")
        ok = ok and (a == b)

    from datetime import date
    ses = date(2026, 7, 24)
    tikrinti("ataskaita D-1 patenka i [D-2, D]",
             ataskaita_lange(["2026-07-23"], ses, -2, 0), 1.0)
    tikrinti("ataskaita D+1 (ateitis) NEPATENKA",
             ataskaita_lange(["2026-07-25"], ses, -2, 0), 0.0)
    tikrinti("be kalendoriaus - NaN (nezinoma), ne 0",
             bool(np.isnan(ataskaita_lange([], ses, -2, 0))), True)

    # dienos kontekstas: 6 akcijos, viena krenta; rinka kyla
    idx = pd.bdate_range("2025-01-01", periods=300)
    r = np.random.default_rng(1)
    dien = {}
    for i in range(6):
        c = 100 * np.cumprod(1 + np.full(300, 0.002) + r.normal(0, 0.001, 300))
        v = np.full(300, 1e6)
        dien[f"T{i}"] = pd.DataFrame(dict(Open=c, High=c * 1.01, Low=c * 0.99,
                                          Close=c, Volume=v), index=idx)
    # T0 krito paskutines 5 dienas ir D-1 turejo 3x apyvarta
    d0 = dien["T0"].copy()
    d0.iloc[-6:-1, d0.columns.get_loc("Close")] *= np.linspace(0.98, 0.90, 5)
    d0.iloc[-2, d0.columns.get_loc("Volume")] = 3e6
    dien["T0"] = d0
    sekt = {f"T{i}": "A" for i in range(6)}
    k = dienos_kontekstas(dien, sekt)
    kd = k["T0"].iloc[-1]           # D = paskutine diena, mato iki D-1
    tikrinti("rinka kilo -> Z2 'rinka nekrito' = 1", float(kd["ret5_rinka"] >= 0), 1.0)
    tikrinti("sektorius (be pacios akcijos) kilo -> Z3 = 1",
             float(kd["ret5_sekt"] >= 0), 1.0)
    tikrinti("kritimo dienos (D-1) apyvarta 3x matoma dienoje D",
             round(float(kd["relvol"]), 1), 3.0)
    # ateitis: pakeitus PASKUTINE diena (D), D eilute nesikeicia
    d1 = {t: x.copy() for t, x in dien.items()}
    d1["T0"].iloc[-1, d1["T0"].columns.get_loc("Volume")] = 50e6
    d1["T0"].iloc[-1, d1["T0"].columns.get_loc("Close")] *= 0.5
    k1 = dienos_kontekstas(d1, sekt)
    tikrinti("dienos D pozymiai nepriklauso nuo pacios D (tik iki D-1)",
             bool(np.allclose(k1["T0"].iloc[-1].fillna(-9).values,
                              kd.fillna(-9).values)), True)
    # bet D+1 eilute jau mato D (testas ne tuscias)
    # sektoriaus mediana T0 = TIKSLI T1..T5 mediana (be pacios T0)
    cl = pd.DataFrame({t: x["Close"] for t, x in dien.items()})
    r5 = (cl / cl.shift(5) - 1).shift(1).iloc[-1]
    tikrinti("sektoriaus mediana = kitu sektoriaus akciju mediana (be pacios)",
             round(float(k["T0"].iloc[-1]["ret5_sekt"]), 10),
             round(float(r5[[f"T{i}" for i in range(1, 6)]].median()), 10))

    # 52 sav.: vakar prie aukstumos
    tikrinti("52 sav. aukstuma skaiciuojama (kylanti serija - prie aukstumos)",
             float(k["T1"].iloc[-1]["iki_52s"] >= 0.95), 1.0)

    # zymos vienam signalui
    z = zymos_signalui(1, kd, ["2026-07-23"], ses)
    tikrinti("Z6 = kulminacija IR be ataskaitos (cia ataskaita buvo -> 0)",
             z["Z6"], 0.0)
    z2 = zymos_signalui(1, kd, ["2026-01-01"], ses)
    tikrinti("Z6 = 1, kai kulminacija ir ataskaitos nebuvo", z2["Z6"], 1.0)
    tikrinti("Z1: RYTOJAUS ataskaita (ateitis) signalo nezymi",
             zymos_signalui(1, kd, ["2026-07-25"], ses)["Z1"], 0.0)
    tikrinti("Z7: RYTOJAUS ataskaita (ateitis) signalo nezymi",
             zymos_signalui(2, kd, ["2026-07-25"], ses)["Z7"], 0.0)
    z3 = zymos_signalui(2, kd, ["2026-07-24"], ses)
    tikrinti("Z7: ataskaita tą pacia diena -> 1", z3["Z7"], 1.0)

    # verdiktas
    import io, contextlib
    def dal(sk_pp):
        eil = []
        rng = np.random.default_rng(3)
        for i in range(300):
            s = f"2026-0{1 + i % 8}-{1 + i % 27:02d}"
            zy = float(i % 2)
            p = 0.5 + (sk_pp / 200 if zy else -sk_pp / 200)
            eil.append(dict(sesija=s, tipas=1, tikras=float(rng.random() < p),
                            Z1=zy, Z2=zy, Z3=zy, Z4=zy, Z5=zy, Z6=zy))
        return pd.DataFrame(eil)
    for pav, sk, laukta in (("aiskus 30 p.p. skirtumas -> PADEDA", 30, True),
                            ("jokio skirtumo -> nepadeda", 0, False)):
        dd = {m: dal(sk) for m in ("EU matyta", "EU NEMATYTA",
                                   "JAV matyta", "JAV NEMATYTA")}
        with contextlib.redirect_stdout(io.StringIO()):
            pad = vertinti(dd)
        tikrinti(f"verdiktas: {pav}", any(x.startswith("Z2") for x in pad), laukta)

    print("-" * 60)
    print("SAVITIKRA: VISKAS GERAI" if ok else "SAVITIKRA: YRA KLAIDU")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--savitikra", action="store_true")
    ap.add_argument("--dienos", type=int, default=60)
    a = ap.parse_args()
    if a.savitikra:
        sys.exit(savitikra())
    D._KURSAS.clear()
    paleisti(a.dienos)
