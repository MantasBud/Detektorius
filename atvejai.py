#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATVEJU TYRIMAS  —  2026-09-27
=============================

Mantas nurode penkis KONKRECIUS ralius, kuriuos detektorius privalo pagauti:

    ADYEN   2026-08-10
    AMD     IBIS, 2026-08-14 .. 08-21  (kelios dienos!)
    CAP     SBF,  2026-07-23
    PTX     IBIS, 2026-08-03
    SAP     2026-07-23

Tai pirmas kartas projekte, kai turime ZYMETUS TEIGIAMUS PAVYZDZIUS. Iki
siol visi ~60 testu buvo paieska be etalono. Su etalonu klausimas pasikeicia
is "kas veikia" i "kodel mano taisykles sito nepagavo" - o tai atsakoma
tiksliai, be statistikos.

KA SIS SKRIPTAS PADARO
----------------------
1. Kiekvienam atvejui - dienos anatomija (tarpas, kelias per diena, kada
   buvo maksimumas, kaip uzdare, kas buvo kita diena).
2. Perleidzia scenarijus_1/2/3 per KIEKVIENA to sesijos bara ir parodo,
   kuris scenarijus suveike ir kada.
3. Jei scenarijus 2 NESUVEIKE - parodo POSALYGIU diagnostika: kuri butent
   salyga pasake "ne" tame bare, kur kaina prasilauze i naujos sesijos
   maksimuma. Tai tiesioginis atsakymas i klausima "issiaiskink kaip tokius
   detektuoti".
4. Suskaiciuoja MFE (kiek daugiausia buvo naudai) trimis horizontais:
   ta pati sesija / +1 sesija / +3 sesijos. Eurais nuo 18 000 minus 10.
   Tai patikrina Manto pastaba, kad realus sandoriai duoda simtus euru.
5. Paleidzia dabartini baigtis() ta paciai situacijai ir parodo, kur mano
   slenkantis stop'as ISEJO ir kiek liko ant stalo. Tai patikrina hipoteze,
   kad ralio scenarijaus minusas atejo is ISEJIMO taisykles, ne is detekcijos.

KONTEKSTAS SKAICIUOJAMAS TOS DIENOS ATZVILGIU
---------------------------------------------
detektorius.py v3 kalibracijoje ATR, SMA200, kritimo dienos ir rinkos rezimas
buvo SIANDIENOS reiksmes, taikomos visoms 60 sesijoms. Cia kontekstas
skaiciuojamas tik is duomenu IKI tos sesijos. Sitas failas kartu yra to
taisymo prototipas v4.
"""

import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("KLAIDA: reikia yfinance")

import detektorius as D

POZICIJA = D.POZICIJA
SANAUDOS = D.SANAUDOS_EUR

# (etikete, tikeriu kandidatai, pradzios data, pabaigos data)
ATVEJAI = [
    ("ADYEN",     ["ADYEN.AS", "ADYEN.DE"],        "2026-08-10", "2026-08-10"),
    ("AMD IBIS",  ["AMD.DE", "AMD.F", "AMD.BE"],   "2026-08-14", "2026-08-21"),
    ("CAP SBF",   ["CAP.PA", "CAPP.PA", "CGM.DE"], "2026-07-23", "2026-07-23"),
    ("PTX IBIS",  ["PTX.DE", "PTX.F", "PTX.BE"],   "2026-08-03", "2026-08-03"),
    ("SAP",       ["SAP.DE", "SAP.F"],             "2026-07-23", "2026-07-23"),
]

UZDARYMAS_MIN = 17 * 60 + 30     # EU sesija
TZ = "Europe/Berlin"


# ---------------------------------------------------------------- pagalbines

def vienas(df, t):
    """yfinance kartais grazina MultiIndex net vienam tikeriui."""
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        lygiai = df.columns.get_level_values(0)
        if t in set(lygiai):
            d = df[t]
        else:
            d = df.droplevel(0, axis=1) if len(set(lygiai)) == 1 else None
    else:
        d = df
    if d is None:
        return None
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    return d if len(d) else None


def rasti_tikeri(kandidatai):
    """Grazina (tikeris, dienos_df). Isbando kandidatus, praneša kuris tiko."""
    for t in kandidatai:
        try:
            raw = yf.download(t, period="3y", interval="1d",
                              auto_adjust=False, progress=False)
            d = vienas(raw, t)
            if d is not None and len(d) > 250:
                med = float((d["Close"] * d["Volume"]).tail(20).median())
                print(f"    tikeris {t}: OK, {len(d)} dienu, "
                      f"apyvarta ~{med/1e6:.1f} mln.")
                return t, d
            print(f"    tikeris {t}: per mazai duomenu "
                  f"({0 if d is None else len(d)} d.)")
        except Exception as e:
            print(f"    tikeris {t}: klaida {e}")
    return None, None


def kontekstas_sesijai(d, ses):
    """Kontekstas TIK is duomenu iki ses (be tos dienos).

    Tai v3 kalibracijos klaidos taisymas: ten atr/sma200/kritimo_dienu buvo
    paskutines dienos reiksmes, taikomos visoms 60 sesijoms.
    """
    dd = d[[x.date() < ses for x in d.index]]
    if len(dd) < 220:
        return None
    c, h, l, v = dd["Close"], dd["High"], dd["Low"], dd["Volume"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = float((tr.rolling(20).mean() / c).iloc[-1] * 100)
    sma200 = c.rolling(200).mean()
    kyla = bool(sma200.iloc[-1] > sma200.iloc[-21])
    apyv = float((c * v).rolling(20).median().iloc[-1])
    pok = (c / prev - 1.0).iloc[-6:]
    serija = 0
    for x in reversed(pok.tolist()):
        if x < 0:
            serija += 1
        else:
            break
    return dict(atr=atr, sma200_kyla=kyla, apyvarta=apyv,
                vakar_uzdarymas=float(c.iloc[-1]), kritimo_dienu=serija,
                uzdarymai=c)


# ---------------------------------------------------------------- anatomija

def look_ahead_matas(d, t, n=60):
    """Kiek v3 kalibracija klydo, taikydama SIANDIENOS konteksta 60-ciai sesiju.

    detektorius.py v3 dienos_kontekstas() skaiciuoja atr/sma200_kyla/
    kritimo_dienu tik paskutinei eilutei ir ta viena reiksme naudoja visoms
    kalibracijos sesijoms. Cia matuojam, kiek tai skiriasi nuo teisingo,
    tos dienos konteksto.
    """
    sesijos = [x.date() for x in d.index][-n:]
    siandien = kontekstas_sesijai(d, date.today())
    if siandien is None:
        return
    santykiai, sma_nesutampa, blok_klaidingai, blok_praleista = [], 0, 0, 0
    for ses in sesijos:
        k = kontekstas_sesijai(d, ses)
        if k is None:
            continue
        santykiai.append(k["atr"] / max(1e-9, siandien["atr"]))
        if k["sma200_kyla"] != siandien["sma200_kyla"]:
            sma_nesutampa += 1
            if siandien["sma200_kyla"]:
                blok_klaidingai += 1      # v3 praleido diena, kuria turejo blokuoti
            else:
                blok_praleista += 1       # v3 blokavo diena, kuria turejo praleisti
    if not santykiai:
        return
    s = np.array(santykiai)
    print(f"\n    ZVILGSNIO I ATEITI MATAS ({t}, {len(s)} sesijos):")
    print(f"      ATR tos dienos / ATR siandien:  mediana {np.median(s):.2f}, "
          f"ribos {s.min():.2f}..{s.max():.2f}")
    print(f"      sesiju, kur nuokrypis >20%: "
          f"{(np.abs(s-1) > 0.20).mean()*100:.0f}%")
    print(f"      SMA200 filtras skyresi {sma_nesutampa}/{len(s)} sesiju "
          f"({sma_nesutampa/len(s)*100:.0f}%)")
    if blok_klaidingai:
        print(f"        is ju {blok_klaidingai} dienu v3 PRALEIDO, nors "
              f"SMA200 tada nekilo (butent sita kryptis pagrazina rezultata)")


def dienu_anatomija(d, nuo, iki):
    """Dienos bary anatomija per nurodyta tarpa +/- kontekstas."""
    idx = [x.date() for x in d.index]
    d = d.copy()
    d["_ses"] = idx
    pries = d[d["_ses"] < nuo].tail(5)
    lang = d[(d["_ses"] >= nuo) & (d["_ses"] <= iki)]
    po = d[d["_ses"] > iki].head(3)
    if lang.empty:
        print("    DIENOS BARU NERA tam tarpui")
        return None

    print(f"\n    {'data':<12}{'atid':>9}{'max':>9}{'min':>9}{'uzd':>9}"
          f"{'tarpas%':>9}{'diena%':>9}{'uzd.vietoje':>13}{'apyv x':>8}")
    mv = d[d["_ses"] < nuo]["Volume"].tail(20).median()
    med_v = float(mv) if np.isfinite(mv) and mv > 0 else 1.0
    visos = pd.concat([pries, lang, po])
    prev_c = None
    for _, r in visos.iterrows():
        o, h, l, c, v = (float(r["Open"]), float(r["High"]), float(r["Low"]),
                         float(r["Close"]), float(r["Volume"]))
        tarpas = (o / prev_c - 1) * 100 if prev_c else float("nan")
        diena = (c / o - 1) * 100
        vieta = (c - l) / (h - l) * 100 if h > l else float("nan")
        zyme = "  <<<" if nuo <= r["_ses"] <= iki else ""
        print(f"    {str(r['_ses']):<12}{o:>9.2f}{h:>9.2f}{l:>9.2f}{c:>9.2f}"
              f"{tarpas:>9.2f}{diena:>9.2f}{vieta:>12.0f}%{v/max(1,med_v):>8.1f}"
              f"{zyme}")
        prev_c = c

    pask = lang.iloc[-1]
    baze = (float(pries["Close"].iloc[-1]) if len(pries)
            else float(lang["Open"].iloc[0]))
    print(f"\n    RALIO APIMTIS: nuo {baze:.2f} (pries) iki "
          f"{float(lang['High'].max()):.2f} (max) = "
          f"{(float(lang['High'].max())/baze-1)*100:+.2f}%   "
          f"iki uzdarymo {float(pask['Close']):.2f} = "
          f"{(float(pask['Close'])/baze-1)*100:+.2f}%   "
          f"({len(lang)} sesijos)")
    return lang


# ------------------------------------------------- scenarijaus 2 diagnostika

def s2_salygos(langas, kont, iki_uzdarymo):
    """Tos pacios salygos kaip scenarijus_2, bet su etiketemis.

    Pabaigoje tikrinama, ar diagnostika ir tikrasis scenarijus_2 sutaria.
    Jei nesutaria - tai KODO DUBLIAVIMO klaida, ir skriptas apie tai rekia.
    Butent tokio dubliavimo (trys Z-balo apibrezimai) mes ir mokomes issisaugoti.
    """
    atr = kont["atr"]
    dab = langas.iloc[-1]
    kaina = float(dab["Close"])
    baze = langas.iloc[-(D.S2_BAZE_BARU + 1):-1]
    b_max = float(baze["High"].max()) if len(baze) else float("nan")
    b_min = float(baze["Low"].min()) if len(baze) else float("nan")
    pries = langas.iloc[:-1]
    sant = dab.get("apyv_santykis", np.nan)
    vwap = dab.get("vwap", np.nan)
    atid = float(dab["sesijos_atidarymas"])
    vu = kont.get("vakar_uzdarymas") or 0
    baze_pl = (b_max - b_min) / b_max * 100.0 if b_max > 0 else float("nan")
    nuo_atid = (kaina - atid) / atid * 100.0 if atid > 0 else float("nan")

    s = []
    s.append(("baru pakanka", len(langas) >= D.S2_BAZE_BARU + 3,
              f"{len(langas)} >= {D.S2_BAZE_BARU + 3}"))
    s.append(("ne paskutines min", iki_uzdarymo >= D.S2_NERODYTI_PASKUTINES_MIN,
              f"iki uzdarymo {iki_uzdarymo} min"))
    s.append(("baze rami", baze_pl <= D.S2_BAZE_MAX_ATR * atr,
              f"bazes plotis {baze_pl:.2f}% <= {D.S2_BAZE_MAX_ATR*atr:.2f}%"))
    s.append(("virs bazes", kaina > b_max, f"{kaina:.2f} > {b_max:.2f}"))
    s.append(("sesijos maks.", not len(pries) or kaina > float(pries["High"].max()),
              f"{kaina:.2f} > {float(pries['High'].max()) if len(pries) else 0:.2f}"))
    s.append(("apyvartos suolis", bool(np.isfinite(sant) and sant >= D.S2_APYVARTOS_SUOLIS),
              f"x{sant:.2f} >= {D.S2_APYVARTOS_SUOLIS}"))
    s.append(("virs VWAP", bool(np.isfinite(vwap) and kaina > vwap),
              f"{kaina:.2f} > {vwap:.2f}" if np.isfinite(vwap) else "vwap nera"))
    s.append(("auksteja dugnai", D._auksteja_dugnai(langas.iloc[-4:], 2), "3 barai"))
    s.append(("ne per toli nuo atid.", nuo_atid <= D.S2_MAX_NUO_ATIDARYMO_ATR * atr,
              f"{nuo_atid:+.2f}% <= {D.S2_MAX_NUO_ATIDARYMO_ATR*atr:.2f}%"))
    s.append(("virs vakar uzd.", not (vu > 0) or (kaina / vu - 1.0) >= 0,
              f"{kaina:.2f} vs {vu:.2f}"))
    return s


def s2_tikrinti_sutarima(langas, kont, iki):
    tikras = D.scenarijus_2(langas, kont, iki) is not None
    diag = all(ok for _, ok, _ in s2_salygos(langas, kont, iki))
    if tikras != diag:
        print("    !!! DIAGNOSTIKA NESUTINKA SU scenarijus_2 - kodo dubliavimas")
    return tikras


# ---------------------------------------------------------------- sekimas

def sekti_sesija(sd, kont, etikete, ses):
    """Perleidzia visus tris scenarijus per kiekviena bara."""
    print(f"\n    --- {ses}: scenariju sekimas per {len(sd)} baru ---")
    print(f"    kontekstas TOS DIENOS: ATR {kont['atr']:.2f}%  "
          f"SMA200 {'kyla' if kont['sma200_kyla'] else 'NEKYLA'}  "
          f"kritimo dienu {kont['kritimo_dienu']}  "
          f"apyvarta {kont['apyvarta']/1e6:.1f} mln  "
          f"vakar uzd. {kont['vakar_uzdarymas']:.2f}")

    suveike = {}
    pirmas_proverzis = None
    for i in range(6, len(sd)):
        langas = sd.iloc[:i + 1]
        iki = UZDARYMAS_MIN - int(langas["minute"].iloc[-1])
        laikas = f"{int(langas['minute'].iloc[-1])//60:02d}:" \
                 f"{int(langas['minute'].iloc[-1])%60:02d}"
        kaina = float(langas["Close"].iloc[-1])

        # pirmas naujos sesijos maksimumas - cia ralis vizualiai prasideda
        if pirmas_proverzis is None and i >= D.S2_BAZE_BARU + 3:
            pries = langas.iloc[:-1]
            if kaina > float(pries["High"].max()):
                pirmas_proverzis = (i, laikas)

        for f, args, nr in ((D.scenarijus_1, (langas, kont), 1),
                            (D.scenarijus_2, (langas, kont, iki), 2),
                            (D.scenarijus_3, (langas, kont,
                                              kont["vakar_uzdarymas"]), 3)):
            if nr in suveike:
                continue
            try:
                s = f(*args)
            except Exception:
                s = None
            if not s:
                continue
            kl = D.kietieji_filtrai(kont, s, "neutral", None, None)
            suveike[nr] = dict(i=i, laikas=laikas, sig=s, kliutys=kl)
            print(f"    SUVEIKE {nr} ({s['scenarijus']}) {laikas} bare {i}: "
                  f"kaina {s['ieina']:.2f} tikslas "
                  f"{s['tikslas'] if s['tikslas'] else '-'} stop {s['stop']:.2f} "
                  f"progresas {s['progresas']*100:.0f}%"
                  + (f"   BLOKUOTA: {', '.join(kl)}" if kl else "   PRALEISTA"))

    for nr in (1, 2, 3):
        if nr not in suveike:
            print(f"    scenarijus {nr}: nesuveike ne karto")

    # jei ralis nesuveike - kodel butent proverzio bare?
    if 2 not in suveike and pirmas_proverzis:
        i, laikas = pirmas_proverzis
        langas = sd.iloc[:i + 1]
        iki = UZDARYMAS_MIN - int(langas["minute"].iloc[-1])
        s2_tikrinti_sutarima(langas, kont, iki)
        print(f"\n    KODEL ralis nesuveike? Pirmas naujos sesijos maksimumas "
              f"{laikas} (baras {i}), kaina {float(langas['Close'].iloc[-1]):.2f}:")
        for nm, ok, det in s2_salygos(langas, kont, iki):
            print(f"      {'+' if ok else 'NE'}  {nm:<24} {det}")
    return suveike


# ---------------------------------------------------------------- MFE / baigtis

def mfe_ir_baigtis(visos_sesijos, ses, i, sig):
    """MFE trimis horizontais + dabartinis baigtis()."""
    ieina = sig["ieina"]
    sesijos = list(visos_sesijos.groupby("sesija", sort=True))
    idx = [k for k, _ in sesijos]
    if ses not in idx:
        return
    p = idx.index(ses)
    sd = sesijos[p][1]
    likutis = sd.iloc[i + 1:]

    def mfe_mae(fr):
        if fr.empty:
            return float("nan"), float("nan")
        return ((float(fr["High"].max()) / ieina - 1) * 100,
                (float(fr["Low"].min()) / ieina - 1) * 100)

    horizontai = [("ta pati sesija", likutis)]
    for k in (1, 3):
        papild = [sesijos[p + j][1] for j in range(1, k + 1) if p + j < len(sesijos)]
        horizontai.append((f"+{k} sesijos",
                           pd.concat([likutis] + papild) if papild else likutis))

    print(f"\n    MFE / MAE nuo ieina={ieina:.2f} (pozicija {POZICIJA:.0f} EUR, "
          f"sanaudos {SANAUDOS:.0f} EUR):")
    print(f"      {'horizontas':<16}{'MFE %':>9}{'MFE EUR':>10}"
          f"{'MAE %':>9}{'MAE EUR':>10}")
    for nm, fr in horizontai:
        mfe, mae = mfe_mae(fr)
        print(f"      {nm:<16}{mfe:>9.2f}{mfe/100*POZICIJA-SANAUDOS:>10.0f}"
              f"{mae:>9.2f}{mae/100*POZICIJA-SANAUDOS:>10.0f}")

    rytas = sesijos[p + 1][1].iloc[:1] if p + 1 < len(sesijos) else sd.iloc[:0]
    b = D.baigtis(likutis, rytas, sig)
    eur = b["pelnas_pct"] / 100 * POZICIJA - SANAUDOS
    mfe_ses, _ = mfe_mae(likutis)
    print(f"\n    DABARTINE baigtis(): {b['baigtis']} po {b['minuciu']} min, "
          f"{b['pelnas_pct']:+.2f}% = {eur:+.0f} EUR")
    if np.isfinite(mfe_ses):
        print(f"    ANT STALO LIKO: MFE buvo {mfe_ses:+.2f}%, "
              f"isejau su {b['pelnas_pct']:+.2f}% -> "
              f"{(mfe_ses-b['pelnas_pct'])/100*POZICIJA:+.0f} EUR nepaimta")


# ---------------------------------------------------------------- vienas atvejis

def atvejis(etikete, kandidatai, nuo_s, iki_s):
    nuo = date.fromisoformat(nuo_s)
    iki = date.fromisoformat(iki_s)
    print("\n" + "=" * 78)
    print(f"{etikete}   {nuo_s}" + (f" .. {iki_s}" if iki != nuo else ""))
    print("=" * 78)

    t, dien = rasti_tikeri(kandidatai)
    if t is None:
        print("    NEPAVYKO rasti tikerio - praleidziu")
        return

    lang = dienu_anatomija(dien, nuo, iki)
    if lang is None:
        return
    look_ahead_matas(dien, t)

    kont = kontekstas_sesijai(dien, nuo)
    if kont is None:
        print("    per mazai dienos istorijos kontekstui (reikia 220 d.)")
        return

    dienu_skirtumas = (date.today() - nuo).days
    if dienu_skirtumas > 58:
        print(f"\n    5 MIN DUOMENU NEBERA ({dienu_skirtumas} d. > 60 d. yahoo riba).")
        print("    Dienos anatomija virsuje - tai viskas, ka galima pamatyti.")
        print("    Valandiniai barai yra, bet scenarijai aprasyti 5 min barais")
        print("    (baze = 12 baru = 1 val.), tad juos leisti per 1h barus")
        print("    reikstu matuoti visai kita dalyka. Neleidziu.")
        return

    raw = yf.download(t, period="60d", interval="5m", auto_adjust=False,
                      progress=False, prepost=False)
    b5 = vienas(raw, t)
    if b5 is None or len(b5) < 100:
        print("    5 min baru negauta")
        return
    visos = D.sesijos_rodikliai(b5, "eu")

    for ses, sd in visos.groupby("sesija", sort=True):
        if not (nuo <= ses <= iki) or len(sd) < 12:
            continue
        k = kontekstas_sesijai(dien, ses) or kont
        suveike = sekti_sesija(sd, k, etikete, ses)
        if 2 in suveike:
            mfe_ir_baigtis(visos, ses, suveike[2]["i"], suveike[2]["sig"])
        elif 1 in suveike:
            mfe_ir_baigtis(visos, ses, suveike[1]["i"], suveike[1]["sig"])
        else:
            # net jei nesuveike - kiek diena apskritai dave nuo ketvirtadalio
            i = max(12, len(sd) // 4)
            sig = dict(tipas=2, ieina=float(sd["Close"].iloc[i]),
                       tikslas=None,
                       stop=float(sd["Low"].iloc[max(0, i - 12):i].min()))
            print(f"\n    (scenarijus nesuveike - rodau, ka bendrai dave diena "
                  f"nuo baro {i})")
            mfe_ir_baigtis(visos, ses, i, sig)


if __name__ == "__main__":
    print("ATVEJU TYRIMAS - Manto nurodyti raliai")
    print(f"pozicija {POZICIJA:.0f} EUR, sanaudos {SANAUDOS:.0f} EUR uz cikla "
          f"(luzio taskas {SANAUDOS/POZICIJA*100:.4f}%)")
    for a in ATVEJAI:
        try:
            atvejis(*a)
        except Exception as e:
            import traceback
            print(f"\n    KLAIDA atvejyje {a[0]}: {e}")
            traceback.print_exc()
    print("\n" + "=" * 78)
    print("BAIGTA")
