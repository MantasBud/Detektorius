#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIKLAS - detektorius kas 5 min vienoje GitHub pamainoje  -  2026-09-29
=====================================================================

Kodel: GitHub planuoklis "*/5" realiai paleisdavo kas 10-20 min (serveriu
apkrova), todel korteles signalus pamatydavo veliau arba visai nepamatydavo
(SOI 2026-09-28). Cia viena pamaina (iki ~5 val.) pati suka rata kas 5 min.

Kiekvienas ratas:
  1. git pull            - naujausias kodas, busena ir zurnalas is repo;
                           jei diena metu ikelsi nauja detektorius.py,
                           kitas ratas jau naudos ji.
  2. savitikra           - jei nepraeina, sis ratas praleidziamas (sugedes
                           kodas puslapio nesugadina), pamaina pabaigoje
                           pazymima raudonai.
  3. detektorius --live  - ATSKIRAS procesas: kas rata nauji duomenys is
                           yahoo, is ankstesnio rato atmintyje nelieka nieko.
  4. commit + push       - kaip anksciau darydavo workflow.

Ratai vyksta kas 5 min + 60 s (kad yahoo spetu pateikti ka tik uzsidariusi
bara). Pamaina baigiasi ties savo riba (UTC 11:30, 16:15, 21:05) arba po
MAX_MIN minuciu - kas pirmiau; tada ja perima kita pamaina, kuri jau laukia
eileje (concurrency).

Paleidimas:
    python ciklas.py --savitikra
    python ciklas.py
"""

import argparse
import hashlib
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

RIBOS_UTC = [(11, 30), (16, 15), (21, 5)]   # pamainu pabaigos
MAX_MIN = 320                                # < 360 min GitHub darbo ribos
ZINGSNIS_S = 5 * 60
POSLINKIS_S = 60
LIVE_TIMEOUT_S = 240

FAILAI_COMMITUI = ["docs", "busena.json", "dividendai.json", "ataskaitos.json",
                   "kursas.json"]


def pabaiga(pradzia):
    """Pirma pamainos riba bent 60 min po pradzios; ne ilgiau MAX_MIN.

    60 min, o ne maziau: jei pamaina paleista laiku, bet pries ja niekas
    nevyko (ankstesne nukrito), ji neturi baigtis po keliu minuciu ir palikti
    likusios dienos be detektoriaus. Arti dienos pabaigos - iki paskutines ribos.
    """
    kand = [pradzia.replace(hour=h, minute=m, second=0, microsecond=0)
            for h, m in RIBOS_UTC]
    tinka = ([k for k in kand if k > pradzia + timedelta(minutes=60)]
             or [k for k in kand if k > pradzia])
    riba = tinka[0] if tinka else pradzia          # po paskutines ribos - nieko
    return min(riba, pradzia + timedelta(minutes=MAX_MIN))


def kitas_laikas(dabar):
    """Artimiausias 5 min zymuo + POSLINKIS, grieztai velesnis uz dabar."""
    epoch = dabar.timestamp()
    k = (epoch - POSLINKIS_S) // ZINGSNIS_S * ZINGSNIS_S + ZINGSNIS_S + POSLINKIS_S
    return datetime.fromtimestamp(k, tz=timezone.utc)


def sh(cmd, timeout=None):
    """Grazina (kodas, isvestis). Niekada nemeta isimties."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT po {timeout} s: {cmd}"
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def irasyti(zinute="Detektorius", bandymu=3):
    """commit + push; jei kas nors ikele tarp ratu - pull --rebase ir vel."""
    for f in FAILAI_COMMITUI:
        sh(f"[ -e {f} ] && git add -A {f}")
    k, _ = sh("git diff --cached --quiet")
    if k == 0:
        return True, "nera pakeitimu"
    k, out = sh(f'git commit -q -m "{zinute}"')
    if k != 0:
        return False, out
    for _ in range(bandymu):
        k, out = sh("git push -q")
        if k == 0:
            return True, "irasyta"
        k2, out2 = sh("git pull --rebase -q")
        if k2 != 0:
            # Konfliktas: kas nors pakeite ta pati faila, kuri raso ciklas.
            # Niekada nepaliekam repo pusiau sujungto - nutraukiam; kitas
            # ratas prasides tiksliai nuo nutolusio repo ir viska perskaiciuos.
            sh("git rebase --abort")
            return False, "konfliktas su nutolusiu repo - sio rato rezultatas atmestas"
    return False, out


def sinchronizuoti():
    """Darbo aplankas = TIKSLIAI nutolusio repo busena.

    Nesekmingo rato likuciai (pusiau irasyti failai, neissiustas commit,
    nutrauktas sujungimas) niekada neblokuoja kito rato: nutolusis repo
    visada laimi, o ciklo isvestys perskaiciuojamos is nauju duomenu.
    Jei nutolusio repo pasiekti nepavyksta - tesiama su turimu.
    """
    sh("git rebase --abort")
    sh("git merge --abort")
    k, out = sh("git fetch -q")
    if k == 0:
        k, out = sh("git reset -q --hard @{u}")
    if k != 0:
        sh("git reset -q --hard HEAD")
    return k == 0, out


def ratas(cmd_savitikra, cmd_live, zurnalas=print):
    """Vienas ratas. Grazina True, jei viskas pavyko."""
    ok_s, out = sinchronizuoti()
    if not ok_s:
        zurnalas(f"  nutolusio repo pasiekti nepavyko (tesiama su turimu): {out.strip()[-300:]}")
    kodas = hashlib.sha1(open("detektorius.py", "rb").read()).hexdigest()[:8] \
        if _yra("detektorius.py") else "nera"
    k, out = sh(cmd_savitikra, timeout=120)
    if k != 0:
        zurnalas(f"  SAVITIKRA NEPRAEJO (kodas {kodas}) - ratas praleistas:\n"
                 + out[-1500:])
        return False
    k, out = sh(cmd_live, timeout=LIVE_TIMEOUT_S)
    zurnalas(out.rstrip()[-2500:])
    if k != 0:
        zurnalas(f"  LIVE KLAIDA (kodas {k}) - puslapis siame rate neatnaujintas")
        return False
    ok, info = irasyti()
    zurnalas(f"  kodas {kodas}; {info}")
    return ok


def _yra(f):
    try:
        open(f, "rb").close()
        return True
    except OSError:
        return False


def paleisti():
    pradzia = datetime.now(timezone.utc)
    pab = pabaiga(pradzia)
    print(f"Pamaina: pradzia {pradzia:%H:%M:%S} UTC, pabaiga {pab:%H:%M} UTC")
    if pab <= pradzia:
        print("Po paskutines pamainos ribos - nieko nedarau.")
        return 0
    ratu, nesekmiu = 0, 0
    while datetime.now(timezone.utc) < pab:
        ratu += 1
        t0 = datetime.now(timezone.utc)
        print(f"\n===== ratas {ratu}  {t0:%H:%M:%S} UTC =====", flush=True)
        if not ratas("python detektorius.py --savitikra",
                     "python detektorius.py --live --rinka abi",
                     lambda s: print(s, flush=True)):
            nesekmiu += 1
        trukme = (datetime.now(timezone.utc) - t0).total_seconds()
        print(f"  trukme {trukme:.0f} s", flush=True)
        kitas = kitas_laikas(datetime.now(timezone.utc))
        if kitas >= pab:
            break
        time.sleep(max(0.0, (kitas - datetime.now(timezone.utc)).total_seconds()))
    print(f"\nPamaina baigta: {ratu} ratu, nesekmingu {nesekmiu}.")
    return 1 if nesekmiu else 0


# ============================================================ savitikra

def savitikra():
    import os
    import tempfile
    ok = True

    def tikrinti(s, a, b):
        nonlocal ok
        g = (a == b)
        print(f"  {'OK ' if g else 'BLOGAI'}  {s}{'' if g else f'  (gauta {a}, laukta {b})'}")
        ok = ok and g

    U = timezone.utc
    d = lambda h, m, s=0: datetime(2026, 9, 29, h, m, s, tzinfo=U)
    tikrinti("pamaina 06:45 -> iki 11:30", pabaiga(d(6, 45)), d(11, 30))
    tikrinti("pamaina 11:15 -> iki 16:15", pabaiga(d(11, 15)), d(16, 15))
    tikrinti("pamaina 16:00 -> iki 21:05", pabaiga(d(16, 0)), d(21, 5))
    tikrinti("velavusi pamaina 11:20 (<60 min iki ribos) -> iki 16:15",
             pabaiga(d(11, 20)), d(16, 15))
    tikrinti("pamaina laiku 15:58, pries ja nieko -> iki 21:05, ne 16:15",
             pabaiga(d(15, 58)), d(21, 5))
    tikrinti("labai velavusi 10:40 -> ne ilgiau MAX_MIN (16:00)",
             pabaiga(d(10, 40)), d(16, 0))
    tikrinti("rankinis 20:10 -> iki 21:05", pabaiga(d(20, 10)), d(21, 5))
    tikrinti("ankstyva pamaina 02:00 -> ne ilgiau MAX_MIN",
             pabaiga(d(2, 0)), d(2, 0) + timedelta(minutes=MAX_MIN))
    tikrinti("po 21:05 -> nieko", pabaiga(d(21, 10)), d(21, 10))
    tikrinti("jokia pamaina netrunka ilgiau MAX_MIN (visos minutes paroje)",
             all((pabaiga(d(0, 0) + timedelta(minutes=i)) - (d(0, 0) + timedelta(minutes=i)))
                 <= timedelta(minutes=MAX_MIN) for i in range(24 * 60)), True)
    tikrinti("kitas laikas po 10:07:30 -> 10:11:00", kitas_laikas(d(10, 7, 30)), d(10, 11))
    tikrinti("kitas laikas tiksliai 10:11:00 -> 10:16:00 (grieztai veliau)",
             kitas_laikas(d(10, 11)), d(10, 16))
    tikrinti("kitas laikas 10:10:59 -> 10:11:00", kitas_laikas(d(10, 10, 59)), d(10, 11))

    # git: ratas su tikru repo ir "nutolusiu" (bare) repo
    senas = os.getcwd()
    tmp = tempfile.mkdtemp()
    try:
        sh(f"git init -q --bare {tmp}/nut.git")
        sh(f"git clone -q {tmp}/nut.git {tmp}/a")
        os.chdir(f"{tmp}/a")
        sh('git config user.email t@t && git config user.name t')
        open("detektorius.py", "w").write("# v1\n")
        sh("mkdir -p docs && echo 0 > docs/x && git add -A && git commit -qm init "
           "&& git push -q origin HEAD:main && git branch -u origin/main 2>/dev/null; "
           "git push -q -u origin HEAD")
        # kitas zmogus ikelia nauja koda
        sh(f"git clone -q {tmp}/nut.git {tmp}/b")
        os.chdir(f"{tmp}/b")
        sh('git config user.email m@m && git config user.name m')
        open("detektorius.py", "w").write("# v2\n")
        sh("git commit -qam v2 && git push -q")
        os.chdir(f"{tmp}/a")
        zin = []
        r = ratas("true", "echo 1 >> docs/x; cat detektorius.py > docs/kodas", zin.append)
        tikrinti("ratas: live PALEISTAS jau su nauju kodu (pull pries paleidima)",
                 open("docs/kodas").read(), "# v2\n")
        tikrinti("ratas: rezultatas nusiunciamas i nutolusi repo", r, True)
        k, out = sh(f"git --git-dir={tmp}/nut.git log -1 --format=%s")
        tikrinti("ratas: paskutinis commit - Detektorius", out.strip(), "Detektorius")
        # tarp pull ir push kitas ikelia - push'as turi pavykti po rebase
        os.chdir(f"{tmp}/b")
        sh("git pull -q && echo '# v3' > detektorius.py && git commit -qam v3 && git push -q")
        os.chdir(f"{tmp}/a")
        sh("echo 2 >> docs/x")
        ok_i, _ = irasyti()
        tikrinti("irasyti: konfliktas su nauju ikelimu issprendziamas rebase",
                 ok_i, True)
        # KONFLIKTAS: kitas pakeicia ta pati faila, kuri raso ciklas
        os.chdir(f"{tmp}/b")
        sh("git pull -q && echo RANKA > docs/x && git commit -qam ranka && git push -q")
        os.chdir(f"{tmp}/a")
        sh("echo CIKLAS >> docs/x")
        ok_k, info_k = irasyti()
        k_r, _ = sh("git status | grep -qi rebase")
        tikrinti("konfliktas: rato rezultatas atmestas, repo NElieka pusiau sujungtas",
                 (ok_k, k_r != 0), (False, True))
        r = ratas("true", "echo PO >> docs/x; cat detektorius.py > docs/kodas", zin.append)
        k, out = sh(f"git --git-dir={tmp}/nut.git show HEAD:docs/x")
        tikrinti("po konflikto kitas ratas prasideda nuo nutolusio repo ir issiuncia",
                 (r, out.strip()), (True, "RANKA\nPO"))

        # savitikra nepraeina -> live nepaleidziamas
        zin = []
        r = ratas("false", "echo SUGADINTA > docs/x", zin.append)
        tikrinti("ratas: nepraejus savitikrai live NEpaleidziamas",
                 (r, "SUGADINTA" in open("docs/x").read()), (False, False))
        # live klaida -> nieko neirasoma
        k0, h0 = sh(f"git --git-dir={tmp}/nut.git rev-parse HEAD")
        r = ratas("true", "echo 9 >> docs/x; exit 3", zin.append)
        k1, h1 = sh(f"git --git-dir={tmp}/nut.git rev-parse HEAD")
        tikrinti("ratas: live klaida -> False ir nieko nenusiusta", (r, h0 == h1), (False, True))
        # po nesekmingo rato (nesvarus aplankas) kitas ratas vis tiek gauna nauja koda
        os.chdir(f"{tmp}/b")
        sh("git pull -q && echo '# v4' > detektorius.py && git commit -qam v4 && git push -q")
        os.chdir(f"{tmp}/a")
        r = ratas("true", "echo 5 >> docs/x; cat detektorius.py > docs/kodas", zin.append)
        tikrinti("ratas: po nesekmingo rato live paleistas su nauju kodu",
                 (r, open("docs/kodas").read()), (True, "# v4\n"))
        # live per ilgas -> nutraukiamas
        global LIVE_TIMEOUT_S
        sen_t = LIVE_TIMEOUT_S
        LIVE_TIMEOUT_S = 1
        try:
            r = ratas("true", "sleep 5", zin.append)
        finally:
            LIVE_TIMEOUT_S = sen_t
        tikrinti("ratas: uzstriges live nutraukiamas pagal laika", r, False)
    finally:
        os.chdir(senas)
        sh(f"rm -rf {tmp}")

    print("-" * 60)
    print("SAVITIKRA: VISKAS GERAI" if ok else "SAVITIKRA: YRA KLAIDU")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--savitikra", action="store_true")
    a = ap.parse_args()
    sys.exit(savitikra() if a.savitikra else paleisti())
