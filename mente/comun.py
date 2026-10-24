# -*- coding: utf-8 -*-
"""Lo poco que MenteAnima comparte: la carpeta del programa y los mensajes por consola."""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(msg, flush=True)


def fallar(msg: str) -> None:
    print("\nERROR: " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def consola_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
