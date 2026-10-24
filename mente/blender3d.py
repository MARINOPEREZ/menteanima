# -*- coding: utf-8 -*-
"""MenteAnima v2 · render con Blender: arma la configuración, ejecuta Blender sin ventana con
mente/blender_escena.py y convierte los fotogramas PNG con alfa al formato pedido (WebM, MOV, GIF, APNG)."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from mente.comun import RAIZ

GUION = RAIZ / "mente" / "blender_escena.py"


def buscar_blender() -> Path | None:
    """blender.exe: en el PATH, en BLENDER o en Archivos de programa (la versión más nueva)."""
    for cand in (os.environ.get("BLENDER"), shutil.which("blender")):
        if cand and Path(cand).exists():
            return Path(cand)
    raices = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation",
              Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Blender Foundation"]
    encontrados = []
    for r in raices:
        encontrados += glob.glob(str(r / "Blender*" / "blender.exe"))
    return Path(sorted(encontrados)[-1]) if encontrados else None


def renderizar(modelo: str | Path, secuencia: list, salida_cuadros: str | Path, fps: int = 30,
               resolucion: tuple = (1080, 1080), vista: str = "tres-cuartos", orbita: float = 0.0,
               elevacion: float = 7.0, archivos_movimiento: list | None = None, muestras: int = 32,
               progreso=None) -> list:
    """Renderiza y devuelve la lista de PNG (con alfa) en orden."""
    blender = buscar_blender()
    if blender is None:
        raise RuntimeError("No encuentro Blender. Instálalo (winget install BlenderFoundation.Blender) o define "
                           "la variable BLENDER con la ruta a blender.exe.")
    salida_cuadros = Path(salida_cuadros)
    if salida_cuadros.exists():
        shutil.rmtree(salida_cuadros)
    salida_cuadros.mkdir(parents=True)
    cfg = {"modelo": str(Path(modelo).resolve()), "secuencia": list(secuencia), "fps": int(fps),
           "resolucion": [int(resolucion[0]), int(resolucion[1])], "salida": str(salida_cuadros.resolve()),
           "camara": {"vista": vista, "orbita": float(orbita), "elevacion": float(elevacion)},
           "archivos_movimiento": archivos_movimiento or [], "muestras": int(muestras)}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False)
        ruta_cfg = fh.name
    try:
        proc = subprocess.Popen([str(blender), "-b", "--factory-startup", "-P", str(GUION), "--", ruta_cfg],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace")
        registro = []
        for linea in proc.stdout:
            registro.append(linea.rstrip())
            if progreso and (linea.startswith("MENTEANIMA:") or "Fra:" in linea and "Rendering" in linea):
                progreso(linea.strip())
        proc.wait()
    finally:
        os.unlink(ruta_cfg)
    cuadros = sorted(salida_cuadros.glob("*.png"))
    if proc.returncode != 0 or not cuadros:
        cola = "\n".join(l for l in registro[-25:])
        raise RuntimeError(f"Blender no terminó bien (código {proc.returncode}).\n{cola}")
    return cuadros


def fotogramas(cuadros: list):
    """PNG → BGRA uint8, de a uno (para an.exportar)."""
    for c in cuadros:
        img = cv2.imdecode(np.fromfile(str(c), np.uint8), cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        yield img
