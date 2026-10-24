# -*- coding: utf-8 -*-
"""MenteAnima v2 · Cuaderno de Google Colab que convierte la hoja de personaje en un modelo 3D con TRELLIS
(Microsoft, licencia MIT): malla con UNA textura coherente (sin ojos repetidos), lista para el esqueleto de
Mixamo y el render en Blender. Tu portátil no tiene GPU NVIDIA; Colab presta una T4 gratis."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from mente.vistas import ROLES

ORDEN_TRELLIS = ("frente", "perfil_der", "espalda", "perfil_izq")


def exportar_vistas(vistas: dict, carpeta: str | Path, lado: int = 1024) -> list:
    """Cada vista real como PNG cuadrado, transparente y centrado (la entrada de TRELLIS). Las vistas inventadas
    en espejo no se exportan: TRELLIS completa lo que falta mejor que un reflejo."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = []
    for i, rol in enumerate(ORDEN_TRELLIS):
        v = vistas.get(rol)
        if v is None or v.espejo:
            continue
        a = v.alfa > 20
        ys, xs = np.nonzero(a)
        rec = v.bgra[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = rec.shape[:2]
        s = 0.86 * lado / max(h, w)
        rec = cv2.resize(rec, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        lienzo = np.zeros((lado, lado, 4), np.uint8)
        y0, x0 = (lado - rec.shape[0]) // 2, (lado - rec.shape[1]) // 2
        lienzo[y0:y0 + rec.shape[0], x0:x0 + rec.shape[1]] = rec
        ruta = carpeta / f"{i + 1}_{rol}.png"
        ok, buf = cv2.imencode(".png", lienzo)
        buf.tofile(str(ruta))
        rutas.append(ruta)
    return rutas


def _md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n").splitlines(keepends=True)}


def _codigo(texto: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": texto.strip("\n").splitlines(keepends=True)}


def cuaderno(destino: str | Path, nombre: str = "mascota") -> Path:
    celdas = [
        _md(f"""
# MenteAnima · Modelo 3D de «{nombre}» con TRELLIS (Google Colab, GPU gratis)
Convierte las vistas de la hoja de personaje (frente, perfiles, espalda) en **un modelo 3D con una sola
textura** (`.glb`). Después: esqueleto con dedos en Mixamo y render en Blender con `menteanima.py`.

**Antes de empezar**
1. *Entorno de ejecución → Cambiar tipo de entorno → **GPU T4***.
2. Ten a mano los PNG que exportó `python menteanima.py vistas <hoja> --exportar …` (1_frente.png, 2_perfil_der.png…).
3. Ejecuta las celdas **en orden**. La celda 2 tarda ~15–25 min la primera vez (compila partes en CUDA).
   Si Colab pide **reiniciar la sesión** después de la celda 2, acepta y sigue con la celda 3.

**Privacidad:** las imágenes solo viven en tu sesión de Colab (se borran al cerrarla).
**Licencia:** TRELLIS es MIT (uso comercial libre). El modelo que sale es tuyo.
"""),
        _codigo("""
#@title 1 · Verificar la GPU y Python
!nvidia-smi -L || echo "SIN GPU: Entorno de ejecución > Cambiar tipo de entorno > GPU T4"
import sys; print("Python", sys.version.split()[0])
"""),
        _codigo("""
#@title 2 · Instalar TRELLIS (PyTorch 2.4 + CUDA 12.1, como lo prueba el proyecto)
%%bash
set -e
cd /content
pip -q install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
[ -d TRELLIS ] || git clone -q --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS
# T4 no admite flash-attn: se usa xformers. Una opción a la vez: si una falla, se ve cuál.
for opcion in --basic --xformers --spconv --kaolin --nvdiffrast --diffoctreerast --mipgaussian; do
  echo "==== $opcion"
  . ./setup.sh $opcion || echo "!!!! Falló $opcion"
done
python -c "import kaolin" 2>/dev/null || pip -q install git+https://github.com/NVIDIAGameWorks/kaolin.git
echo "LISTO. Si Colab pide reiniciar la sesión, acepta y sigue con la celda 3."
"""),
        _codigo("""
#@title 3 · Subir las vistas (PNG transparentes; el orden lo da el nombre: 1_frente, 2_perfil_der, …)
import os, sys
os.chdir("/content/TRELLIS"); sys.path.insert(0, "/content/TRELLIS")
from google.colab import files
from PIL import Image
subidas = files.upload()
nombres = sorted(subidas)
imagenes = [Image.open(n).convert("RGBA") for n in nombres]
print("Vistas:", nombres)
"""),
        _codigo("""
#@title 4 · Generar el modelo 3D
SEMILLA = 7  #@param {type:"integer"}
PASOS = 20  #@param {type:"integer"}
TEXTURA = 2048  #@param [1024, 2048]
SIMPLIFICAR = 0.9  #@param {type:"number"}
import os
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"
import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils
import imageio, numpy as np
pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipe.cuda()
if len(imagenes) > 1:
    salida = pipe.run_multi_image(imagenes, seed=SEMILLA,
                                  sparse_structure_sampler_params={"steps": PASOS, "cfg_strength": 7.5},
                                  slat_sampler_params={"steps": PASOS, "cfg_strength": 3})
else:
    salida = pipe.run(imagenes[0], seed=SEMILLA)
glb = postprocessing_utils.to_glb(salida["gaussian"][0], salida["mesh"][0], simplify=SIMPLIFICAR,
                                  texture_size=TEXTURA)
glb.export("/content/modelo_trellis.glb")
video = render_utils.render_video(salida["gaussian"][0])["color"]
imageio.mimsave("/content/vista_previa.mp4", video, fps=30)
print("Listo: /content/modelo_trellis.glb")
"""),
        _codigo("""
#@title 5 · Ver la vista previa (giro de 360°)
from IPython.display import Video
Video("/content/vista_previa.mp4", embed=True, width=420)
"""),
        _codigo("""
#@title 6 · Descargar el modelo
from google.colab import files
files.download("/content/modelo_trellis.glb")
"""),
        _md("""
**Siguiente paso en tu portátil**

    python menteanima.py importar modelo_trellis.glb

¿No convence? Cambia la **SEMILLA** en la celda 4 y vuelve a ejecutar solo esa celda (y la 5–6): cada semilla da
una variante. Con la hoja en **pose A** (brazos separados del cuerpo, manos abiertas) el resultado y el esqueleto
de Mixamo salen mucho mejor.
"""),
    ]
    nb = {"cells": celdas, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                        "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                        "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return destino
