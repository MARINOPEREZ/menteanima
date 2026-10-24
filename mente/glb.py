# -*- coding: utf-8 -*-
"""Exporta el modelo 3D de MenteAnima a glTF binario (.glb): malla con textura (atlas horneado desde las vistas),
esqueleto con pesos y una animación por movimiento. Se abre en Blender, en visores web (three.js,
<model-viewer>), en motores de juego y en editores que aceptan glTF."""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np

from mente import manos3d
from mente import modelo3d as m3
from mente.animacion import HUESOS, ORDEN

FLOTANTE, BYTE_SIN_SIGNO, ENTERO_SIN_SIGNO = 5126, 5121, 5125


def _quat(R: np.ndarray) -> list:
    """Matriz de rotación 3x3 → cuaternión [x, y, z, w]."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return list(q / np.linalg.norm(q))


def hornear_atlas(m: m3.Modelo, celda: int = 7, lado_max: int = 4096) -> tuple:
    """Atlas de textura: cada par de triángulos ocupa una celda; cada texel toma el color de las vistas como en
    el render. Devuelve (imagen BGR, UV por esquina (f, 3, 2))."""
    F = len(m.T)
    celdas = (F + 1) // 2
    por_fila = math.ceil(math.sqrt(celdas))
    while por_fila * celda > lado_max and celda > 4:
        celda -= 1
    lado = 1 << math.ceil(math.log2(max(64, por_fila * celda)))
    por_fila = lado // celda
    e, g = 0.9, 1.2                                   # margen al borde de la celda y separación en la diagonal
    lg = celda - 2 * e - g                            # largo de los catetos
    iy, ix = np.divmod(np.arange(lado * lado), lado)
    cx, cy = ix // celda, iy // celda
    cel = cy * por_fila + cx
    a, b = (ix % celda) + 0.5, (iy % celda) + 0.5
    arriba = a + b > celda
    f = 2 * cel + arriba
    ok = f < F
    l1 = np.where(arriba, (celda - e - a) / lg, (a - e) / lg)
    l2 = np.where(arriba, (celda - e - b) / lg, (b - e) / lg)
    lam = np.stack([1 - l1 - l2, l1, l2], 1)
    lam = np.clip(lam, 0, None)
    lam /= np.maximum(lam.sum(1, keepdims=True), 1e-9)
    img = np.zeros((lado * lado, 3), np.float32)
    sel = np.flatnonzero(ok)
    for ini in range(0, len(sel), 1_000_000):
        s = sel[ini:ini + 1_000_000]
        tv = m.T[f[s]]
        P = m3._interp(lam[s], tv, m.V)
        N = m3._interp(lam[s], tv, m.N)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        m.grupo_punto = m.grupo[tv[:, 0]] == m3.GRUPOS3D.index("cabeza")
        m.mano_punto = m.es_mano[tv[:, 0]] if m.es_mano is not None else None
        m.respaldo_punto = m3._interp(lam[s], tv, m.color_vertice) if m.color_vertice is not None else None
        img[s] = m3.colorear(m, P, N)[:, :3]
        m.grupo_punto = m.mano_punto = m.respaldo_punto = None
    img = np.clip(img, 0, 255).astype(np.uint8).reshape(lado, lado, 3)
    # UV de las esquinas de cada triángulo
    fi = np.arange(F)
    c = fi // 2
    ox, oy = (c % por_fila) * celda, (c // por_fila) * celda
    abajo = np.array([[e, e], [celda - e - g, e], [e, celda - e - g]])
    sobre = np.array([[celda - e, celda - e], [e + g, celda - e], [celda - e, e + g]])
    esq = np.where((fi % 2 == 1)[:, None, None], sobre[None], abajo[None])
    uv = (esq + np.stack([ox, oy], 1)[:, None, :]) / lado
    return img, uv.astype(np.float32)


class _Bufer:
    def __init__(self):
        self.datos = bytearray()
        self.vistas, self.accesores = [], []

    def agregar(self, arr: np.ndarray, tipo: str, comp: int, objetivo: int | None = None, minmax: bool = False,
                normalizado: bool = False) -> int:
        while len(self.datos) % 4:
            self.datos.append(0)
        crudo = np.ascontiguousarray(arr).tobytes()
        vista = {"buffer": 0, "byteOffset": len(self.datos), "byteLength": len(crudo)}
        if objetivo:
            vista["target"] = objetivo
        self.datos += crudo
        self.vistas.append(vista)
        acc = {"bufferView": len(self.vistas) - 1, "componentType": comp, "count": int(arr.shape[0]), "type": tipo}
        if normalizado:
            acc["normalized"] = True
        if minmax:
            acc["min"] = [float(x) for x in np.atleast_1d(arr.min(0))]
            acc["max"] = [float(x) for x in np.atleast_1d(arr.max(0))]
        self.accesores.append(acc)
        return len(self.accesores) - 1

    def imagen(self, png: bytes) -> int:
        while len(self.datos) % 4:
            self.datos.append(0)
        self.vistas.append({"buffer": 0, "byteOffset": len(self.datos), "byteLength": len(png)})
        self.datos += png
        return len(self.vistas) - 1


def exportar_glb(m: m3.Modelo, ruta: str | Path, animaciones: dict | None = None, fps: float = 30,
                 progreso=None) -> Path:
    """animaciones: {nombre: lista de diccionarios de pose} (por defecto, todos los movimientos 3D)."""
    from mente import movimientos3d as mv3
    ruta = Path(ruta)
    if animaciones is None:
        animaciones = {}
        for nombre in mv3.MOVIMIENTOS:
            poses, _ = mv3.poses_de(m, mv3.preparar_pasos([nombre]), fps)
            animaciones[nombre] = poses
    escala = 1.0 / m.altura                           # en metros: el personaje mide 1 m
    if progreso:
        progreso("Horneando la textura…")
    atlas, uv = hornear_atlas(m)
    ok, png = cv2.imencode(".png", atlas)
    b = _Bufer()
    F = len(m.T)
    pos = (m.V[m.T].reshape(-1, 3) * escala).astype(np.float32)
    nor = m.N[m.T].reshape(-1, 3).astype(np.float32)
    tex = uv.reshape(-1, 2)
    W = m.W[m.T].reshape(-1, m.W.shape[1])
    orden_w = np.argsort(-W, axis=1)[:, :4]
    pesos = np.take_along_axis(W, orden_w, 1)
    pesos /= np.maximum(pesos.sum(1, keepdims=True), 1e-9)
    huesos = m3.huesos_de(m)                         # columnas de W = articulaciones de la piel, en orden
    juntas = orden_w.astype(np.uint8)
    pesos[pesos < 1e-4] = 0
    a_pos = b.agregar(pos, "VEC3", FLOTANTE, 34962, minmax=True)
    a_nor = b.agregar(nor, "VEC3", FLOTANTE, 34962)
    a_uv = b.agregar(tex, "VEC2", FLOTANTE, 34962)
    a_j = b.agregar(juntas, "VEC4", BYTE_SIN_SIGNO, 34962)
    a_w = b.agregar(pesos.astype(np.float32), "VEC4", FLOTANTE, 34962)
    # esqueleto: nodo 0 = raíz (mueve todo el cuerpo), luego un nodo por hueso
    padre_de = m.padre or {hb: HUESOS[hb][2] for hb in ORDEN}
    pivote = m.pivote or {hb: np.asarray(m.J[HUESOS[hb][0]], float) for hb in ORDEN}
    nodos = [{"name": "raiz", "children": []}]
    for hb in huesos:
        padre = padre_de[hb]
        pp = pivote[padre] if padre else np.zeros(3)
        nodos.append({"name": hb, "translation": [float(x) for x in (pivote[hb] - pp) * escala], "children": []})
    for i, hb in enumerate(huesos):
        padre = padre_de[hb]
        (nodos[huesos.index(padre) + 1] if padre else nodos[0])["children"].append(i + 1)
    for n in nodos:
        if not n["children"]:
            del n["children"]
    ibm = np.stack([np.eye(4) for _ in huesos])
    for i, hb in enumerate(huesos):
        ibm[i, :3, 3] = -pivote[hb] * escala
    a_ibm = b.agregar(ibm.transpose(0, 2, 1).reshape(len(huesos), 16).astype(np.float32), "MAT4", FLOTANTE)
    i_img = b.imagen(png.tobytes())
    malla = {"name": "personaje", "primitives": [{
        "attributes": {"POSITION": a_pos, "NORMAL": a_nor, "TEXCOORD_0": a_uv, "JOINTS_0": a_j,
                       "WEIGHTS_0": a_w}, "material": 0}]}
    nodos.append({"name": "malla", "mesh": 0, "skin": 0})
    # animaciones
    anims = []
    for nombre, poses in animaciones.items():
        if progreso:
            progreso(f"Animación «{nombre}»…")
        n = len(poses)
        tiempos = (np.arange(n) / fps).astype(np.float32)
        a_t = b.agregar(tiempos, "SCALAR", FLOTANTE, minmax=True)
        canales, muestreos = [], []
        rot = {hb: [] for hb in huesos}
        raiz_t, raiz_r, raiz_s = [], [], []
        for pose in poses:
            G = m3.transformacion_global(m, pose)
            completa = manos3d.expandir(m, pose)
            for hb in huesos:
                rot[hb].append(_quat(m3.rotacion_local(m, hb, completa.get(hb))))
            L = G[:3, :3]
            s = np.linalg.norm(L, axis=0)
            raiz_t.append(G[:3, 3] * escala)
            raiz_r.append(_quat(L / s))
            raiz_s.append(s)
        def canal(nodo, camino, valores, tipo):
            muestreos.append({"input": a_t, "output": b.agregar(np.asarray(valores, np.float32), tipo, FLOTANTE),
                              "interpolation": "LINEAR"})
            canales.append({"sampler": len(muestreos) - 1, "target": {"node": nodo, "path": camino}})
        canal(0, "translation", raiz_t, "VEC3")
        canal(0, "rotation", raiz_r, "VEC4")
        canal(0, "scale", raiz_s, "VEC3")
        for i, hb in enumerate(huesos):
            canal(i + 1, "rotation", rot[hb], "VEC4")
        anims.append({"name": nombre, "channels": canales, "samplers": muestreos})
    gltf = {
        "asset": {"version": "2.0", "generator": "MenteAnima (MenteXYZ Studio)"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0, "scenes": [{"nodes": [0, len(nodos) - 1]}],
        "nodes": nodos, "meshes": [malla],
        "skins": [{"joints": list(range(1, len(huesos) + 1)), "skeleton": 0, "inverseBindMatrices": a_ibm}],
        "materials": [{"name": "dibujo", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                                                   "metallicFactor": 0.0, "roughnessFactor": 1.0},
                       "extensions": {"KHR_materials_unlit": {}}}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],   # sin mipmaps: no mezclan celdas
        "images": [{"bufferView": i_img, "mimeType": "image/png"}],
        "animations": anims,
        "buffers": [{"byteLength": 0}],
        "bufferViews": b.vistas, "accessors": b.accesores,
    }
    while len(b.datos) % 4:
        b.datos.append(0)
    gltf["buffers"][0]["byteLength"] = len(b.datos)
    js = json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(b.datos)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<I4s", len(js), b"JSON"))
        fh.write(js)
        fh.write(struct.pack("<I4s", len(b.datos), b"BIN\x00"))
        fh.write(bytes(b.datos))
    return ruta
