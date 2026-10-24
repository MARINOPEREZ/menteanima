# -*- coding: utf-8 -*-
"""MenteAnima v2 · Forma de TRELLIS + dibujos originales.

TRELLIS (Microsoft, MIT) da una malla 3D coherente (visera, capucha, zapatillas con volumen real), pero inventa
detalles (logos, espalda) y su textura es borrosa. Aquí se usa SOLO su forma:
  1. leer_glb       malla, UV y textura del .glb de TRELLIS
  2. alinear        prueba las 4 orientaciones y ajusta escala y posición para que su silueta coincida con la
                    del frente y la del perfil de la hoja (el mismo espacio 3D que las vistas)
  3. desde_trellis  copia el esqueleto, los pesos, las manos y los dedos del modelo propio (vértice más cercano)
                    y pinta con los dibujos originales (proyección con prueba de profundidad); la textura de
                    TRELLIS solo rellena lo que ninguna vista ve.
Con una hoja en pose A (hoja_pose_a: el modelo propio con los brazos abiertos, para que TRELLIS no los pegue al
torso), a_reposo devuelve la malla a la postura del dibujo original: así se usan los dibujos, las manos y los
dedos del modelo propio, y los pesos se calculan con los brazos todavía separados del cuerpo.
"""
from __future__ import annotations

import copy
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np

from mente import modelo3d as m3
from mente.animacion import HUESOS

TIPOS = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
COMPONENTES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def leer_glb(ruta: str | Path) -> tuple:
    """(V (n,3), T (f,3), UV (n,2) o None, textura BGR o None) de la primera malla del .glb."""
    datos = Path(ruta).read_bytes()
    magia, _, _ = struct.unpack("<4sII", datos[:12])
    if magia != b"glTF":
        raise ValueError("No es un archivo .glb")
    largo = struct.unpack("<I", datos[12:16])[0]
    js = json.loads(datos[20:20 + largo])
    p = 20 + largo
    bin_largo = struct.unpack("<I", datos[p:p + 4])[0]
    binario = datos[p + 8:p + 8 + bin_largo]

    def acc(i):
        a = js["accessors"][i]
        vista = js["bufferViews"][a["bufferView"]]
        ini = vista.get("byteOffset", 0) + a.get("byteOffset", 0)
        n = COMPONENTES[a["type"]]
        tipo = TIPOS[a["componentType"]]
        arr = np.frombuffer(binario, tipo, a["count"] * n, ini)
        return arr.reshape(a["count"], n) if n > 1 else arr

    prim = js["meshes"][0]["primitives"][0]
    V = acc(prim["attributes"]["POSITION"]).astype(np.float64)
    T = acc(prim["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in prim else \
        np.arange(len(V)).reshape(-1, 3)                   # malla sin índices: un vértice por esquina
    UV = acc(prim["attributes"]["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in prim["attributes"] else None
    tex = None
    try:
        img = js["images"][js["textures"][js["materials"][prim["material"]]["pbrMetallicRoughness"]
                                         ["baseColorTexture"]["index"]]["source"]]
        vista = js["bufferViews"][img["bufferView"]]
        crudo = binario[vista.get("byteOffset", 0):vista.get("byteOffset", 0) + vista["byteLength"]]
        tex = cv2.imdecode(np.frombuffer(crudo, np.uint8), cv2.IMREAD_COLOR)
    except (KeyError, IndexError, TypeError):
        pass
    # transformación del nodo de la malla (si la hay)
    for nodo in js.get("nodes", []):
        if nodo.get("mesh") == 0 and "matrix" in nodo:
            M = np.array(nodo["matrix"], float).reshape(4, 4).T
            V = V @ M[:3, :3].T + M[:3, 3]
    return V, T, UV, tex


def _silueta(V: np.ndarray, T: np.ndarray, vista, forma: tuple) -> np.ndarray:
    """Silueta de la malla vista desde una vista de la hoja, del tamaño `forma` (alto, ancho) reducido."""
    h, w = vista.alfa.shape
    fy, fx = forma[0] / h, forma[1] / w
    u, v = vista.proyectar(V)
    Q = np.stack([u * fx, v * fy, vista.profundidad(V)], 1)
    ids, _, _ = m3.rasterizar(Q, T, forma[1], forma[0], dos_caras=True)
    return ids >= 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def alinear(V0: np.ndarray, T: np.ndarray, vistas: dict, altura: float) -> tuple:
    """(V en el espacio de las vistas, puntaje). Prueba girar 0/90/180/270° alrededor de Y y ajusta escala y
    desplazamientos para que las siluetas coincidan con el frente y el perfil."""
    fr = vistas["frente"]
    lado = next((vistas[r] for r in ("perfil_der", "perfil_izq") if r in vistas and not vistas[r].espejo), None)
    e = 0.33
    m_fr = cv2.resize((fr.alfa > 127).astype(np.uint8), None, fx=e, fy=e, interpolation=cv2.INTER_AREA) > 0
    m_la = cv2.resize((lado.alfa > 127).astype(np.uint8), None, fx=e, fy=e, interpolation=cv2.INTER_AREA) > 0 \
        if lado is not None else None
    mejor = (-1.0, None)
    for k in range(4):
        a = k * math.pi / 2
        R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        V = V0 @ R.T
        s = altura / max(np.ptp(V[:, 1]), 1e-9)
        V = V * s
        V[:, 1] -= V[:, 1].min()                          # los pies en el suelo
        V[:, 0] -= (V[:, 0].min() + V[:, 0].max()) / 2
        V[:, 2] -= (V[:, 2].min() + V[:, 2].max()) / 2
        # desplazamientos que mejor calzan (búsqueda gruesa y luego fina)
        def puntaje(Vc):
            p = _iou(_silueta(Vc, T, fr, m_fr.shape), m_fr)
            if m_la is not None:
                p = 0.5 * p + 0.5 * _iou(_silueta(Vc, T, lado, m_la.shape), m_la)
            return p
        actual = V.copy()
        base = puntaje(actual)
        for paso in (0.04 * altura, 0.015 * altura, 0.005 * altura):
            mejora = True
            while mejora:
                mejora = False
                for eje in (0, 2):
                    for sgn in (-1, 1):
                        prueba = actual.copy()
                        prueba[:, eje] += sgn * paso
                        p = puntaje(prueba)
                        if p > base + 1e-4:
                            actual, base, mejora = prueba, p, True
        for f in (0.97, 1.03, 0.985, 1.015):                # escala fina (alrededor de los pies)
            prueba = actual.copy()
            c = np.array([prueba[:, 0].mean(), 0.0, prueba[:, 2].mean()])
            prueba = (prueba - c) * f + c
            p = puntaje(prueba)
            if p > base + 1e-4:
                actual, base = prueba, p
        if base > mejor[0]:
            mejor = (base, actual)
    return mejor[1], mejor[0]


def _vecinos(base: np.ndarray, consulta: np.ndarray, k: int = 6) -> tuple:
    """Los k vértices de `base` más cercanos a cada punto de `consulta` (FLANN de OpenCV)."""
    indice = cv2.flann_Index(base.astype(np.float32), {"algorithm": 1, "trees": 4})
    idx, dist2 = indice.knnSearch(consulta.astype(np.float32), k, params={"checks": 64})
    return idx.astype(np.int64), np.sqrt(np.maximum(dist2, 0))


def _pesos_geodesicos(base: m3.Modelo, V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Pesos por distancia A LO LARGO de la superficie de la malla (no por el aire): el costado de la sudadera
    queda lejos del brazo porque para llegar hay que pasar por la axila. Cada hueso parte de los vértices que
    están a su radio (el grosor de esa parte en el modelo propio)."""
    from mente.animacion import _dijkstra
    huesos = m3.huesos_de(base)
    # el .glb duplica los vértices en las costuras de la textura: se sueldan los que están en el mismo lugar
    # para que la superficie sea una sola pieza (si no, son cientos de trocitos sueltos)
    clave = np.round(V / max(1e-6, 1e-4 * base.altura)).astype(np.int64)
    _, unico, soldado = np.unique(clave, axis=0, return_index=True, return_inverse=True)
    soldado = soldado.ravel()
    Vo, V = V, V[unico]
    T = soldado[T]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 2] != T[:, 0])]
    n = len(V)
    aristas = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
    largo = np.linalg.norm(V[aristas[:, 0]] - V[aristas[:, 1]], axis=1)
    vecinos = [[] for _ in range(n)]
    for (i, j), c in zip(aristas.tolist(), largo.tolist()):
        vecinos[i].append((j, c))
        vecinos[j].append((i, c))
    def segmentos(hb):
        segs = [(np.asarray(base.J[a], float), np.asarray(base.J[b], float))
                for a, b in m3.SEG3.get(hb, [(HUESOS[hb][0], HUESOS[hb][1])])]
        if hb.startswith(("brazo_", "muslo_")):        # el hombro y la cadera quedan dentro del cuerpo: sin
            a, b = segs[0]                              # su primer tramo, el pecho no «nace» del brazo
            segs = [(a + 0.3 * (b - a), b)]
        return segs

    propios_huesos = [k for k, hb in enumerate(huesos) if hb in HUESOS]
    DE = np.full((n, len(huesos)), np.inf)                # distancia por el aire a cada hueso
    for k in propios_huesos:
        DE[:, k] = np.min([m3._dist_seg3(V, a, b) for a, b in segmentos(huesos[k])], axis=0)
    cercano = np.argmin(DE, axis=1)
    D = np.full((n, len(huesos)), np.inf)
    for k in propios_huesos:                             # muñeca y dedos: se copian del modelo propio
        de = DE[:, k]
        propios = base.W[:, k] > 0.9
        if propios.any():
            db = np.min([m3._dist_seg3(base.V[propios], a, b) for a, b in segmentos(huesos[k])], axis=0)
            radio = float(np.percentile(db, 60))
        else:
            radio = float(np.percentile(de, 2))
        # cada hueso nace de los vértices a su radio que no están más cerca de otro hueso
        ini = np.where((de < 1.25 * radio) & (cercano == k), de, np.inf)
        if not np.isfinite(ini).any():
            cerca = np.argsort(de)[:20]
            ini[cerca] = de[cerca]
        D[:, k] = _dijkstra(vecinos, ini)
    banda = 0.04 * base.altura           # más ancha reparte el estiramiento de la axila, pero deforma el pecho
    with np.errstate(invalid="ignore"):
        W = np.clip(1 - (D - D.min(axis=1, keepdims=True)) / banda, 0, 1) ** 2
    W[~np.isfinite(W)] = 0
    sin = W.sum(1) == 0                                  # piezas sueltas: el hueso más cercano por el aire
    if sin.any():
        W[sin] = 0
        W[np.flatnonzero(sin), cercano[sin]] = 1
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    return W[soldado]                                    # de vuelta a los vértices originales (con su textura)


def _pesos(base: m3.Modelo, V: np.ndarray, T: np.ndarray) -> tuple:
    """(W, es_mano): pesos por la superficie; en las manos, copiados del modelo propio (muñeca y falanges)."""
    idx, dist = _vecinos(base.V, V, 6)
    w = 1.0 / np.maximum(dist, 1e-3)
    w /= w.sum(1, keepdims=True)
    W_copia = np.einsum("nk,nkb->nb", w, base.W[idx])
    W_copia /= np.maximum(W_copia.sum(1, keepdims=True), 1e-9)
    W = _pesos_geodesicos(base, V, T)
    # «mano» solo si el vértice está pegado a una mano del modelo propio Y, por la superficie, es del antebrazo
    # (en pose A los guantes cuelgan junto a la cadera: por cercanía, trozos del pantalón se volverían mano)
    antebrazos = [i for i, h in enumerate(m3.huesos_de(base)) if h.startswith("antebrazo_")]
    es_mano = np.zeros(len(V), bool)
    if base.es_mano is not None:
        es_mano = (base.es_mano[idx[:, 0]] & (dist[:, 0] < 0.03 * base.altura) &
                   np.isin(np.argmax(W, axis=1), antebrazos))
    W[es_mano] = W_copia[es_mano]
    return W, es_mano


def vistas_de_malla(V: np.ndarray, T: np.ndarray, altura: float) -> dict:
    """Siluetas de frente y de perfil de una malla, como si fueran las vistas de una hoja (para alinear)."""
    from mente.vistas import Vista
    ancho, alto = int(1.3 * altura), int(1.15 * altura)
    vistas = {}
    for rol in ("frente", "perfil_der"):
        v = Vista(rol, np.zeros((alto, ancho, 4), np.uint8), ancho / 2, 1.08 * altura)
        v.bgra[..., 3] = _silueta(V, T, v, (alto, ancho)).astype(np.uint8) * 255
        vistas[rol] = v
    return vistas


def a_reposo(base: m3.Modelo, V0: np.ndarray, T: np.ndarray, pose: dict) -> tuple:
    """(V, W, es_mano, puntaje): la malla de TRELLIS hecha desde una hoja dibujada con el modelo propio en
    `pose`, devuelta a la postura del dibujo original. Se alinea con el modelo propio en esa pose, los pesos se
    calculan ahí (brazos separados del cuerpo: la axila no se pega al brazo) y se deshace la pose de cada vértice.
    Los vértices no cambian de orden: los pesos sirven igual para la malla devuelta."""
    P = m3.posar3d(base, pose)
    V, iou = alinear(V0, T, vistas_de_malla(P.V, base.T, base.altura), base.altura)
    posado = copy.copy(base)
    posado.V, posado.J = P.V, P.J
    W, es_mano = _pesos(posado, V, T)
    M = m3.matrices3d(base, pose)
    Ms = np.stack([P.G @ M[hb] for hb in m3.huesos_de(base)])
    B = np.einsum("nb,bij->nij", W, Ms)                       # matriz de cada vértice: de reposo a la pose
    Vh = np.hstack([V, np.ones((len(V), 1))])[..., None]
    return np.linalg.solve(B, Vh)[:, :3, 0], W, es_mano, iou


def desde_trellis(base: m3.Modelo, ruta_glb: str | Path, progreso=None, suavizado: int = 0,
                  pose_hoja: dict | None = None) -> m3.Modelo:
    """Modelo con la forma de TRELLIS y todo lo demás (esqueleto, pesos, dedos, vistas) del modelo propio.
    `pose_hoja`: la pose con la que se dibujó la hoja que recibió TRELLIS (POSE_A) si no es la del dibujo."""
    def avisar(t):
        if progreso:
            progreso(t)
    avisar("Leyendo el modelo de TRELLIS…")
    V0, T, UV, tex = leer_glb(ruta_glb)
    W = None
    if pose_hoja is not None:
        avisar("Devolviéndolo de la pose de la hoja a la del dibujo original…")
        V0, W, es_mano, iou0 = a_reposo(base, V0, T, pose_hoja)
        avisar(f"  (coincidencia con el modelo propio en esa pose: {iou0:.0%})")
    avisar("Alineándolo con las vistas de la hoja…")
    V, iou = alinear(V0, T, base.vistas, base.altura)
    avisar(f"Alineado (coincidencia de siluetas {iou:.0%}).")
    if W is None:
        avisar("Calculando los pesos sobre la malla…")
        W, es_mano = _pesos(base, V, T)
    # suavizado opcional (0 = ninguno)
    if suavizado:
        aa, bb = T[:, [0, 1, 2]].ravel(), T[:, [1, 2, 0]].ravel()
        vec_a, vec_b = np.concatenate([aa, bb]), np.concatenate([bb, aa])
        grado = np.maximum(np.bincount(vec_a, minlength=len(V)), 1)[:, None]
        for _ in range(int(suavizado)):
            prom = np.zeros_like(W)
            np.add.at(prom, vec_a, W[vec_b])
            W = 0.5 * W + 0.5 * prom / grado
        W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    huesos = m3.huesos_de(base)
    grupo_hueso = np.array([m3.GRUPOS3D.index(HUESOS[h][4]) if h in HUESOS else
                            m3.GRUPOS3D.index("brazo_" + h.split("_")[1]) for h in huesos])
    grupo = grupo_hueso[np.argmax(W, axis=1)]
    m = copy.copy(base)
    m.V, m.T = V, T
    m.N = m3.normales(V, T)
    m.W, m.grupo = W, grupo
    m.texturas, m.profundidad = {}, {}
    m.es_mano = es_mano if base.es_mano is not None else None
    m.pies = {}
    for lado in ("izq", "der"):
        sel = np.flatnonzero(grupo == m3.GRUPOS3D.index("pierna_" + lado))
        if len(sel):
            m.pies[lado] = sel[np.argsort(V[sel, 1])[:12]]
    # color propio de TRELLIS por vértice: solo para lo que ninguna vista ve
    m.color_vertice = None
    if UV is not None and tex is not None:
        h, ww = tex.shape[:2]
        x = np.clip(UV[:, 0] * (ww - 1), 0, ww - 1).astype(np.float32)
        y = np.clip(UV[:, 1] * (h - 1), 0, h - 1).astype(np.float32)
        m.color_vertice = m3._muestrear(tex.astype(np.float32), x, y)[:, :3]
    m.origen = "trellis"
    avisar("Preparando las texturas desde los dibujos…")
    m3.preparar_texturas(m)
    return m


# --------------------------------------------------------------------------------------------
# Pedir el modelo a TRELLIS (demo pública en Hugging Face) y hoja en pose A
# --------------------------------------------------------------------------------------------
ESPACIO = "trellis-community/TRELLIS"
# en los ejes del dibujo («dibujo»): la misma pose al dibujar la hoja y al devolver la malla a reposo
POSE_A = {"brazo_izq": (0, 38, 0), "brazo_der": (0, 38, 0), "antebrazo_izq": (6, -6, 0), "antebrazo_der": (6, -6, 0),
          "muslo_izq": (0, 7, 0), "muslo_der": (0, 7, 0), "pierna_izq": (0, -3, 0), "pierna_der": (0, -3, 0),
          "apoyo": 1.0, "dibujo": True}


def token_hf() -> str | None:
    """Token de Hugging Face (más cupo de GPU): variable HF_TOKEN o credenciales/hf_token.txt (no va a Git)."""
    import os
    from mente.comun import RAIZ
    t = os.environ.get("HF_TOKEN")
    archivo = RAIZ / "credenciales" / "hf_token.txt"
    if not t and archivo.exists():
        t = archivo.read_text(encoding="utf-8-sig").strip()          # el Bloc de notas puede agregar BOM
    return t or None


def pedir_modelo(vistas_png: list, destino: str | Path, semilla: int = 7, simplificar: float = 0.7,
                 textura: int = 2048, progreso=None) -> tuple:
    """Pide el modelo a la demo pública de TRELLIS. Con varias vistas usa el modo multivista; si se acaba el cupo
    gratuito de GPU, lo intenta solo con el frente. Devuelve (ruta del .glb, «multivista» | «frente»)."""
    import shutil
    import httpx
    from gradio_client import Client, handle_file

    def avisar(t):
        if progreso:
            progreso(t)

    def ruta(x):
        if isinstance(x, str):
            return x
        if isinstance(x, (list, tuple)):
            return ruta(x[0])
        if isinstance(x, dict):
            return ruta(x["image"]) if "image" in x else x.get("path")

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tok = token_hf()
    intentos = [("multivista", [Path(p) for p in vistas_png])] if len(vistas_png) > 1 else []
    intentos.append(("frente", [Path(vistas_png[0])]))
    ultimo = None
    for modo, imgs in intentos:
        try:
            c = Client(ESPACIO, verbose=False, httpx_kwargs={"timeout": httpx.Timeout(900.0)},
                       **({"token": tok} if tok else {}))
            c.predict(api_name="/start_session")
            if modo == "multivista":
                c.predict(api_name="/lambda_1")
                prep = c.predict([{"image": handle_file(str(p)), "caption": None} for p in imgs],
                                 api_name="/preprocess_images")
                galeria = [{"image": handle_file(ruta(p)), "caption": None} for p in prep]
                principal = handle_file(ruta(prep[0]))
            else:
                c.predict(api_name="/lambda")
                principal = handle_file(ruta(c.predict(handle_file(str(imgs[0])), api_name="/preprocess_image")))
                galeria = []
            avisar(f"TRELLIS ({modo}) generando…")
            res = c.predict(principal, galeria, semilla, 7.5, 12, 3.0, 12, "stochastic", simplificar, textura,
                            api_name="/generate_and_extract_glb")
            glb = res[2] if isinstance(res[2], str) else res[1]
            shutil.copy(glb, destino)
            return destino, modo
        except Exception as e:                      # cupo agotado, cola llena, sin conexión…
            ultimo = e
            avisar(f"TRELLIS ({modo}) no pudo: {str(e)[:160]}")
    raise RuntimeError(f"TRELLIS no respondió: {ultimo}")


def hoja_pose_a(m: m3.Modelo, alto_px: int = 900) -> np.ndarray:
    """Hoja de personaje en pose A (frente, perfil, espalda, perfil) dibujada desde el modelo 3D: brazos
    separados del cuerpo para que TRELLIS y los esqueletos automáticos no los peguen al torso."""
    from mente import render_gl
    gl = render_gl.renderizador(m)
    escala = alto_px / m.altura
    W, H = int(0.85 * alto_px), int(1.12 * alto_px)
    celdas = []
    for az in (0, 90, 180, 270):
        cam = m3.Camara(azimut=az, elevacion=0, fov=0, objetivo=(0, m.altura / 2, 0))
        if gl is not None:
            r = gl.render(POSE_A, cam, escala, (-W / 2, -H / 2), (W, H), luz=0.0)
        else:
            r = m3.renderizar3d(m, m3.posar3d(m, POSE_A), cam, escala, (-W / 2, -H / 2), (W, H), ss=2, luz=0.0)
        ys, xs = np.nonzero(r[..., 3] > 10)
        celdas.append(r[:, max(0, xs.min() - 30):xs.max() + 30])
    hueco = np.zeros((H, 60, 4), np.uint8)
    return np.hstack(sum([[c, hueco] for c in celdas], [])[:-1])
