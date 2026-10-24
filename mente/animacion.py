# -*- coding: utf-8 -*-
"""MenteAnima — de una imagen a un personaje que se mueve, con fondo transparente. Local y gratis (OpenCV).

Etapas:
  1. recorte     Separa la figura del fondo: usa la transparencia del PNG si la trae; si no, estima el fondo
                 desde los bordes (aunque sea un degradado), lo refina con GrabCut y suaviza el borde con un
                 filtro guiado por la imagen.
  2. esqueleto   «Exoesqueleto» de 15 articulaciones estimado de la silueta (pies, entrepierna, cuello, manos);
                 se corrige a mano en la ventana y se guarda en <imagen>.esqueleto.json.
  3. malla       Triangula la silueta (Delaunay) y calcula los pesos de la piel con distancias GEODÉSICAS (por
                 dentro de la figura, y más largas al cruzar un borde de la imagen): la pierna izquierda no
                 arrastra a la derecha aunque se toquen. Donde dos partes se tocan lejos de su articulación
                 (brazo pegado al cuerpo, piernas juntas) la malla se «corta» como un títere de papel.
  4. pose        Cinemática directa 2D: cada hueso gira en su articulación; los pies se apoyan en el suelo y, al
                 desplazarse, el pie de apoyo no patina.
  5. render      Rasteriza la malla deformada (índice de triángulo por píxel + remap) en capas (piernas detrás
                 del torso, cabeza encima, brazos delante) con supermuestreo para bordes suaves.

Los movimientos están en mente/movimientos.py; la ventana en mente/animador_ventana.py; el programa en
menteanima.py.
"""
from __future__ import annotations

import heapq
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------------------------
# Esqueleto: articulaciones y huesos («izq»/«der» = lado de la imagen, como la ves)
# --------------------------------------------------------------------------------------------
ARTICULACIONES = ["cabeza", "cuello", "hombro_izq", "codo_izq", "mano_izq", "hombro_der", "codo_der", "mano_der",
                  "cadera", "cadera_izq", "rodilla_izq", "pie_izq", "cadera_der", "rodilla_der", "pie_der"]

NOMBRES = {"cabeza": "cabeza", "cuello": "cuello", "hombro_izq": "hombro izq.", "codo_izq": "codo izq.",
           "mano_izq": "mano izq.", "hombro_der": "hombro der.", "codo_der": "codo der.", "mano_der": "mano der.",
           "cadera": "pelvis", "cadera_izq": "cadera izq.", "rodilla_izq": "rodilla izq.", "pie_izq": "pie izq.",
           "cadera_der": "cadera der.", "rodilla_der": "rodilla der.", "pie_der": "pie der."}

# hueso: (articulación donde gira, articulación final, hueso padre, capa de dibujo, grupo)
HUESOS = {
    "pelvis":        ("cadera", "cadera", None, 2.0, "nucleo"),
    "torso":         ("cadera", "cuello", None, 2.1, "nucleo"),
    "cabeza":        ("cuello", "cabeza", "torso", 3.0, "cabeza"),
    "brazo_izq":     ("hombro_izq", "codo_izq", "torso", 4.0, "brazo_izq"),
    "antebrazo_izq": ("codo_izq", "mano_izq", "brazo_izq", 4.1, "brazo_izq"),
    "brazo_der":     ("hombro_der", "codo_der", "torso", 5.0, "brazo_der"),
    "antebrazo_der": ("codo_der", "mano_der", "brazo_der", 5.1, "brazo_der"),
    "muslo_izq":     ("cadera_izq", "rodilla_izq", "pelvis", 0.0, "pierna_izq"),
    "pierna_izq":    ("rodilla_izq", "pie_izq", "muslo_izq", 0.1, "pierna_izq"),
    "muslo_der":     ("cadera_der", "rodilla_der", "pelvis", 1.0, "pierna_der"),
    "pierna_der":    ("rodilla_der", "pie_der", "muslo_der", 1.1, "pierna_der"),
}
ORDEN = list(HUESOS)           # los padres van antes que los hijos

# Segmentos que «llevan» la piel de cada hueso (para los pesos)
SEGMENTOS = {
    "pelvis": [("cadera", "cadera_izq"), ("cadera", "cadera_der")],
    "torso": [("cadera", "cuello"), ("cuello", "hombro_izq"), ("cuello", "hombro_der")],
}
for _h, (_a, _b, *_r) in HUESOS.items():
    SEGMENTOS.setdefault(_h, [(_a, _b)])

# Articulación que une dos grupos: cerca de ella la piel se dobla; lejos, se corta.
BISAGRAS = {frozenset(("nucleo", "cabeza")): "cuello",
            frozenset(("nucleo", "brazo_izq")): "hombro_izq", frozenset(("nucleo", "brazo_der")): "hombro_der",
            frozenset(("nucleo", "pierna_izq")): "cadera_izq", frozenset(("nucleo", "pierna_der")): "cadera_der"}

LINEAS = [("cadera", "cuello"), ("cuello", "cabeza"), ("cuello", "hombro_izq"), ("hombro_izq", "codo_izq"),
          ("codo_izq", "mano_izq"), ("cuello", "hombro_der"), ("hombro_der", "codo_der"), ("codo_der", "mano_der"),
          ("cadera", "cadera_izq"), ("cadera_izq", "rodilla_izq"), ("rodilla_izq", "pie_izq"),
          ("cadera", "cadera_der"), ("cadera_der", "rodilla_der"), ("rodilla_der", "pie_der")]


# --------------------------------------------------------------------------------------------
# Imagen
# --------------------------------------------------------------------------------------------
def leer_imagen(ruta: str | Path) -> np.ndarray:
    """BGRA uint8. Acepta PNG, JPG, WEBP… (también rutas con tildes en Windows)."""
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def guardar_png(ruta: str | Path, bgra: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        raise ValueError(f"No se pudo escribir {ruta}")
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(ruta))


def tiene_transparencia(bgra: np.ndarray) -> bool:
    a = bgra[..., 3]
    return bool((a < 250).mean() > 0.02 and (a > 5).mean() > 0.02)


# --------------------------------------------------------------------------------------------
# 1. Recorte de la figura
# --------------------------------------------------------------------------------------------
def _componentes_principales(m: np.ndarray, minimo: float = 0.12, solo_junto: bool = True) -> np.ndarray:
    """Se queda con la figura más grande (y las piezas grandes que la acompañan) y tapa huecos diminutos.
    Con solo_junto=False conserva todas las piezas grandes aunque estén lejos."""
    n, et, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m.astype(bool)
    areas = st[1:, cv2.CC_STAT_AREA]
    g = int(np.argmax(areas))
    gx, gy, gw, gh = st[1 + g, :4]
    margen = 0.1 * gh
    # piezas grandes junto a la figura principal (no otras figuras de la misma imagen)
    junto = [i for i in range(len(areas)) if areas[i] >= minimo * areas[g] and (not solo_junto or (
             st[1 + i, 0] < gx + gw + margen and st[1 + i, 0] + st[1 + i, 2] > gx - margen and
             st[1 + i, 1] < gy + gh + margen and st[1 + i, 1] + st[1 + i, 3] > gy - margen))]
    sal = np.isin(et, 1 + np.array(junto))
    # huecos: componentes del fondo que no tocan el borde y son pequeños → figura
    fondo = (~sal).astype(np.uint8)
    n2, et2, st2, _ = cv2.connectedComponentsWithStats(fondo, connectivity=4)
    h, w = m.shape
    for i in range(1, n2):
        x, y, bw, bh, area = st2[i]
        toca = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not toca and area < 0.0015 * sal.sum():
            sal[et2 == i] = True
    return sal


def _modelo_de_fondo(img: np.ndarray, banda: int) -> np.ndarray:
    """Fondo suave (sirve para degradados) estimado solo con los bordes de la imagen."""
    h, w = img.shape[:2]
    lado = 40
    s = lado / max(h, w)
    sw, sh = max(8, round(w * s)), max(8, round(h * s))
    chico = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    b = max(1, round(banda * s))
    desconocido = np.full((sh, sw), 255, np.uint8)
    desconocido[b:-b, b:-b] = 0
    desconocido = 255 - desconocido
    # los píxeles del borde que se salen mucho de su lado (la figura toca el borde) también se rellenan
    borde = desconocido == 0
    med = np.median(chico[borde].reshape(-1, 3), axis=0)
    dist = np.linalg.norm(chico.astype(np.float32) - med, axis=2)
    mad = np.median(dist[borde]) + 1e-3
    desconocido[borde & (dist > max(40.0, 4 * mad))] = 255
    relleno = cv2.inpaint(chico, desconocido, 3, cv2.INPAINT_TELEA)
    relleno = cv2.GaussianBlur(relleno, (0, 0), 1.0)
    return cv2.resize(relleno, (w, h), interpolation=cv2.INTER_CUBIC)


def recortar(bgra: np.ndarray, sensibilidad: float = 1.0, todas: bool = False) -> np.ndarray:
    """Alfa (uint8) de la figura. Si la imagen ya trae transparencia, la respeta. Con todas=True conserva
    todas las figuras (una hoja de personaje con varias vistas)."""
    if tiene_transparencia(bgra):
        a = bgra[..., 3].copy()
        m = _componentes_principales(a > 127, solo_junto=not todas)
        return np.where(cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0, a, 0).astype(np.uint8)
    img = bgra[..., :3]
    h, w = img.shape[:2]
    s = min(1.0, 720 / max(h, w))
    peq = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    ph, pw = peq.shape[:2]
    banda = max(2, round(0.012 * max(ph, pw)))
    # 1) dónde está la figura: lo que se aparta del fondo estimado desde los bordes
    fondo = _modelo_de_fondo(peq, banda)
    lab = cv2.cvtColor(peq, cv2.COLOR_BGR2LAB).astype(np.float32)
    labf = cv2.cvtColor(fondo, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = cv2.GaussianBlur(np.linalg.norm(lab - labf, axis=2), (0, 0), 1.0)
    borde = np.ones((ph, pw), bool)
    borde[banda:-banda, banda:-banda] = False
    t = max(12.0, 2.5 * float(np.percentile(d[borde], 90))) / sensibilidad
    burda = _componentes_principales(cv2.morphologyEx((d > t).astype(np.uint8), cv2.MORPH_OPEN,
                                                      np.ones((5, 5), np.uint8)) > 0, 0.12, solo_junto=False)
    if burda.sum() < 0.002 * ph * pw:
        burda = d > t
    yy, xx = np.nonzero(burda)
    if len(xx) == 0:
        raise ValueError("No se distingue ninguna figura del fondo. Prueba con un PNG con fondo transparente.")
    mx, my = round(0.04 * pw), round(0.04 * ph)
    x0, y0 = max(1, xx.min() - mx), max(1, yy.min() - my)
    x1, y1 = min(pw - 2, xx.max() + mx), min(ph - 2, yy.max() + my)
    # 2) GrabCut dentro de ese rectángulo: aprende los colores de la figura y del fondo (sirve para ropa
    #    oscura sobre fondo oscuro, donde la simple diferencia de color no alcanza)
    mascara = np.zeros((ph, pw), np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(peq, mascara, (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), bg, fg, 6,
                    cv2.GC_INIT_WITH_RECT)
        m = (mascara == cv2.GC_FGD) | (mascara == cv2.GC_PR_FGD)
    except cv2.error:
        m = burda
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    m = _componentes_principales(m, solo_junto=not todas)
    # a tamaño completo, con borde suave que sigue los bordes de la imagen
    mf = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    r = max(2, round(2 / s))
    try:
        mf = cv2.ximgproc.guidedFilter(img, mf, r, (0.03 * 255) ** 2)
    except Exception:
        mf = cv2.GaussianBlur(mf, (0, 0), r / 2)
    a = np.clip((mf - 0.5) * 2.2 + 0.5, 0, 1)
    a[a < 0.04] = 0
    return (a * 255).astype(np.uint8)


# --------------------------------------------------------------------------------------------
# 2. Esqueleto automático
# --------------------------------------------------------------------------------------------
def _cerca_dentro(m: np.ndarray, p) -> tuple:
    """El punto de la figura más cercano a p."""
    x, y = int(round(p[0])), int(round(p[1]))
    h, w = m.shape
    if 0 <= x < w and 0 <= y < h and m[y, x]:
        return float(p[0]), float(p[1])
    ys, xs = np.nonzero(m)
    i = np.argmin((xs - p[0]) ** 2 + (ys - p[1]) ** 2)
    return float(xs[i]), float(ys[i])


def _tramos(fila: np.ndarray) -> list:
    """Tramos [a, b] de píxeles de figura en una fila."""
    d = np.diff(np.concatenate([[0], fila.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def detectar_esqueleto(alfa: np.ndarray) -> dict:
    """Estimación inicial para una figura de frente y de pie (se corrige a mano en la ventana)."""
    m0 = alfa > 127
    ys, xs = np.nonzero(m0)
    if len(xs) < 50:
        raise ValueError("No se encontró ninguna figura en la imagen.")
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    H = y1 - y0 + 1
    # se trabaja con la figura a ~400 px de alto
    s = min(1.0, 400 / H)
    m = cv2.resize(m0.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST) > 0 if s < 1 else m0
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    H = y1 - y0 + 1
    filas = {y: _tramos(m[y]) for y in range(y0, y1 + 1)}
    cx = float(np.median(xs[ys > y0 + 0.5 * H])) if (ys > y0 + 0.5 * H).any() else float(xs.mean())

    # pies: lo más bajo a cada lado del centro
    banda = (ys > y1 - 0.07 * H)
    izq, der = banda & (xs < cx), banda & (xs >= cx)
    if izq.sum() < 3 or der.sum() < 3:
        izq, der = banda & (xs < np.median(xs[banda])), banda & (xs >= np.median(xs[banda]))
    pie_i = (float(xs[izq].mean()), float(y1 - 0.045 * H))
    pie_d = (float(xs[der].mean()), float(y1 - 0.045 * H))
    medio = (pie_i[0] + pie_d[0]) / 2

    # entrepierna: subiendo desde los pies, hasta donde hay hueco entre las piernas
    yc = None
    for y in range(int(y1 - 0.03 * H), int(y0 + 0.3 * H), -1):
        tr = filas.get(y, [])
        hueco = any(a <= medio <= b for a, b in tr) is False and \
            any(b < medio for a, b in tr) and any(a > medio for a, b in tr)
        if hueco:
            yc = y
        elif yc is not None and y < yc - 2:
            break
    if yc is None or yc > y1 - 0.12 * H:
        yc = int(y0 + 0.62 * H)
    # centro de cada pierna un poco bajo la entrepierna
    yk = min(int(yc + 0.06 * H), int(y1))
    tr = filas.get(yk, [])
    ti = [((a + b) / 2) for a, b in tr if (a + b) / 2 < medio]
    td = [((a + b) / 2) for a, b in tr if (a + b) / 2 >= medio]
    lx = ti[-1] if ti else (pie_i[0] + medio) / 2
    rx = td[0] if td else (pie_d[0] + medio) / 2
    if len(ti) == 0 or len(td) == 0:      # piernas sin hueco: a un cuarto del ancho de la fila
        ancho = [(a, b) for a, b in tr if a <= medio <= b]
        if ancho:
            a, b = ancho[0]
            lx, rx = a + 0.27 * (b - a), b - 0.27 * (b - a)
    y_cad = yc - 0.02 * H
    cad_i, cad_d = (float(lx), float(y_cad)), (float(rx), float(y_cad))
    pelvis = ((lx + rx) / 2, y_cad - 0.03 * H)

    # cuello: bajando desde la cabeza, donde la figura se angosta (antes de que se ensanchen los hombros)
    anchos = {y: (tr[-1][1] - tr[0][0] + 1 if tr else 0) for y, tr in filas.items()}
    yn, maximo = None, 0
    for y in range(int(y0), int(y0 + 0.5 * H)):
        maximo = max(maximo, anchos.get(y, 0))
        if y > y0 + 0.06 * H and anchos.get(y, 0) < 0.8 * maximo:
            ventana = [v for v in range(y, int(y + 0.1 * H)) if anchos.get(v, 0)]
            yn = min(ventana, key=lambda v: anchos[v]) if ventana else y
            break
    if yn is None:                     # sin cuello visible: lo más angosto bajo lo más ancho de la cabeza
        sup = [y for y in range(int(y0 + 0.03 * H), int(y0 + 0.4 * H)) if y in anchos]
        y_cab = max(sup, key=lambda y: anchos[y]) if sup else int(y0 + 0.15 * H)
        cand = [y for y in range(y_cab + int(0.03 * H), int(min(yc - 0.12 * H, y0 + 0.6 * H)))
                if anchos.get(y, 0)]
        yn = min(cand, key=lambda y: anchos[y] + 0.15 * abs(y - (y_cab + 0.12 * H))) if cand else int(y0 + 0.3 * H)
    trn = filas.get(yn, [])
    cxn = ((trn[0][0] + trn[-1][1]) / 2) if trn else cx
    cuello = (float(cxn), float(yn))
    sobre = m[y0:yn]
    yy, xx = np.nonzero(sobre)
    cabeza = (float(xx.mean()) if len(xx) else cxn, float(y0 + (yy.mean() if len(yy) else 0.15 * H)))

    # manos: los extremos laterales entre el cuello y un poco bajo la entrepierna
    lo, hi = int(yn + 0.05 * H), int(min(y1 - 0.1 * H, yc + 0.3 * H))
    sel = (ys >= lo) & (ys <= hi)
    if sel.sum() < 10:
        sel = ys > yn
    xi, yi = xs[sel], ys[sel]
    k_i, k_d = np.argmin(xi), np.argmax(xi)
    r = 0.05 * H
    def centro(px, py):
        cerca = (xs - px) ** 2 + (ys - py) ** 2 < r * r
        return float(xs[cerca].mean()), float(ys[cerca].mean())
    mano_i = centro(xi[k_i], yi[k_i])
    mano_d = centro(xi[k_d], yi[k_d])

    # hombros: bajo el cuello, hacia dentro del borde
    ys_h = int(yn + 0.06 * H)
    trh = filas.get(ys_h, []) or trn
    borde_i, borde_d = (trh[0][0], trh[-1][1]) if trh else (cxn - 0.15 * H, cxn + 0.15 * H)
    grosor = min(0.07 * H, 0.22 * (borde_d - borde_i))
    hom_i = (float(borde_i + 0.6 * grosor), float(ys_h))
    hom_d = (float(borde_d - 0.6 * grosor), float(ys_h))
    if mano_i[0] > hom_i[0]:      # mano metida hacia dentro: el hombro manda
        mano_i = (min(mano_i[0], hom_i[0]), mano_i[1])
    if mano_d[0] < hom_d[0]:
        mano_d = (max(mano_d[0], hom_d[0]), mano_d[1])

    def medio_dentro(a, b, f=0.5):
        return _cerca_dentro(m, (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
    codo_i, codo_d = medio_dentro(hom_i, mano_i), medio_dentro(hom_d, mano_d)
    rod_i, rod_d = medio_dentro(cad_i, pie_i, 0.48), medio_dentro(cad_d, pie_d, 0.48)

    J = {"cabeza": cabeza, "cuello": cuello, "hombro_izq": hom_i, "codo_izq": codo_i, "mano_izq": mano_i,
         "hombro_der": hom_d, "codo_der": codo_d, "mano_der": mano_d, "cadera": pelvis,
         "cadera_izq": cad_i, "rodilla_izq": rod_i, "pie_izq": pie_i,
         "cadera_der": cad_d, "rodilla_der": rod_d, "pie_der": pie_d}
    return {k: (round(v[0] / s, 1), round(v[1] / s, 1)) for k, v in J.items()}


def guardar_esqueleto(ruta: str | Path, J: dict, tamano: tuple) -> None:
    """Guarda el esqueleto junto con el tamaño (ancho, alto) de la imagen en que se midió."""
    datos = {"imagen": [int(tamano[0]), int(tamano[1])],
             "articulaciones": {k: [round(float(J[k][0]), 1), round(float(J[k][1]), 1)] for k in ARTICULACIONES}}
    Path(ruta).write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")


def cargar_esqueleto(ruta: str | Path, tamano: tuple | None = None) -> dict:
    d = json.loads(Path(ruta).read_text(encoding="utf-8"))
    art = d.get("articulaciones", d)
    falta = [k for k in ARTICULACIONES if k not in art]
    if falta:
        raise ValueError(f"Al esqueleto le faltan articulaciones: {', '.join(falta)}")
    sx = sy = 1.0
    if tamano and d.get("imagen"):
        sx, sy = tamano[0] / d["imagen"][0], tamano[1] / d["imagen"][1]
    return {k: (float(art[k][0]) * sx, float(art[k][1]) * sy) for k in ARTICULACIONES}


def dibujar_esqueleto(bgra: np.ndarray, J: dict, alfa: np.ndarray | None = None) -> np.ndarray:
    """Vista de control: la figura sobre cuadros grises con el esqueleto encima (BGR)."""
    h, w = bgra.shape[:2]
    a = (alfa if alfa is not None else bgra[..., 3]).astype(np.float32)[..., None] / 255
    lado = max(8, w // 40)
    iy, ix = np.indices((h, w))
    cuadros = (((iy // lado) + (ix // lado)) % 2).astype(np.float32)[..., None] * 40 + 180
    out = (bgra[..., :3] * a + cuadros * (1 - a)).astype(np.uint8)
    g = max(2, round(max(h, w) / 300))
    for a_, b_ in LINEAS:
        p, q = (round(J[a_][0]), round(J[a_][1])), (round(J[b_][0]), round(J[b_][1]))
        cv2.line(out, p, q, (255, 255, 255), g + 2, cv2.LINE_AA)
        color = (60, 200, 255) if a_.endswith("izq") or b_.endswith("izq") else (255, 140, 60)
        cv2.line(out, p, q, color, g, cv2.LINE_AA)
    for k in ARTICULACIONES:
        p = (round(J[k][0]), round(J[k][1]))
        cv2.circle(out, p, 2 * g + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, p, 2 * g, (40, 40, 220), -1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------------------------
# 3. Malla, pesos y capas
# --------------------------------------------------------------------------------------------
GRUPOS = {"pierna_izq": 0, "pierna_der": 1, "nucleo": 2, "cabeza": 3, "brazo_izq": 4, "brazo_der": 5}


def _muestrear_contorno(c: np.ndarray, paso: float) -> list:
    c = c.reshape(-1, 2).astype(np.float64)
    if len(c) < 3:
        return []
    seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    acum = np.concatenate([[0], np.cumsum(seg)])
    n = max(3, int(acum[-1] / paso))
    i = np.searchsorted(acum, np.linspace(0, acum[-1], n, endpoint=False), side="right") - 1
    return [tuple(q) for q in np.round(c[i % len(c)]).astype(int)]


def _dist_segmento(P: np.ndarray, a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _dijkstra(vecinos: list, inicial: np.ndarray) -> np.ndarray:
    dist = inicial.copy()
    cola = [(d, i) for i, d in enumerate(dist) if np.isfinite(d)]
    heapq.heapify(cola)
    while cola:
        d, i = heapq.heappop(cola)
        if d > dist[i]:
            continue
        for j, w in vecinos[i]:
            nd = d + w
            if nd < dist[j]:
                dist[j] = nd
                heapq.heappush(cola, (nd, j))
    return dist


def _pesos(D: np.ndarray, altura: float) -> np.ndarray:
    """Pesos suaves solo cerca de los límites entre huesos: el resto se mueve rígido y el dibujo no se deforma."""
    banda = 0.035 * altura
    with np.errstate(invalid="ignore"):
        W = np.clip(1 - (D - D.min(axis=1, keepdims=True)) / banda, 0, 1) ** 2
    W[~np.isfinite(W)] = 0
    s = W.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return W / s


def _dentro_poligono(P: np.ndarray, poli: list) -> np.ndarray:
    c = np.array(poli, np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0 for x, y in P])


@dataclass
class Capa:
    """Una parte del cuerpo (tronco, cabeza, un brazo, una pierna) con su propia textura recortada."""
    nombre: str
    orden: float
    tris: np.ndarray                 # (k, 3) índices de los vértices
    sub: np.ndarray                  # orden de dibujo de cada triángulo dentro de la capa
    W: np.ndarray                    # (n, huesos) pesos de los vértices para esta capa
    tex: np.ndarray                  # BGRA float32 premultiplicada, recortada
    off: tuple                       # (x, y) del recorte en la imagen
    _mips: dict = field(default_factory=dict, repr=False)

    def textura_a(self, escala: float) -> tuple:
        """Textura reducida cuando la capa se dibuja mucho más chica (evita el serrucho)."""
        if escala >= 0.7:
            return self.tex, 1.0
        nivel = max(0.05, round(escala * 1.25, 2))
        if nivel not in self._mips:
            h, w = self.tex.shape[:2]
            self._mips[nivel] = cv2.resize(self.tex, (max(1, round(w * nivel)), max(1, round(h * nivel))),
                                           interpolation=cv2.INTER_AREA)
        return self._mips[nivel], nivel


@dataclass
class Personaje:
    """Todo lo necesario para animar: esqueleto en reposo, malla y capas con sus pesos."""
    J: dict                          # articulaciones en reposo (px de la imagen)
    V: np.ndarray                    # vértices (n, 2)
    T: np.ndarray                    # triángulos (f, 3)
    capas: list                      # Capa, de atrás hacia adelante
    altura: float                    # alto de la figura en reposo (px)
    suelo: float                     # y del suelo en reposo
    pies: dict = field(default_factory=dict)     # capa y vértices más bajos de cada pierna
    reposo: dict = field(default_factory=dict)   # ángulo de cada hueso en el dibujo
    neutro: dict = field(default_factory=dict)   # ángulo «de pie» de cada hueso (base de los movimientos)


def _malla(md: np.ndarray, J: dict, paso: float) -> tuple:
    h, w = md.shape
    puntos = set()
    contornos, _ = cv2.findContours(md, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for c in contornos:
        if cv2.contourArea(c) > paso * paso:
            puntos.update(_muestrear_contorno(c, paso))
    dentro = cv2.erode(md, np.ones((3, 3), np.uint8), iterations=max(1, int(paso * 0.6)))
    g = max(2, int(paso * 1.5))
    for y in range(g // 2, h, g):
        for x in range(g // 2 + (g // 2 if (y // g) % 2 else 0), w, g):
            if dentro[y, x]:
                puntos.add((x, y))
    for k in ARTICULACIONES:                         # las articulaciones también son vértices
        x, y = int(round(J[k][0])), int(round(J[k][1]))
        if 0 <= x < w and 0 <= y < h and dentro[y, x]:
            puntos.add((x, y))
    puntos = sorted(puntos)
    indice = {p: i for i, p in enumerate(puntos)}
    sub = cv2.Subdiv2D((-2, -2, w + 4, h + 4))
    sub.insert([(float(x), float(y)) for x, y in puntos])
    tris = []
    for t in sub.getTriangleList():
        p = [(int(round(t[0])), int(round(t[1]))), (int(round(t[2])), int(round(t[3]))),
             (int(round(t[4])), int(round(t[5])))]
        if not all(q in indice for q in p):
            continue
        P = np.array(p, float)
        area = (P[1, 0] - P[0, 0]) * (P[2, 1] - P[0, 1]) - (P[2, 0] - P[0, 0]) * (P[1, 1] - P[0, 1])
        if abs(area) < 0.5:
            continue
        # el triángulo entero debe caer dentro de la figura (no cruzar huecos entre piernas o brazos)
        pr = [P.mean(0)] + [P[a_] + f * (P[b_] - P[a_]) for a_, b_ in ((0, 1), (1, 2), (2, 0))
                            for f in (0.2, 0.4, 0.6, 0.8)]
        if all(md[min(h - 1, int(round(y))), min(w - 1, int(round(x)))] for x, y in pr):
            tris.append([indice[q] for q in p])
    if not tris:
        raise ValueError("No se pudo triangular la figura (¿es demasiado pequeña?).")
    V = np.array(puntos, float)
    T = np.array(tris, int)
    usados = np.unique(T)
    nuevo = -np.ones(len(V), int)
    nuevo[usados] = np.arange(len(usados))
    return V[usados], nuevo[T]


def _rasterizar_reposo(V: np.ndarray, T: np.ndarray, forma: tuple) -> tuple:
    """Triángulo y coordenadas baricéntricas de cada píxel de la figura en reposo."""
    h, w = forma
    ids = np.full((h, w), -1, np.int32)
    pts = np.round(V[T] * 16).astype(np.int32)
    for f in range(len(T)):
        cv2.fillConvexPoly(ids, pts[f], int(f), cv2.LINE_8, 4)
        cv2.polylines(ids, [pts[f]], True, int(f), 1, cv2.LINE_8, 4)
    yy, xx = np.nonzero(ids >= 0)
    f = ids[yy, xx]
    A = np.concatenate([V[T], np.ones((len(T), 3, 1))], axis=2).transpose(0, 2, 1)   # columnas = vértices
    B = np.linalg.pinv(A)                                                              # (f, 3, 3)
    lam = np.einsum("pij,pj->pi", B[f], np.stack([xx, yy, np.ones_like(xx)], 1).astype(float))
    return yy, xx, f, lam


def geodesico(bgra: np.ndarray, alfa: np.ndarray, J: dict, detalle: float = 1.0, huesos: list | None = None,
              poligono: bool = True) -> dict:
    """Malla de la silueta y distancia geodésica (por dentro de la figura) de cada vértice a cada hueso.
    El torso nace de todo el cuadro hombros-caderas (poligono=True; en un perfil no aplica), la cabeza de su
    círculo y cada antebrazo de la mano entera: así los costados del torso no se van con el brazo."""
    huesos = list(huesos or HUESOS)
    m = alfa > 100
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        raise ValueError("La figura está vacía.")
    altura = float(ys.max() - ys.min() + 1)
    paso = max(3.0, altura / (80 * detalle))
    h, w = m.shape
    # la malla cubre todo lo que tenga algo de opacidad (sin dientes en el borde)
    r = max(1, round(altura / 500))
    md = cv2.dilate((alfa > 6).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    md = _componentes_principales(md > 0, 0.02).astype(np.uint8)
    V, T = _malla(md, J, paso)
    n = len(V)

    # grafo de la malla; cruzar un borde fuerte del dibujo «cuesta» más (separa manga y torso)
    gris = cv2.cvtColor(np.ascontiguousarray(bgra[..., :3]), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gris = cv2.GaussianBlur(gris, (0, 0), max(1.0, paso / 4))
    gx, gy = cv2.Sobel(gris, cv2.CV_32F, 1, 0), cv2.Sobel(gris, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx * gx + gy * gy)
    grad /= max(float(np.percentile(grad[m], 95)), 1e-3)
    aristas = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
    medio = ((V[aristas[:, 0]] + V[aristas[:, 1]]) / 2).astype(int)
    costo = np.linalg.norm(V[aristas[:, 0]] - V[aristas[:, 1]], axis=1) * \
        (1 + 2.5 * np.minimum(2.0, grad[medio[:, 1].clip(0, h - 1), medio[:, 0].clip(0, w - 1)]))
    vecinos = [[] for _ in range(n)]
    for (i, j), c in zip(aristas.tolist(), costo.tolist()):
        vecinos[i].append((j, c))
        vecinos[j].append((i, c))

    torso_poli = [J["hombro_izq"], J["hombro_der"], J["cadera_der"], J["cadera_izq"]]
    cab_r = 0.75 * math.dist(J["cabeza"], J["cuello"])
    D = np.zeros((n, len(huesos)))
    for k, hb in enumerate(huesos):
        de = np.min([_dist_segmento(V, J[a_], J[b_]) for a_, b_ in SEGMENTOS[hb]], axis=0)
        if hb == "torso" and poligono:
            de[_dentro_poligono(V, torso_poli)] = 0
        elif hb == "cabeza":
            de = np.minimum(de, np.maximum(0, np.linalg.norm(V - np.asarray(J["cabeza"]), axis=1) - cab_r))
        elif hb.startswith("antebrazo_"):            # la mano entera (un guante grande) va con el antebrazo
            mano = J["mano_" + hb[-3:]]
            r_mano = min(0.045 * altura, 0.6 * math.dist(mano, J["codo_" + hb[-3:]]))
            de = np.minimum(de, np.maximum(0, np.linalg.norm(V - np.asarray(mano), axis=1) - r_mano))
        ini = np.where(de < paso * 1.2, de, np.inf)
        if not np.isfinite(ini).any():
            cerca = np.argsort(de)[:3]
            ini[cerca] = de[cerca]
        D[:, k] = _dijkstra(vecinos, ini)
        sin = ~np.isfinite(D[:, k])                 # piezas sueltas: distancia directa
        D[sin, k] = de[sin] * 3 + altura
    return {"V": V, "T": T, "D": D, "paso": paso, "altura": altura, "huesos": huesos, "m": m}


def etiquetas(bgra: np.ndarray, alfa: np.ndarray, J: dict, huesos: list | None = None,
              poligono: bool = True) -> np.ndarray:
    """A qué hueso pertenece cada píxel de la figura (índice en `huesos`; -1 = fondo)."""
    g = geodesico(bgra, alfa, J, 1.0, huesos, poligono)
    yy, xx, fpx, lam = _rasterizar_reposo(g["V"], g["T"], alfa.shape)
    dpx = np.einsum("pi,pib->pb", lam, g["D"][g["T"][fpx]])
    lab = np.full(alfa.shape, -1, np.int32)
    lab[yy, xx] = np.argmin(dpx, axis=1)
    lab[alfa < 60] = -1
    return lab


def crear_personaje(bgra: np.ndarray, alfa: np.ndarray, J: dict, detalle: float = 1.0) -> Personaje:
    """Triangula la silueta, calcula los pesos y separa el cuerpo en capas que se cortan limpio donde las
    partes solo se tocan (brazo pegado al cuerpo, piernas juntas) y se doblan en las articulaciones."""
    g = geodesico(bgra, alfa, J, detalle)
    V, T, D, paso, altura, huesos, m = g["V"], g["T"], g["D"], g["paso"], g["altura"], g["huesos"], g["m"]
    n = len(V)
    h, w = m.shape
    ys = np.nonzero(m)[0]
    W = _pesos(D, altura)

    # distancia de cada vértice a cada grupo y, por interpolación, de cada píxel
    nombres = list(GRUPOS)
    grupo_h = np.array([HUESOS[hb][4] for hb in huesos])
    Dg = np.stack([D[:, grupo_h == g].min(axis=1) for g in nombres], 1)              # (n, grupos)
    yy, xx, fpx, lam = _rasterizar_reposo(V, T, (h, w))
    dpx = np.einsum("pi,pig->pg", lam, Dg[T[fpx]])                                    # (píxeles, grupos)

    a = alfa.astype(np.float32)[..., None] / 255
    tex = np.dstack([bgra[..., :3].astype(np.float32) * a, a * 255]).astype(np.float32)
    capa_h = np.array([HUESOS[hb][3] for hb in huesos])
    suave, extra = 1.2, 2.5          # px de difuminado del corte; la capa de atrás se extiende bajo la de adelante
    r_in, r_out = 0.045 * altura, 0.095 * altura
    capas = []
    for gi, g in enumerate(nombres):
        # máscara de la parte, con borde suave
        mg = np.ones(len(yy), np.float32)
        for hi_, h_ in enumerate(nombres):
            if h_ == g:
                continue
            ext = extra if GRUPOS[g] < GRUPOS[h_] else 0.0
            mg = np.minimum(mg, np.clip(0.5 - (dpx[:, gi] - dpx[:, hi_] - ext) / (2 * suave), 0, 1))
        if mg.max() <= 0:
            continue
        mascara = np.zeros((h, w), np.float32)
        mascara[yy, xx] = mg
        sel = np.nonzero(mascara > 0)
        y0_, y1_ = sel[0].min(), sel[0].max() + 1
        x0_, x1_ = sel[1].min(), sel[1].max() + 1
        tex_g = tex[y0_:y1_, x0_:x1_] * mascara[y0_:y1_, x0_:x1_, None]
        # triángulos que tocan la parte
        delta = Dg[:, gi] - np.min(np.where(np.arange(len(nombres)) == gi, np.inf, Dg), axis=1)
        tris_g = T[(delta[T] < 1.5 * paso + extra).any(axis=1)]
        # pesos de la capa: cerca de su articulación, los compartidos (la piel se dobla); lejos, solo los
        # huesos de esta parte (la parte se separa limpia, como un títere de papel)
        Wg = _pesos(np.where(grupo_h == g, D, np.inf), altura)
        for par, art in BISAGRAS.items():
            if g in par:     # en la bisagra solo se mezclan las dos partes que une (no la mano que pasa cerca)
                pareja = next(x for x in par if x != g)
                cerca = 1 - np.clip((np.linalg.norm(V - np.asarray(J[art]), axis=1) - r_in) / (r_out - r_in), 0, 1)
                cerca = (cerca * cerca * (3 - 2 * cerca))[:, None]
                Ws = _pesos(np.where((grupo_h == g) | (grupo_h == pareja), D, np.inf), altura)
                Wg = (1 - cerca) * Wg + cerca * Ws
        dom = np.argmin(D[tris_g].mean(axis=1), axis=1)
        capas.append(Capa(g, float(GRUPOS[g]), tris_g, capa_h[dom], Wg, np.ascontiguousarray(tex_g),
                          (int(x0_), int(y0_))))
    capas.sort(key=lambda c: c.orden)

    pies = {}
    for c in capas:                                   # vértices más bajos de cada pierna (apoyo en el suelo)
        if c.nombre.startswith("pierna_"):
            k = huesos.index(c.nombre)
            sel = np.unique(c.tris)
            sel = sel[c.W[sel, k] > 0.5]
            if len(sel):
                pies[c.nombre[7:]] = (c.nombre, sel[np.argsort(-V[sel, 1])[:6]])
    suelo = float(max(V[i, 1].max() for _, i in pies.values())) if pies else float(ys.max())
    rep = angulos_reposo(J)
    return Personaje(dict(J), V, T, capas, altura, suelo, pies, rep, angulos_neutros(rep))


# --------------------------------------------------------------------------------------------
# 4. Pose (cinemática directa 2D)
# --------------------------------------------------------------------------------------------
# Ángulos en grados, ANTIHORARIOS tal como se ve la imagen, medidos desde «hacia abajo». Con el personaje
# avanzando hacia la derecha: pierna o brazo hacia adelante = positivo; torso inclinado hacia adelante =
# negativo; doblar la rodilla (la pierna va hacia atrás) = negativo; doblar el codo hacia adelante = positivo.
LIMITES = {"brazo_izq": (-35, -7), "brazo_der": (7, 35), "muslo_izq": (-10, 2), "muslo_der": (-2, 10)}
DOBLEZ = {"antebrazo_izq": 20, "antebrazo_der": 20, "pierna_izq": 8, "pierna_der": 8}


def _norm(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def angulos_reposo(J: dict) -> dict:
    rep = {}
    for hb, (a, b, *_r) in HUESOS.items():
        dx, dy = J[b][0] - J[a][0], J[b][1] - J[a][1]
        rep[hb] = 0.0 if a == b else math.degrees(math.atan2(dx, dy))
    return rep


def angulos_neutros(rep: dict) -> dict:
    """Postura de pie que sirve de base: brazos un poco abiertos y piernas casi rectas, aunque el dibujo los
    tenga en otra posición."""
    neu = dict(rep)
    for hb, (lo, hi) in LIMITES.items():
        neu[hb] = min(max(rep[hb], lo), hi)
    for hb, lim in DOBLEZ.items():
        padre = HUESOS[hb][2]
        neu[hb] = neu[padre] + max(-lim, min(lim, _norm(rep[hb] - rep[padre])))
    return neu


def _rot(grados: float) -> np.ndarray:
    r = math.radians(grados)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _tras(x: float, y: float) -> np.ndarray:
    return np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])


def _local(p: Personaje, hb: str) -> float:
    padre = HUESOS[hb][2]
    base_n = p.neutro[padre] if padre else 0.0
    base_r = p.reposo[padre] if padre else 0.0
    return _norm((p.neutro[hb] - base_n) - (p.reposo[hb] - base_r))


def matrices(p: Personaje, pose: dict) -> dict:
    """Matriz (3x3) de cada hueso: de la imagen en reposo a la pose."""
    M = {}
    for hb in ORDEN:
        a, b, padre, *_r = HUESOS[hb]
        Mp = M[padre] if padre else np.eye(3)
        x, y = p.J[a]
        M[hb] = Mp @ _tras(x, y) @ _rot(_local(p, hb) + float(pose.get(hb, 0.0))) @ _tras(-x, -y)
    return M


def pose_del_dibujo(p: Personaje) -> dict:
    """La pose que deja al personaje exactamente como en la imagen."""
    pose = {"apoyo": 0.0}
    pose.update({hb: -_local(p, hb) for hb in ORDEN})
    return pose


@dataclass
class Pose:
    """Vértices posados de cada capa y articulaciones (px de la imagen original)."""
    capas: list
    J: dict

    def limites(self) -> tuple:
        todos = np.vstack(self.capas)
        return todos.min(axis=0), todos.max(axis=0)

    def mover(self, dx: float, dy: float = 0.0) -> "Pose":
        d = np.array([dx, dy])
        return Pose([v + d for v in self.capas], {k: (v[0] + dx, v[1] + dy) for k, v in self.J.items()})


def posar(p: Personaje, pose: dict) -> Pose:
    M = matrices(p, pose)
    Ms = np.stack([M[hb] for hb in ORDEN])[:, :2, :]               # (huesos, 2, 3)
    Vh = np.hstack([p.V, np.ones((len(p.V), 1))])
    por_capa = [np.einsum("nb,bij,nj->ni", c.W, Ms, Vh) for c in p.capas]
    Jp = {}
    for j in ARTICULACIONES:
        hb = next((h_ for h_, d in HUESOS.items() if d[0] == j), None) or \
            next(h_ for h_, d in HUESOS.items() if d[1] == j)
        Jp[j] = (M[hb] @ np.array([p.J[j][0], p.J[j][1], 1.0]))[:2]
    # apoyo: los pies tocan el suelo (si la pose lo pide)
    apoyo = float(pose.get("apoyo", 1.0))
    dy = 0.0
    if apoyo > 0 and p.pies:
        idx = {c.nombre: i for i, c in enumerate(p.capas)}
        bajo = max(float(por_capa[idx[nc]][v, 1].max()) for nc, v in p.pies.values())
        dy = apoyo * (p.suelo - bajo)
    e = float(pose.get("estirar", 1.0))
    cx = p.J["cadera"][0]
    giro = float(pose.get("giro", 0.0))
    G = _tras(0, dy)
    if e != 1.0:                                    # estirar y aplastar desde el suelo
        G = _tras(cx, p.suelo) @ np.diag([1 / math.sqrt(e), e, 1.0]) @ _tras(-cx, -p.suelo) @ G
    if giro:
        c = (cx, p.suelo - 0.45 * p.altura)
        G = _tras(*c) @ _rot(giro) @ _tras(-c[0], -c[1]) @ G
    G = _tras(float(pose.get("x", 0.0)) * p.altura, -float(pose.get("y", 0.0)) * p.altura) @ G
    L, t = G[:2, :2], G[:2, 2]
    return Pose([v @ L.T + t for v in por_capa], {k: tuple(L @ v + t) for k, v in Jp.items()})


# --------------------------------------------------------------------------------------------
# 5. Render
# --------------------------------------------------------------------------------------------
def renderizar(p: Personaje, pose: Pose, escala: float, origen: tuple, lienzo: tuple, ss: int = 2) -> np.ndarray:
    """Dibuja el personaje posado en un lienzo transparente (ancho, alto). Posición en el lienzo:
    px_imagen * escala - origen. Devuelve BGRA uint8 con alfa recta."""
    ancho, alto = lienzo
    out = np.zeros((alto, ancho, 4), np.float32)
    o = np.asarray(origen, float)
    Qc = [v * escala - o for v in pose.capas]
    todos = np.vstack(Qc)
    x0 = max(0, int(math.floor(todos[:, 0].min())) - 2)
    y0 = max(0, int(math.floor(todos[:, 1].min())) - 2)
    x1 = min(ancho, int(math.ceil(todos[:, 0].max())) + 3)
    y1 = min(alto, int(math.ceil(todos[:, 1].max())) + 3)
    if x1 <= x0 or y1 <= y0:
        return out.astype(np.uint8)
    rw, rh = (x1 - x0) * ss, (y1 - y0) * ss
    acum = np.zeros((rh, rw, 4), np.float32)
    for c, Q in zip(p.capas, Qc):
        Qs = (Q - [x0, y0] + 0.5) * ss - 0.5                       # px de la región supermuestreada
        usados = Qs[np.unique(c.tris)]
        bx0 = max(0, int(math.floor(usados[:, 0].min())) - 1)
        by0 = max(0, int(math.floor(usados[:, 1].min())) - 1)
        bx1 = min(rw, int(math.ceil(usados[:, 0].max())) + 2)
        by1 = min(rh, int(math.ceil(usados[:, 1].max())) + 2)
        if bx1 <= bx0 or by1 <= by0:
            continue
        Qs = Qs - [bx0, by0]
        ids = np.full((by1 - by0, bx1 - bx0), -1, np.int32)
        pts = np.round(Qs[c.tris] * 16).astype(np.int32)          # (k, 3, 2) con 4 bits de subpíxel
        for f in np.argsort(c.sub, kind="stable"):
            cv2.fillConvexPoly(ids, pts[f], int(f), cv2.LINE_8, 4)
            cv2.polylines(ids, [pts[f]], True, int(f), 1, cv2.LINE_8, 4)
        tex, nivel = c.textura_a(escala * ss)
        A = np.concatenate([Qs[c.tris], np.ones((len(c.tris), 3, 1))], axis=2)     # (k, 3, 3)
        S = (p.V[c.tris] - c.off + 0.5) * nivel - 0.5                              # (k, 3, 2) en la textura
        ok = np.abs(np.linalg.det(A)) > 1e-6
        C = np.zeros((len(c.tris), 3, 2))
        C[ok] = np.linalg.solve(A[ok], S[ok])
        yy, xx = np.nonzero(ids >= 0)
        f = ids[yy, xx]
        mx = np.full(ids.shape, -10.0, np.float32)
        my = np.full(ids.shape, -10.0, np.float32)
        mx[yy, xx] = C[f, 0, 0] * xx + C[f, 1, 0] * yy + C[f, 2, 0]
        my[yy, xx] = C[f, 0, 1] * xx + C[f, 1, 1] * yy + C[f, 2, 1]
        capa = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        zona = acum[by0:by1, bx0:bx1]
        zona *= 1 - capa[..., 3:4] / 255
        zona += capa
    if ss > 1:
        acum = cv2.resize(acum, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    out[y0:y1, x0:x1] = acum
    a = out[..., 3:4]
    rgb = np.where(a > 0.5, out[..., :3] * 255 / np.maximum(a, 1e-3), 0)
    return np.dstack([np.clip(rgb, 0, 255), np.clip(a, 0, 255)]).astype(np.uint8)


def encuadre(p: Personaje, poses: list, escala: float, lienzo: tuple | None = None) -> tuple:
    """(origen, (ancho, alto)). Sin lienzo: el justo para todo el movimiento. Con lienzo fijo: el suelo al
    92 % del alto y el recorrido centrado."""
    lo = np.min([P.limites()[0] for P in poses], axis=0) * escala
    hi = np.max([P.limites()[1] for P in poses], axis=0) * escala
    if lienzo is None:
        m = 0.03 * p.altura * escala
        ancho, alto = int(math.ceil(hi[0] - lo[0] + 2 * m)), int(math.ceil(hi[1] - lo[1] + 2 * m))
        return (lo[0] - m, lo[1] - m), (ancho + ancho % 2, alto + alto % 2)
    ancho, alto = lienzo
    y0 = p.suelo * escala - 0.92 * alto
    if lo[1] - y0 < 0:                 # un salto alto: se baja el suelo lo justo
        y0 = max(lo[1] - 0.02 * alto, p.suelo * escala - 0.99 * alto)
    return ((lo[0] + hi[0]) / 2 - ancho / 2, y0), (int(ancho), int(alto))


# --------------------------------------------------------------------------------------------
# 6. Cargar una imagen y guardar el resultado
# --------------------------------------------------------------------------------------------
LADO_MAX = 1800                       # imágenes más grandes se reducen (más rápido, igual de nítido en video)


def ruta_esqueleto(imagen: str | Path) -> Path:
    imagen = Path(imagen)
    return imagen.with_name(imagen.stem + ".esqueleto.json")


def preparar(imagen: str | Path, esqueleto: str | Path | None = None, rehacer: bool = False) -> tuple:
    """(bgra, alfa, J, ruta del esqueleto). Usa el esqueleto guardado junto a la imagen si existe."""
    bgra = leer_imagen(imagen)
    h, w = bgra.shape[:2]
    if max(h, w) > LADO_MAX:
        s = LADO_MAX / max(h, w)
        bgra = cv2.resize(bgra, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    alfa = recortar(bgra)
    ruta = Path(esqueleto) if esqueleto else ruta_esqueleto(imagen)
    J = None
    if ruta.exists() and not rehacer:
        J = cargar_esqueleto(ruta, bgra.shape[1::-1])
    if J is None:
        J = detectar_esqueleto(alfa)
    return bgra, alfa, J, ruta


FORMATOS = {".webm": "WebM con transparencia (VP9) — navegadores, CapCut, DaVinci",
            ".mov": "MOV ProRes 4444 con transparencia — Premiere, DaVinci, Final Cut",
            ".gif": "GIF animado (transparencia sin bordes suaves)",
            ".apng": "PNG animado con transparencia completa",
            "": "carpeta con un PNG transparente por fotograma"}


def exportar(fotogramas, ruta: str | Path, fps: float, lienzo: tuple, progreso=None) -> Path:
    """Guarda los fotogramas BGRA (iterable) con fondo transparente según la extensión de `ruta`."""
    ruta = Path(ruta)
    ext = ruta.suffix.lower()
    if ext not in FORMATOS:
        raise ValueError(f"Formato «{ext}» sin transparencia o no soportado. Usa: .webm, .mov, .gif, .apng "
                         "o una carpeta (MP4 no guarda transparencia).")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ancho, alto = lienzo
    n = 0
    if ext in (".webm", ".mov"):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Falta ffmpeg para .webm/.mov (usa .gif, .apng o una carpeta de PNG).")
        codec = (["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "28", "-row-mt", "1",
                  "-deadline", "good", "-cpu-used", "5", "-threads", "8", "-auto-alt-ref", "0"] if ext == ".webm" else
                 ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0",
                  "-qscale:v", "9"])
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{ancho}x{alto}",
               "-framerate", str(fps), "-i", "-", *codec, str(ruta)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for f in fotogramas:
                proc.stdin.write(np.ascontiguousarray(f).tobytes())
                n += 1
                if progreso:
                    progreso(n)
        finally:
            proc.stdin.close()
            err = proc.stderr.read().decode("utf-8", "replace")
            proc.stderr.close()
            proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg falló: " + err.strip()[-400:])
        return ruta
    if ext == "":
        ruta.mkdir(parents=True, exist_ok=True)
        for f in fotogramas:
            n += 1
            guardar_png(ruta / f"{n:04d}.png", f)
            if progreso:
                progreso(n)
        return ruta
    from PIL import Image
    imgs = []
    for f in fotogramas:
        rgba = cv2.cvtColor(f, cv2.COLOR_BGRA2RGBA)
        if ext == ".gif":
            im = Image.fromarray(rgba[..., :3]).quantize(255, method=Image.Quantize.FASTOCTREE)
            idx = np.array(im)
            idx[rgba[..., 3] < 128] = 255
            pal = im.getpalette()[:765] + [0, 0, 0]
            im = Image.fromarray(idx, "P")
            im.putpalette(pal)
            im.info["transparency"] = 255
            imgs.append(im)
        else:
            imgs.append(Image.fromarray(rgba, "RGBA"))
        n += 1
        if progreso:
            progreso(n)
    if not imgs:
        raise ValueError("No hay fotogramas.")
    dur = max(20, round(1000 / fps))
    if ext == ".gif":
        imgs[0].save(ruta, save_all=True, append_images=imgs[1:], duration=dur, loop=0, disposal=2,
                     transparency=255, optimize=False)
    else:
        imgs[0].save(ruta, format="PNG", save_all=True, append_images=imgs[1:], duration=dur, loop=0,
                     disposal=0, blend=0)
    return ruta
