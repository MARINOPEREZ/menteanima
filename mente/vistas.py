# -*- coding: utf-8 -*-
"""Hoja de personaje → vistas (frente, perfiles, espalda) alineadas en un mismo espacio 3D.

Espacio del mundo (en px de la vista de frente, ya a escala común):
  X = derecha de la imagen de frente (lado IZQUIERDO del personaje), Y = arriba (0 = suelo),
  Z = hacia donde mira el personaje.
Cada vista es una proyección ortogonal: frente mira desde +Z, espalda desde -Z, «perfil_der» desde +X (el
personaje mira hacia la izquierda de la imagen) y «perfil_izq» desde -X (mira hacia la derecha).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mente.animacion import _componentes_principales, recortar, tiene_transparencia

ROLES = ("frente", "perfil_der", "espalda", "perfil_izq")
NOMBRE_ROL = {"frente": "FRENTE", "perfil_der": "PERFIL (mira a la izquierda)", "espalda": "ESPALDA",
              "perfil_izq": "PERFIL (mira a la derecha)"}
# hacia dónde está la cámara de cada vista
DIRECCION = {"frente": np.array([0.0, 0.0, 1.0]), "espalda": np.array([0.0, 0.0, -1.0]),
             "perfil_der": np.array([1.0, 0.0, 0.0]), "perfil_izq": np.array([-1.0, 0.0, 0.0])}


@dataclass
class Vista:
    rol: str
    bgra: np.ndarray            # la figura recortada (BGRA, alfa limpio) a la escala común, con margen
    c: float                    # columna de la imagen donde está el 0 de su eje horizontal (X o Z)
    suelo: float                # fila del suelo (Y = 0)
    espejo: bool = False        # vista inventada reflejando otra (falta en la hoja)

    @property
    def alfa(self) -> np.ndarray:
        return self.bgra[..., 3]

    def horizontal(self, P: np.ndarray) -> np.ndarray:
        X, Z = P[..., 0], P[..., 2]
        return {"frente": X, "espalda": -X, "perfil_der": -Z, "perfil_izq": Z}[self.rol]

    def proyectar(self, P: np.ndarray) -> tuple:
        """Puntos del mundo (…, 3) → (u, v) en la imagen de la vista."""
        return self.c + self.horizontal(P), self.suelo - P[..., 1]

    def profundidad(self, P: np.ndarray) -> np.ndarray:
        """Cuánto se acerca el punto a la cámara de la vista (mayor = más cerca)."""
        return P @ DIRECCION[self.rol]


# --------------------------------------------------------------------------------------------
# Separar la hoja
# --------------------------------------------------------------------------------------------
def _tramos(v: np.ndarray) -> list:
    d = np.diff(np.concatenate([[0], v.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def separar(bgra: np.ndarray) -> list:
    """Recorta cada figura de la hoja (de izquierda a derecha) → lista de BGRA con alfa limpio."""
    alfa = bgra[..., 3] if tiene_transparencia(bgra) else recortar(bgra, todas=True)
    m = _componentes_principales(alfa > 127, 0.05, solo_junto=False)
    occ = cv2.blur(m.sum(0).astype(np.float32)[None, :], (5, 1))[0]
    tramos = [(a, b) for a, b in _tramos(occ > 0.015 * occ.max()) if occ[a:b].sum() > 0.04 * occ.sum()]
    # figuras que se tocan: se parte por el valle más hondo de un tramo demasiado ancho
    alto = np.ptp(np.nonzero(m)[0]) + 1
    cambio = True
    while cambio:
        cambio = False
        anchos = sorted(b - a for a, b in tramos)
        ref = anchos[len(anchos) // 2] if len(tramos) > 1 else 0.6 * alto
        for i, (a, b) in enumerate(tramos):
            if b - a > 1.7 * ref and b - a > 0.5 * alto:
                medio = occ[a + (b - a) // 4: b - (b - a) // 4]
                corte = a + (b - a) // 4 + int(np.argmin(medio))
                if occ[corte] < 0.5 * min(occ[a:corte].max(), occ[corte:b].max()):
                    tramos[i:i + 1] = [(a, corte), (corte, b)]
                    cambio = True
                    break
    figuras = []
    for a, b in tramos:
        sub = np.zeros_like(m)
        sub[:, a:b] = m[:, a:b]
        sub = _componentes_principales(sub, 0.08)
        ys, xs = np.nonzero(sub)
        if len(xs) < 100:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        recorte = bgra[y0:y1, x0:x1].copy()
        recorte[..., 3] = np.where(cv2.dilate(sub[y0:y1, x0:x1].astype(np.uint8), np.ones((3, 3))) > 0,
                                   alfa[y0:y1, x0:x1], 0)
        figuras.append(recorte)
    return figuras


def _ancho_torso(alfa: np.ndarray) -> float:
    m = alfa > 127
    ys = np.nonzero(m.any(1))[0]
    y0, y1 = ys.min(), ys.max()
    filas = [m[y] for y in range(int(y0 + 0.4 * (y1 - y0)), int(y0 + 0.6 * (y1 - y0)))]
    anchos = [np.ptp(np.nonzero(f)[0]) + 1 for f in filas if f.any()]
    return float(np.median(anchos)) / (y1 - y0 + 1)


def mira_a_la_izquierda(alfa: np.ndarray) -> float:
    """> 0 si el perfil mira hacia la izquierda de la imagen (las puntas de los pies salen hacia adelante)."""
    m = alfa > 127
    ys, xs = np.nonzero(m)
    y0, y1 = ys.min(), ys.max()
    H = y1 - y0 + 1
    pie = xs[ys > y1 - 0.06 * H]
    pierna = xs[(ys > y1 - 0.26 * H) & (ys < y1 - 0.12 * H)]
    if len(pie) == 0 or len(pierna) == 0:
        return 0.0
    return float(pierna.mean() - pie.mean()) / H


def asignar_roles(figuras: list) -> dict:
    """{rol: figura}. Orden habitual de una hoja: frente, perfil, espalda, perfil. Las vistas anchas (hombros)
    son frente y espalda; la primera es el frente. Los perfiles se distinguen por hacia dónde apuntan los pies."""
    n = len(figuras)
    if n == 0:
        raise ValueError("No se encontró ninguna figura en la imagen.")
    if n == 1:
        return {"frente": figuras[0]}
    figuras = figuras[:4]
    anchos = [_ancho_torso(f[..., 3]) for f in figuras]
    orden = np.argsort(anchos)[::-1]
    if n == 2:
        anchas, angostas = [0], [1]
    elif n == 3:
        angostas = [int(orden[-1])]
        anchas = sorted(i for i in range(3) if i not in angostas)
    else:
        anchas, angostas = sorted(int(i) for i in orden[:2]), sorted(int(i) for i in orden[2:])
    roles = {"frente": figuras[anchas[0]]}
    if len(anchas) > 1:
        roles["espalda"] = figuras[anchas[1]]
    lados = [(mira_a_la_izquierda(figuras[i][..., 3]), i) for i in angostas]
    if len(lados) == 2:
        (s0, i0), (s1, i1) = lados
        if s0 >= s1:
            roles["perfil_der"], roles["perfil_izq"] = figuras[i0], figuras[i1]
        else:
            roles["perfil_der"], roles["perfil_izq"] = figuras[i1], figuras[i0]
    elif lados:
        s, i = lados[0]
        roles["perfil_der" if s >= 0 else "perfil_izq"] = figuras[i]
    return roles


# --------------------------------------------------------------------------------------------
# Escala común y alineación
# --------------------------------------------------------------------------------------------
def _a_escala(fig: np.ndarray, alto: int, margen: float = 0.12) -> tuple:
    """Escala la figura a `alto` px y le da margen transparente. Devuelve (bgra, fila del suelo)."""
    m = fig[..., 3] > 127
    ys = np.nonzero(m.any(1))[0]
    s = alto / (ys.max() - ys.min() + 1)
    h, w = fig.shape[:2]
    img = cv2.resize(fig, (max(1, round(w * s)), max(1, round(h * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    p = int(round(margen * alto))
    img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    ys = np.nonzero((img[..., 3] > 127).any(1))[0]
    return img, float(ys.max())


def _centro(alfa: np.ndarray, desde: float = 0.35, hasta: float = 0.6) -> float:
    m = alfa > 127
    ys = np.nonzero(m.any(1))[0]
    y0, y1 = ys.min(), ys.max()
    medios = []
    for y in range(int(y0 + desde * (y1 - y0)), int(y0 + hasta * (y1 - y0))):
        xs = np.nonzero(m[y])[0]
        if len(xs):
            medios.append((xs.min() + xs.max()) / 2)
    return float(np.median(medios))


def _registrar(a: np.ndarray, b: np.ndarray, suelo_a: float, suelo_b: float) -> float:
    """Desplazamiento horizontal dx que mejor superpone la silueta b (ya reflejada si hace falta) sobre a,
    con los suelos alineados. Devuelve dx tal que u_a = u_b + dx."""
    s = 160 / max(a.shape[0], b.shape[0])
    def chica(m, suelo):
        x = cv2.resize((m > 127).astype(np.float32), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return x, suelo * s
    A, sa = chica(a, suelo_a)
    B, sb = chica(b, suelo_b)
    dy = int(round(sa - sb))
    H = max(A.shape[0], B.shape[0] + abs(dy)) + 4
    W = A.shape[1] + 2 * B.shape[1]
    lienzo_a = np.zeros((H, W), np.float32)
    lienzo_a[:A.shape[0], B.shape[1]:B.shape[1] + A.shape[1]] = A
    mejor, dx_m = -1.0, 0
    for dx in range(0, A.shape[1] + B.shape[1]):
        lb = np.zeros_like(lienzo_a)
        y0 = max(0, dy)
        hb = min(B.shape[0], H - y0)
        lb[y0:y0 + hb, dx:dx + B.shape[1]] = B[:hb]
        inter = np.minimum(lienzo_a, lb).sum()
        union = np.maximum(lienzo_a, lb).sum()
        if union and inter / union > mejor:
            mejor, dx_m = inter / union, dx
    return (dx_m - B.shape[1]) / s


def preparar_vistas(bgra: np.ndarray, alto: int = 900) -> dict:
    """Separa la hoja, asigna roles y alinea todas las vistas. {rol: Vista}; las que faltan se inventan
    reflejando otra (y se marcan con espejo=True)."""
    roles = asignar_roles(separar(bgra))
    vistas = {}
    for rol, fig in roles.items():
        img, suelo = _a_escala(fig, alto)
        vistas[rol] = Vista(rol, img, 0.0, suelo)
    fr = vistas["frente"]
    fr.c = _centro(fr.alfa, 0.45, 0.6)
    # perfiles inventados si faltan: el otro lado en espejo; si no hay ninguno, el frente «de canto»
    if "perfil_der" in vistas and "perfil_izq" not in vistas:
        v = vistas["perfil_der"]
        vistas["perfil_izq"] = Vista("perfil_izq", v.bgra[:, ::-1].copy(), 0.0, v.suelo, True)
    elif "perfil_izq" in vistas and "perfil_der" not in vistas:
        v = vistas["perfil_izq"]
        vistas["perfil_der"] = Vista("perfil_der", v.bgra[:, ::-1].copy(), 0.0, v.suelo, True)
    if "perfil_der" in vistas:
        pd = vistas["perfil_der"]
        pd.c = _centro(pd.alfa)
        pi = vistas["perfil_izq"]
        # u_der = c_der - Z ; u_izq = c_izq + Z → el reflejo de la izquierda se superpone a la derecha
        espejo = pi.bgra[..., 3][:, ::-1]
        dx = _registrar(pd.alfa, espejo, pd.suelo, pi.suelo)
        w = pi.bgra.shape[1]
        # columna reflejada: u' = (w - 1 - u_izq); u_der = u' + dx → c_izq = (w - 1) + dx - c_der
        pi.c = (w - 1) + dx - pd.c
    if "espalda" in vistas:
        es = vistas["espalda"]
        dx = _registrar(fr.alfa, es.alfa[:, ::-1], fr.suelo, es.suelo)
        w = es.bgra.shape[1]
        es.c = (w - 1) + dx - fr.c
    else:
        vistas["espalda"] = Vista("espalda", fr.bgra[:, ::-1].copy(), (fr.bgra.shape[1] - 1) - fr.c, fr.suelo, True)
    return vistas


def resumen(vistas: dict, alto: int = 360) -> np.ndarray:
    """Tira de control con cada vista, su rol y su eje (BGR)."""
    celdas = []
    for rol in ROLES:
        if rol not in vistas:
            continue
        v = vistas[rol]
        s = alto / v.bgra.shape[0]
        img = cv2.resize(v.bgra, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        a = img[..., 3:4].astype(np.float32) / 255
        out = (img[..., :3] * a + 235 * (1 - a)).astype(np.uint8)
        x = int(round(v.c * s))
        cv2.line(out, (x, 0), (x, alto), (0, 0, 255), 1)
        cv2.line(out, (0, int(v.suelo * s)), (out.shape[1], int(v.suelo * s)), (0, 160, 0), 1)
        texto = NOMBRE_ROL[rol] + (" (espejo)" if v.espejo else "")
        cv2.putText(out, texto, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        celdas.append(out)
    return np.hstack(celdas)
