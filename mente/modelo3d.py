# -*- coding: utf-8 -*-
"""MenteAnima 3D — de una hoja de personaje (frente, perfiles, espalda) a un modelo 3D con huesos que se mueve.

Etapas:
  1. vistas       mente/vistas.py separa la hoja, reconoce cada vista y las alinea en un mismo espacio.
  2. esqueleto    Las 15 articulaciones del frente (X, Y) reciben su profundidad Z de los perfiles.
  3. partes       Cada vista se separa en huesos (distancias geodésicas, como en el modo 2D). De cada hueso se
                  mide, a lo largo de su eje, el ancho en el frente y la profundidad en el perfil.
  4. volumen      Cada hueso es un sólido de secciones redondeadas (superelipse) con esos anchos y esas
                  profundidades, recortado por las siluetas reales. Cada parte del cuerpo (tronco, cabeza,
                  brazos, piernas) es una malla cerrada aparte (superficie por «surface nets» + suavizado).
  5. textura      Al dibujar, cada píxel toma el color de las vistas que lo ven de frente (con prueba de
                  profundidad para no pintar con lo que tapa): nítido como el dibujo original, sin atlas.
  6. pose         Cinemática directa 3D (adelante, afuera, giro por hueso), pies apoyados en el suelo.
  7. render       Rasterizador propio con z-buffer (numpy), cámara con perspectiva suave y supermuestreo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from mente.animacion import ARTICULACIONES, HUESOS, ORDEN, detectar_esqueleto, etiquetas
from mente.vistas import DIRECCION, Vista

GRUPOS3D = ["nucleo", "cabeza", "brazo_izq", "brazo_der", "pierna_izq", "pierna_der"]
CENTRALES = ["pelvis", "torso", "cabeza"]
ARRIBA = {"torso", "cabeza"}          # huesos que apuntan hacia arriba


def _lado(nombre: str) -> str | None:
    return nombre[-3:] if nombre[-3:] in ("izq", "der") else None


# --------------------------------------------------------------------------------------------
# 2. Esqueleto 3D
# --------------------------------------------------------------------------------------------
def _z_en_fila(v: Vista, Y: float, z_ref: float = 0.0) -> float | None:
    """Centro (en Z del mundo) del tramo de silueta del perfil a la altura Y más cercano a z_ref."""
    fila = int(round(v.suelo - Y))
    if not 0 <= fila < v.bgra.shape[0]:
        return None
    m = v.alfa[fila] > 127
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    tramos = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))
    if not tramos:
        return None
    signo = -1.0 if v.rol == "perfil_der" else 1.0          # u = c - Z  /  u = c + Z
    centros = [signo * ((a + b) / 2 - v.c) for a, b in tramos]
    return float(min(centros, key=lambda z: abs(z - z_ref)))


def esqueleto_3d(vistas: dict, J2: dict) -> dict:
    """Articulaciones 3D: X, Y del frente; Z de los perfiles (centro del cuerpo a esa altura)."""
    fr = vistas["frente"]
    P = {k: np.array([u - fr.c, fr.suelo - v, 0.0]) for k, (u, v) in J2.items()}
    if "perfil_der" not in vistas:
        return P
    def z(k, ref=0.0):
        lado = _lado(k)
        roles = ["perfil_der"] if lado == "der" else ["perfil_izq"] if lado == "izq" else ["perfil_der", "perfil_izq"]
        vals = [_z_en_fila(vistas[r], P[k][1], ref) for r in roles if r in vistas]
        vals = [x for x in vals if x is not None]
        return float(np.mean(vals)) if vals else ref
    for k in ("cadera", "cuello", "cadera_izq", "cadera_der", "hombro_izq", "hombro_der"):
        P[k][2] = z(k)
    P["cabeza"][2] = z("cabeza", P["cuello"][2])
    for lado in ("izq", "der"):
        P["codo_" + lado][2] = z("codo_" + lado, P["hombro_" + lado][2])
        P["mano_" + lado][2] = z("mano_" + lado, P["codo_" + lado][2])
        P["rodilla_" + lado][2] = z("rodilla_" + lado, P["cadera_" + lado][2])
        P["pie_" + lado][2] = z("pie_" + lado, P["rodilla_" + lado][2])
    return P


# --------------------------------------------------------------------------------------------
# 3. Perfiles de cada hueso
# --------------------------------------------------------------------------------------------
def _huesos_vista(rol: str) -> list:
    if rol == "frente":
        return list(HUESOS)
    lado = "der" if rol == "perfil_der" else "izq"
    return CENTRALES + [h for h in HUESOS if _lado(h) == lado]


def etiquetar_vistas(vistas: dict, J3: dict) -> dict:
    """{rol: (etiquetas por píxel, lista de huesos)} para el frente y los perfiles reales."""
    sal = {}
    for rol in ("frente",):
        v = vistas.get(rol)
        if v is None or v.espejo:
            continue
        J2 = {k: tuple(float(x) for x in v.proyectar(np.asarray(P))) for k, P in J3.items()}
        hs = _huesos_vista(rol)
        sal[rol] = (etiquetas(v.bgra, v.alfa, J2, hs, poligono=(rol == "frente")), hs)
    return sal


def _segmento(hb: str) -> tuple:
    a, b = HUESOS[hb][0], HUESOS[hb][1]
    if hb == "pelvis":
        return "cadera", "cuello"            # la pelvis va dentro del sólido del torso
    return a, b


@dataclass
class Perfil:
    hueso: str
    A: np.ndarray                    # eje 3D
    B: np.ndarray
    t: np.ndarray                    # muestras a lo largo del eje
    lim: dict                        # rol → (lo, hi) desplazamientos perpendiculares en esa vista


def _medir(mask: np.ndarray, a2: np.ndarray, b2: np.ndarray, t: np.ndarray, R: float) -> tuple:
    """Para cada t, el tramo de la máscara sobre la perpendicular al eje (lo, hi) más cercano al eje."""
    d = b2 - a2
    L = np.linalg.norm(d)
    n = np.array([-d[1], d[0]]) / max(L, 1e-9)
    s = np.arange(-R, R + 1, 1.0)
    P = a2[None, None, :] + t[:, None, None] * d[None, None, :] + s[None, :, None] * n[None, None, :]
    h, w = mask.shape
    xi = np.clip(np.round(P[..., 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(P[..., 1]).astype(int), 0, h - 1)
    dentro = mask[yi, xi] & (P[..., 0] >= 0) & (P[..., 0] < w) & (P[..., 1] >= 0) & (P[..., 1] < h)
    lo = np.full(len(t), np.nan)
    hi = np.full(len(t), np.nan)
    for i in range(len(t)):
        fila = dentro[i]
        if not fila.any():
            continue
        dd = np.diff(np.concatenate([[0], fila.astype(np.int8), [0]]))
        ini, fin = np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1
        dist = np.where((ini <= R) & (fin >= R), 0, np.minimum(abs(ini - R), abs(fin - R)))
        j = int(np.argmin(dist))
        if dist[j] > 0.25 * R:
            continue
        lo[i], hi[i] = s[ini[j]] - 0.5, s[fin[j]] + 0.5
    return lo, hi


def _suavizar(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Rellena huecos (interpolando) y suaviza con mediana + media móvil."""
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return x
    idx = np.arange(len(x))
    y = np.interp(idx, idx[ok], x[ok])
    p = k // 2
    yp = np.pad(y, p, mode="edge")
    y = np.array([np.median(yp[i:i + k]) for i in range(len(y))])
    y = np.convolve(np.pad(y, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    y[~ok] = np.where(np.cumsum(ok)[~ok] == 0, np.nan, y[~ok])         # antes del primer dato: nada
    y[~ok & (np.cumsum(ok[::-1])[::-1] == 0)] = np.nan                  # después del último: nada
    return y


def perfiles(vistas: dict, J3: dict, etiq: dict, altura: float, manos: dict | None = None) -> list:
    """Anchos (frente) y profundidades (perfil) de cada hueso a lo largo de su eje.
    De lado los brazos quedan dentro del cuerpo (no tienen silueta propia): se hacen redondos. El tronco, la
    cabeza y las piernas toman la profundidad de la silueta del perfil."""
    sal = []
    fr = vistas["frente"]
    lab, hs = etiq["frente"]
    for hb in HUESOS:
        if hb == "pelvis":
            continue
        a, b = _segmento(hb)
        A, B = np.asarray(J3[a], float), np.asarray(J3[b], float)
        lado = _lado(hb)
        # rango del eje: hasta donde llegan los píxeles del hueso en el frente
        ids = [hs.index(hb)] + ([hs.index("pelvis")] if hb == "torso" else [])
        mask_f = np.isin(lab, ids)
        a2, b2 = np.array(fr.proyectar(A)), np.array(fr.proyectar(B))
        d2 = b2 - a2
        L2 = max(float(d2 @ d2), 1e-9)
        yy, xx = np.nonzero(mask_f)
        if len(xx) < 10:
            continue
        tp = ((np.stack([xx, yy], 1) - a2) @ d2) / L2
        t0, t1 = float(np.percentile(tp, 0.5)), float(np.percentile(tp, 99.5))
        t0, t1 = max(t0, -0.8), min(t1, 3.0)
        largo = math.dist(A, B) * (t1 - t0)
        t = np.linspace(t0, t1, max(16, int(largo / 2.5)))
        R = 0.45 * altura
        lo, hi = _medir(mask_f, a2, b2, t, R)
        lim = {"frente": (_suavizar(lo), _suavizar(hi))}
        mano = (manos or {}).get(lado) if hb.startswith("antebrazo") else None
        if mano is not None:                     # el guante: su forma de perfil sale de la mano en el perfil
            vp = mano["vista"]
            a2p, b2p = np.array(vp.proyectar(A)), np.array(vp.proyectar(B))
            lo, hi = _medir(mano["mascara"], a2p, b2p, t, R)
            antes = t < mano["t_muneca"] + 0.03
            lo[antes], hi[antes] = np.nan, np.nan
            lim["perfil"] = (lo, hi, vp.rol)
        elif not hb.startswith(("brazo", "antebrazo")):
            reales = [r for r in (("perfil_" + lado,) if lado else ("perfil_der", "perfil_izq"))
                      if r in vistas and not vistas[r].espejo]
            if not reales:
                reales = [r for r in ("perfil_der", "perfil_izq") if r in vistas and not vistas[r].espejo]
            if reales:
                vp = vistas[reales[0]]
                a2p, b2p = np.array(vp.proyectar(A)), np.array(vp.proyectar(B))
                if np.linalg.norm(b2p - a2p) > 0.3 * math.dist(A, B):
                    lo, hi = _medir(vp.alfa > 127, a2p, b2p, t, R)
                    lim["perfil"] = (_suavizar(lo), _suavizar(hi), reales[0])
        sal.append(Perfil(hb, A, B, t, lim))
    return sal


# --------------------------------------------------------------------------------------------
# 4. Volumen y malla
# --------------------------------------------------------------------------------------------
def _perp(d: np.ndarray) -> np.ndarray:
    return np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)


def _dentro(pf: Perfil, P: np.ndarray, vistas: dict, n: float = 2.5, extender: float = 0.0) -> np.ndarray:
    """¿Los puntos P (N, 3) están dentro del sólido del hueso?"""
    AB = pf.B - pf.A
    t = ((P - pf.A) @ AB) / max(float(AB @ AB), 1e-9)
    t0, t1 = pf.t[0] - extender, pf.t[-1]
    tc = np.clip(t, pf.t[0], pf.t[-1])
    ok = (t >= t0) & (t <= t1)
    coords = []
    fr = vistas["frente"]
    a2, b2 = np.array(fr.proyectar(pf.A)), np.array(fr.proyectar(pf.B))
    nf = _perp(b2 - a2)
    u, v = fr.proyectar(P)
    s = (np.stack([u, v], 1) - (a2 + t[:, None] * (b2 - a2))) @ nf
    lo, hi = (np.interp(tc, pf.t, x, left=np.nan, right=np.nan) for x in pf.lim["frente"])
    ok &= np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    mitad = np.maximum((hi - lo) / 2, 1e-3)
    coords.append((s - (lo + hi) / 2) / mitad)
    redondo = 0.95 * mitad                       # sección redonda cuando no hay medida de perfil
    if "perfil" in pf.lim:
        lo_p, hi_p, rol_p = pf.lim["perfil"]
        vp = vistas[rol_p]
        a2p, b2p = np.array(vp.proyectar(pf.A)), np.array(vp.proyectar(pf.B))
        npf = _perp(b2p - a2p)
        up, vv = vp.proyectar(P)
        sp = (np.stack([up, vv], 1) - (a2p + t[:, None] * (b2p - a2p))) @ npf
        lo2 = np.interp(tc, pf.t, lo_p, left=np.nan, right=np.nan)
        hi2 = np.interp(tc, pf.t, hi_p, left=np.nan, right=np.nan)
        falta = ~(np.isfinite(lo2) & np.isfinite(hi2) & (hi2 > lo2))
        lo2 = np.where(falta, -redondo, lo2)
        hi2 = np.where(falta, redondo, hi2)
        if pf.hueso not in CENTRALES:            # una pierna no es mucho más profunda que ancha (ni al revés)
            centro = (lo2 + hi2) / 2
            mit = np.clip((hi2 - lo2) / 2, 0.6 * mitad, 1.7 * mitad)
            lo2, hi2 = centro - mit, centro + mit
        coords.append((sp - (lo2 + hi2) / 2) / np.maximum((hi2 - lo2) / 2, 1e-3))
    else:                                        # brazos (o sin perfil): sección redonda
        zc = pf.A[2] + t * (pf.B[2] - pf.A[2])
        coords.append((P[:, 2] - zc) / np.maximum(redondo, 1e-3))
    q = np.abs(coords[0]) ** n + np.abs(coords[1]) ** n
    return ok & (q <= 1.0)


def _silueta(v: Vista, P: np.ndarray) -> np.ndarray:
    u, w = v.proyectar(P)
    h, ww = v.alfa.shape
    ui, wi = np.round(u).astype(int), np.round(w).astype(int)
    ok = (ui >= 0) & (ui < ww) & (wi >= 0) & (wi < h)
    sal = np.zeros(len(P), bool)
    sal[ok] = v.alfa[wi[ok], ui[ok]] > 60
    return sal


def _difuminar3(F: np.ndarray, veces: int = 2) -> np.ndarray:
    for _ in range(veces):
        for eje in range(3):
            a = np.moveaxis(F, eje, 0)
            b = a.copy()
            b[1:-1] = 0.25 * a[:-2] + 0.5 * a[1:-1] + 0.25 * a[2:]
            F = np.moveaxis(b, 0, eje)
    return F


_ESQ = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
_ARISTAS = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(_ESQ[a] - _ESQ[b]).sum() == 1]


def surface_nets(F: np.ndarray, iso: float = 0.5) -> tuple:
    """Superficie de la región F > iso: vértices (en índices de la grilla) y triángulos con normal hacia afuera."""
    nx, ny, nz = F.shape
    c = [F[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k] for i, j, k in _ESQ]
    dentro = [ci > iso for ci in c]
    alguno = np.logical_or.reduce(dentro)
    todos = np.logical_and.reduce(dentro)
    activo = alguno & ~todos
    celdas = np.argwhere(activo)
    if len(celdas) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    suma = np.zeros((len(celdas), 3))
    cuenta = np.zeros(len(celdas))
    sel = tuple(celdas.T)
    for a, b in _ARISTAS:
        va, vb = c[a][sel], c[b][sel]
        cruza = (va > iso) != (vb > iso)
        tt = np.where(cruza, (iso - va) / np.where(vb - va == 0, 1, vb - va), 0)
        pos = _ESQ[a][None, :] + tt[:, None] * (_ESQ[b] - _ESQ[a])[None, :]
        suma += pos * cruza[:, None]
        cuenta += cruza
    V = celdas + suma / np.maximum(cuenta, 1)[:, None]
    indice = -np.ones((nx - 1, ny - 1, nz - 1), np.int64)
    indice[sel] = np.arange(len(celdas))
    caras = []
    D = F > iso
    # aristas en x: celdas vecinas en (y, z)
    for eje in range(3):
        if eje == 0:
            e = D[:-1, 1:-1, 1:-1] != D[1:, 1:-1, 1:-1]
            adentro = D[:-1, 1:-1, 1:-1]
            i, j, k = np.nonzero(e)
            j, k = j + 1, k + 1
            q = [indice[i, j - 1, k - 1], indice[i, j, k - 1], indice[i, j, k], indice[i, j - 1, k]]
            adentro = adentro[i, j - 1, k - 1]
        elif eje == 1:
            e = D[1:-1, :-1, 1:-1] != D[1:-1, 1:, 1:-1]
            adentro = D[1:-1, :-1, 1:-1]
            i, j, k = np.nonzero(e)
            i, k = i + 1, k + 1
            q = [indice[i - 1, j, k - 1], indice[i - 1, j, k], indice[i, j, k], indice[i, j, k - 1]]
            adentro = adentro[i - 1, j, k - 1]
        else:
            e = D[1:-1, 1:-1, :-1] != D[1:-1, 1:-1, 1:]
            adentro = D[1:-1, 1:-1, :-1]
            i, j, k = np.nonzero(e)
            i, j = i + 1, j + 1
            q = [indice[i - 1, j - 1, k], indice[i, j - 1, k], indice[i, j, k], indice[i - 1, j, k]]
            adentro = adentro[i - 1, j - 1, k]
        q = np.stack(q, 1)
        q[~adentro] = q[~adentro][:, ::-1]
        ok = (q >= 0).all(1)
        q = q[ok]
        caras.append(np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]))
    return V, np.concatenate(caras)


def _vecindad(T: np.ndarray, n: int) -> tuple:
    a = np.concatenate([T[:, 0], T[:, 1], T[:, 2], T[:, 1], T[:, 2], T[:, 0]])
    b = np.concatenate([T[:, 1], T[:, 2], T[:, 0], T[:, 0], T[:, 1], T[:, 2]])
    grado = np.bincount(a, minlength=n).astype(float)
    return a, b, np.maximum(grado, 1)


def suavizar_malla(V: np.ndarray, T: np.ndarray, veces: int = 8) -> np.ndarray:
    """Suavizado de Taubin (quita los escalones de la grilla sin encoger la malla)."""
    a, b, grado = _vecindad(T, len(V))
    V = V.copy()
    for _ in range(veces):
        for f in (0.5, -0.53):
            prom = np.zeros_like(V)
            np.add.at(prom, a, V[b])
            V += f * (prom / grado[:, None] - V)
    return V


def normales(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V)
    for i in range(3):
        np.add.at(N, T[:, i], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)


def _dist_seg3(P: np.ndarray, a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


SEG3 = {"pelvis": [("cadera", "cadera_izq"), ("cadera", "cadera_der")],
        "torso": [("cadera", "cuello"), ("cuello", "hombro_izq"), ("cuello", "hombro_der")]}


@dataclass
class Modelo:
    V: np.ndarray                    # vértices en reposo (n, 3)
    N: np.ndarray                    # normales en reposo
    T: np.ndarray                    # triángulos (f, 3)
    grupo: np.ndarray                # grupo de cada vértice (índice en GRUPOS3D)
    W: np.ndarray                    # pesos (n, huesos)
    J: dict                          # articulaciones 3D en reposo
    altura: float
    vistas: dict = field(repr=False, default_factory=dict)
    texturas: dict = field(repr=False, default_factory=dict)    # rol → BGRA premultiplicada float32
    profundidad: dict = field(repr=False, default_factory=dict)  # rol → mapa de profundidad en reposo
    pies: dict = field(default_factory=dict)                     # lado → índices de los vértices más bajos
    J2: dict = field(default_factory=dict)                       # esqueleto 2D del frente (para corregirlo)
    grupo_punto: np.ndarray | None = field(default=None, repr=False)   # (atlas) ¿cada punto es de la cabeza?
    huesos: list = field(default_factory=list)                   # huesos en orden (padres primero)
    padre: dict = field(default_factory=dict)
    pivote: dict = field(default_factory=dict)                   # punto de giro de cada hueso (reposo)
    ejes: dict = field(default_factory=dict)                     # dedos: (eje de cierre, eje de abrir)
    manos: dict = field(default_factory=dict)                    # manos con muñeca y falanges
    mascaras_mano: dict = field(default_factory=dict, repr=False)  # rol → dónde está el guante en esa vista
    es_mano: np.ndarray | None = field(default=None, repr=False)   # ¿el vértice es de una mano?
    color_mano: np.ndarray | None = None                          # color propio del guante (BGR)
    mano_punto: np.ndarray | None = field(default=None, repr=False)  # (atlas) ¿cada punto es de una mano?
    color_vertice: np.ndarray | None = field(default=None, repr=False)  # color de respaldo por vértice (TRELLIS)
    respaldo_punto: np.ndarray | None = field(default=None, repr=False)  # (atlas) ese color en cada punto
    origen: str = "siluetas"                                     # «siluetas» (propio) o «trellis»
    neutra: dict = field(default_factory=dict)                   # giros que llevan el dibujo a la postura de pie
    guante: dict | None = field(default=None, repr=False)        # guante propio (manos3d.parametros_guante)


def construir(vistas: dict, J2: dict | None = None, detalle: float = 1.0, progreso=None) -> Modelo:
    """Hoja ya separada (vistas) → modelo 3D con huesos y pesos."""
    fr = vistas["frente"]
    if J2 is None:
        J2 = detectar_esqueleto(fr.alfa)
    def avisar(txt):
        if progreso:
            progreso(txt)
    J3 = esqueleto_3d(vistas, J2)
    avisar("Separando el cuerpo en partes en cada vista…")
    etiq = etiquetar_vistas(vistas, J3)
    ys = np.nonzero((fr.alfa > 127).any(1))[0]
    altura = float(ys.max() - ys.min() + 1)
    from mente import manos3d
    manos = manos3d.detectar_manos(vistas, J3)
    perf = perfiles(vistas, J3, etiq, altura, manos)
    avisar("Construyendo el volumen…")
    h = altura / (200 * detalle)                                   # tamaño del vóxel
    # caja del personaje
    ufr = np.nonzero((fr.alfa > 60).any(0))[0]
    xs = (ufr.min() - fr.c - 4 * h, ufr.max() - fr.c + 4 * h)
    zs = [-0.5 * altura, 0.5 * altura]
    if "perfil_der" in vistas:
        pd = vistas["perfil_der"]
        upd = np.nonzero((pd.alfa > 60).any(0))[0]
        zs = [pd.c - upd.max() - 4 * h, pd.c - upd.min() + 4 * h]
    gx = np.arange(xs[0], xs[1], h)
    gy = np.arange(-2 * h, altura + 4 * h, h)
    gz = np.arange(zs[0], zs[1], h)
    PX, PY, PZ = np.meshgrid(gx, gy, gz, indexing="ij")
    P = np.stack([PX.ravel(), PY.ravel(), PZ.ravel()], 1)
    casco = _silueta(fr, P)
    lados = [vistas[r] for r in ("perfil_der", "perfil_izq") if r in vistas and not vistas[r].espejo]
    if lados:
        casco &= np.logical_or.reduce([_silueta(v, P) for v in lados])
    campos = {g: np.zeros(len(P), np.float32) for g in GRUPOS3D}
    for pf in perf:
        g = HUESOS[pf.hueso][4]
        padre = HUESOS[pf.hueso][2]
        une = padre is not None and HUESOS[padre][4] != g          # se une a otra parte: entra un poco en ella
        dentro = _dentro(pf, P, vistas, extender=0.3 if une else 0.0) & casco
        campos[g] = np.maximum(campos[g], dentro.astype(np.float32))
    avisar("Armando la malla…")
    Vs, Ts, Gs = [], [], []
    base = 0
    forma = (len(gx), len(gy), len(gz))
    for gi, g in enumerate(GRUPOS3D):
        F = _difuminar3(campos[g].reshape(forma), 2)
        if F.max() < 0.5:
            continue
        Vg, Tg = surface_nets(F, 0.5)
        if len(Tg) == 0:
            continue
        Vg = np.stack([xs[0] + Vg[:, 0] * h, -2 * h + Vg[:, 1] * h, zs[0] + Vg[:, 2] * h], 1)
        Vg = suavizar_malla(Vg, Tg, 6)
        Vs.append(Vg)
        Ts.append(Tg + base)
        Gs.append(np.full(len(Vg), gi))
        base += len(Vg)
    V = np.vstack(Vs)
    T = np.vstack(Ts)
    grupo = np.concatenate(Gs)
    # pesos: distancia a los huesos de su propia parte (rígido salvo cerca de los codos y rodillas)
    huesos = list(HUESOS)
    D = np.full((len(V), len(huesos)), np.inf)
    for k, hb in enumerate(huesos):
        sel = grupo == GRUPOS3D.index(HUESOS[hb][4])
        segs = SEG3.get(hb, [(HUESOS[hb][0], HUESOS[hb][1])])
        D[sel, k] = np.min([_dist_seg3(V[sel], J3[a], J3[b]) for a, b in segs], axis=0)
    banda = 0.04 * altura
    with np.errstate(invalid="ignore"):
        W = np.clip(1 - (D - D.min(axis=1, keepdims=True)) / banda, 0, 1) ** 2
    W[~np.isfinite(W)] = 0
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    m = Modelo(V, normales(V, T), T, grupo, W, J3, altura, vistas)
    m.J2 = dict(J2)
    m.neutra = postura_neutra(J3)
    m.huesos = list(ORDEN)
    m.padre = {hb: HUESOS[hb][2] for hb in ORDEN}
    m.pivote = {hb: np.asarray(J3[HUESOS[hb][0]], float) for hb in ORDEN}
    if manos:
        avisar("Armando muñecas y dedos…")
        manos3d.agregar(m, manos)
        manos3d.mascaras_por_vista(m, manos)
    for lado in ("izq", "der"):
        sel = np.flatnonzero(grupo == GRUPOS3D.index("pierna_" + lado))
        if len(sel):
            m.pies[lado] = sel[np.argsort(V[sel, 1])[:12]]
    avisar("Preparando las texturas…")
    preparar_texturas(m)
    return m


def ruta_esqueleto3d(imagen) -> "Path":
    from pathlib import Path
    imagen = Path(imagen)
    return imagen.with_name(imagen.stem + ".esqueleto3d.json")


def cargar(imagen, esqueleto=None, detalle: float = 1.0, rehacer: bool = False, progreso=None) -> tuple:
    """Imagen (hoja de personaje o una sola vista) → (modelo, vistas, ruta del esqueleto).
    El esqueleto del frente se guarda/lee en <imagen>.esqueleto3d.json (coordenadas de la vista de frente)."""
    from pathlib import Path
    from mente.animacion import cargar_esqueleto, leer_imagen
    from mente.vistas import preparar_vistas
    if progreso:
        progreso("Separando las vistas de la hoja…")
    vistas = preparar_vistas(leer_imagen(imagen))
    ruta = Path(esqueleto) if esqueleto else ruta_esqueleto3d(imagen)
    fr = vistas["frente"]
    J2 = None
    if ruta.exists() and not rehacer:
        J2 = cargar_esqueleto(ruta, fr.bgra.shape[1::-1])
    if J2 is None:
        J2 = detectar_esqueleto(fr.alfa)
    return construir(vistas, J2, detalle, progreso), vistas, ruta


# --------------------------------------------------------------------------------------------
# Rasterizador con z-buffer
# --------------------------------------------------------------------------------------------
def rasterizar(Q: np.ndarray, T: np.ndarray, ancho: int, alto: int, w: np.ndarray | None = None,
               dos_caras: bool = False) -> tuple:
    """Q: (n, 3) = x, y de pantalla (px, y hacia abajo) y profundidad (mayor = más cerca).
    Devuelve (ids (alto, ancho) con -1 = vacío, baricéntricas (alto, ancho, 3), profundidad).
    Z-buffer en una pasada: profundidad cuantizada e índice del triángulo van juntos en un entero de 64 bits
    y gana el máximo de cada píxel."""
    tri = Q[T].astype(np.float64)
    x, y, zt = tri[..., 0], tri[..., 1], tri[..., 2]
    area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    vis = (area < -1e-9) if not dos_caras else (np.abs(area) > 1e-9)
    xmin = np.ceil(x.min(1)).astype(np.int64)
    xmax = np.floor(x.max(1)).astype(np.int64)
    ymin = np.ceil(y.min(1)).astype(np.int64)
    ymax = np.floor(y.max(1)).astype(np.int64)
    vis &= (xmax >= 0) & (ymax >= 0) & (xmin < ancho) & (ymin < alto) & (xmax >= xmin) & (ymax >= ymin)
    caras = np.flatnonzero(vis)
    vacio = (np.full((alto, ancho), -1, np.int64), np.zeros((alto, ancho, 3), np.float32),
             np.full((alto, ancho), -np.inf))
    if len(caras) == 0:
        return vacio
    x, y, zt, area = x[caras], y[caras], zt[caras], area[caras]
    # coeficientes: l1 = a1*px + b1*py + c1, l2 = …, z = az*px + bz*py + cz
    inv = 1.0 / area
    a1, b1 = (y[:, 2] - y[:, 0]) * inv, -(x[:, 2] - x[:, 0]) * inv
    c1 = -(a1 * x[:, 0] + b1 * y[:, 0])
    a2, b2 = -(y[:, 1] - y[:, 0]) * inv, (x[:, 1] - x[:, 0]) * inv
    c2 = -(a2 * x[:, 0] + b2 * y[:, 0])
    az = (zt[:, 1] - zt[:, 0]) * a1 + (zt[:, 2] - zt[:, 0]) * a2
    bz = (zt[:, 1] - zt[:, 0]) * b1 + (zt[:, 2] - zt[:, 0]) * b2
    cz = zt[:, 0] + (zt[:, 1] - zt[:, 0]) * c1 + (zt[:, 2] - zt[:, 0]) * c2
    x0, x1 = np.clip(xmin[caras], 0, ancho - 1), np.clip(xmax[caras], 0, ancho - 1)
    y0, y1 = np.clip(ymin[caras], 0, alto - 1), np.clip(ymax[caras], 0, alto - 1)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    cuenta = bw * bh
    zmin, zmax = float(zt.min()), float(zt.max())
    zesc = (2 ** 22 - 2) / max(zmax - zmin, 1e-9)
    buf = np.full(alto * ancho, -1, np.int64)
    acum = np.cumsum(cuenta)
    lim = 2_000_000
    inicio = 0
    while inicio < len(caras):
        fin = int(np.searchsorted(acum, (acum[inicio - 1] if inicio else 0) + lim, side="right"))
        fin = max(fin, inicio + 1)
        cc = cuenta[inicio:fin]
        k = np.repeat(np.arange(inicio, fin), cc)
        off = np.arange(cc.sum()) - np.repeat(np.cumsum(cc) - cc, cc)
        fila, col = np.divmod(off, bw[k])
        px = (x0[k] + col).astype(np.float64)
        py = (y0[k] + fila).astype(np.float64)
        l1 = a1[k] * px + b1[k] * py + c1[k]
        l2 = a2[k] * px + b2[k] * py + c2[k]
        dentro = (l1 >= -1e-7) & (l2 >= -1e-7) & (l1 + l2 <= 1 + 1e-7)
        k, px, py = k[dentro], px[dentro], py[dentro]
        z = az[k] * px + bz[k] * py + cz[k]
        clave = (np.clip((z - zmin) * zesc, 0, 2 ** 22 - 1).astype(np.int64) << 40) | k
        np.maximum.at(buf, py.astype(np.int64) * ancho + px.astype(np.int64), clave)
        inicio = fin
    sel = np.flatnonzero(buf >= 0)
    k = buf[sel] & ((1 << 40) - 1)
    py, px = np.divmod(sel, ancho)
    l1 = a1[k] * px + b1[k] * py + c1[k]
    l2 = a2[k] * px + b2[k] * py + c2[k]
    lam = np.stack([1 - l1 - l2, l1, l2], 1)
    if w is not None:                         # perspectiva: baricéntricas corregidas
        wi = w[T[caras[k]]]
        lam = lam / wi
        lam /= lam.sum(1, keepdims=True)
    ids = vacio[0].ravel()
    ids[sel] = caras[k]
    L = vacio[1].reshape(-1, 3)
    L[sel] = lam
    Z = vacio[2].ravel()
    Z[sel] = az[k] * px + bz[k] * py + cz[k]
    return ids.reshape(alto, ancho), L.reshape(alto, ancho, 3), Z.reshape(alto, ancho)


def _interp(lam: np.ndarray, tv: np.ndarray, A: np.ndarray) -> np.ndarray:
    return lam[:, 0:1] * A[tv[:, 0]] + lam[:, 1:2] * A[tv[:, 1]] + lam[:, 2:3] * A[tv[:, 2]]


# --------------------------------------------------------------------------------------------
# 5. Texturas desde las vistas
# --------------------------------------------------------------------------------------------
def preparar_texturas(m: Modelo) -> None:
    for rol, v in m.vistas.items():
        a = v.bgra[..., 3:4].astype(np.float32) / 255
        m.texturas[rol] = np.dstack([v.bgra[..., :3].astype(np.float32) * a, a * 255]).astype(np.float32)
        h, w = v.alfa.shape
        u, vv = v.proyectar(m.V)
        Q = np.stack([u, vv, v.profundidad(m.V)], 1)
        _, _, z = rasterizar(Q, m.T, w, h)
        z = np.where(np.isfinite(z), z, -1e9).astype(np.float32)
        m.profundidad[rol] = cv2.dilate(z, np.ones((3, 3), np.uint8))   # tolerante en los bordes


def _muestrear(img: np.ndarray, u: np.ndarray, v: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
    n = len(u)
    ancho = 1024
    filas = (n + ancho - 1) // ancho
    mu = np.full(filas * ancho, -10, np.float32)
    mv_ = np.full(filas * ancho, -10, np.float32)
    mu[:n], mv_[:n] = u, v
    out = cv2.remap(img, mu.reshape(filas, ancho), mv_.reshape(filas, ancho), interp,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape(filas * ancho, -1)[:n]


NITIDEZ = 10.0          # qué tan rápido pasa de una vista a otra al girar la cámara
NITIDEZ_CABEZA = 28.0   # en la cabeza, casi un interruptor: una sola vista pinta toda la cara


def _suave_paso(x: np.ndarray, a: float, b: float) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def colorear(m: Modelo, P: np.ndarray, N: np.ndarray, C: np.ndarray | None = None,
             cabeza: np.ndarray | None = None, mano: np.ndarray | None = None,
             respaldo: np.ndarray | None = None) -> np.ndarray:
    """Color (BGR recto + alfa 255) de puntos de la superficie en reposo P con normales N.
    Con C (dirección hacia la cámara de cada punto, en el espacio de reposo) manda la vista que más se parece a
    lo que ve la cámara: de frente, el dibujo del frente; de perfil, el del perfil (así no se suman los ojos del
    frente y el del perfil). Sin C (atlas del .glb) mandan el frente y la espalda; los perfiles rellenan los
    costados que ellos no ven. Con prueba de profundidad: una vista no pinta lo que tiene tapado.
    En la cabeza (cabeza=True) y con cámara, UNA sola vista pinta toda la cara (la más parecida a la cámara, sin
    prueba de profundidad): los rasgos de vistas distintas no caen en el mismo sitio del volumen y, mezclados,
    darían ojos de más."""
    if cabeza is None or C is None:
        cabeza = np.zeros(len(P), bool)
    if mano is None:
        mano = m.mano_punto if m.mano_punto is not None else np.zeros(len(P), bool)
    hay_mano = bool(mano.any()) and bool(m.mascaras_mano)

    def en_guante(rol, idx, u, vv):
        """1 fuera de las manos; dentro, 1 solo si ese píxel de la vista es del guante."""
        if not hay_mano or rol not in m.mascaras_mano:
            return 1.0
        mk = m.mascaras_mano[rol]
        ui = np.clip(np.round(u).astype(int), 0, mk.shape[1] - 1)
        vi = np.clip(np.round(vv).astype(int), 0, mk.shape[0] - 1)
        return np.where(mano[idx], mk[vi, ui], 1.0)
    tol = m.altura / 110
    n = len(P)
    acum = np.zeros((n, 4), np.float32)
    peso = np.zeros(n, np.float32)
    P = P.astype(np.float32)
    cubre = np.zeros(n, np.float32)                       # cuánto cubren frente y espalda (modo atlas)
    orden = sorted(m.vistas, key=lambda r: r.startswith("perfil"))
    for rol in orden:
        v = m.vistas[rol]
        d = DIRECCION[rol].astype(np.float32)
        coseno = N @ d
        idx = np.flatnonzero(coseno > np.where(cabeza, -0.3, 0.02))   # lo que esta vista mira de frente
        if len(idx) == 0:
            continue
        Pi = P[idx]
        u, vv = v.proyectar(Pi)
        col = _muestrear(m.texturas[rol], u, vv)
        zmap = _muestrear(m.profundidad[rol], u, vv, cv2.INTER_NEAREST)[:, 0]
        cab = cabeza[idx]
        visible = ((v.profundidad(Pi) >= zmap - tol) | cab) * (col[:, 3] / 255) * (0.6 if v.espejo else 1.0) \
            * en_guante(rol, idx, u, vv)
        if C is not None:
            alin = C[idx] @ d - 1
            w = np.where(cab, (0.3 + np.clip(coseno[idx], 0, None)) * np.exp(NITIDEZ_CABEZA * alin),
                         np.clip(coseno[idx], 0, None) ** 2 * np.exp(NITIDEZ * alin))
            if rol.startswith("perfil"):
                # con la cámara más de frente (o de espalda), el perfil pinta solo el costado de la cabeza
                # (audífonos, oreja), que el frente vería estirado; la cara sigue siendo del frente
                Ni, Ci = N[idx], C[idx]
                relleno = _suave_paso(coseno[idx], 0.55, 0.8) * (1 - _suave_paso(np.abs(Ni[:, 2]), 0.12, 0.3))
                w = w + np.where(cab & (np.abs(Ci[:, 2]) > np.abs(Ci[:, 0])), relleno, 0)
            w = w * visible
        elif rol.startswith("perfil"):
            w = _suave_paso(coseno[idx], 0.1, 0.4) * visible * (1 - cubre[idx])
        else:                          # en la cara, el frente cubre hasta casi el costado (sin ojos del perfil)
            w = np.where(m.grupo_punto[idx] if m.grupo_punto is not None else False,
                         _suave_paso(coseno[idx], 0.03, 0.18), _suave_paso(coseno[idx], 0.12, 0.35)) * visible
            cubre[idx] = np.maximum(cubre[idx], w)
        acum[idx] += w[:, None] * col
        peso[idx] += w
    if respaldo is None:
        respaldo = m.respaldo_punto
    if respaldo is not None:                             # lo que ningún dibujo ve: el color propio del modelo
        falta = np.flatnonzero(peso < 1e-7)
        acum[falta] = np.hstack([respaldo[falta], np.full((len(falta), 1), 255.0, np.float32)])
        peso[falta] = 1.0
    for holgura in (4.0, None):                          # nadie lo ve bien (axilas, entrepierna, coronilla…)
        falta = np.flatnonzero(peso < 1e-7)
        if len(falta) == 0:
            break
        Pf, Nf = P[falta], N[falta]
        for rol, v in m.vistas.items():
            u, vv = v.proyectar(Pf)
            col = _muestrear(m.texturas[rol], u, vv)
            wr = np.clip(Nf @ DIRECCION[rol].astype(np.float32) + 0.6, 0, None) ** 2 * (col[:, 3] / 255)
            if holgura:                                  # primero, solo lo que esa vista casi ve
                zmap = _muestrear(m.profundidad[rol], u, vv, cv2.INTER_NEAREST)[:, 0]
                wr = wr * (v.profundidad(Pf) >= zmap - holgura * tol)
            wr = wr * en_guante(rol, falta, u, vv)
            acum[falta] += wr[:, None] * col
            peso[falta] += wr
    sin_color = np.flatnonzero(mano & (peso < 1e-7))
    if len(sin_color) and m.color_mano is not None:      # la mano que nadie ve: el color del guante
        acum[sin_color] = np.append(m.color_mano, 255.0)
        peso[sin_color] = 1.0
    col = acum / np.maximum(peso, 1e-9)[:, None]
    # opaco por dentro: el color se des-premultiplica y el alfa lo da la cobertura
    a = np.maximum(col[:, 3:4], 1e-3)
    rgb = np.where(a > 1, col[:, :3] * 255 / a, 0)
    from mente import manos3d
    if manos3d.GUANTE_PROPIO and mano.any():             # 23-oct: la mano, con el guante propio (no el dibujo)
        idx = np.flatnonzero(mano)
        gc = manos3d.color_guante(m, P[idx], N[idx])
        if gc is not None:
            rgb[idx] = gc
    return np.hstack([rgb, np.full((n, 1), 255, np.float32)])


# --------------------------------------------------------------------------------------------
# 6. Pose 3D
# --------------------------------------------------------------------------------------------
def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotacion(hb: str, valor) -> np.ndarray:
    """(adelante, afuera, giro) en grados → matriz 3x3. Adelante: el extremo va hacia donde mira el
    personaje; afuera: se aleja del cuerpo (en el torso y la cabeza: inclinarse hacia +X); giro: sobre su eje."""
    if not isinstance(valor, (tuple, list)):
        valor = (valor, 0.0, 0.0)
    ad, af, gi = (math.radians(float(x)) for x in (list(valor) + [0, 0])[:3])
    arriba = hb in ARRIBA
    lado = _lado(hb)
    pitch = ad if arriba else -ad
    roll = -af if arriba or lado is None else (af if lado == "der" else -af)
    return _rz(roll) @ _rx(pitch) @ _ry(gi)


def _m4(R=None, t=None):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    if t is not None:
        M[:3, 3] = t
    return M


def huesos_de(m: Modelo) -> list:
    return m.huesos or list(ORDEN)


def postura_neutra(J3: dict) -> dict:
    """Si el dibujo trae los brazos abiertos (pose A o T) o las piernas muy separadas, los giros que los llevan
    a la postura de pie de la que parten los movimientos (caminar con los brazos abajo, no en cruz)."""
    neutra = {}
    for lado, signo in (("izq", -1.0), ("der", 1.0)):
        v = np.asarray(J3["codo_" + lado], float) - np.asarray(J3["hombro_" + lado], float)
        abierto = math.degrees(math.atan2(signo * v[0], -v[1]))          # hacia afuera, desde «abajo»
        if abierto > 18:
            neutra["brazo_" + lado] = (0.0, -(abierto - 10.0), 0.0)
        v = np.asarray(J3["rodilla_" + lado], float) - np.asarray(J3["cadera_" + lado], float)
        abierto = math.degrees(math.atan2(signo * v[0], -v[1]))
        if abierto > 8:
            neutra["muslo_" + lado] = (0.0, -(abierto - 3.0), 0.0)
    return neutra


# Extremidades con ejes anatómicos: el hombro y la cadera giran en el marco del torso, pero el giro sobre su eje
# va por el eje del hueso; el codo y la rodilla son bisagras (solo se doblan, en un plano, dentro de sus límites).
RAIZ_EXTREMIDAD = {"brazo": ("hombro", "codo"), "muslo": ("cadera", "rodilla")}
EXTREMIDAD = {"brazo": "brazo", "antebrazo": "brazo", "muneca": "brazo", "muslo": "muslo", "pierna": "muslo"}
BISAGRAS = {"antebrazo": (-5.0, 150.0), "pierna": (-150.0, 5.0)}        # «adelante» permitido, en grados


def _rotacion_minima(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """La rotación más corta que lleva la dirección a a la b (vectores unitarios)."""
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999999:
        return _rx(math.pi)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def marco_extremidad(m: Modelo, raiz: str, lado: str) -> tuple:
    """(N, H): N lleva el hueso raíz (brazo o muslo) del dibujo a la postura neutra; H lleva «colgando derecho»
    (−Y) a esa dirección neutra. En el marco H, X es el eje de la bisagra del codo o la rodilla."""
    a, b = RAIZ_EXTREMIDAD[raiz]
    u = np.asarray(m.J[f"{b}_{lado}"], float) - np.asarray(m.J[f"{a}_{lado}"], float)
    u /= max(float(np.linalg.norm(u)), 1e-9)
    n = (getattr(m, "neutra", None) or {}).get(f"{raiz}_{lado}")
    N = rotacion(f"{raiz}_{lado}", n) if n is not None else np.eye(3)
    return N, _rotacion_minima(np.array([0.0, -1.0, 0.0]), N @ u)


def rotacion_local(m: Modelo, hb: str, valor) -> np.ndarray:
    """Giro del hueso en su articulación: primero a la postura neutra, luego el del movimiento.
    En brazos y piernas, «giro» > 0 es rotación hacia adentro en los dos lados."""
    from mente import manos3d
    if hb in m.ejes:
        return manos3d.rotacion_dedo(m, hb, valor if valor is not None else 0.0)
    valor = valor if valor is not None else (0, 0, 0)
    lado = _lado(hb)
    parte = hb[:-4] if lado else hb
    if parte not in EXTREMIDAD:
        R = rotacion(hb, valor)
        n = getattr(m, "neutra", None) or {}
        return R @ rotacion(hb, n[hb]) if hb in n else R
    if not isinstance(valor, (tuple, list)):
        valor = (valor, 0.0, 0.0)
    ad, af, gi = (float(x) for x in (list(valor) + [0, 0])[:3])
    if lado == "der":
        gi = -gi
    raiz = EXTREMIDAD[parte]
    N, H = marco_extremidad(m, raiz, lado)
    if parte == raiz:                     # hombro / cadera: el giro, sobre el eje del propio hueso
        return rotacion(hb, (ad, af, 0.0)) @ H @ _ry(math.radians(gi)) @ H.T @ N
    if parte in BISAGRAS:                 # codo / rodilla: solo se doblan (no hay «afuera»)
        ad, af = float(np.clip(ad, *BISAGRAS[parte])), 0.0
    F = N.T @ H                           # ejes anatómicos de la extremidad, en coordenadas del dibujo
    return F @ rotacion(hb, (ad, af, gi)) @ F.T


def matrices3d(m: Modelo, pose: dict) -> dict:
    from mente import manos3d
    dibujo = bool(pose.get("dibujo"))                # ejes del dibujo, sin postura neutra ni bisagras
    pose = manos3d.expandir(m, pose)
    M = {}
    for hb in huesos_de(m):
        padre = m.padre[hb] if m.padre else HUESOS[hb][2]
        Mp = M[padre] if padre else np.eye(4)
        p = m.pivote[hb] if m.pivote else np.asarray(m.J[HUESOS[hb][0]], float)
        valor = pose.get(hb)
        if isinstance(valor, np.ndarray) and valor.shape == (3, 3):   # captura real (mente/mocap.py): el giro
            R = valor                                                  # ya viene hecho, en el espacio del dibujo
        elif dibujo and hb not in m.ejes:
            R = rotacion(hb, valor if valor is not None else (0, 0, 0))
        else:
            R = rotacion_local(m, hb, valor)
        M[hb] = Mp @ _m4(t=p) @ _m4(R) @ _m4(t=-p)
    return M


@dataclass
class Pose3D:
    V: np.ndarray                    # vértices posados (n, 3)
    J: dict                          # articulaciones posadas
    G: np.ndarray                    # transformación global (4x4) aplicada después de los huesos
    R: np.ndarray | None = None      # rotación de cada vértice (n, 3, 3): de reposo a la pose


def _posar_vertices(m: Modelo, M: dict, idx=None) -> np.ndarray:
    Ms = np.stack([M[hb] for hb in huesos_de(m)])[:, :3, :]
    V = m.V if idx is None else m.V[idx]
    W = m.W if idx is None else m.W[idx]
    Vh = np.hstack([V, np.ones((len(V), 1))])
    return np.einsum("nb,bij,nj->ni", W, Ms, Vh)


def transformacion_global(m: Modelo, pose: dict, M: dict | None = None) -> np.ndarray:
    """4x4 que se aplica después de los huesos: pies al suelo, estirar, rumbo y desplazamiento.
    Solo posa los vértices de los pies (rápido)."""
    M = M or matrices3d(m, pose)
    apoyo = float(pose.get("apoyo", 1.0))
    dy = 0.0
    if apoyo > 0 and m.pies:
        idx = np.concatenate(list(m.pies.values()))
        dy = -apoyo * float(_posar_vertices(m, M, idx)[:, 1].min())
    cad = np.asarray(m.J["cadera"], float)
    e = float(pose.get("estirar", 1.0))
    G = _m4(t=[0, dy, 0])
    if e != 1.0:
        k = 1 / math.sqrt(e)
        G = _m4(t=[cad[0], 0, cad[2]]) @ np.diag([k, e, k, 1.0]) @ _m4(t=[-cad[0], 0, -cad[2]]) @ G
    rumbo = float(pose.get("rumbo", 0.0))
    if rumbo:
        G = _m4(t=[cad[0], 0, cad[2]]) @ _m4(_ry(math.radians(rumbo))) @ _m4(t=[-cad[0], 0, -cad[2]]) @ G
    return _m4(t=[float(pose.get("x", 0)) * m.altura, float(pose.get("y", 0)) * m.altura,
                  float(pose.get("z", 0)) * m.altura]) @ G


def posar3d(m: Modelo, pose: dict, idx=None) -> Pose3D:
    """Vértices posados (todos, o solo los de `idx`), articulaciones y transformación global."""
    M = matrices3d(m, pose)
    V = _posar_vertices(m, M, idx)
    Jp = {}
    for j in ARTICULACIONES:
        hb = next((h_ for h_, d in HUESOS.items() if d[0] == j), None) or \
            next(h_ for h_, d in HUESOS.items() if d[1] == j)
        Jp[j] = (M[hb] @ np.append(m.J[j], 1.0))[:3]
    G = transformacion_global(m, pose, M)
    V = V @ G[:3, :3].T + G[:3, 3]
    Jp = {k: G[:3, :3] @ v + G[:3, 3] for k, v in Jp.items()}
    R = None
    if idx is None:
        Ms = np.stack([G[:3, :3] @ M[hb][:3, :3] for hb in huesos_de(m)])
        R = np.einsum("nb,bij->nij", m.W, Ms)
    return Pose3D(V, Jp, G, R)


# --------------------------------------------------------------------------------------------
# 7. Cámara y render
# --------------------------------------------------------------------------------------------
@dataclass
class Camara:
    azimut: float = 0.0              # 0 = de frente; 90 = desde la derecha de la imagen (+X)
    elevacion: float = 6.0           # grados por encima
    fov: float = 18.0                # 0 = ortográfica
    objetivo: tuple = (0.0, 0.0, 0.0)

    def ejes(self) -> tuple:
        a, e = math.radians(self.azimut), math.radians(self.elevacion)
        d = np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
        r = np.array([math.cos(a), 0.0, -math.sin(a)])
        u = np.cross(d, r)
        return r, u, d

    def posicion(self, altura: float) -> np.ndarray | None:
        if self.fov <= 0:
            return None
        dist = 0.5 * altura / math.tan(math.radians(self.fov) / 2) + 0.5 * altura
        return np.asarray(self.objetivo, float) + dist * self.ejes()[2]

    def hacia_camara(self, P: np.ndarray, altura: float) -> np.ndarray:
        """Dirección unitaria desde cada punto (mundo) hacia la cámara."""
        pos = self.posicion(altura)
        if pos is None:
            return np.broadcast_to(self.ejes()[2], P.shape).copy()
        D = pos - P
        return D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)

    def proyectar(self, P: np.ndarray, escala: float, altura: float) -> tuple:
        """(x, y, profundidad, w) en px relativos al objetivo (y hacia abajo)."""
        r, u, d = self.ejes()
        Q = P - np.asarray(self.objetivo, float)
        xc, yc, zc = Q @ r, Q @ u, Q @ d
        if self.fov > 0:
            dist = 0.5 * altura / math.tan(math.radians(self.fov) / 2) + 0.5 * altura
            w = (dist - zc) / dist
        else:
            w = np.ones_like(zc)
        return escala * xc / w, -escala * yc / w, zc, w


def renderizar3d(m: Modelo, pose: Pose3D, cam: Camara, escala: float, origen: tuple, lienzo: tuple,
                 ss: int = 2, luz: float = 0.18) -> np.ndarray:
    """BGRA uint8 (alfa recta) del tamaño lienzo=(ancho, alto). Posición: proyección*escala - origen."""
    ancho, alto = lienzo
    x, y, z, w = cam.proyectar(pose.V, escala, m.altura)
    Q = np.stack([(x - origen[0] + 0.5) * ss - 0.5, (y - origen[1] + 0.5) * ss - 0.5, z], 1)
    ids, lam, _ = rasterizar(Q, m.T, ancho * ss, alto * ss, w if cam.fov > 0 else None)
    sel = ids >= 0
    out = np.zeros((alto * ss, ancho * ss, 4), np.float32)
    if sel.any():
        f = ids[sel]
        l = lam[sel]
        tv = m.T[f]
        P = _interp(l, tv, m.V)
        N = _interp(l, tv, m.N)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        C = None
        if pose.R is not None:
            Cv = cam.hacia_camara(pose.V, m.altura)
            Cv = np.einsum("nji,nj->ni", pose.R, Cv)                 # Rᵀ · c: al espacio de reposo
            C = _interp(l, tv, Cv)
            C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
        col = colorear(m, P, N, C, m.grupo[tv[:, 0]] == GRUPOS3D.index("cabeza"),
                       m.es_mano[tv[:, 0]] if m.es_mano is not None else None,
                       _interp(l, tv, m.color_vertice) if m.color_vertice is not None else None)
        if luz:
            Np = normales(pose.V, m.T)
            Nq = _interp(l, tv, Np)
            Nq /= np.maximum(np.linalg.norm(Nq, axis=1, keepdims=True), 1e-9)
            r_, u_, d_ = cam.ejes()
            L = 0.6 * d_ + 0.55 * u_ - 0.35 * r_
            L /= np.linalg.norm(L)
            sombra = (1 - luz) + luz * np.clip(Nq @ L, 0, 1) * 1.25
            if m.es_mano is not None:                    # el guante propio no trae sombras pintadas: más volumen
                from mente import manos3d
                if manos3d.GUANTE_PROPIO:
                    mano = m.es_mano[tv[:, 0]]
                    lam = np.clip(Nq @ L, 0, 1)
                    tinte = manos3d.SOMBRA_GUANTE[None, :] + (1 - manos3d.SOMBRA_GUANTE[None, :]) * lam[:, None]
                    col[mano, :3] *= tinte[mano]                     # sombra lavanda en el guante
                    sombra = np.where(mano, 0.68 + 0.36 * lam, sombra)
            col[:, :3] *= np.clip(sombra, 0, 1.2)[:, None]
        out[sel] = col
    if ss > 1:
        out = cv2.resize(out, (ancho, alto), interpolation=cv2.INTER_AREA)
    a = out[..., 3:4]
    # el color ya viene recto; la cobertura del supermuestreo lo premultiplicó
    rgb = np.where(a > 0.5, out[..., :3] * 255 / np.maximum(a, 1e-3), 0)
    return np.dstack([np.clip(rgb, 0, 255), np.clip(a, 0, 255)]).astype(np.uint8)
