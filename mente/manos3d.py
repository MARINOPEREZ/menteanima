# -*- coding: utf-8 -*-
"""Manos explícitas para MenteAnima 3D: muñeca y falanges que se doblan.

En la vista donde se ve el dorso de cada mano (el perfil de ese lado) se busca el guante por su color, la muñeca
donde termina la manga y cada dedo por las protuberancias del borde de la mano (nudillos o puntas). Cada dedo
recibe 3 falanges (nudillo, articulación media y última) que se doblan hacia la palma («cierre») y se abren en
abanico («abrir»). Si la mano no se distingue (mismo color que la manga) o no hay perfiles, no se agregan dedos.
Las falanges solo doblan lo que el dibujo muestra: abrir un puño dibujado no inventa una palma nueva.
"""
from __future__ import annotations

import math
import warnings

import cv2
import numpy as np

FALANGES = (0.0, 0.45, 0.75)       # nudillo, articulación media y última (fracción del largo del dedo)
REPARTO = (0.4, 0.35, 0.25)        # cuánto del cierre se dobla en cada articulación
LARGO_DEDO = 0.55                  # largo del dedo respecto del largo de la mano (muñeca → punta)


def _suave(x, a, b):
    t = np.clip((np.asarray(x, float) - a) / max(b - a, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _vista_mano(vistas: dict, lado: str):
    """El perfil de ese lado (ve el dorso de la mano); si no hay perfiles, nada."""
    v = vistas.get("perfil_" + lado)
    if v is None:
        return None
    return v


def detectar(v, A3: np.ndarray, B3: np.ndarray) -> dict | None:
    """Mano en la vista v, con el antebrazo de A3 (codo) a B3 (mano) en 3D."""
    a2 = np.array(v.proyectar(np.asarray(A3, float)), float)
    b2 = np.array(v.proyectar(np.asarray(B3, float)), float)
    L = float(np.linalg.norm(b2 - a2))
    if L < 8:
        return None
    d = (b2 - a2) / L
    n = np.array([-d[1], d[0]])
    h, w = v.alfa.shape
    R = int(1.0 * L)
    x0, y0 = max(0, int(b2[0]) - R), max(0, int(b2[1]) - R)
    x1, y1 = min(w, int(b2[0]) + R), min(h, int(b2[1]) + R)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    sub = np.ascontiguousarray(v.bgra[y0:y1, x0:x1, :3])
    alfa = v.alfa[y0:y1, x0:x1]
    lab = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB).astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cerca = ((xx - b2[0]) ** 2 + (yy - b2[1]) ** 2 < (0.35 * L) ** 2) & (alfa > 200)
    if cerca.sum() < 30:
        return None
    pix = lab[cerca].reshape(-1, 3)
    k = 3 if len(pix) >= 30 else 1
    _, etq, centros = cv2.kmeans(pix, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3,
                                 cv2.KMEANS_PP_CENTERS)
    cuenta = np.bincount(etq.ravel(), minlength=k)
    c0 = centros[int(np.argmax(cuenta))]
    if cuenta.max() < 0.3 * len(pix):
        return None
    parecido = np.linalg.norm(lab - c0, axis=2) < 26
    # relleno que sigue degradados suaves (la sombra de los nudillos) y se detiene en bordes fuertes
    lab8 = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB)
    relleno = np.zeros((lab8.shape[0] + 2, lab8.shape[1] + 2), np.uint8)
    sy, sx = np.nonzero(cerca & parecido)
    for j in np.linspace(0, len(sx) - 1, min(25, len(sx))).astype(int):
        if not relleno[sy[j] + 1, sx[j] + 1]:
            cv2.floodFill(lab8, relleno, (int(sx[j]), int(sy[j])), 0, (9, 7, 7), (9, 7, 7),
                          4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    crudo = (parecido | (relleno[1:-1, 1:-1] > 0)) & (alfa > 127)
    ker = max(3, int(0.09 * L)) | 1
    m = cv2.morphologyEx(crudo.astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ker, ker)))   # tapa costuras y rayas
    ncomp, lbl = cv2.connectedComponents(m)
    if ncomp <= 1:
        return None
    pys, pxs = np.nonzero(m)
    i = int(np.argmin((pxs + x0 - b2[0]) ** 2 + (pys + y0 - b2[1]) ** 2))
    mano = lbl == lbl[pys[i], pxs[i]]
    ys, xs = np.nonzero(mano)
    P = np.stack([xs + x0, ys + y0], 1) - a2
    e, s = P @ d, P @ n
    e_w, e_t = float(np.percentile(e, 2)), float(e.max())
    largo = e_t - e_w
    area = len(xs)
    # la «mano» no puede subir por la manga ni ser diminuta o enorme
    if e_w < 0.35 * L or largo < 0.18 * L or not 0.04 * L * L < area < 3.0 * L * L:
        return None
    mascara = np.zeros((h, w), bool)
    mascara[ys + y0, xs + x0] = True
    # dedos: protuberancias del borde más alejado de la muñeca (sobre la máscara sin cerrar: el cierre taparía
    # las muescas entre nudillos)
    abierto = mano & crudo
    ys2, xs2 = np.nonzero(abierto)
    P2 = np.stack([xs2 + x0, ys2 + y0], 1) - a2
    e, s = P2 @ d, P2 @ n
    s0, s1 = int(math.floor(s.min())), int(math.ceil(s.max()))
    emax = np.full(s1 - s0 + 1, -np.inf)
    np.maximum.at(emax, np.round(s - s0).astype(int), e)
    ok = np.isfinite(emax)
    emax = np.interp(np.arange(len(emax)), np.flatnonzero(ok), emax[ok])
    sig = max(1.0, 0.025 * L)
    k1 = np.exp(-0.5 * (np.arange(-3 * int(sig) - 1, 3 * int(sig) + 2) / sig) ** 2)
    suave = np.convolve(np.pad(emax, len(k1) // 2, mode="edge"), k1 / k1.sum(), mode="valid")[:len(emax)]
    ancho = len(suave)
    rad = max(2, int(0.07 * ancho))
    candidatos = [j for j in range(len(suave)) if suave[j] >= suave[max(0, j - rad):j + rad + 1].max() - 1e-6
                  and suave[j] > e_w + 0.6 * largo]
    picos = []
    for j in candidatos:                                    # prominencia: lo que baja antes del vecino más alto
        izq = suave[:j][::-1]
        der = suave[j + 1:]
        def valle(lado_):
            vmin = suave[j]
            for x in lado_:
                if x > suave[j]:
                    break
                vmin = min(vmin, x)
            return vmin
        prom = suave[j] - max(valle(izq), valle(der))
        if prom >= max(1.5, 0.035 * largo) and (not picos or j - picos[-1] > rad):
            picos.append(j)
    dedos = []
    # 1) pliegues oscuros entre los dedos, cerca del borde (un puño dibujado: nudillos separados por rayas)
    lab_v = cv2.cvtColor(np.ascontiguousarray(v.bgra[..., :3]), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    s_vals = np.arange(s0, s1 + 1, dtype=float)
    filas = []
    for f in np.linspace(0.1, 0.3, 7):
        q = a2[None] + (e_t - f * largo) * d[None] + s_vals[:, None] * n[None]
        xi = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
        filas.append(np.where(mascara[yi, xi], lab_v[yi, xi], np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        luz = np.nanmedian(np.array(filas), axis=0)
    validos = np.isfinite(luz)
    limites = []
    if validos.sum() > 10:
        luz = np.interp(np.arange(len(luz)), np.flatnonzero(validos), luz[validos])
        luz = np.convolve(np.pad(luz, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        borde = max(2, int(0.06 * len(luz)))
        rad2 = max(2, int(0.08 * len(luz)))
        for j in range(borde, len(luz) - borde):
            ventana = luz[max(0, j - rad2):j + rad2 + 1]
            if luz[j] <= ventana.min() + 1e-6:
                alto_izq, alto_der = luz[max(0, j - 3 * rad2):j].max(), luz[j + 1:j + 3 * rad2 + 1].max()
                if min(alto_izq, alto_der) - luz[j] >= 22 and (not limites or j - limites[-1] > 2 * rad2):
                    limites.append(j)
    if 1 <= len(limites) <= 4:
        cortes = [0] + limites + [len(s_vals) - 1]
        anchos_ = np.diff(cortes)
        for q in range(len(cortes) - 1):
            if anchos_[q] < 0.4 * np.median(anchos_):
                continue                         # un borde demasiado angosto no es un dedo
            j0, j1 = cortes[q], cortes[q + 1]
            jm = j0 + int(np.argmax(suave[j0:j1 + 1])) if j1 > j0 else j0
            dedos.append({"s": float(s_vals[(j0 + j1) // 2]), "e": float(suave[min(jm, len(suave) - 1)]),
                          "s0": float(s_vals[j0]), "s1": float(s_vals[j1])})
    # 2) si no hay pliegues: protuberancias del contorno (una mano abierta, dedos separados)
    if len(dedos) < 2 and 2 <= len(picos) <= 6:
        dedos = []
        picos = sorted(sorted(picos, key=lambda j: -suave[j])[:5])
        lims = [0] + [picos[q] + int(np.argmin(suave[picos[q]:picos[q + 1] + 1])) for q in range(len(picos) - 1)] \
            + [ancho - 1]
        for q, j in enumerate(picos):
            dedos.append({"s": j + s0, "e": float(suave[j]), "s0": lims[q] + s0, "s1": lims[q + 1] + s0})
    if len(dedos) < 2:
        dedos = []
    color = cv2.cvtColor(np.uint8([[np.clip(c0, 0, 255)]]), cv2.COLOR_LAB2BGR)[0, 0].astype(np.float32)
    return {"vista": v, "a2": a2, "d": d, "n": n, "L": L, "t_muneca": e_w / L, "t_punta": e_t / L,
            "e_w": e_w, "e_t": e_t, "dedos": dedos, "mascara": mascara, "color": color}


def detectar_manos(vistas: dict, J3: dict) -> dict:
    sal = {}
    for lado in ("izq", "der"):
        v = _vista_mano(vistas, lado)
        if v is None:
            continue
        info = detectar(v, J3["codo_" + lado], J3["mano_" + lado])
        if info:
            sal[lado] = info
    return sal


def _a_mundo(info: dict, e: float, s: float, A3: np.ndarray, B3: np.ndarray) -> np.ndarray:
    """Punto de la vista de la mano (a lo largo e, a lo ancho s) → punto 3D sobre la mano."""
    v = info["vista"]
    q = info["a2"] + e * info["d"] + s * info["n"]
    t = e / info["L"]
    X = A3[0] + t * (B3[0] - A3[0])
    Y = v.suelo - q[1]
    hz = q[0] - v.c
    Z = -hz if v.rol == "perfil_der" else hz
    return np.array([X, Y, Z])


def _rot_eje(eje: np.ndarray, grados: float) -> np.ndarray:
    a = math.radians(grados)
    k = np.asarray(eje, float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def mascaras_por_vista(m, manos: dict) -> None:
    """La mano (el guante) en cada vista: la mano solo toma color de sus propios píxeles (al girar se ve su
    parte de abajo, que el dibujo no muestra, y si no tomaría el color del pantalón que tiene detrás)."""
    from mente.modelo3d import GRUPOS3D
    m.mascaras_mano = {}
    for rol, v in m.vistas.items():
        union = np.zeros(v.alfa.shape, bool)
        for lado in manos:
            info = detectar(v, m.J["codo_" + lado], m.J["mano_" + lado])
            if info is not None:
                union |= cv2.dilate(info["mascara"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        m.mascaras_mano[rol] = union
    m.color_mano = np.mean([info["color"] for info in manos.values()], axis=0)
    m.es_mano = np.zeros(len(m.V), bool)
    for lado, info in manos.items():
        A3, B3 = np.asarray(m.J["codo_" + lado], float), np.asarray(m.J["mano_" + lado], float)
        AB = B3 - A3
        sel = np.flatnonzero(m.grupo == GRUPOS3D.index("brazo_" + lado))
        t = ((m.V[sel] - A3) @ AB) / max(float(AB @ AB), 1e-9)
        m.es_mano[sel[t > info["t_muneca"] - 0.02]] = True


def agregar(m, manos: dict) -> None:
    """Agrega al modelo los huesos de muñeca y falanges, y reparte los pesos de la mano entre ellos."""
    from mente.modelo3d import GRUPOS3D
    for lado, info in manos.items():
        A3, B3 = np.asarray(m.J["codo_" + lado], float), np.asarray(m.J["mano_" + lado], float)
        AB = B3 - A3
        t_w = info["t_muneca"]
        muneca = "muneca_" + lado
        m.huesos.append(muneca)
        m.padre[muneca] = "antebrazo_" + lado
        m.pivote[muneca] = A3 + t_w * AB
        largo = info["e_t"] - info["e_w"]
        Lf = LARGO_DEDO * largo
        palma = np.array([-1.0 if lado == "der" else 1.0, 0.0, 0.0])          # la palma mira hacia el cuerpo
        dedos3 = []
        for dd in info["dedos"]:
            base = dd["e"] - Lf
            juntas = [_a_mundo(info, base + f * Lf, dd["s"], A3, B3) for f in FALANGES]
            punta = _a_mundo(info, dd["e"], dd["s"], A3, B3)
            dedos3.append((dd, base, juntas, punta))
        dedos3.sort(key=lambda x: -x[3][2])                                    # de adelante hacia atrás
        if not dedos3:
            m.manos[lado] = {"info": info, "falanges": []}
        else:
            # el pulgar: el de más adelante si no llega tan lejos como los demás
            pulgar = 0 if len(dedos3) >= 3 and dedos3[0][0]["e"] < info["e_t"] - 0.18 * largo else None
            indice = 1 if pulgar == 0 else 0
            falanges = []
            for i, (dd, base, juntas, punta) in enumerate(dedos3):
                vdir = punta - juntas[0]
                vdir /= max(np.linalg.norm(vdir), 1e-9)
                flex = np.cross(vdir, palma)
                flex /= max(np.linalg.norm(flex), 1e-9)
                nombres = []
                padre = muneca
                for k in range(3):
                    hb = f"dedo{i + 1}_{lado}_{k + 1}"
                    m.huesos.append(hb)
                    m.padre[hb] = padre
                    m.pivote[hb] = juntas[k]
                    m.ejes[hb] = (flex, palma)
                    nombres.append(hb)
                    padre = hb
                falanges.append(nombres)
            centro = (len(dedos3) - 1) / 2
            m.manos[lado] = {"info": info, "falanges": falanges, "indice": indice, "pulgar": pulgar,
                             "pos": [(i - centro) / max(centro, 1) for i in range(len(dedos3))],
                             "dedos": [(dd, base) for dd, base, _, _ in dedos3]}
    # pesos
    nuevos = len(m.huesos) - m.W.shape[1]
    if nuevos <= 0:
        return
    m.W = np.hstack([m.W, np.zeros((len(m.W), nuevos))])
    col = {hb: i for i, hb in enumerate(m.huesos)}
    for lado, mano in m.manos.items():
        info = mano["info"]
        A3, B3 = np.asarray(m.J["codo_" + lado], float), np.asarray(m.J["mano_" + lado], float)
        AB = B3 - A3
        sel = np.flatnonzero(m.grupo == GRUPOS3D.index("brazo_" + lado))
        V = m.V[sel]
        t = ((V - A3) @ AB) / max(float(AB @ AB), 1e-9)
        hmano = _suave(t, info["t_muneca"] - 0.05, info["t_muneca"] + 0.05)
        Wm = np.zeros((len(sel), m.W.shape[1]))
        Wm[:, col["muneca_" + lado]] = 1.0
        if mano["falanges"]:
            u, vv = info["vista"].proyectar(V)
            Q = np.stack([u, vv], 1) - info["a2"]
            e, s = Q @ info["d"], Q @ info["n"]
            largo = info["e_t"] - info["e_w"]
            Lf = LARGO_DEDO * largo
            total = np.zeros(len(sel))
            for (dd, base), nombres in zip(mano["dedos"], mano["falanges"]):
                wb = 0.2 * (dd["s1"] - dd["s0"])
                f = _suave(s, dd["s0"] - wb / 2, dd["s0"] + wb / 2) * (1 - _suave(s, dd["s1"] - wb / 2, dd["s1"] + wb / 2))
                g = _suave(e, base - 0.12 * Lf, base + 0.12 * Lf)
                fg = f * g
                juntas = [base + x * Lf for x in FALANGES] + [np.inf]
                b = 0.08 * Lf
                for k, hb in enumerate(nombres):
                    pk = _suave(e, juntas[k] - b, juntas[k] + b) * (1 - (_suave(e, juntas[k + 1] - b, juntas[k + 1] + b)
                                                                       if np.isfinite(juntas[k + 1]) else 0))
                    Wm[:, col[hb]] += fg * pk
                total += fg
            Wm[:, col["muneca_" + lado]] = np.clip(1 - total, 0, 1)
            Wm /= np.maximum(Wm.sum(1, keepdims=True), 1e-9)
        m.W[sel] = (1 - hmano[:, None]) * m.W[sel] + hmano[:, None] * Wm


# --------------------------------------------------------------------------------------------
# Guante propio (23-oct): la mano ya no se pinta proyectando el dibujo (un puño visto desde un lado), que al abrir
# los dedos o girar la muñeca dejaba manchas y trozos «rotos». Se pinta con un material propio: el blanco del guante
# del dibujo, con luz según la pose, y sus 3 costuras en el dorso, que viajan pegadas a la superficie.
# --------------------------------------------------------------------------------------------
GUANTE_PROPIO = True
COSTURAS = (0.3, 0.5, 0.7)          # dónde van las 3 costuras, a lo ancho del dorso
TRAMO_COSTURA = (0.14, 0.47)        # de dónde a dónde van, a lo largo de la mano (fracción muñeca → punta)
GROSOR_COSTURA = 0.032              # grosor, en fracción del largo de la mano
COLOR_COSTURA = np.array([38.0, 30.0, 34.0])     # BGR
SOMBRA_GUANTE = np.array([0.92, 0.78, 0.80])      # la sombra del guante tira a lavanda, como en el dibujo (BGR)


def parametros_guante(m) -> dict:
    """Por cada mano: e(P) = ke·P + e0 (a lo largo, muñeca → punta) y s(P) = ks·P + s0 (a lo ancho), en el espacio
    de reposo; el tramo de las costuras, sus posiciones, el grosor y hacia dónde mira el dorso. Se calcula una vez."""
    guardado = getattr(m, "guante", None)
    if guardado is not None:
        return guardado
    from mente.vistas import DIRECCION
    sal = {}
    for lado, mano in (getattr(m, "manos", None) or {}).items():
        info = mano.get("info")
        if not info:
            continue
        v = info["vista"]
        base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        u, vv = v.proyectar(base)
        q = np.stack([np.asarray(u, float), np.asarray(vv, float)], 1)
        Jq = (q[1:] - q[0]).T                                      # (2, 3): cuánto se mueve el punto en la vista
        d, n, a2 = np.asarray(info["d"], float), np.asarray(info["n"], float), np.asarray(info["a2"], float)
        ke, e0 = Jq.T @ d, float((q[0] - a2) @ d)
        ks, s0 = Jq.T @ n, float((q[0] - a2) @ n)
        largo = float(info["e_t"] - info["e_w"])
        ys, xs = np.nonzero(info["mascara"])
        Q = np.stack([xs, ys], 1).astype(float) - a2
        e_pix, s_pix = Q @ d, Q @ n
        dorso_pix = (e_pix > info["e_w"] + TRAMO_COSTURA[0] * largo) & (e_pix < info["e_w"] + TRAMO_COSTURA[1] * largo)
        s_px = s_pix[dorso_pix] if dorso_pix.sum() > 10 else s_pix
        s_lo, s_hi = (float(np.percentile(s_px, 8)), float(np.percentile(s_px, 92))) if len(s_px) else (-1.0, 1.0)
        pix = v.bgra[ys, xs, :3].astype(float) if len(xs) else np.zeros((0, 3))
        blanco = np.percentile(pix, 80, axis=0) if len(pix) else (m.color_mano if m.color_mano is not None
                                                                   else np.array([235.0, 235.0, 240.0]))
        # pliegues entre los dedos: a medio camino entre un dedo y el siguiente, de su base a la punta
        dedos = sorted(mano.get("dedos") or [], key=lambda x: x[0]["s"])
        pliegues = [((a["s1"] + b["s0"]) / 2, min(ba, bb)) for (a, ba), (b, bb) in zip(dedos, dedos[1:])]
        sal[lado] = {"ke": ke, "e0": e0, "ks": ks, "s0": s0, "pliegues": pliegues[:3], "e_punta": float(info["e_t"]),
                     "e_lo": float(info["e_w"] + TRAMO_COSTURA[0] * largo),
                     "e_hi": float(info["e_w"] + TRAMO_COSTURA[1] * largo),
                     "lineas": [s_lo + f * (s_hi - s_lo) for f in COSTURAS],
                     "grosor": GROSOR_COSTURA * largo, "dorso": np.asarray(DIRECCION["perfil_" + lado], float),
                     "color": np.asarray(blanco, float)}
    m.guante = sal
    return sal


def _suave_np(x, a, b):
    t = np.clip((x - a) / max(b - a, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def color_guante(m, P: np.ndarray, N: np.ndarray) -> np.ndarray | None:
    """Color BGR (n, 3) del guante propio en los puntos de mano P (reposo) con normales N; None si no hay datos."""
    g = parametros_guante(m)
    if not g:
        return None
    x_centro = float(np.asarray(m.J["cadera"], float)[0])
    lados = np.where(P[:, 0] < x_centro, "izq", "der")
    col = np.zeros((len(P), 3), np.float32)
    for lado in ("izq", "der"):
        sel = np.flatnonzero(lados == lado)
        if not len(sel):
            continue
        p = g.get(lado) or next(iter(g.values()))
        Pi, Ni = P[sel].astype(float), N[sel].astype(float)
        e = Pi @ p["ke"] + p["e0"]
        s = Pi @ p["ks"] + p["s0"]
        gr = p["grosor"]
        tramo = _suave_np(e, p["e_lo"], p["e_lo"] + gr) * (1 - _suave_np(e, p["e_hi"] - gr, p["e_hi"]))
        dorso = _suave_np(Ni @ p["dorso"], 0.15, 0.4)
        linea = np.zeros(len(sel))
        for sl in p["lineas"]:
            linea = np.maximum(linea, 1 - _suave_np(np.abs(s - sl), 0.5 * gr, gr))
        k = linea * tramo * dorso
        for sp, base in p.get("pliegues", []):                     # los dedos se ven separados
            en_dedo = _suave_np(e, base, base + gr) * (1 - _suave_np(e, p["e_punta"] - gr, p["e_punta"]))
            k = np.maximum(k, 0.85 * (1 - _suave_np(np.abs(s - sp), 0.35 * gr, 0.8 * gr)) * en_dedo)
        k = k[:, None]
        col[sel] = (1 - k) * p["color"] + k * COLOR_COSTURA
    return col


def expandir(m, pose: dict) -> dict:
    """Traduce «dedos_der»: (cierre, abrir) e «indice_der»: cierre a giros de cada falange."""
    if not getattr(m, "manos", None):
        return pose
    p = dict(pose)
    for lado, mano in m.manos.items():
        valor = pose.get("dedos_" + lado, (0.0, 0.0))
        cierre, abre = (valor if isinstance(valor, (tuple, list)) else (valor, 0.0))[:2]
        ind = pose.get("indice_" + lado)
        for i, nombres in enumerate(mano["falanges"]):
            c = ind if (ind is not None and i == mano["indice"]) else cierre
            for k, hb in enumerate(nombres):
                p.setdefault(hb, (c * REPARTO[k], abre * mano["pos"][i] if k == 0 else 0.0))
    return p


def rotacion_dedo(m, hb: str, valor) -> np.ndarray:
    """(cierre, abrir) en grados → matriz 3x3 alrededor de los ejes propios del dedo."""
    if not isinstance(valor, (tuple, list)):
        valor = (valor, 0.0)
    flex, abre = (list(valor) + [0.0, 0.0])[:2]
    eje_flex, eje_abre = m.ejes[hb]
    return _rot_eje(eje_abre, abre) @ _rot_eje(eje_flex, flex)
