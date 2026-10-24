# -*- coding: utf-8 -*-
"""Movimientos de captura real (mocap) para MenteAnima 3D: BVH y ASF/AMC (el formato de CMU) → poses del modelo.

23-oct (Marino: «empieza con los cambios… libre albedrío»): los movimientos hechos a mano se veían rígidos (el brazo
izquierdo «no era natural»). Aquí el movimiento sale de una persona real capturada, sin Mixamo ni servicios:
  1. leer_bvh / leer_asf_amc   jerarquía + cuadros → posición de cada articulación en cada cuadro (cinemática directa)
  2. articulaciones            nombres de la captura → las 15 articulaciones del modelo (lado: el «izq» del modelo
                               es la izquierda de la IMAGEN, o sea el lado derecho del personaje: Right / r…)
  3. poses_desde_captura       por cada hueso, la DIRECCIÓN de la captura (no sus ángulos): así sirve con cualquier
                               esqueleto y con cualquier proporción; la pelvis y el tronco se orientan con dos
                               líneas (columna + caderas u hombros) para no perder el giro. El salto sale de la
                               altura real de los pies.
Las poses traen matrices 3x3 por hueso (modelo3d.matrices3d las usa tal cual); los dedos siguen siendo los del modelo.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mente.animacion import HUESOS


@dataclass
class Captura:
    nombre: str
    fps: float
    posiciones: dict = field(default_factory=dict)      # articulación de la captura → (F, 3)

    @property
    def cuadros(self) -> int:
        return len(next(iter(self.posiciones.values()))) if self.posiciones else 0

    @property
    def segundos(self) -> float:
        return self.cuadros / self.fps if self.fps else 0.0


# --------------------------------------------------------------------------------------------
# Rotaciones
# --------------------------------------------------------------------------------------------
def _rot(eje: str, grados: np.ndarray) -> np.ndarray:
    """(F, 3, 3) rotaciones de `grados` (F,) alrededor de x, y o z."""
    a = np.radians(np.asarray(grados, float))
    c, s = np.cos(a), np.sin(a)
    o, u = np.zeros_like(a), np.ones_like(a)
    if eje == "x":
        filas = [[u, o, o], [o, c, -s], [o, s, c]]
    elif eje == "y":
        filas = [[c, o, s], [o, u, o], [-s, o, c]]
    else:
        filas = [[c, -s, o], [s, c, o], [o, o, u]]
    return np.moveaxis(np.array(filas), [0, 1], [-2, -1])


def _rot_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """La rotación más corta que lleva la dirección a a la b."""
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999999:                                   # opuestas: media vuelta sobre un eje perpendicular
        p = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def _marco(principal: np.ndarray, secundario: np.ndarray, orden: str) -> np.ndarray:
    """Base ortonormal (columnas x, y, z) con un eje exacto y el otro lo más cerca posible.
    orden "y": principal = arriba (y), secundario = derecha del personaje (−x); "x": al revés."""
    if orden == "y":
        y = principal / np.linalg.norm(principal)
        x = -secundario - (-secundario @ y) * y
        x /= max(np.linalg.norm(x), 1e-12)
    else:
        x = -principal / np.linalg.norm(principal)
        y = secundario - (secundario @ x) * x
        y /= max(np.linalg.norm(y), 1e-12)
    return np.stack([x, y, np.cross(x, y)], axis=1)


# --------------------------------------------------------------------------------------------
# 1. Lectores
# --------------------------------------------------------------------------------------------
def leer_bvh(ruta: str | Path, texto: str | None = None) -> Captura:
    """BVH (Mixamo, Rokoko, CMU convertido…): jerarquía, canales y cuadros → posiciones por articulación."""
    texto = texto if texto is not None else Path(ruta).read_text(encoding="utf-8", errors="replace")
    fichas = texto.split()
    i = 0
    nodos = []                                         # (nombre, padre, offset, canales)
    pila: list = []
    ultimo = None
    while i < len(fichas):
        f = fichas[i]
        if f in ("ROOT", "JOINT"):
            nodos.append([fichas[i + 1], pila[-1] if pila else None, np.zeros(3), []])
            ultimo = len(nodos) - 1
            i += 2
        elif f == "End":                               # End Site: un punto final sin canales
            padre = pila[-1]
            nodos.append([nodos[padre][0] + "_fin", padre, np.zeros(3), []])
            ultimo = len(nodos) - 1
            i += 2
        elif f == "{":
            pila.append(ultimo)
            i += 1
        elif f == "}":
            pila.pop()
            i += 1
        elif f == "OFFSET":
            nodos[ultimo][2] = np.array([float(x) for x in fichas[i + 1:i + 4]])
            i += 4
        elif f == "CHANNELS":
            n = int(fichas[i + 1])
            nodos[ultimo][3] = fichas[i + 2:i + 2 + n]
            i += 2 + n
        elif f == "MOTION":
            i += 1
            break
        else:
            i += 1
    n_cuadros = int(fichas[fichas.index("Frames:", i) + 1])
    k = fichas.index("Time:", i) + 1
    paso = float(fichas[k])
    total = sum(len(nd[3]) for nd in nodos)
    valores = np.array([float(x) for x in fichas[k + 1:k + 1 + n_cuadros * total]]).reshape(n_cuadros, total)
    G, pos, col = {}, {}, 0
    for idx, (nombre, padre, offset, canales) in enumerate(nodos):
        F = n_cuadros
        R = np.tile(np.eye(3), (F, 1, 1))
        t = np.tile(offset, (F, 1))
        for c in canales:
            v = valores[:, col]
            col += 1
            cl = c.lower()
            if cl.endswith("position"):
                t[:, "xyz".index(cl[0])] += v
            elif cl.endswith("rotation"):
                R = R @ _rot(cl[0], v)                 # orden de los canales = orden de la multiplicación
        if padre is None:
            Gr, Gt = R, t
        else:
            Rp, tp = G[padre]
            Gt = tp + np.einsum("fij,fj->fi", Rp, t)
            Gr = Rp @ R
        G[idx] = (Gr, Gt)
        pos[nombre] = Gt
    return Captura(Path(ruta).stem if ruta else "bvh", 1.0 / paso if paso > 0 else 30.0, pos)


def _secciones_asf(texto: str) -> dict:
    sec, actual = {}, None
    for linea in texto.splitlines():
        l = linea.split("#", 1)[0].strip()
        if not l:
            continue
        if l.startswith(":"):
            partes = l.split(None, 1)
            actual = partes[0][1:].lower()
            sec.setdefault(actual, [])
            if len(partes) > 1:
                sec[actual].append(partes[1])
        elif actual:
            sec[actual].append(l)
    return sec


def leer_asf_amc(asf: str | Path, amc: str | Path, texto_asf: str | None = None, texto_amc: str | None = None,
                 fps: float = 120.0) -> Captura:
    """ASF (esqueleto) + AMC (cuadros), el formato de la base de CMU (120 cuadros por segundo)."""
    texto_asf = texto_asf if texto_asf is not None else Path(asf).read_text(encoding="utf-8", errors="replace")
    texto_amc = texto_amc if texto_amc is not None else Path(amc).read_text(encoding="utf-8", errors="replace")
    sec = _secciones_asf(texto_asf)
    huesos: dict = {}
    actual = None
    for l in sec.get("bonedata", []):
        p = l.split()
        if p[0] == "begin":
            actual = {"dof": [], "axis": np.zeros(3), "direction": np.zeros(3), "length": 0.0}
        elif p[0] == "end" and actual is not None:
            huesos[actual["name"]] = actual
            actual = None
        elif actual is not None:
            if p[0] == "name":
                actual["name"] = p[1]
            elif p[0] == "direction":
                actual["direction"] = np.array([float(x) for x in p[1:4]])
            elif p[0] == "length":
                actual["length"] = float(p[1])
            elif p[0] == "axis":
                actual["axis"] = np.array([float(x) for x in p[1:4]])
            elif p[0] == "dof":
                actual["dof"] = [x.lower() for x in p[1:]]
    padres = {}
    for l in sec.get("hierarchy", []):
        p = l.split()
        if p[0] in ("begin", "end"):
            continue
        for hijo in p[1:]:
            padres[hijo] = p[0]
    raiz = {"order": ["tx", "ty", "tz", "rx", "ry", "rz"], "orientation": np.zeros(3), "position": np.zeros(3)}
    for l in sec.get("root", []):
        p = l.split()
        if p[0] == "order":
            raiz["order"] = [x.lower() for x in p[1:]]
        elif p[0] in ("orientation", "position"):
            raiz[p[0]] = np.array([float(x) for x in p[1:4]])
    # cuadros del AMC
    cuadros: list = []
    for linea in texto_amc.splitlines():
        l = linea.strip()
        if not l or l.startswith("#") or l.startswith(":"):
            continue
        p = l.split()
        if len(p) == 1 and p[0].isdigit():
            cuadros.append({})
        elif cuadros:
            cuadros[-1][p[0]] = [float(x) for x in p[1:]]
    F = len(cuadros)

    def _c(eje_xyz):                                   # matriz del «axis» del ASF (estático x, luego y, luego z)
        ax, ay, az = (np.full(1, v) for v in eje_xyz)
        return (_rot("z", az) @ _rot("y", ay) @ _rot("x", ax))[0]

    def _dofs(nombre, orden):
        ang = {"rx": np.zeros(F), "ry": np.zeros(F), "rz": np.zeros(F), "tx": np.zeros(F), "ty": np.zeros(F),
               "tz": np.zeros(F)}
        for f, cu in enumerate(cuadros):
            vals = cu.get(nombre)
            if vals:
                for k, d in enumerate(orden[:len(vals)]):
                    ang[d][f] = vals[k]
        return ang

    ang = _dofs("root", raiz["order"])
    C = _c(raiz["orientation"])
    Rm = C @ (_rot("z", ang["rz"]) @ _rot("y", ang["ry"]) @ _rot("x", ang["rx"])) @ C.T
    tr = np.stack([ang["tx"], ang["ty"], ang["tz"]], 1) + raiz["position"]
    G = {"root": (Rm, tr)}
    pos = {"root": tr}
    pendientes = list(huesos)
    while pendientes:
        avance = False
        for nombre in list(pendientes):
            padre = padres.get(nombre, "root")
            if padre not in G:
                continue
            h = huesos[nombre]
            Rp, tp = G[padre]
            a = _dofs(nombre, h["dof"])
            C = _c(h["axis"])
            R = Rp @ C @ (_rot("z", a["rz"]) @ _rot("y", a["ry"]) @ _rot("x", a["rx"])) @ C.T
            fin = tp + h["length"] * np.einsum("fij,j->fi", R, h["direction"])
            G[nombre] = (R, fin)
            pos[nombre] = fin                          # en ASF, la posición de un hueso es la de su extremo
            pendientes.remove(nombre)
            avance = True
        if not avance:
            break
    return Captura(Path(amc).stem if amc else "amc", fps, pos)


def leer(ruta: str | Path, asf: str | Path | None = None) -> Captura:
    """Una captura por su extensión: .bvh, o .amc con su .asf (si no se da, el .asf de la misma carpeta)."""
    ruta = Path(ruta)
    if ruta.suffix.lower() == ".bvh":
        return leer_bvh(ruta)
    if ruta.suffix.lower() == ".amc":
        if asf is None:
            sujeto = ruta.stem.split("_")[0]
            candidatos = [ruta.with_name(sujeto + ".asf")] + sorted(ruta.parent.glob("*.asf"))
            asf = next((c for c in candidatos if c.exists()), None)
            if asf is None:
                raise FileNotFoundError(f"Falta el esqueleto .asf de {ruta.name} (en la misma carpeta).")
        return leer_asf_amc(asf, ruta)
    raise ValueError(f"No reconozco «{ruta.suffix}»: usa .bvh o .amc (con su .asf).")


# --------------------------------------------------------------------------------------------
# 2. Nombres de la captura → articulaciones del modelo
# --------------------------------------------------------------------------------------------
# «izq» del modelo = izquierda de la imagen = lado DERECHO del personaje (mira a la cámara)
_LADO = {"izq": ("right", "r"), "der": ("left", "l")}
_BVH = {"cadera": ["upleg", "thigh", "upperleg", "hip"], "rodilla": ["leg", "shin", "lowerleg", "calf", "knee"],
        "pie": ["foot", "ankle"], "hombro": ["arm", "upperarm", "shoulder"], "codo": ["forearm", "lowerarm", "elbow"],
        "mano": ["hand", "wrist"]}
_CMU = {"cadera": "hipjoint", "rodilla": "femur", "pie": "tibia", "hombro": "clavicle", "codo": "humerus",
        "mano": "radius"}


def _limpio(nombre: str) -> str:
    return re.sub(r"[^a-z0-9]", "", nombre.lower().split(":")[-1])


def articulaciones(cap: Captura) -> dict:
    """Las 15 articulaciones del modelo, (F, 3) cada una, desde los nombres de la captura (BVH o CMU)."""
    P = cap.posiciones
    limpio = {_limpio(k): k for k in P}
    out = {}
    es_cmu = "lhumerus" in limpio or "rfemur" in limpio
    for lado, (largo, corto) in _LADO.items():
        for parte in ("cadera", "rodilla", "pie", "hombro", "codo", "mano"):
            clave = None
            if es_cmu:
                clave = limpio.get(corto + _CMU[parte])
            else:
                for base in _BVH[parte]:
                    for pref in (largo, corto, largo + "_", corto + "_"):
                        clave = clave or limpio.get(_limpio(pref + base))
            if clave is None:
                raise ValueError(f"La captura no trae «{parte} {lado}» ({cap.nombre}).")
            out[f"{parte}_{lado}"] = P[clave]
    if es_cmu:
        out["cadera"] = P["root"]
        out["cuello"] = P[limpio["thorax"]] if "thorax" in limpio else P[limpio["lowerneck"]]
        out["cabeza"] = P[limpio["head"]]
    else:
        raiz = next((limpio[k] for k in ("hips", "pelvis", "hip", "root") if k in limpio), None)
        out["cadera"] = P[raiz] if raiz else (out["cadera_izq"] + out["cadera_der"]) / 2
        cuello = next((limpio[k] for k in ("neck", "neck1", "neckbase") if k in limpio), None)
        cabeza = next((limpio[k] for k in ("headfin", "headtopend", "headtop", "head") if k in limpio), None)
        if cuello is None or cabeza is None:
            raise ValueError(f"La captura no trae cuello y cabeza ({cap.nombre}).")
        out["cuello"], out["cabeza"] = P[cuello], P[cabeza]
        if cabeza == limpio.get("head"):              # sin punto final: la cabeza mide como el cuello
            out["cabeza"] = P[cabeza] + (P[cabeza] - P[cuello])
    # la pelvis del modelo está entre las dos caderas (la raíz de la captura puede estar más atrás)
    out["cadera"] = (out["cadera_izq"] + out["cadera_der"]) / 2
    return out


# --------------------------------------------------------------------------------------------
# 3. Captura → poses del modelo
# --------------------------------------------------------------------------------------------
def _remuestrear(A: dict, fps_in: float, fps_out: float, inicio: float, fin: float | None) -> dict:
    F = len(next(iter(A.values())))
    t_fin = min(fin if fin is not None else F / fps_in, F / fps_in)
    t = np.arange(max(0.0, inicio), t_fin - 1e-9, 1.0 / fps_out)
    x = t * fps_in
    i0 = np.clip(np.floor(x).astype(int), 0, F - 1)
    i1 = np.clip(i0 + 1, 0, F - 1)
    w = (x - i0)[:, None]
    return {k: (1 - w) * v[i0] + w * v[i1] for k, v in A.items()}


def _arriba(A: dict) -> np.ndarray:
    """Eje vertical de la captura (Y en casi todas; Z en algunas exportaciones)."""
    d = (A["cabeza"] - (A["pie_izq"] + A["pie_der"]) / 2).mean(0)
    e = np.zeros(3)
    e[int(np.argmax(np.abs(d)))] = np.sign(d[int(np.argmax(np.abs(d)))])
    return e


def a_modelo(A: dict) -> tuple:
    """Pasa la captura al espacio del modelo: el personaje mira a +Z (la cámara), su derecha es −X y arriba +Y.
    Devuelve (articulaciones en ese espacio, altura de la captura)."""
    up = _arriba(A)
    derecha = (A["cadera_izq"][0] - A["cadera_der"][0]) + (A["hombro_izq"][0] - A["hombro_der"][0])
    derecha = derecha - (derecha @ up) * up
    derecha /= max(np.linalg.norm(derecha), 1e-12)
    adelante = np.cross(up, derecha)
    B = np.stack([-derecha, up, adelante])             # filas: los ejes del modelo en la captura
    out = {k: v @ B.T for k, v in A.items()}
    pies = np.minimum(out["pie_izq"][:, 1], out["pie_der"][:, 1])
    altura = float(np.percentile(out["cabeza"][:, 1] - pies, 90))
    return out, max(altura, 1e-9)


def rumbo_suave(A: dict, fps: float, ventana: float = 1.0) -> np.ndarray:
    """Hacia dónde mira el cuerpo en cada cuadro (grados, alrededor de Y), suavizado: el rumbo general, sin el
    vaivén de la cadera en cada paso. A en el espacio del modelo."""
    lado = (A["cadera_izq"] - A["cadera_der"]) + (A["hombro_izq"] - A["hombro_der"])     # hacia la derecha (−X)
    ang = np.unwrap(np.arctan2(lado[:, 2], -lado[:, 0]))
    k = max(1, int(round(ventana * fps)) | 1)
    pad = np.pad(ang, k // 2, mode="edge")
    return np.degrees(np.convolve(pad, np.ones(k) / k, mode="valid"))


def sin_giro(A: dict, fps: float) -> dict:
    """La captura sin su rumbo general: el personaje sigue mirando a la cámara aunque la persona diera la vuelta
    (las caminatas de CMU van y vuelven). El avance también se gira, para que siga yendo hacia adelante."""
    r = np.radians(-rumbo_suave(A, fps))
    c, s_ = np.cos(r), np.sin(r)
    centro = A["cadera"].copy()
    out = {}
    for k, v in A.items():
        d = v - centro
        out[k] = np.stack([c * d[:, 0] + s_ * d[:, 2], d[:, 1], -s_ * d[:, 0] + c * d[:, 2]], 1) + centro
    paso = np.diff(centro, axis=0, prepend=centro[:1])                  # el avance, en el marco del cuerpo
    paso = np.stack([c * paso[:, 0] + s_ * paso[:, 2], paso[:, 1], -s_ * paso[:, 0] + c * paso[:, 2]], 1)
    nuevo = centro[0] + np.cumsum(paso, axis=0)
    return {k: v - centro + nuevo for k, v in out.items()}


def poses_desde_captura(m, cap: Captura, fps: float = 30.0, inicio: float = 0.0, fin: float | None = None,
                        desplazar: bool = False, manos: dict | None = None, girar: bool = False) -> tuple:
    """(poses, desplazamientos (dx, dz)) por fotograma, listas para movimientos3d / render_gl.
    `manos`: lo que hacen los dedos («dedos_izq»: (cierre, abrir)…); por defecto, relajados.
    `girar`: conserva los giros de la persona (si no, mira siempre a la cámara; ver sin_giro)."""
    A = _remuestrear(articulaciones(cap), cap.fps, fps, inicio, fin)
    A, alto = a_modelo(A)
    if not girar:
        A = sin_giro(A, fps)
    J = {k: np.asarray(v, float) for k, v in m.J.items()}

    def dir_reposo(hb):
        a, b = HUESOS[hb][0], HUESOS[hb][1]
        return J[b] - J[a]

    marco_pelvis0 = _marco(J["cadera_izq"] - J["cadera_der"], J["cuello"] - J["cadera"], "x")
    marco_torso0 = _marco(J["cuello"] - J["cadera"], J["hombro_izq"] - J["hombro_der"], "y")
    suelo = float(np.min(np.minimum(A["pie_izq"][:, 1], A["pie_der"][:, 1])))
    raiz0 = A["cadera"][0]
    manos = manos if manos is not None else {"dedos_izq": (25.0, 0.0), "dedos_der": (25.0, 0.0)}
    poses, desp = [], []
    n = len(A["cadera"])
    for f in range(n):
        Jf = {k: v[f] for k, v in A.items()}
        G = {}
        G["pelvis"] = _marco(Jf["cadera_izq"] - Jf["cadera_der"], Jf["cuello"] - Jf["cadera"], "x") @ marco_pelvis0.T
        G["torso"] = _marco(Jf["cuello"] - Jf["cadera"], Jf["hombro_izq"] - Jf["hombro_der"], "y") @ marco_torso0.T
        pose = {}
        for hb in ("pelvis", "torso", "cabeza", "brazo_izq", "antebrazo_izq", "brazo_der", "antebrazo_der",
                   "muslo_izq", "pierna_izq", "muslo_der", "pierna_der"):
            padre = HUESOS[hb][2]
            if hb not in G:
                a, b = HUESOS[hb][0], HUESOS[hb][1]
                Gp = G[padre]
                G[hb] = _rot_min(Gp @ dir_reposo(hb), Jf[b] - Jf[a]) @ Gp
            pose[hb] = G[hb] if padre is None else G[padre].T @ G[hb]
        # salto: la altura real de los pies sobre el suelo de la captura (los pies del modelo se apoyan solos)
        libre = float(min(Jf["pie_izq"][1], Jf["pie_der"][1]) - suelo)
        pose["y"] = max(0.0, libre) / alto
        pose["apoyo"] = 1.0
        pose.update(manos)
        poses.append(pose)
        if desplazar:
            d = (Jf["cadera"] - raiz0) / alto * m.altura
            desp.append((float(d[0]), float(d[2])))
        else:
            desp.append((0.0, 0.0))
    return poses, desp


# --------------------------------------------------------------------------------------------
# 4. Capturas de CMU: catálogo, descarga verificada y búsqueda en las carpetas
# --------------------------------------------------------------------------------------------
# mocap.cs.cmu.edu: «This dataset of motions is free for all uses»; se pueden usar en productos, pero no revender los
# archivos, así que el programa no los trae: los baja de CMU y comprueba cada uno con su huella SHA-256.
AGRADECIMIENTO_CMU = ("The data used in this project was obtained from mocap.cs.cmu.edu. The database was created "
                      "with funding from NSF EIA-0196217.")
CMU = {   # archivo: (qué hace, sha256)
    "141.asf": ("esqueleto de la sesión 141", "e32993b8079c0914ed4d865ad53f7b9646398808855233e6f650d4f4417df2bb"),
    "141_19.amc": ("caminar", "ea82ad04c1ee54134b411993ccca52006dbd63d1cacddda76471ca34539a71b5"),
    "141_01.amc": ("correr", "f0127a9067895d4fc6e2c82c0c3e8329ab102e392c579e6eaad2e7661ad3d180"),
    "141_05.amc": ("saltar de lado", "d5a8665dfe131128480668cb10d473244c812274172164ad51492cfa524b76f2"),
    "141_16.amc": ("saludar", "755f065106adb3b40ba6bd118d6aa6f66858e2553851fa40b530705b12dee1ca"),
    "141_12.amc": ("bailar", "f3dc6f547373714baaa3b158de3d3a13400373a10b903022fefd8230f893ab77"),
    "141_21.amc": ("encoger los hombros", "5d7a1fa9d3d805cbf233a33db13030b600bc928cc52744a720a65a2216a0a9b9"),
    "141_20.amc": ("esperar", "223616f94b053b965bdcf650c77fcb8ef00423405d819419a7d7af62428b06c2"),
    "14.asf": ("esqueleto de la sesión 14", "96a22ac52ac15bd464c296028936be0ae7d2f0ee4a056202b1aa17f4ad02ec7f"),
    "14_24.amc": ("dirigir el tráfico, saludar y señalar", "e6b8926fbb14d786ab38628f36b4593b29b244597c5d80b23cf3c8e96206bfe8"),
}
CARPETAS_CAPTURAS = ("capturas", "material_mocap")


def _url_cmu(archivo: str, esquema: str) -> str:
    return f"{esquema}://mocap.cs.cmu.edu/subjects/{archivo.split('_')[0].split('.')[0]}/{archivo}"


def descargar_cmu(destino: str | Path, progreso=None) -> list:
    """Baja las capturas del catálogo a `destino` (las que ya están y coinciden, no las vuelve a bajar).
    Primero por https; si el certificado del servidor no se puede comprobar, por http. En los dos casos el archivo
    solo se guarda si su SHA-256 es el del catálogo (así nadie en el camino puede cambiarlo)."""
    import hashlib
    import urllib.error
    import urllib.request
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    listos = []
    for archivo, (que, huella) in CMU.items():
        ruta = destino / archivo
        if ruta.exists() and hashlib.sha256(ruta.read_bytes()).hexdigest() == huella:
            listos.append(ruta)
            continue
        datos = None
        for esquema in ("https", "http"):
            try:
                with urllib.request.urlopen(_url_cmu(archivo, esquema), timeout=60) as r:
                    datos = r.read()
                break
            except urllib.error.URLError:
                continue
        if datos is None:
            raise ConnectionError(f"No pude bajar {archivo} de mocap.cs.cmu.edu (¿hay internet?).")
        if hashlib.sha256(datos).hexdigest() != huella:
            raise ValueError(f"{archivo} no coincide con su huella SHA-256: no lo guardo.")
        ruta.write_bytes(datos)
        listos.append(ruta)
        if progreso:
            progreso(f"{archivo} · {que} ({len(datos) / 1024:.0f} KB)")
    return listos


def buscar_capturas(raiz: str | Path) -> dict:
    """{etiqueta: ruta} de las capturas (.bvh, .amc) en las carpetas capturas/ y material_mocap/ del programa."""
    out = {}
    for carpeta in CARPETAS_CAPTURAS:
        base = Path(raiz) / carpeta
        if not base.is_dir():
            continue
        for ruta in sorted(list(base.rglob("*.amc")) + list(base.rglob("*.bvh"))):
            que = CMU.get(ruta.name, (None,))[0]
            out[f"{que} · CMU {ruta.stem}" if que else ruta.stem] = ruta
    return out
