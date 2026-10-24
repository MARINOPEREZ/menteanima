# -*- coding: utf-8 -*-
"""Movimientos 3D de MenteAnima: caminar, correr, brincar, señalar, saludar, bailar, celebrar, girar y respirar.

Cada movimiento es f(t, modelo) → pose. En la pose, cada hueso recibe (adelante, afuera, giro) en grados:
  · adelante: el extremo del hueso va hacia donde mira el personaje (en el torso y la cabeza: inclinarse o
    asentir hacia adelante). Doblar la rodilla = la pierna (canilla) va hacia atrás → negativo.
  · afuera: el extremo se aleja del cuerpo hacia su lado (en el torso y la cabeza: inclinarse hacia +X).
  · giro: sobre su propio eje (torso y cabeza: mirar hacia +X; brazos y piernas: > 0 rota hacia adentro).
El codo y la rodilla son bisagras: solo «adelante» (el codo de 0 a 150°, la rodilla de 0 a −150°); su «afuera» no
existe. Para llevar el antebrazo de lado (mano en la cintura, saludo) se rota el brazo sobre su eje en el hombro,
como en el cuerpo humano.
Claves del cuerpo entero: "x", "y", "z" (en alturas; y = arriba), "rumbo" (hacia dónde mira, grados; 0 = a la
cámara de frente, 90 = hacia la derecha de la imagen), "estirar" y "apoyo" (1 = pies en el suelo).
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import os
import pickle
from dataclasses import dataclass
from typing import Callable

import numpy as np

from mente import modelo3d as m3
from mente.movimientos import _suave, _tramo

PI2 = 2 * math.pi
LADOS = (("izq", 0.0), ("der", math.pi))


def _agachar(k: float) -> dict:
    """Flexión de rodillas hacia adelante (los pies quedan bajo el cuerpo y el torso compensa)."""
    return {"muslo_izq": (58 * k, 4 * k, 0), "pierna_izq": (-108 * k, 0, 0),
            "muslo_der": (58 * k, 4 * k, 0), "pierna_der": (-108 * k, 0, 0), "torso": (16 * k, 0, 0),
            "cabeza": (-8 * k, 0, 0)}


def _sumar(*poses: dict) -> dict:
    sal = {}
    for q in poses:
        for k, v in q.items():
            if isinstance(v, tuple):
                a = sal.get(k, (0.0, 0.0, 0.0))
                sal[k] = tuple(x + y for x, y in zip(a, v))
            else:
                sal[k] = sal.get(k, 0.0) + v
    return sal


def _escalar(pose: dict, f: float) -> dict:
    return {k: (tuple(x * f for x in v) if isinstance(v, tuple) else v * f) for k, v in pose.items()
            if k not in ("estirar", "apoyo")}


def respirar(t: float, m) -> dict:
    w = PI2 * t / 3.0
    return {"torso": (1.2 * math.sin(w), 0, 0), "cabeza": (-1.5 * math.sin(w + 0.7), 0, 1.5 * math.sin(w / 2)),
            "brazo_izq": (0, 3 + 2.5 * math.sin(w + 0.3), 0), "brazo_der": (0, 3 + 2.5 * math.sin(w + 0.3), 0),
            "antebrazo_izq": (4 + 2 * math.sin(w + 0.9), 0, 0), "antebrazo_der": (4 + 2 * math.sin(w + 0.9), 0, 0),
            "estirar": 1 + 0.008 * math.sin(w)}


def caminar(t: float, m) -> dict:
    fase = PI2 * t / 1.0                       # 1 s = dos pasos
    pose = {}
    for lado, d in LADOS:
        q = fase + d
        pose["muslo_" + lado] = (3 + 25 * math.sin(q), 2, 0)
        pose["pierna_" + lado] = (-(5 + 52 * max(0.0, math.cos(q)) ** 1.6), 0, 0)
        pose["brazo_" + lado] = (-24 * math.sin(q), 6, 0)                   # brazo contrario a la pierna
        pose["antebrazo_" + lado] = (12 + 18 * max(0.0, -math.sin(q)), 0, 0)
    pose["torso"] = (4, 0, -7 * math.sin(fase))                             # hombros contra la cadera
    pose["pelvis"] = (0, 0, 5 * math.sin(fase))
    pose["cabeza"] = (-2, 0, 7 * math.sin(fase))
    return pose


def correr(t: float, m) -> dict:
    fase = PI2 * t / 0.68
    pose = {}
    for lado, d in LADOS:
        q = fase + d
        pose["muslo_" + lado] = (14 + 42 * math.sin(q), 3, 0)
        pose["pierna_" + lado] = (-(16 + 95 * max(0.0, math.cos(q - 0.35)) ** 1.3), 0, 0)
        pose["brazo_" + lado] = (-42 * math.sin(q), 10, 8)                 # las manos van un poco al centro
        pose["antebrazo_" + lado] = (88 + 12 * math.sin(q), 0, 0)
    pose["torso"] = (12, 0, -10 * math.sin(fase))
    pose["pelvis"] = (0, 0, 7 * math.sin(fase))
    pose["cabeza"] = (-8, 0, 10 * math.sin(fase))
    pose["y"] = 0.035 * math.sin(fase) ** 2
    return pose


def brincar(t: float, m) -> dict:
    u = min(1.0, max(0.0, t / 1.4))
    k = 0.8 * _tramo(u, 0.04, 0.3) * (1 - _tramo(u, 0.32, 0.4))            # anticipación
    v = (u - 0.4) / 0.38
    vuelo = 0.0 <= v <= 1.0
    if vuelo:
        k = 0.35 * math.sin(math.pi * v)                                    # recoge las piernas
    k += 0.65 * _tramo(u, 0.78, 0.85) * (1 - _tramo(u, 0.86, 1.0))         # amortigua la caída
    altura = 0.4 * 4 * v * (1 - v) if vuelo else 0.0
    arriba = _tramo(u, 0.3, 0.45) * (1 - _tramo(u, 0.72, 0.95))
    atras = _tramo(u, 0.04, 0.3) * (1 - _tramo(u, 0.3, 0.4))
    estirar = 1 - 0.07 * k * (not vuelo) + 0.07 * _tramo(u, 0.36, 0.42) * (1 - _tramo(u, 0.44, 0.56))
    apoyo = 1 - _tramo(u, 0.38, 0.42) + _tramo(u, 0.76, 0.8)
    pose = _agachar(k)
    for lado in ("izq", "der"):
        pose["brazo_" + lado] = (165 * arriba - 45 * atras, 12 * arriba + 6, 0)
        pose["antebrazo_" + lado] = (10 * arriba + 15 * atras, 0, 0)
    pose = _sumar(pose, {"cabeza": (-6 * arriba, 0, 0)})
    pose.update({"y": altura, "estirar": estirar, "apoyo": min(1.0, apoyo)})
    return pose


def senalar(t: float, m) -> dict:
    u = min(1.0, max(0.0, t / 2.6))
    k = _tramo(u, 0.0, 0.17) * (1 - _tramo(u, 0.84, 1.0))
    golpe = 5 * (math.exp(-((u - 0.36) / 0.035) ** 2) + math.exp(-((u - 0.56) / 0.035) ** 2))
    # la otra mano en la cintura: el hombro se abre, va un poco atrás y rota hacia adentro; el codo se dobla
    pose = {"brazo_der": ((40 + golpe) * k, 72 * k, 0), "antebrazo_der": (0, 0, 0),
            "brazo_izq": (-16 * k, 36 * k, 72 * k), "antebrazo_izq": (100 * k, 0, 0),
            "torso": (2 * k, 0, 12 * k), "cabeza": (0, -4 * k, 18 * k)}
    return _sumar(pose, _escalar(respirar(t, m), 1 - k))


def saludar(t: float, m) -> dict:
    u = min(1.0, max(0.0, t / 2.6))
    k = _tramo(u, 0.0, 0.15) * (1 - _tramo(u, 0.86, 1.0))
    ola = 24 * math.sin(PI2 * 2.4 * t) * _tramo(u, 0.1, 0.2) * (1 - _tramo(u, 0.8, 0.88))
    # brazo de lado y antebrazo arriba (rotación hacia afuera en el hombro); la ola, rotando el hombro
    return {"brazo_der": (5 * k, 90 * k, (-75 + 0.6 * ola) * k), "antebrazo_der": (75 * k, 0, 55 * k),
            "torso": (0, -3 * k, 6 * k), "cabeza": (0, -6 * k, 8 * k), "brazo_izq": (0, 5 * k, 0),
            "estirar": 1 + 0.01 * math.sin(PI2 * t)}


def bailar(t: float, m) -> dict:
    b = PI2 * t * 2.0                          # 2 golpes por segundo
    lado = math.sin(b / 2)
    pose = _agachar(0.18 * (0.5 - 0.5 * math.cos(b)))
    arr = 0.5 + 0.5 * math.sin(b / 4)
    pose = _sumar(pose, {"torso": (0, 7 * lado, 12 * math.sin(b / 4)), "cabeza": (3 * math.sin(b), -6 * lado, 0),
                         "brazo_der": (20, 50 + 95 * arr, 0), "antebrazo_der": (30 + 20 * math.sin(b), 0, 0),
                         "brazo_izq": (20, 50 + 95 * (1 - arr), 0), "antebrazo_izq": (30 - 20 * math.sin(b), 0, 0)})
    pose["x"] = 0.025 * lado
    return pose


def celebrar(t: float, m) -> dict:
    u = min(1.0, max(0.0, t / 2.4))
    k = _tramo(u, 0.0, 0.12) * (1 - _tramo(u, 0.88, 1.0))
    salto = max(0.0, math.sin(PI2 * 2.0 * t)) * _tramo(u, 0.1, 0.18) * (1 - _tramo(u, 0.8, 0.88))
    puno = 16 * math.sin(PI2 * 2.0 * t)
    pose = _agachar(0.12 * (1 - salto) * k)
    pose = _sumar(pose, {"brazo_izq": (20 * k, (140 + 8 * salto) * k, 0),
                         "brazo_der": (20 * k, (140 + 8 * salto) * k, 0),
                         "antebrazo_izq": ((20 + puno) * k, 0, 0), "antebrazo_der": ((20 + puno) * k, 0, 0),
                         "cabeza": (-8 * k, 4 * math.sin(PI2 * t) * k, 0)})
    pose.update({"y": 0.06 * salto, "estirar": 1 + 0.03 * salto})
    return pose


def girar(t: float, m) -> dict:
    """Da una vuelta completa sobre sí mismo con un saltito: se luce que es 3D."""
    u = min(1.0, max(0.0, t / 2.0))
    g = _tramo(u, 0.12, 0.88)
    v = (u - 0.12) / 0.76
    salto = math.sin(math.pi * v) if 0 <= v <= 1 else 0.0
    k = _tramo(u, 0.0, 0.12) * (1 - _tramo(u, 0.88, 1.0))
    pose = _agachar(0.25 * (_tramo(u, 0.0, 0.1) * (1 - _tramo(u, 0.1, 0.16)) +
                            _tramo(u, 0.86, 0.92) * (1 - _tramo(u, 0.92, 1.0))))
    pose = _sumar(pose, {"brazo_izq": (0, 55 * k, 0), "brazo_der": (0, 55 * k, 0),
                         "antebrazo_izq": (20 * k, 0, 0), "antebrazo_der": (20 * k, 0, 0)})
    pose.update({"rumbo": 360 * g, "y": 0.08 * salto})
    return pose


# --------------------------------------------------------------------------------------------
# Manos: muñeca y dedos (si el personaje tiene manos con dedos dibujados; si no, estas claves no hacen nada)
# «dedos_der»: (cierre, abrir) — cierre > 0 cierra el puño, < 0 lo abre; «indice_der»: cierre solo del índice;
# «muneca_der»: (adelante, afuera, giro) como cualquier hueso. Rangos moderados: abrir un puño dibujado deja ver
# la palma, que ninguna vista dibuja.
# --------------------------------------------------------------------------------------------
def _ambas(cierre: float, abrir: float = 0.0) -> dict:
    return {"dedos_izq": (cierre, abrir), "dedos_der": (cierre, abrir)}


def _manos_respirar(t: float, m) -> dict:
    return _ambas(3 + 3 * math.sin(PI2 * t / 3.0 + 1.2))


def _manos_caminar(t: float, m) -> dict:
    fase = PI2 * t / 1.0
    pose = {}
    for lado, d in LADOS:
        q = fase + d
        pose["dedos_" + lado] = (6 + 5 * math.sin(q), 0.0)
        pose["muneca_" + lado] = (8 * math.sin(q + 0.9), 0.0, 0.0)          # la mano va un poco atrasada
    return pose


def _manos_correr(t: float, m) -> dict:
    fase = PI2 * t / 0.68
    pose = _ambas(26)                                                       # puños
    for lado, d in LADOS:
        pose["muneca_" + lado] = (6 * math.sin(fase + d + 0.9), 0.0, 0.0)
    return pose


def _manos_brincar(t: float, m) -> dict:
    u = min(1.0, max(0.0, t / 1.4))
    arriba = _tramo(u, 0.3, 0.45) * (1 - _tramo(u, 0.72, 0.95))
    atras = _tramo(u, 0.04, 0.3) * (1 - _tramo(u, 0.3, 0.4))
    return _ambas(-16 * arriba + 14 * atras, 7 * arriba)                    # arriba, manos abiertas


def _manos_senalar(t: float, m) -> dict:
    u = min(1.0, max(0.0, t / 2.6))
    k = _tramo(u, 0.0, 0.17) * (1 - _tramo(u, 0.84, 1.0))
    return {"indice_der": -40 * k, "dedos_der": (28 * k, 0.0), "dedos_izq": (12 * k, 0.0),
            "muneca_der": (0.0, -4 * k, 0.0)}


def _manos_saludar(t: float, m) -> dict:
    u = min(1.0, max(0.0, t / 2.6))
    k = _tramo(u, 0.0, 0.15) * (1 - _tramo(u, 0.86, 1.0))
    ola = 24 * math.sin(PI2 * 2.4 * t) * _tramo(u, 0.1, 0.2) * (1 - _tramo(u, 0.8, 0.88))
    return {"dedos_der": (-16 * k, 7 * k), "muneca_der": (0.0, 0.7 * ola * k, 0.0)}   # saluda con la muñeca


def _manos_bailar(t: float, m) -> dict:
    b = PI2 * t * 2.0
    return _ambas(10 + 12 * math.sin(2 * b))                                # chasquidos al ritmo


def _manos_celebrar(t: float, m) -> dict:
    u = min(1.0, max(0.0, t / 2.4))
    k = _tramo(u, 0.0, 0.12) * (1 - _tramo(u, 0.88, 1.0))
    return _ambas(30 * k + 6 * math.sin(PI2 * 2.0 * t) * k)


def _manos_girar(t: float, m) -> dict:
    u = min(1.0, max(0.0, t / 2.0))
    k = _tramo(u, 0.0, 0.12) * (1 - _tramo(u, 0.88, 1.0))
    return _ambas(-12 * k, 6 * k)


def _con_manos(cuerpo, manos):
    def f(t: float, m) -> dict:
        return _sumar(cuerpo(t, m), manos(t, m))
    f.__name__ = cuerpo.__name__
    f.__doc__ = cuerpo.__doc__
    return f


@dataclass
class Movimiento:
    funcion: Callable
    segundos: float
    periodo: float | None
    desplaza: bool
    descripcion: str


MOVIMIENTOS = {
    "caminar": Movimiento(_con_manos(caminar, _manos_caminar), 3.0, 1.0, True, "camina (en el sitio o avanzando)"),
    "correr": Movimiento(_con_manos(correr, _manos_correr), 2.04, 0.68, True, "corre con los puños cerrados"),
    "brincar": Movimiento(_con_manos(brincar, _manos_brincar), 1.4, 1.4, False,
                          "se agacha, salta (manos abiertas) y cae amortiguando"),
    "señalar": Movimiento(_con_manos(senalar, _manos_senalar), 2.6, None, False,
                          "señala con el índice, mirando hacia allá"),
    "saludar": Movimiento(_con_manos(saludar, _manos_saludar), 2.6, None, False, "saluda moviendo la muñeca"),
    "bailar": Movimiento(_con_manos(bailar, _manos_bailar), 4.0, 2.0, False, "baila al ritmo, con chasquidos"),
    "celebrar": Movimiento(_con_manos(celebrar, _manos_celebrar), 2.4, None, False, "puños arriba y saltitos"),
    "girar": Movimiento(_con_manos(girar, _manos_girar), 2.0, None, False, "da una vuelta completa sobre sí mismo"),
    "respirar": Movimiento(_con_manos(respirar, _manos_respirar), 3.0, 3.0, False, "quieto, respirando (para pausas)"),
}
SINONIMOS = {"saltar": "brincar", "salto": "brincar", "senalar": "señalar", "apuntar": "señalar",
             "andar": "caminar", "caminata": "caminar", "trotar": "correr", "carrera": "correr",
             "hola": "saludar", "baile": "bailar", "festejar": "celebrar", "quieto": "respirar",
             "reposo": "respirar", "vuelta": "girar", "voltear": "girar"}
DIRECCIONES = {"frente": 0.0, "derecha": 90.0, "izquierda": -90.0, "atras": 180.0, "atrás": 180.0}


def nombre_movimiento(texto: str) -> str:
    t = texto.strip().lower()
    t = SINONIMOS.get(t, t)
    if t not in MOVIMIENTOS:
        raise ValueError(f"Movimiento desconocido: «{texto}». Disponibles: {', '.join(MOVIMIENTOS)}")
    return t


def preparar_pasos(nombres: list, segundos: float | None = None) -> list:
    pasos = []
    for n in nombres:
        nombre = nombre_movimiento(n)
        mv = MOVIMIENTOS[nombre]
        d = float(segundos) if segundos else mv.segundos
        if mv.periodo:
            d = max(1, round(d / mv.periodo)) * mv.periodo
        pasos.append((nombre, d))
    return pasos


def _mezclar(a: dict, b: dict, w: float) -> dict:
    sal = {}
    for k in set(a) | set(b):
        base = 1.0 if k in ("estirar", "apoyo") else 0.0
        va, vb = a.get(k), b.get(k)
        if isinstance(va, tuple) or isinstance(vb, tuple):
            va = va if isinstance(va, tuple) else (0.0, 0.0, 0.0)
            vb = vb if isinstance(vb, tuple) else (0.0, 0.0, 0.0)
            sal[k] = tuple((1 - w) * x + w * y for x, y in zip(va, vb))
        else:
            sal[k] = (1 - w) * (base if va is None else va) + w * (base if vb is None else vb)
    return sal


def poses_de(m, pasos: list, fps: float = 30, rumbo: float = 0.0, desplazar: bool = False,
             transicion: float = 0.3) -> tuple:
    """(lista de diccionarios de pose, lista de desplazamientos (dx, dz)) por fotograma."""
    inicios, t = [], 0.0
    for _, d in pasos:
        inicios.append(t)
        t += d
    n = max(1, int(round(t * fps)))
    poses, peso = [], []
    for i in range(n):
        t = i / fps
        k = max(j for j in range(len(pasos)) if inicios[j] <= t + 1e-9)
        nombre = pasos[k][0]
        pose = MOVIMIENTOS[nombre].funcion(t - inicios[k], m)
        p_desp = float(MOVIMIENTOS[nombre].desplaza)
        if k > 0 and t - inicios[k] < transicion / 2:
            ant = pasos[k - 1][0]
            w = _suave(0.5 + (t - inicios[k]) / transicion)
            pose = _mezclar(MOVIMIENTOS[ant].funcion(t - inicios[k - 1], m), pose, w)
            p_desp = (1 - w) * MOVIMIENTOS[ant].desplaza + w * p_desp
        elif k + 1 < len(pasos) and inicios[k + 1] - t < transicion / 2:
            sig = pasos[k + 1][0]
            w = _suave(0.5 - (inicios[k + 1] - t) / transicion)
            pose = _mezclar(pose, MOVIMIENTOS[sig].funcion(t - inicios[k + 1], m), w)
            p_desp = (1 - w) * p_desp + w * MOVIMIENTOS[sig].desplaza
        pose["rumbo"] = pose.get("rumbo", 0.0) + rumbo
        poses.append(pose)
        peso.append(p_desp if desplazar else 0.0)
    desp = _avanzar(m, poses, peso) if desplazar and any(peso) else [(0.0, 0.0)] * n
    return poses, desp


def _avanzar(m, poses: list, peso: list) -> list:
    """Desplazamiento (dx, dz) para que el pie de apoyo no patine."""
    if len(m.pies) < 2:
        return [(0.0, 0.0)] * len(poses)
    inc = np.zeros((len(poses), 2))
    previo = None
    lados = list(m.pies)
    todos = np.concatenate([m.pies[l] for l in lados])
    cortes = np.cumsum([len(m.pies[l]) for l in lados])[:-1]
    for i, pose in enumerate(poses):
        V = m3.posar3d(m, pose, todos).V
        pies = dict(zip(lados, np.split(V, cortes)))
        apoyo = min(pies, key=lambda l: pies[l][:, 1].min())
        xz = pies[apoyo][:, [0, 2]].mean(0)
        if previo and previo[0] == apoyo:
            inc[i] = -(xz - previo[1])
        elif i > 0:
            inc[i] = inc[i - 1]
        previo = (apoyo, xz)
    if len(inc) > 1:
        inc[0] = inc[1]
    for c in range(2):
        inc[:, c] = np.convolve(np.pad(inc[:, c], 1, mode="edge"), np.ones(3) / 3, mode="valid") * np.asarray(peso)
    acum = np.cumsum(inc, axis=0)
    return [(float(a), float(b)) for a, b in acum]


# --------------------------------------------------------------------------------------------
# Animar: poses → fotogramas (en varios procesos)
# --------------------------------------------------------------------------------------------
_MODELO = None


def _iniciar(datos: bytes) -> None:
    global _MODELO
    _MODELO = pickle.loads(datos)


def _cuadro(args) -> np.ndarray:
    pose, desp, cam, escala, origen, lienzo, ss, luz = args
    P = m3.posar3d(_MODELO, pose)
    P.V[:, 0] += desp[0]
    P.V[:, 2] += desp[1]
    return m3.renderizar3d(_MODELO, P, cam, escala, origen, lienzo, ss, luz)


def camaras(n: int, vista: str = "tres-cuartos", orbita: float = 0.0, altura: float = 1.0) -> list:
    """Una cámara por fotograma. vista: frente | tres-cuartos | perfil | espalda; orbita: grados que gira la
    cámara alrededor del personaje durante la animación."""
    az0 = {"frente": 0.0, "tres-cuartos": 30.0, "perfil": 90.0, "espalda": 180.0}.get(vista, 30.0)
    return [m3.Camara(azimut=az0 + orbita * i / max(1, n), elevacion=7.0, objetivo=(0.0, 0.5 * altura, 0.0))
            for i in range(n)]


def encuadre3d(m, poses: list, desp: list, cams: list, escala: float, lienzo: tuple | None = None) -> tuple:
    lo, hi = np.array([np.inf, np.inf]), np.array([-np.inf, np.inf * -1])
    paso = max(1, len(poses) // 60)
    muestra = np.arange(0, len(m.V), 5)
    for i in list(range(0, len(poses), paso)) + [len(poses) - 1]:
        V = m3.posar3d(m, poses[i], muestra).V
        V[:, 0] += desp[i][0]
        V[:, 2] += desp[i][1]
        x, y, _, _ = cams[i].proyectar(V, escala, m.altura)
        lo = np.minimum(lo, [x.min(), y.min()])
        hi = np.maximum(hi, [x.max(), y.max()])
    if lienzo is None:
        mg = 0.04 * m.altura * escala
        ancho, alto = int(math.ceil(hi[0] - lo[0] + 2 * mg)), int(math.ceil(hi[1] - lo[1] + 2 * mg))
        return (lo[0] - mg, lo[1] - mg), (ancho + ancho % 2, alto + alto % 2)
    ancho, alto = lienzo
    # el suelo cerca del borde de abajo y el recorrido centrado
    suelo = cams[0].proyectar(np.array([[0.0, 0.0, 0.0]]), escala, m.altura)[1][0]
    y0 = suelo - 0.93 * alto
    if lo[1] - y0 < 0:
        y0 = max(lo[1] - 0.02 * alto, suelo - 0.99 * alto)
    return ((lo[0] + hi[0]) / 2 - ancho / 2, y0), (int(ancho), int(alto))


def animar(m, nombres: list, segundos: float | None = None, fps: float = 30, direccion: str = "frente",
           desplazar: bool = False, altura: float | None = None, lienzo: tuple | None = None, vista: str = "tres-cuartos",
           orbita: float = 0.0, ss: int = 2, luz: float = 0.18, procesos: int | None = None,
           gpu: bool = True, captura=None, desde: float = 0.0, hasta: float | None = None,
           girar: bool = False) -> tuple:
    """(generador de fotogramas BGRA, (ancho, alto), cantidad). Usa la tarjeta gráfica (OpenGL) si hay; si no,
    el rasterizador de numpy repartido en varios procesos. Con `captura` (mente/mocap.py: un BVH o un AMC de
    CMU ya leído) el movimiento sale de esa persona real, del segundo `desde` al `hasta`."""
    rumbo = DIRECCIONES.get(direccion, 0.0) if isinstance(direccion, str) else float(direccion)
    if captura is not None:
        from mente import mocap as cp
        poses, desp = cp.poses_desde_captura(m, captura, fps, desde, hasta, desplazar, girar=girar)
        if rumbo:
            for pose in poses:
                pose["rumbo"] = pose.get("rumbo", 0.0) + rumbo
    else:
        poses, desp = poses_de(m, preparar_pasos(nombres, segundos), fps, rumbo, desplazar)
    n = len(poses)
    if altura is None:
        altura = 0.6 * lienzo[1] if lienzo else 720
    escala = float(altura) / m.altura
    cams = camaras(n, vista, orbita, m.altura)
    origen, tam = encuadre3d(m, poses, desp, cams, escala, lienzo)
    tareas = [(poses[i], desp[i], cams[i], escala, origen, tam, ss, luz) for i in range(n)]
    procesos = procesos if procesos is not None else max(1, min(8, (os.cpu_count() or 2) - 2))

    def generar():
        from mente import render_gl
        gl = render_gl.renderizador(m) if gpu else None
        if gl is not None:                       # tarjeta gráfica: milisegundos por cuadro
            for pose, d, cam, esc, org, tm, _ss, lz in tareas:
                yield gl.render(pose, cam, esc, org, tm, d, lz)
            return
        if procesos <= 1 or n < 4:
            global _MODELO
            _MODELO = m
            for tarea in tareas:
                yield _cuadro(tarea)
            return
        datos = pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL)
        with cf.ProcessPoolExecutor(procesos, initializer=_iniciar, initargs=(datos,)) as ex:
            yield from ex.map(_cuadro, tareas, chunksize=2)
    return generar(), tam, n
