# -*- coding: utf-8 -*-
"""Movimientos de MenteAnima: caminar, correr, brincar, señalar, saludar, bailar, celebrar y respirar.

Cada movimiento es una función f(t, personaje) → pose, donde la pose es un diccionario con:
  · el giro de cada hueso en grados (antihorario a la vista) respecto de la postura neutra de pie. Avanzando
    hacia la derecha: pierna o brazo hacia adelante = positivo; doblar la rodilla = negativo; doblar el codo
    hacia adelante = positivo; inclinar el torso hacia adelante = negativo.
  · "x", "y": desplazamiento del cuerpo en alturas del personaje (y positivo = hacia arriba)
  · "estirar": estirar (>1) o aplastar (<1) desde el suelo · "giro": giro de todo el cuerpo
  · "apoyo": 1 = los pies se apoyan en el suelo; 0 = libre (en el aire)
Los movimientos están pensados mirando a la derecha; hacia la izquierda se usan en espejo.
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from mente.animacion import Personaje, Pose, encuadre, posar, renderizar

PI2 = 2 * math.pi


def _suave(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _tramo(u: float, a: float, b: float) -> float:
    """0 antes de a, 1 después de b, curva suave entre ambos."""
    return _suave((u - a) / (b - a)) if b > a else float(u >= a)


def _agachar(k: float) -> dict:
    """Rodillas hacia afuera (vista de frente): los pies quedan bajo las caderas y el cuerpo baja."""
    return {"muslo_izq": -34 * k, "pierna_izq": 68 * k, "muslo_der": 34 * k, "pierna_der": -68 * k}


def apuntar(p: Personaje, lado: str, angulo: float, torso: float = 0.0) -> tuple:
    """Giros de brazo y antebrazo para que el brazo quede recto en `angulo` (grados desde «hacia abajo»,
    antihorario: 90 = a la derecha, -90 = a la izquierda, 180 = arriba)."""
    brazo = angulo - p.neutro["brazo_" + lado] - torso
    antebrazo = -(p.neutro["antebrazo_" + lado] - p.neutro["brazo_" + lado])
    return brazo, antebrazo


def _sumar(*poses: dict) -> dict:
    sal = {}
    for q in poses:
        for k, v in q.items():
            sal[k] = sal.get(k, 0.0) + v
    return sal


# --------------------------------------------------------------------------------------------
# Movimientos
# --------------------------------------------------------------------------------------------
def respirar(t: float, p: Personaje) -> dict:
    w = PI2 * t / 3.0
    return {"torso": 1.2 * math.sin(w), "cabeza": -1.6 * math.sin(w + 0.7),
            "brazo_izq": -2.5 * math.sin(w + 0.3), "brazo_der": 2.5 * math.sin(w + 0.3),
            "antebrazo_izq": -2 * math.sin(w + 0.9), "antebrazo_der": 2 * math.sin(w + 0.9),
            "estirar": 1 + 0.008 * math.sin(w)}


def caminar(t: float, p: Personaje) -> dict:
    fase = PI2 * t / 1.0                      # 1 s = dos pasos
    pose = {}
    for lado, d in (("izq", 0.0), ("der", math.pi)):
        q = fase + d
        pose["muslo_" + lado] = 3 + 24 * math.sin(q)
        pose["pierna_" + lado] = -(5 + 50 * max(0.0, math.cos(q)) ** 1.6)     # la rodilla se dobla al avanzar
        pose["brazo_" + lado] = -20 * math.sin(q)                             # brazo contrario a la pierna
        pose["antebrazo_" + lado] = 10 + 14 * max(0.0, -math.sin(q))
    pose["torso"] = -3 + 1.2 * math.cos(2 * fase)
    pose["cabeza"] = 2 - 1.5 * math.cos(2 * fase)
    return pose


def correr(t: float, p: Personaje) -> dict:
    fase = PI2 * t / 0.68
    pose = {}
    for lado, d in (("izq", 0.0), ("der", math.pi)):
        q = fase + d
        pose["muslo_" + lado] = 12 + 40 * math.sin(q)
        pose["pierna_" + lado] = -(14 + 88 * max(0.0, math.cos(q - 0.35)) ** 1.3)
        pose["brazo_" + lado] = -36 * math.sin(q)
        pose["antebrazo_" + lado] = 78 + 14 * math.sin(q)
    pose["torso"] = -9 + 1.5 * math.cos(2 * fase)
    pose["cabeza"] = 6
    pose["y"] = 0.035 * math.sin(fase) ** 2                                   # fase de vuelo
    return pose


def brincar(t: float, p: Personaje) -> dict:
    u = min(1.0, max(0.0, t / 1.4))
    # agacharse (anticipación) → impulso → vuelo → caída que amortigua
    k = 0.85 * _tramo(u, 0.04, 0.3) * (1 - _tramo(u, 0.32, 0.4))
    v = (u - 0.4) / 0.38
    vuelo = 0.0 <= v <= 1.0
    if vuelo:
        k = 0.3 * math.sin(math.pi * v)                                        # recoge las piernas
    k += 0.7 * _tramo(u, 0.78, 0.85) * (1 - _tramo(u, 0.86, 1.0))
    altura = 0.38 * 4 * v * (1 - v) if vuelo else 0.0
    arriba = _tramo(u, 0.3, 0.45) * (1 - _tramo(u, 0.72, 0.95))
    atras = _tramo(u, 0.04, 0.3) * (1 - _tramo(u, 0.3, 0.4))
    estirar = 1 - 0.08 * k * (not vuelo) + 0.07 * _tramo(u, 0.36, 0.42) * (1 - _tramo(u, 0.44, 0.56))
    apoyo = 1 - _tramo(u, 0.38, 0.42) + _tramo(u, 0.76, 0.8)
    pose = _agachar(k)
    pose.update({"brazo_izq": -150 * arriba - 20 * atras, "brazo_der": 150 * arriba + 20 * atras,
                 "antebrazo_izq": -15 * arriba, "antebrazo_der": 15 * arriba,
                 "torso": -4 * atras, "cabeza": 5 * atras - 4 * arriba,
                 "y": altura, "estirar": estirar, "apoyo": min(1.0, apoyo)})
    return pose


def senalar(t: float, p: Personaje) -> dict:
    u = min(1.0, max(0.0, t / 2.6))
    k = _tramo(u, 0.0, 0.17) * (1 - _tramo(u, 0.84, 1.0))
    golpe = 5 * (math.exp(-((u - 0.36) / 0.035) ** 2) + math.exp(-((u - 0.56) / 0.035) ** 2))
    torso = -4 * k
    b, a = apuntar(p, "der", 100 + golpe, torso)
    pose = {"brazo_der": b * k, "antebrazo_der": a * k,
            "brazo_izq": -32 * k, "antebrazo_izq": 88 * k,                     # la otra mano en la cintura
            "torso": torso, "cabeza": -6 * k}
    return _sumar(pose, {k_: v * (1 - k) for k_, v in respirar(t, p).items() if k_ != "estirar"})


def saludar(t: float, p: Personaje) -> dict:
    u = min(1.0, max(0.0, t / 2.6))
    k = _tramo(u, 0.0, 0.15) * (1 - _tramo(u, 0.86, 1.0))
    b, _ = apuntar(p, "der", 140)
    ola = 22 * math.sin(PI2 * 2.4 * t) * _tramo(u, 0.1, 0.2) * (1 - _tramo(u, 0.8, 0.88))
    return {"brazo_der": b * k, "antebrazo_der": (35 + ola) * k, "torso": 2.5 * k, "cabeza": -5 * k,
            "brazo_izq": -4 * k, "estirar": 1 + 0.01 * math.sin(PI2 * t)}


def bailar(t: float, p: Personaje) -> dict:
    b = PI2 * t * 2.0                          # 2 golpes por segundo
    lado = math.sin(b / 2)
    pose = _agachar(0.2 * (0.5 - 0.5 * math.cos(b)))
    arr_d = 0.5 + 0.5 * math.sin(b / 4)
    pose.update({"x": 0.025 * lado, "torso": 7 * lado, "cabeza": -6 * lado + 3 * math.sin(b),
                 "brazo_der": 40 + 100 * arr_d, "antebrazo_der": 40 + 20 * math.sin(b),
                 "brazo_izq": -40 - 100 * (1 - arr_d), "antebrazo_izq": -40 - 20 * math.sin(b)})
    return pose


def celebrar(t: float, p: Personaje) -> dict:
    u = min(1.0, max(0.0, t / 2.4))
    k = _tramo(u, 0.0, 0.12) * (1 - _tramo(u, 0.88, 1.0))
    salto = max(0.0, math.sin(PI2 * 2.0 * t)) * _tramo(u, 0.1, 0.18) * (1 - _tramo(u, 0.8, 0.88))
    puño = 16 * math.sin(PI2 * 2.0 * t)
    pose = _agachar(0.12 * (1 - salto) * k)
    pose.update({"brazo_izq": (-135 - 8 * salto) * k, "brazo_der": (135 + 8 * salto) * k,
                 "antebrazo_izq": (-25 - puño) * k, "antebrazo_der": (25 + puño) * k,
                 "cabeza": 4 * math.sin(PI2 * t) * k, "y": 0.06 * salto, "estirar": 1 + 0.03 * salto})
    return pose


@dataclass
class Movimiento:
    funcion: Callable
    segundos: float           # duración por defecto
    periodo: float | None     # para repetir sin cortes
    desplaza: bool            # con --desplazar avanza por el lienzo
    descripcion: str


MOVIMIENTOS = {
    "caminar": Movimiento(caminar, 3.0, 1.0, True, "camina (en el sitio o avanzando)"),
    "correr": Movimiento(correr, 2.04, 0.68, True, "corre con el cuerpo inclinado"),
    "brincar": Movimiento(brincar, 1.4, 1.4, False, "se agacha, salta y cae amortiguando"),
    "señalar": Movimiento(senalar, 2.6, None, False, "señala hacia adelante con el brazo recto"),
    "saludar": Movimiento(saludar, 2.6, None, False, "saluda con la mano"),
    "bailar": Movimiento(bailar, 4.0, 2.0, False, "baila al ritmo (2 golpes por segundo)"),
    "celebrar": Movimiento(celebrar, 2.4, None, False, "brazos arriba y saltitos"),
    "respirar": Movimiento(respirar, 3.0, 3.0, False, "quieto, respirando (para pausas)"),
}
SINONIMOS = {"saltar": "brincar", "salto": "brincar", "senalar": "señalar", "apuntar": "señalar",
             "andar": "caminar", "caminata": "caminar", "trotar": "correr", "carrera": "correr",
             "hola": "saludar", "baile": "bailar", "festejar": "celebrar", "quieto": "respirar",
             "reposo": "respirar"}


def nombre_movimiento(texto: str) -> str:
    t = texto.strip().lower()
    t = SINONIMOS.get(t, t)
    if t not in MOVIMIENTOS:
        raise ValueError(f"Movimiento desconocido: «{texto}». Disponibles: {', '.join(MOVIMIENTOS)}")
    return t


def espejo(pose: dict) -> dict:
    """La misma pose mirando hacia la izquierda."""
    sal = {}
    for k, v in pose.items():
        if k in ("estirar", "apoyo", "y"):
            sal[k] = v
        elif k.endswith("_izq"):
            sal[k[:-4] + "_der"] = -v
        elif k.endswith("_der"):
            sal[k[:-4] + "_izq"] = -v
        else:
            sal[k] = -v
    return sal


def _mezclar(a: dict, b: dict, w: float) -> dict:
    sal = {}
    for k in set(a) | set(b):
        base = 1.0 if k in ("estirar", "apoyo") else 0.0
        sal[k] = (1 - w) * a.get(k, base) + w * b.get(k, base)
    return sal


# --------------------------------------------------------------------------------------------
# Secuencia → poses de cada fotograma
# --------------------------------------------------------------------------------------------
def preparar_pasos(nombres: list, segundos: float | None = None) -> list:
    """[(movimiento, segundos)]. Si se pide una duración, los movimientos que se repiten la redondean a
    ciclos completos (así el video se puede repetir sin saltos)."""
    pasos = []
    for n in nombres:
        m = MOVIMIENTOS[nombre_movimiento(n)]
        d = float(segundos) if segundos else m.segundos
        if m.periodo:
            d = max(1, round(d / m.periodo)) * m.periodo
        pasos.append((nombre_movimiento(n), d))
    return pasos


def poses_de(p: Personaje, pasos: list, fps: float = 30, direccion: int = 1, desplazar: bool = False,
             transicion: float = 0.3) -> list:
    """Una Pose por fotograma. `direccion` 1 = mira a la derecha, -1 = a la izquierda."""
    inicios, t = [], 0.0
    for _, d in pasos:
        inicios.append(t)
        t += d
    total = t
    n = max(1, int(round(total * fps)))
    salida, avance = [], []
    for i in range(n):
        t = i / fps
        k = max(j for j in range(len(pasos)) if inicios[j] <= t + 1e-9)
        nombre, _ = pasos[k]
        pose = MOVIMIENTOS[nombre].funcion(t - inicios[k], p)
        peso_desplaza = float(MOVIMIENTOS[nombre].desplaza)
        # transiciones suaves entre movimientos
        if k > 0 and t - inicios[k] < transicion / 2:
            ant = pasos[k - 1][0]
            w = _suave(0.5 + (t - inicios[k]) / transicion)
            pose = _mezclar(MOVIMIENTOS[ant].funcion(t - inicios[k - 1], p), pose, w)
            peso_desplaza = (1 - w) * MOVIMIENTOS[ant].desplaza + w * peso_desplaza
        elif k + 1 < len(pasos) and inicios[k + 1] - t < transicion / 2:
            sig = pasos[k + 1][0]
            w = _suave(0.5 - (inicios[k + 1] - t) / transicion)
            pose = _mezclar(pose, MOVIMIENTOS[sig].funcion(t - inicios[k + 1], p), w)
            peso_desplaza = (1 - w) * peso_desplaza + w * MOVIMIENTOS[sig].desplaza
        if direccion < 0:
            pose = espejo(pose)
        salida.append(posar(p, pose))
        avance.append(peso_desplaza if desplazar else 0.0)
    if desplazar and any(avance):
        salida = _avanzar(p, salida, avance)
    return salida


def animar(p: Personaje, nombres: list, segundos: float | None = None, fps: float = 30, direccion: int = 1,
           desplazar: bool = False, altura: float | None = None, lienzo: tuple | None = None, ss: int = 2) -> tuple:
    """(generador de fotogramas BGRA, (ancho, alto), cantidad de fotogramas).
    `altura`: alto del personaje en el video (px); por defecto el del dibujo, hasta 800."""
    poses = poses_de(p, preparar_pasos(nombres, segundos), fps, direccion, desplazar)
    if altura is None:
        altura = 0.6 * lienzo[1] if lienzo else min(p.altura, 800)
    escala = float(altura) / p.altura
    origen, tam = encuadre(p, poses, escala, lienzo)
    for c in p.capas:                         # prepara las texturas reducidas antes de usar hilos
        c.textura_a(escala * ss)

    def generar():
        hilos = max(1, min(6, (os.cpu_count() or 2) - 1))
        with cf.ThreadPoolExecutor(hilos) as ex:
            pendientes = []
            for P in poses:
                pendientes.append(ex.submit(renderizar, p, P, escala, origen, tam, ss))
                if len(pendientes) >= 2 * hilos:
                    yield pendientes.pop(0).result()
            for f in pendientes:
                yield f.result()
    return generar(), tam, len(poses)


def _avanzar(p: Personaje, poses: list, peso: list) -> list:
    """Desplaza el cuerpo para que el pie de apoyo no patine (bloqueo del pie)."""
    idx = {c.nombre: i for i, c in enumerate(p.capas)}
    def pie(P: Pose, lado: str) -> tuple:
        capa, v = p.pies[lado]
        q = P.capas[idx[capa]][v]
        return float(q[:, 1].max()), float(q[:, 0].mean())
    if len(p.pies) < 2:
        return poses
    inc = np.zeros(len(poses))
    previo = None
    for i, P in enumerate(poses):
        (yi, xi), (yd, xd) = pie(P, "izq"), pie(P, "der")
        apoyo = "izq" if yi >= yd else "der"
        x = xi if apoyo == "izq" else xd
        if previo and previo[0] == apoyo:
            inc[i] = -(x - previo[1])
        elif i > 0:
            inc[i] = inc[i - 1]
        previo = (apoyo, x)
    if len(inc) > 1:
        inc[0] = inc[1]
    inc = np.convolve(np.pad(inc, 1, mode="edge"), np.ones(3) / 3, mode="valid") * np.asarray(peso)
    x = np.cumsum(inc)
    return [P.mover(float(dx)) for P, dx in zip(poses, x)]
