# -*- coding: utf-8 -*-
"""Lo ejecuta Blender (no Python normal):  blender -b --factory-startup -P mente/blender_escena.py -- config.json

Arma la escena de MenteAnima v2 y renderiza la animación con fondo transparente:
  · importa el personaje (.glb/.gltf o .fbx de Mixamo) y, si hay, los movimientos en .fbx (sin piel) de Mixamo
  · encadena los movimientos pedidos en la pista NLA con transiciones suaves
  · cámara alrededor del personaje (vista, elevación, órbita) que lo encuadra entero en todo el recorrido
  · luz de estudio suave, colores fieles al dibujo (transformación «Standard»), fondo transparente
  · fotogramas PNG con alfa en la carpeta de salida (menteanima los convierte a WebM/MOV/GIF)
"""
import json
import math
import sys

import bpy
from mathutils import Vector

VISTAS = {"frente": 0.0, "tres-cuartos": 30.0, "perfil": 90.0, "espalda": 180.0}


def log(msg):
    print("MENTEANIMA: " + str(msg), flush=True)


def cfg_de_argumentos():
    argv = sys.argv
    return json.load(open(argv[argv.index("--") + 1], encoding="utf-8"))


def importar(ruta):
    antes = set(bpy.data.objects)
    if ruta.lower().endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=ruta)
    elif ruta.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=ruta, use_anim=True, ignore_leaf_bones=True,
                                 automatic_bone_orientation=False)
    else:
        raise ValueError("Formato no soportado: " + ruta)
    return [o for o in bpy.data.objects if o not in antes]


def esqueleto(objetos):
    for o in objetos:
        if o.type == "ARMATURE":
            return o
    return None


def acciones_del(objetos):
    vistas = []
    for o in objetos:
        ad = o.animation_data
        if ad is None:
            continue
        if ad.action and ad.action not in vistas:
            vistas.append(ad.action)
        for pista in ad.nla_tracks:
            for tira in pista.strips:
                if tira.action and tira.action not in vistas:
                    vistas.append(tira.action)
    return vistas


def limites(objetos, cuadros):
    """Caja (mín, máx) del personaje a lo largo de los cuadros indicados (con la animación aplicada)."""
    escena = bpy.context.scene
    lo, hi = Vector((1e9, 1e9, 1e9)), Vector((-1e9, -1e9, -1e9))
    # solo lo que sale en el render (el importador de glTF agrega una esfera oculta para dibujar los huesos)
    mallas = [o for o in objetos if o.type == "MESH" and not o.hide_render and o.visible_get()]
    if not mallas:
        mallas = [o for o in objetos if o.type == "MESH" and o.parent is not None]
    dep = bpy.context.evaluated_depsgraph_get()
    for f in cuadros:
        escena.frame_set(int(f))
        dep = bpy.context.evaluated_depsgraph_get()
        for o in mallas:
            ev = o.evaluated_get(dep)
            for esq in ev.bound_box:
                p = ev.matrix_world @ Vector(esq)
                lo = Vector((min(lo[i], p[i]) for i in range(3)))
                hi = Vector((max(hi[i], p[i]) for i in range(3)))
    return lo, hi


def main():
    cfg = cfg_de_argumentos()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    escena = bpy.context.scene
    fps = int(cfg.get("fps", 30))
    escena.render.fps = fps
    ancho, alto = cfg.get("resolucion", [1080, 1080])

    personaje = importar(cfg["modelo"])
    arm = esqueleto(personaje)
    acciones = {a.name: a for a in acciones_del(personaje)}
    # movimientos de Mixamo en archivos aparte (mismos nombres de huesos): su acción pasa al personaje
    for extra in cfg.get("archivos_movimiento", []):
        nuevos = importar(extra["archivo"])
        for a in acciones_del(nuevos):
            a.name = extra.get("nombre", a.name)
            acciones[a.name] = a
            a.use_fake_user = True
        for o in nuevos:
            bpy.data.objects.remove(o, do_unlink=True)
    log("acciones: " + ", ".join(sorted(acciones)))

    # secuencia en la pista NLA del esqueleto (o del objeto animado)
    duenio = arm or next((o for o in personaje if o.animation_data), None)
    inicio, fin = 1, 1
    pedidas = cfg.get("secuencia") or []
    if duenio is not None and pedidas:
        ad = duenio.animation_data or duenio.animation_data_create()
        ad.action = None
        for pista in list(ad.nla_tracks):
            ad.nla_tracks.remove(pista)
        # los objetos que el .glb animaba por separado (p. ej. la raíz) también siguen la secuencia
        otros = [o for o in personaje if o is not duenio and o.animation_data]
        for o in otros:
            o.animation_data.action = None
            for pista in list(o.animation_data.nla_tracks):
                o.animation_data.nla_tracks.remove(pista)
        transicion = int(cfg.get("transicion", 8))
        cuadro = 1
        pistas = {}
        for paso, nombre in enumerate(pedidas):
            accion = acciones.get(nombre)
            if accion is None:
                log("AVISO: no existe el movimiento " + nombre)
                continue
            largo = max(1, int(round(accion.frame_range[1] - accion.frame_range[0])))
            veces = max(1, int(cfg.get("repeticiones", {}).get(nombre, 1)))
            for o in [duenio] + otros:
                ado = o.animation_data
                if o not in pistas:
                    pistas[o] = [ado.nla_tracks.new(), ado.nla_tracks.new()]
                pista = pistas[o][paso % 2]          # dos pistas alternadas: las transiciones se superponen
                try:
                    tira = pista.strips.new(nombre, cuadro, accion)
                except RuntimeError:
                    continue
                tira.repeat = veces
                tira.blend_in = tira.blend_out = min(transicion, largo // 3)
                tira.extrapolation = "NOTHING"
            cuadro += largo * veces - transicion
        fin = cuadro + transicion
    elif acciones:
        a = next(iter(acciones.values()))
        inicio, fin = int(a.frame_range[0]), int(a.frame_range[1])
    else:                                   # modelo sin movimientos (p. ej. recién salido de TRELLIS): toma fija
        inicio, fin = 1, int(cfg.get("cuadros", 1))
    escena.frame_start, escena.frame_end = inicio, max(inicio + 1, fin)
    log(f"cuadros {escena.frame_start}-{escena.frame_end}")

    # encuadre: la caja de todo el recorrido
    muestra = range(escena.frame_start, escena.frame_end + 1, max(1, (escena.frame_end - escena.frame_start) // 24))
    lo, hi = limites(personaje, list(muestra) + [escena.frame_end])
    centro = (lo + hi) / 2
    alto_obj = hi.z - lo.z
    ancho_obj = max(hi.x - lo.x, hi.y - lo.y)

    # cámara sobre un pivote (para la órbita)
    cam_cfg = cfg.get("camara", {})
    az = math.radians(VISTAS.get(cam_cfg.get("vista", "tres-cuartos"), 30.0) + float(cam_cfg.get("azimut", 0)))
    el = math.radians(float(cam_cfg.get("elevacion", 7)))
    lente = float(cam_cfg.get("lente", 50))
    camara = bpy.data.objects.new("camara", bpy.data.cameras.new("camara"))
    camara.data.lens = lente
    camara.data.sensor_fit = "AUTO"
    pivote = bpy.data.objects.new("pivote", None)
    pivote.location = centro
    escena.collection.objects.link(pivote)
    escena.collection.objects.link(camara)
    camara.parent = pivote
    # distancia que encuadra al personaje con margen en el lado limitante
    fov_v = 2 * math.atan(0.5 * 36.0 * min(1.0, alto / ancho) / lente)
    fov_h = 2 * math.atan(0.5 * 36.0 * min(1.0, ancho / alto) / lente)
    margen = float(cam_cfg.get("margen", 1.12))
    dist = margen * max((alto_obj / 2) / math.tan(fov_v / 2), (ancho_obj / 2) / math.tan(fov_h / 2)) + ancho_obj / 2
    camara.location = Vector((dist * math.sin(az) * math.cos(el), -dist * math.cos(az) * math.cos(el),
                              dist * math.sin(el)))
    restr = camara.constraints.new("TRACK_TO")
    restr.target = pivote
    restr.track_axis = "TRACK_NEGATIVE_Z"
    restr.up_axis = "UP_Y"
    escena.camera = camara
    orbita = float(cam_cfg.get("orbita", 0))
    if orbita:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"   # giro a velocidad constante
        pivote.rotation_euler = (0, 0, 0)
        pivote.keyframe_insert("rotation_euler", frame=escena.frame_start)
        pivote.rotation_euler = (0, 0, math.radians(orbita))
        pivote.keyframe_insert("rotation_euler", frame=escena.frame_end + 1)

    # luz de estudio suave (la textura ya trae el dibujo; la luz solo da volumen)
    mundo = bpy.data.worlds.new("mundo")
    escena.world = mundo
    mundo.use_nodes = True
    fondo = mundo.node_tree.nodes.get("Background")
    if fondo:
        fondo.inputs[0].default_value = (1, 1, 1, 1)
        fondo.inputs[1].default_value = float(cfg.get("luz_ambiente", 0.9))
    for nombre, energia, rot in (("clave", 2.2, (50, 0, -35)), ("relleno", 0.8, (60, 0, 140)), ("contra", 1.5, (110, 0, 180))):
        sol = bpy.data.objects.new(nombre, bpy.data.lights.new(nombre, "SUN"))
        sol.data.energy = energia * float(cfg.get("luz", 1.0))
        sol.rotation_euler = tuple(math.radians(x) for x in rot)
        escena.collection.objects.link(sol)

    # render
    for motor in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            escena.render.engine = motor
            break
        except TypeError:
            continue
    escena.render.film_transparent = True
    escena.render.resolution_x, escena.render.resolution_y = int(ancho), int(alto)
    escena.render.resolution_percentage = 100
    escena.render.image_settings.file_format = "PNG"
    escena.render.image_settings.color_mode = "RGBA"
    escena.view_settings.view_transform = "Standard"
    escena.view_settings.look = "None"
    escena.render.filepath = cfg["salida"].rstrip("/\\") + "/"
    try:
        escena.eevee.taa_render_samples = int(cfg.get("muestras", 32))
    except AttributeError:
        pass
    log(f"render {ancho}x{alto} a {fps} fps, motor {escena.render.engine}")
    bpy.ops.render.render(animation=True)
    log("LISTO")


if __name__ == "__main__":
    main()
