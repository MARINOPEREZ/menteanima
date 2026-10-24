#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MenteAnima — de una imagen de tu personaje a un personaje 3D que camina, corre, brinca, señala, saluda, baila,
celebra y gira. El resultado sale con FONDO TRANSPARENTE (sin plantillas de fondo) y también como modelo 3D
(.glb) con esqueleto y animaciones.

Cómo funciona (todo local y gratis, sin IA en la nube):
  1. Separa la hoja de personaje en sus vistas (frente, perfiles, espalda) y las alinea.
  2. Arma un «exoesqueleto» de 15 articulaciones en el frente (corregible en la ventana) y le da profundidad
     con los perfiles.
  3. Construye el volumen 3D de cada parte del cuerpo con el ancho del frente y la profundidad del perfil, y lo
     pinta con las 4 vistas: cada lado del modelo lleva el dibujo real de ese lado.
  4. Lo mueve con huesos 3D (movimientos hechos a mano o copiados de una persona real: capturas BVH o las de
     Carnegie Mellon) y lo dibuja con una cámara que puede mirarlo de frente, de tres cuartos, de perfil o girar
     a su alrededor.

Uso:
  python menteanima.py                                    # ventana: subir imagen, ajustar, ver en 3D y exportar
  python menteanima.py animar hoja.png -m caminar,señalar,brincar
  python menteanima.py animar hoja.png -m correr --direccion derecha --desplazar --vista frente -o corre.webm
  python menteanima.py animar hoja.png -m girar --orbita 360 --lienzo 1080x1920 -o vuelta.mov
  python menteanima.py modelo hoja.png -o mascota.glb      # modelo 3D con esqueleto y todas las animaciones
  python menteanima.py visor mascota.glb                   # lo abre en el navegador para girarlo
  python menteanima.py vistas hoja.png                     # cómo reconoció cada vista (imagen de control)
  python menteanima.py animar figura.png --2d              # modo títere 2D (una sola imagen, sin volumen)
  python menteanima.py capturas                            # baja movimientos de personas reales (CMU, gratis)
  python menteanima.py animar hoja.png --captura capturas/cmu/141_16.amc --vista frente -o saluda.webm

Mejor resultado: una hoja de personaje con frente, perfil y espalda (cuerpo entero, de pie, alineados al mismo
suelo), en PNG con fondo transparente o liso. Con una sola imagen también funciona: el volumen se infla y la
espalda se refleja del frente (un 3D aproximado).
Formatos con transparencia: .webm (VP9), .mov (ProRes 4444), .gif, .apng o una carpeta de PNG.
Requisitos: Python 3.9+, numpy, opencv-contrib-python, pillow; ffmpeg para .webm/.mov.
Opcional: moderngl (usa la tarjeta gráfica: la vista 3D se gira en vivo y los videos salen en segundos).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    print("Faltan dependencias. Ejecuta:  pip install numpy opencv-contrib-python pillow", file=sys.stderr)
    sys.exit(1)

from mente import animacion as an
from mente import movimientos as mv
from mente.comun import RAIZ, consola_utf8, fallar, log

SALIDA = RAIZ / "salida" / "animaciones"
VISTAS_CAMARA = ("tres-cuartos", "frente", "perfil", "espalda")


def _lienzo(texto: str | None) -> tuple | None:
    if not texto:
        return None
    try:
        w, h = (int(v) for v in texto.lower().replace("×", "x").split("x"))
        return w + w % 2, h + h % 2
    except ValueError:
        fallar(f"--lienzo debe ser ANCHOxALTO (por ejemplo 1920x1080), no «{texto}».")


def _existe(imagen: str) -> None:
    if not Path(imagen).exists():
        fallar(f"No existe la imagen: {imagen}")


def _modelo(imagen: str, esqueleto: str | None, detalle: float):
    from mente import modelo3d as m3
    _existe(imagen)
    t = time.time()
    m, vistas, ruta = m3.cargar(imagen, esqueleto, detalle, progreso=lambda s: log("  " + s))
    reales = [r for r, v in vistas.items() if not v.espejo]
    falta = [r for r, v in vistas.items() if v.espejo]
    log(f"Vistas: {', '.join(reales)}" + (f" (inventadas en espejo: {', '.join(falta)})" if falta else ""))
    if len(reales) == 1:
        log("  Una sola vista: el volumen se infla y la espalda refleja el frente. Para un 3D fiel usa una hoja "
            "con frente, perfil y espalda.")
    origen = "guardado en " + ruta.name if ruta.exists() else "automático (corrígelo en la ventana)"
    log(f"Modelo 3D listo: {len(m.V)} vértices, {len(m.T)} triángulos; esqueleto {origen} ({time.time() - t:.1f} s).")
    return m


def _personaje2d(imagen: str, esqueleto: str | None, detalle: float = 1.0) -> an.Personaje:
    _existe(imagen)
    t = time.time()
    bgra, alfa, J, ruta = an.preparar(imagen, esqueleto)
    origen = "guardado en " + ruta.name if ruta.exists() else "automático (corrígelo con: python menteanima.py)"
    log(f"Figura recortada y esqueleto {origen}.")
    p = an.crear_personaje(bgra, alfa, J, detalle)
    log(f"Títere listo: {len(p.V)} vértices, {len(p.T)} triángulos, {len(p.capas)} capas ({time.time() - t:.1f} s).")
    return p


def _nombres(texto: str, modulo) -> list:
    try:
        nombres = [modulo.nombre_movimiento(m) for m in texto.replace("+", ",").split(",") if m.strip()]
    except ValueError as e:
        fallar(str(e))
    if not nombres:
        fallar("Indica al menos un movimiento con -m (por ejemplo -m caminar).")
    return nombres


def _guardar(fotos, tam, n, salida: Path, fps: float, t0: float) -> None:
    def progreso(i: int) -> None:
        if i % 15 == 0 or i == n:
            print(f"\r  {i}/{n} fotogramas", end="", flush=True)
    try:
        ruta = an.exportar(fotos, salida, fps, tam, progreso)
    except (RuntimeError, ValueError) as e:
        print()
        fallar(str(e))
    print()
    log(f"Listo en {time.time() - t0:.1f} s: {ruta}")
    log(f"  {an.FORMATOS[ruta.suffix.lower()]}")


def cmd_animar(a) -> None:
    lienzo = _lienzo(a.lienzo)
    if a.d2:
        nombres = _nombres(a.movimientos, mv)
        salida = Path(a.salida) if a.salida else SALIDA / f"{Path(a.imagen).stem}-{'-'.join(nombres)}-2d.webm"
        if salida.suffix.lower() not in an.FORMATOS:
            fallar(f"«{salida.suffix}» no guarda transparencia. Usa .webm, .mov, .gif, .apng o una carpeta.")
        p = _personaje2d(a.imagen, a.esqueleto, a.detalle)
        t = time.time()
        izquierda = a.direccion == "izquierda"
        fotos, tam, n = mv.animar(p, nombres, a.segundos, a.fps, -1 if izquierda else 1, a.desplazar,
                                  a.altura, lienzo, 1 if a.rapido else 2)
        log(f"Animando (2D) {' → '.join(nombres)}: {n} fotogramas, lienzo {tam[0]}x{tam[1]} transparente…")
        _guardar(fotos, tam, n, salida, a.fps, t)
        return
    from mente import movimientos3d as mv3
    cap = None
    if a.captura:                             # 23-oct: movimiento de una persona real (BVH o AMC de CMU)
        from mente import mocap as cp
        try:
            cap = cp.leer(a.captura, a.asf)
        except (OSError, ValueError) as e:
            fallar(f"No pude leer la captura: {e}")
        nombres = ["captura-" + Path(a.captura).stem]
        log(f"Captura real: {Path(a.captura).name} · {cap.cuadros} cuadros a {cap.fps:g} fps ({cap.segundos:.1f} s)")
    else:
        nombres = _nombres(a.movimientos, mv3)
    salida = Path(a.salida) if a.salida else SALIDA / f"{Path(a.imagen).stem}-{'-'.join(nombres)}.webm"
    if salida.suffix.lower() not in an.FORMATOS:
        fallar(f"«{salida.suffix}» no guarda transparencia. Usa .webm, .mov, .gif, .apng o una carpeta.")
    m = _modelo(a.imagen, a.esqueleto, a.detalle)
    if a.forma == "trellis":
        from mente import trellis3d as t3
        glb_a, glb_t = _glb_trellis(a.imagen, True), _glb_trellis(a.imagen, False)
        if glb_a.exists():                    # hecho desde la hoja en pose A: se devuelve a la pose del dibujo
            m = t3.desde_trellis(m, glb_a, progreso=lambda t: log("  " + t), pose_hoja=t3.POSE_A)
        elif glb_t.exists():
            m = t3.desde_trellis(m, glb_t, progreso=lambda t: log("  " + t))
        else:
            fallar(f"No hay modelo de TRELLIS para esta hoja ({glb_a.name}). Primero: python menteanima.py "
                   f"trellis {a.imagen} --pose-a")
    t = time.time()
    fotos, tam, n = mv3.animar(m, nombres, a.segundos, a.fps, a.direccion, a.desplazar, a.altura, lienzo,
                               a.vista, a.orbita, 1 if a.rapido else 2, captura=cap, desde=a.desde, hasta=a.hasta,
                               girar=getattr(a, "con_giro", False))
    log(f"Animando en 3D {' → '.join(nombres)} (vista {a.vista}{', órbita %g°' % a.orbita if a.orbita else ''}): "
        f"{n} fotogramas a {a.fps:g} fps, lienzo {tam[0]}x{tam[1]} transparente…")
    _guardar(fotos, tam, n, salida, a.fps, t)


def cmd_modelo(a) -> None:
    from mente import glb
    m = _modelo(a.imagen, a.esqueleto, a.detalle)
    salida = Path(a.salida) if a.salida else SALIDA / f"{Path(a.imagen).stem}.glb"
    if salida.suffix.lower() != ".glb":
        salida = salida.with_suffix(".glb")
    t = time.time()
    ruta = glb.exportar_glb(m, salida, progreso=lambda s: log("  " + s))
    log(f"Modelo 3D: {ruta} ({ruta.stat().st_size / 1e6:.1f} MB, {time.time() - t:.1f} s)")
    log("  Malla con la textura de las vistas, esqueleto de 11 huesos y una animación por movimiento.")
    log("  Se abre en Blender (Archivo → Importar → glTF), en visores web de glTF y en motores de juego.")


VISOR = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{titulo}</title>
<script type="module" src="https://cdn.jsdelivr.net/npm/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
<style>body{{margin:0;background:#2b2d42;font-family:system-ui,sans-serif;color:#edf2f4}}
model-viewer{{width:100vw;height:86vh;--poster-color:transparent}}#b{{display:flex;gap:6px;flex-wrap:wrap;padding:10px}}
button{{padding:8px 12px;border:0;border-radius:6px;background:#3a86ff;color:#fff;cursor:pointer}}</style></head>
<body><model-viewer id="mv" src="{glb}" camera-controls autoplay shadow-intensity="1" camera-orbit="30deg 80deg auto">
</model-viewer><div id="b"></div><script>
const mv=document.getElementById('mv');mv.addEventListener('load',()=>{{for(const a of mv.availableAnimations){{
const x=document.createElement('button');x.textContent=a;x.onclick=()=>{{mv.animationName=a;mv.play();}};
document.getElementById('b').appendChild(x);}}}});</script></body></html>"""


def cmd_visor(a) -> None:
    """Abre el .glb en el navegador (visor <model-viewer>; se sirve solo en este equipo)."""
    import functools
    import http.server
    import threading
    import webbrowser
    ruta = Path(a.modelo).resolve()
    if not ruta.exists():
        fallar(f"No existe el modelo: {ruta}")
    pagina = VISOR.format(titulo=ruta.stem, glb=ruta.name).encode("utf-8")

    class Manejador(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(pagina)
            elif self.path.lstrip("/") == ruta.name:
                super().do_GET()
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass
    servidor = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                               functools.partial(Manejador, directory=str(ruta.parent)))
    url = f"http://127.0.0.1:{servidor.server_address[1]}/"
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    log(f"Visor 3D en {url} (Ctrl+C para cerrar). Arrastra para girar; los botones cambian de animación.")
    webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        servidor.shutdown()


def cmd_vistas(a) -> None:
    from mente import vistas as vs
    _existe(a.imagen)
    vistas = vs.preparar_vistas(an.leer_imagen(a.imagen))
    salida = Path(a.salida) if a.salida else Path(a.imagen).with_name(Path(a.imagen).stem + ".vistas.png")
    ok, buf = cv2.imencode(".png", vs.resumen(vistas))
    buf.tofile(str(salida))
    for rol in vs.ROLES:
        if rol in vistas:
            log(f"  {vs.NOMBRE_ROL[rol]:<30} {'(inventada en espejo)' if vistas[rol].espejo else ''}")
    log(f"Imagen de control: {salida}")
    if a.exportar:
        from mente import colab3d
        rutas = colab3d.exportar_vistas(vistas, a.exportar)
        log(f"Vistas para TRELLIS ({len(rutas)}): {Path(a.exportar).resolve()}")
        for r in rutas:
            log(f"  {r.name}")


def cmd_blender(a) -> None:
    """Versión 2: render en Blender de un modelo con esqueleto (.glb, o .fbx de Mixamo) y sus movimientos."""
    from mente import blender3d
    modelo = Path(a.modelo)
    if not modelo.exists():
        fallar(f"No existe el modelo: {modelo}")
    extra = []
    if a.movimientos_fbx:                # carpeta con los .fbx de Mixamo (sin piel): el nombre del archivo es el movimiento
        for f in sorted(Path(a.movimientos_fbx).glob("*.fbx")):
            extra.append({"archivo": str(f.resolve()), "nombre": f.stem.lower()})
    nombres = [m.strip().lower() for m in a.movimientos.replace("+", ",").split(",") if m.strip()]
    salida = Path(a.salida) if a.salida else SALIDA / f"{modelo.stem}-{'-'.join(nombres)}-blender.webm"
    if salida.suffix.lower() not in an.FORMATOS:
        fallar(f"«{salida.suffix}» no guarda transparencia. Usa .webm, .mov, .gif, .apng o una carpeta.")
    w, h = _lienzo(a.lienzo) or (1080, 1080)
    t = time.time()
    log(f"Blender: {modelo.name} → {' → '.join(nombres)} (vista {a.vista}"
        f"{', órbita %g°' % a.orbita if a.orbita else ''}), {w}x{h} a {a.fps:g} fps…")
    cuadros_dir = SALIDA / f".cuadros_{modelo.stem}"
    try:
        cuadros = blender3d.renderizar(modelo, nombres, cuadros_dir, int(a.fps), (w, h), a.vista, a.orbita,
                                       archivos_movimiento=extra, muestras=a.muestras,
                                       progreso=lambda l: log("  " + l[11:].strip()) if l.startswith("MENTEANIMA:") else None)
    except RuntimeError as e:
        fallar(str(e))
    log(f"  {len(cuadros)} cuadros renderizados en {time.time() - t:.0f} s; guardando…")
    _guardar(blender3d.fotogramas(cuadros), (w, h), len(cuadros), salida, a.fps, t)


def cmd_pose_a(a) -> None:
    """Hoja en pose A dibujada desde el modelo 3D (entrada ideal para TRELLIS)."""
    from mente import trellis3d as t3
    m = _modelo(a.imagen, a.esqueleto, a.detalle)
    hoja = t3.hoja_pose_a(m)
    salida = Path(a.salida) if a.salida else Path(a.imagen).with_name(Path(a.imagen).stem + "_pose_a_3d.png")
    an.guardar_png(salida, hoja)
    log(f"Hoja en pose A: {salida}")
    log("  Úsala con: python menteanima.py trellis " + str(salida))


def _glb_trellis(imagen, pose_a: bool) -> Path:
    """Dónde se guarda el .glb de TRELLIS de una hoja (el de la hoja en pose A lleva «.pose_a»)."""
    return SALIDA / "trellis" / f"{Path(imagen).stem}{'.pose_a' if pose_a else ''}.glb"


def cmd_trellis(a) -> None:
    """Forma 3D de TRELLIS + dibujos originales + esqueleto propio. Compara con el modelo propio.
    Con --pose-a, a TRELLIS va la hoja en pose A dibujada desde el modelo propio (brazos separados del cuerpo)
    y su malla se devuelve a la pose del dibujo: así conserva los dibujos, las manos y los dedos."""
    import shutil
    from mente import colab3d
    from mente import trellis3d as t3
    from mente import vistas as vs
    _existe(a.imagen)
    m = _modelo(a.imagen, a.esqueleto, a.detalle)
    destino = _glb_trellis(a.imagen, a.pose_a)
    if a.glb:
        _existe(a.glb)
        if Path(a.glb).resolve() != destino.resolve():
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(a.glb, destino)                     # donde lo busca «animar --forma trellis»
    else:
        hoja = an.leer_imagen(a.imagen)
        if a.pose_a:
            hoja = t3.hoja_pose_a(m)
            an.guardar_png(SALIDA / "trellis" / f"{Path(a.imagen).stem}_pose_a.png", hoja)
        carpeta = SALIDA / "trellis" / f"vistas_{destino.stem.replace('.', '_')}"
        rutas = colab3d.exportar_vistas(vs.preparar_vistas(hoja), carpeta)
        log(f"Pidiendo el modelo a TRELLIS (demo pública de Hugging Face) con {len(rutas)} vista(s)…")
        log("  Se envían solo las vistas de la hoja; sin token de Hugging Face el cupo gratis de GPU es corto.")
        try:
            destino, modo = t3.pedir_modelo(rutas, destino, progreso=lambda t: log("  " + t))
        except RuntimeError as e:
            fallar(str(e) + "\nOpciones: esperar a que se renueve el cupo (24 h), guardar un token gratuito de "
                   "Hugging Face en credenciales/hf_token.txt, o usar el cuaderno de Colab (menteanima.py colab).")
        log(f"Modelo de TRELLIS ({modo}): {destino}")
    mt = t3.desde_trellis(m, destino, progreso=lambda t: log("  " + t), pose_hoja=t3.POSE_A if a.pose_a else None)
    if a.salida:
        from mente import glb
        r = glb.exportar_glb(mt, a.salida, progreso=lambda t: log("  " + t))
        log(f"Modelo combinado (forma TRELLIS + dibujos + esqueleto): {r}")
    log("Para animarlo: python menteanima.py animar " + a.imagen + " --forma trellis -m caminar")


def cmd_colab(a) -> None:
    """Cuaderno de Google Colab para generar el modelo 3D con TRELLIS (y, si se da la hoja, sus vistas)."""
    from mente import colab3d
    destino = colab3d.cuaderno(a.salida, a.nombre)
    log(f"Cuaderno listo: {destino}")
    if a.imagen:
        from mente import vistas as vs
        _existe(a.imagen)
        carpeta = destino.parent / f"vistas_{Path(a.imagen).stem}"
        rutas = colab3d.exportar_vistas(vs.preparar_vistas(an.leer_imagen(a.imagen)), carpeta)
        log(f"Vistas para subir en la celda 3 ({len(rutas)}): {carpeta}")
    log("Ábrelo en https://colab.research.google.com (Archivo > Subir cuaderno), elige GPU T4 y ejecuta en orden.")


def cmd_esqueleto(a) -> None:
    _existe(a.imagen)
    if a.d2:
        bgra, alfa, J, ruta = an.preparar(a.imagen, a.esqueleto, rehacer=a.rehacer)
    else:
        from mente import modelo3d as m3
        from mente.vistas import preparar_vistas
        vistas = preparar_vistas(an.leer_imagen(a.imagen))
        fr = vistas["frente"]
        bgra, alfa = fr.bgra, fr.alfa
        ruta = Path(a.esqueleto) if a.esqueleto else m3.ruta_esqueleto3d(a.imagen)
        J = an.cargar_esqueleto(ruta, bgra.shape[1::-1]) if ruta.exists() and not a.rehacer else \
            an.detectar_esqueleto(alfa)
    an.guardar_esqueleto(ruta, J, bgra.shape[1::-1])
    vista = ruta.with_suffix(".png")
    ok, buf = cv2.imencode(".png", an.dibujar_esqueleto(bgra, J, alfa))
    buf.tofile(str(vista))
    log(f"Esqueleto: {ruta}\nVista de control: {vista}")
    log("Si algún punto quedó mal, corrígelo en la ventana (python menteanima.py ventana "
        f"\"{a.imagen}\") o edita el .json.")


def cmd_recortar(a) -> None:
    _existe(a.imagen)
    bgra = an.leer_imagen(a.imagen)
    alfa = an.recortar(bgra)
    out = bgra.copy()
    out[..., 3] = alfa
    ys, xs = np.nonzero(alfa)
    out = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    salida = Path(a.salida) if a.salida else Path(a.imagen).with_name(Path(a.imagen).stem + ".recorte.png")
    an.guardar_png(salida, out)
    log(f"Figura sin fondo: {salida}")


def cmd_movimientos(a) -> None:
    from mente import movimientos3d as mv3
    for nombre, m in mv3.MOVIMIENTOS.items():
        ciclo = f", ciclo {m.periodo:g} s" if m.periodo else ""
        log(f"  {nombre:<10} {m.descripcion} ({m.segundos:g} s{ciclo}{', avanza con --desplazar' if m.desplaza else ''})")
    log("Combínalos en orden con comas: -m caminar,señalar,brincar")
    log("Cámara: --vista tres-cuartos | frente | perfil | espalda · --orbita 360 (la cámara da la vuelta)")
    log("Hacia dónde mira: --direccion frente | derecha | izquierda | atras")


def cmd_capturas(a) -> None:
    from mente import mocap as cp
    destino = Path(a.carpeta) if a.carpeta else RAIZ / "capturas" / "cmu"
    log(f"Bajando las capturas de mocap.cs.cmu.edu a {destino} (cada archivo se comprueba con su huella SHA-256)…")
    try:
        listos = cp.descargar_cmu(destino, progreso=lambda t: log("  " + t))
    except (ConnectionError, ValueError) as e:
        fallar(str(e))
    log(f"Listas: {len(listos)} archivos. Úsalas en la ventana («Persona real») o con --captura.")
    log("Si las usas en un video, agradece a CMU: " + cp.AGRADECIMIENTO_CMU)


def cmd_ventana(a) -> None:
    try:
        from mente.animador_ventana import abrir
    except ImportError as e:  # pragma: no cover
        fallar(f"No se pudo abrir la ventana ({e}). Usa: python menteanima.py animar <imagen> -m caminar")
    abrir(a.imagen)


def crear_parser() -> argparse.ArgumentParser:
    from mente import movimientos3d as mv3
    ap = argparse.ArgumentParser(description="MenteAnima — personaje 3D animado desde una imagen, con fondo "
                                             "transparente.")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("animar", help="anima la imagen en 3D y exporta con fondo transparente")
    s.add_argument("imagen")
    s.add_argument("-m", "--movimientos", default="caminar",
                   help="en orden, separados por comas: " + ", ".join(mv3.MOVIMIENTOS))
    s.add_argument("-o", "--salida", help="archivo .webm/.mov/.gif/.apng o carpeta (por defecto salida/animaciones/)")
    s.add_argument("--segundos", type=float, help="duración de cada movimiento (se redondea a ciclos completos)")
    s.add_argument("--fps", type=float, default=30)
    s.add_argument("--altura", type=float, help="alto del personaje en px (por defecto 720, o 60 %% del lienzo)")
    s.add_argument("--lienzo", help="tamaño fijo, p. ej. 1920x1080 (por defecto, el justo para el movimiento)")
    s.add_argument("--vista", choices=VISTAS_CAMARA, default="tres-cuartos", help="desde dónde mira la cámara")
    s.add_argument("--orbita", type=float, default=0.0, help="grados que la cámara gira alrededor (p. ej. 360)")
    s.add_argument("--direccion", default="frente", choices=["frente", "derecha", "izquierda", "atras"],
                   help="hacia dónde mira y avanza el personaje")
    s.add_argument("--desplazar", action="store_true", help="caminar y correr avanzan (si no, en el sitio)")
    s.add_argument("--esqueleto", help="archivo .json del esqueleto (por defecto <imagen>.esqueleto3d.json)")
    s.add_argument("--detalle", type=float, default=0.85, help="densidad del volumen 3D (1 = fino, 0,6 = rápido)")
    s.add_argument("--rapido", action="store_true", help="sin supermuestreo (bordes menos suaves, 3x más rápido)")
    s.add_argument("--2d", dest="d2", action="store_true", help="modo títere 2D (sin volumen)")
    s.add_argument("--forma", choices=["siluetas", "trellis"], default="siluetas",
                   help="volumen propio (siluetas) o la forma de TRELLIS con los dibujos encima")
    s.add_argument("--captura", help="movimiento de captura real: archivo .bvh, o .amc de CMU (con su .asf)")
    s.add_argument("--asf", help="esqueleto .asf del .amc (por defecto, el de la misma carpeta)")
    s.add_argument("--desde", type=float, default=0.0, help="segundo de la captura donde empieza")
    s.add_argument("--hasta", type=float, help="segundo de la captura donde termina (por defecto, al final)")
    s.add_argument("--con-giro", dest="con_giro", action="store_true",
                   help="con la captura, conserva los giros de la persona (si no, mira siempre a la cámara)")
    s.set_defaults(f=cmd_animar)
    s = sub.add_parser("modelo", help="exporta el modelo 3D (.glb) con esqueleto y animaciones")
    s.add_argument("imagen")
    s.add_argument("-o", "--salida")
    s.add_argument("--esqueleto")
    s.add_argument("--detalle", type=float, default=0.75)
    s.set_defaults(f=cmd_modelo)
    s = sub.add_parser("visor", help="abre un modelo .glb en el navegador (girarlo y ver sus animaciones)")
    s.add_argument("modelo")
    s.set_defaults(f=cmd_visor)
    s = sub.add_parser("vistas", help="muestra cómo se reconoció cada vista de la hoja")
    s.add_argument("imagen")
    s.add_argument("-o", "--salida")
    s.add_argument("--exportar", help="carpeta donde guardar cada vista como PNG (entrada de TRELLIS)")
    s.set_defaults(f=cmd_vistas)
    s = sub.add_parser("blender", help="versión 2: render en Blender de un modelo con esqueleto y movimientos")
    s.add_argument("modelo", help=".glb (de menteanima modelo o de TRELLIS) o .fbx con esqueleto (Mixamo)")
    s.add_argument("-m", "--movimientos", default="caminar", help="nombres de las acciones, en orden")
    s.add_argument("--movimientos-fbx", help="carpeta con movimientos .fbx de Mixamo (sin piel)")
    s.add_argument("-o", "--salida")
    s.add_argument("--vista", choices=VISTAS_CAMARA, default="tres-cuartos")
    s.add_argument("--orbita", type=float, default=0.0)
    s.add_argument("--lienzo", help="ANCHOxALTO (por defecto 1080x1080)")
    s.add_argument("--fps", type=float, default=30)
    s.add_argument("--muestras", type=int, default=32, help="calidad del antialias de Blender (16 = rápido)")
    s.set_defaults(f=cmd_blender)
    s = sub.add_parser("pose-a", help="hoja en pose A dibujada desde el modelo 3D (para TRELLIS)")
    s.add_argument("imagen")
    s.add_argument("-o", "--salida")
    s.add_argument("--esqueleto")
    s.add_argument("--detalle", type=float, default=0.85)
    s.set_defaults(f=cmd_pose_a)
    s = sub.add_parser("trellis", help="forma 3D de TRELLIS con los dibujos originales y el esqueleto propio")
    s.add_argument("imagen", help="hoja de personaje original")
    s.add_argument("--pose-a", action="store_true",
                   help="enviar a TRELLIS la hoja en pose A dibujada desde el modelo (recomendado) y devolverla a "
                        "la pose del dibujo: conserva las manos y los dedos")
    s.add_argument("--glb", help="usar este .glb de TRELLIS (p. ej. el del cuaderno de Colab) en vez de pedirlo")
    s.add_argument("-o", "--salida", help="exportar el modelo combinado a .glb")
    s.add_argument("--esqueleto")
    s.add_argument("--detalle", type=float, default=0.85)
    s.set_defaults(f=cmd_trellis)
    s = sub.add_parser("colab", help="cuaderno de Google Colab: modelo 3D con TRELLIS (GPU gratis)")
    s.add_argument("imagen", nargs="?", help="hoja de personaje (exporta también sus vistas)")
    s.add_argument("--nombre", default="mascota")
    s.add_argument("--salida", default=str(SALIDA / "trellis_mentexyz.ipynb"))
    s.set_defaults(f=cmd_colab)
    s = sub.add_parser("esqueleto", help="detecta y guarda el esqueleto (+ imagen de control)")
    s.add_argument("imagen")
    s.add_argument("--esqueleto")
    s.add_argument("--rehacer", action="store_true", help="vuelve a detectarlo aunque ya exista")
    s.add_argument("--2d", dest="d2", action="store_true", help="esqueleto del modo 2D")
    s.set_defaults(f=cmd_esqueleto)
    s = sub.add_parser("recortar", help="guarda solo la figura, en PNG transparente")
    s.add_argument("imagen")
    s.add_argument("-o", "--salida")
    s.set_defaults(f=cmd_recortar)
    sub.add_parser("movimientos", help="lista los movimientos y opciones de cámara").set_defaults(f=cmd_movimientos)
    s = sub.add_parser("capturas", help="baja movimientos de personas reales de Carnegie Mellon (gratis)")
    s.add_argument("--carpeta", help="dónde guardarlas (por defecto, capturas/cmu)")
    s.set_defaults(f=cmd_capturas)
    s = sub.add_parser("ventana", help="ventana para subir la imagen, ajustar el esqueleto, ver en 3D y exportar")
    s.add_argument("imagen", nargs="?")
    s.set_defaults(f=cmd_ventana)
    return ap


def main() -> None:
    consola_utf8()
    a = crear_parser().parse_args()
    if not getattr(a, "f", None):
        a = crear_parser().parse_args(["ventana"])
    a.f(a)


if __name__ == "__main__":
    main()
