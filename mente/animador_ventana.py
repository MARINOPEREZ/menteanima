# -*- coding: utf-8 -*-
"""Ventana de MenteAnima 3D (Tkinter, viene con Python): subir la hoja del personaje, revisar las vistas,
corregir el esqueleto arrastrando sus puntos, elegir movimientos (hechos a mano o de una persona real: capturas
de Carnegie Mellon o un .bvh), ver el personaje en 3D (se gira arrastrando el ratón, la rueda acerca y aleja) y
exportar el video con fondo transparente o el modelo .glb."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from mente import animacion as an
from mente import mocap as cp
from mente import modelo3d as m3
from mente import movimientos3d as mv3
from mente.comun import RAIZ

LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
LIENZOS = {"justo para el movimiento": None, "1920x1080 (horizontal)": (1920, 1080),
           "1080x1920 (vertical, Shorts)": (1080, 1920), "1280x720": (1280, 720), "1080x1080": (1080, 1080)}
COLOR = {"izq": "#ff9f1c", "der": "#3a86ff", "centro": "#e63946"}
FONDO = "#2b2d42"
SIN_CAPTURA = "(ninguna: usa la lista de arriba)"


_CUADROS: dict = {}


def _sobre_cuadros(bgra: np.ndarray, lado: int = 12) -> Image.Image:
    """La imagen con alfa sobre cuadros grises (así se ve qué es transparente). Los cuadros se arman una vez por
    tamaño y la mezcla la hace OpenCV: en numpy con decimales costaba ~110 ms por cuadro y trababa la vista previa."""
    h, w = bgra.shape[:2]
    fondo = _CUADROS.get((h, w, lado))
    if fondo is None:
        iy, ix = np.indices((h, w))
        g = ((((iy // lado) + (ix // lado)) % 2) * 45 + 185).astype(np.uint8)
        fondo = cv2.merge([g, g, g])
        _CUADROS.clear()
        _CUADROS[(h, w, lado)] = fondo
    bgra = np.ascontiguousarray(bgra)
    a = cv2.extractChannel(bgra, 3)
    a3 = cv2.merge([a, a, a])
    figura = cv2.multiply(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB), a3, scale=1 / 255)
    return Image.fromarray(cv2.add(figura, cv2.multiply(fondo, cv2.bitwise_not(a3), scale=1 / 255)))


def _soltar_boton(e) -> str | None:
    """Un botón se activa si se suelta el clic DENTRO de él. Tk exige además que no haya llegado un «salió» en el
    medio, y con grabadores de pantalla, escritorio remoto o controles automáticos llega uno falso: el clic se perdía
    (sobre todo al reproducir)."""
    w = e.widget
    if not w.instate(["pressed"]):
        return None
    w.state(["!pressed"])
    if 0 <= e.x < w.winfo_width() and 0 <= e.y < w.winfo_height() and w.instate(["!disabled"]):
        w.invoke()
    return "break"


class Ventana:
    RADIO = 7

    def __init__(self, raiz: tk.Tk, imagen: str | None = None):
        self.r = raiz
        raiz.title("MenteAnima 3D — anima tu personaje")
        self.esc = max(1.0, raiz.winfo_fpixels("1i") / 96)     # escala de Windows (125 %, 150 %…)
        sw, sh = raiz.winfo_screenwidth(), raiz.winfo_screenheight()
        raiz.geometry(f"{min(int(1240 * self.esc), sw - 60)}x{min(int(860 * self.esc), sh - 90)}")
        raiz.minsize(min(int(760 * self.esc), sw - 60), min(int(620 * self.esc), sh - 90))
        self.cola: queue.Queue = queue.Queue()
        self.ocupado = False
        self.ruta_img: Path | None = None
        self.vistas = None
        self.m = None                 # modelo 3D
        self.J2 = None                # esqueleto 2D del frente (editable)
        self.gl = None
        self.gl_de = None
        self.modo = "esqueleto"
        self.az, self.el, self.zoom = 30.0, 8.0, 1.0
        self.poses, self.desps, self.i = [], [], 0
        self.tocando = False
        self._sonando = threading.Event()      # el hilo que dibuja la reproducción (la ventana queda libre)
        self._hilo_3d = None
        self._cuadro: queue.Queue = queue.Queue(maxsize=1)
        self._tam = (10, 10)
        self.arrastre = None
        self.orbitando = None
        self.k, self.ox, self.oy = 1.0, 0, 0
        self.capturas: dict = {}      # etiqueta → ruta (.amc / .bvh)
        self._leidas: dict = {}       # ruta → Captura ya leída
        self._armar()
        self.r.after(60, self._atender)
        if imagen:
            self.cargar(imagen)

    # ---------------------------------------------------------------------------------------- interfaz
    def _armar(self) -> None:
        estilo = ttk.Style()
        try:
            estilo.theme_use("clam")
        except tk.TclError:
            pass
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 10, "bold"))
        for clase in ("TButton", "TRadiobutton", "TCheckbutton"):
            self.r.bind_class(clase, "<ButtonRelease-1>", _soltar_boton)
        izq = ttk.Frame(self.r)
        izq.pack(side="left", fill="both", expand=True)
        pestanas = ttk.Frame(izq, padding=(6, 4))
        pestanas.pack(fill="x")
        self.v_modo = tk.StringVar(value="esqueleto")
        for valor, texto in (("esqueleto", "1 · Esqueleto"), ("3d", "2 · Ver en 3D"), ("vistas", "Vistas de la hoja")):
            ttk.Radiobutton(pestanas, text=texto, value=valor, variable=self.v_modo, style="Toolbutton",
                            command=lambda: self.cambiar_modo(self.v_modo.get())).pack(side="left", padx=2)
        self.ayuda = ttk.Label(pestanas, text="", foreground="#666")
        self.ayuda.pack(side="right")
        self.lienzo = tk.Canvas(izq, background=FONDO, highlightthickness=0)
        self.lienzo.pack(fill="both", expand=True)
        self.estado = ttk.Label(izq, text="Sube la hoja de tu personaje (frente, perfil y espalda) o una sola imagen.",
                                anchor="w", padding=(8, 4))
        self.estado.pack(fill="x")
        self.barra = ttk.Progressbar(izq, mode="determinate")
        self.barra.pack(fill="x")

        der = ttk.Frame(self.r, padding=12, width=int(330 * self.esc))
        der.pack(side="right", fill="y")
        der.pack_propagate(False)
        ttk.Button(der, text="Subir imagen…", command=self.elegir).pack(fill="x")
        self.l_vistas = ttk.Label(der, text="", wraplength=int(300 * self.esc), foreground="#555")
        self.l_vistas.pack(anchor="w", pady=(4, 0))
        fila = ttk.Frame(der)
        fila.pack(fill="x", pady=(6, 0))
        ttk.Button(fila, text="Detectar esqueleto", command=self.redetectar).pack(side="left", expand=True, fill="x")
        ttk.Button(fila, text="Rehacer 3D", command=self.reconstruir).pack(side="left", expand=True, fill="x",
                                                                           padx=(4, 0))

        ttk.Label(der, text="Movimientos (en orden)", style="Titulo.TLabel").pack(anchor="w", pady=(8, 2))
        fila = ttk.Frame(der)
        fila.pack(fill="x")
        self.v_mov = tk.StringVar(value="caminar")
        ttk.Combobox(fila, textvariable=self.v_mov, values=list(mv3.MOVIMIENTOS), state="readonly",
                     width=14).pack(side="left", expand=True, fill="x")
        ttk.Button(fila, text="Agregar", command=self.agregar).pack(side="left", padx=(4, 0))
        self.lista = tk.Listbox(der, height=3, activestyle="none")
        self.lista.pack(fill="x", pady=4)
        self.lista.insert("end", "caminar")
        fila = ttk.Frame(der)
        fila.pack(fill="x")
        ttk.Button(fila, text="Quitar", command=self.quitar).pack(side="left", expand=True, fill="x")
        ttk.Button(fila, text="Limpiar", command=lambda: self.lista.delete(0, "end")).pack(
            side="left", expand=True, fill="x", padx=(4, 0))

        ttk.Label(der, text="Persona real (captura de movimiento)", style="Titulo.TLabel").pack(anchor="w",
                                                                                              pady=(10, 2))
        fila = ttk.Frame(der)
        fila.pack(fill="x")
        self.v_cap = tk.StringVar(value=SIN_CAPTURA)
        self.c_cap = ttk.Combobox(fila, textvariable=self.v_cap, state="readonly", width=22)
        self.c_cap.pack(side="left", expand=True, fill="x")
        ttk.Button(fila, text="Abrir…", command=self.abrir_captura).pack(side="left", padx=(4, 0))
        self._listar_capturas()
        self.v_cap.trace_add("write", lambda *_: self._preparar_poses())

        ttk.Label(der, text="Ajustes", style="Titulo.TLabel").pack(anchor="w", pady=(8, 2))
        rejilla = ttk.Frame(der)
        rejilla.pack(fill="x")
        self.v_seg = tk.StringVar(value="auto")
        self.v_dir = tk.StringVar(value="frente")
        self.v_cam = tk.StringVar(value="tres-cuartos")
        self.v_orb = tk.StringVar(value="0")
        self.v_desp = tk.BooleanVar(value=False)
        self.v_alt = tk.StringVar(value="720")
        self.v_lienzo = tk.StringVar(value=list(LIENZOS)[0])
        self.v_fps = tk.StringVar(value="30")
        filas = [("Segundos c/u", ttk.Combobox(rejilla, textvariable=self.v_seg, values=["auto", "1", "2", "3", "4", "6"],
                                               width=10)),
                 ("Mira hacia", ttk.Combobox(rejilla, textvariable=self.v_dir, state="readonly", width=10,
                                             values=["frente", "derecha", "izquierda", "atras"])),
                 ("Cámara", ttk.Combobox(rejilla, textvariable=self.v_cam, state="readonly", width=10,
                                         values=["tres-cuartos", "frente", "perfil", "espalda"])),
                 ("Órbita (°)", ttk.Combobox(rejilla, textvariable=self.v_orb, values=["0", "90", "180", "360"],
                                             width=10)),
                 ("Alto (px)", ttk.Spinbox(rejilla, textvariable=self.v_alt, from_=100, to=2000, increment=20, width=10)),
                 ("Cuadros/s", ttk.Combobox(rejilla, textvariable=self.v_fps, values=["24", "25", "30", "60"], width=10)),
                 ("Lienzo", ttk.Combobox(rejilla, textvariable=self.v_lienzo, values=list(LIENZOS), state="readonly",
                                         width=24))]
        for i, (texto, w) in enumerate(filas):
            ttk.Label(rejilla, text=texto).grid(row=i, column=0, sticky="w")
            w.grid(row=i, column=1, sticky="ew", pady=1)
        rejilla.columnconfigure(1, weight=1)
        for v in (self.v_dir, self.v_cam):
            v.trace_add("write", lambda *_: self._preparar_poses())
        ttk.Checkbutton(der, text="Caminar y correr avanzan", variable=self.v_desp,
                        command=self._preparar_poses).pack(anchor="w", pady=(4, 0))

        ttk.Label(der, text="Resultado (fondo transparente)", style="Titulo.TLabel").pack(anchor="w", pady=(8, 2))
        self.b_tocar = ttk.Button(der, text="▶  Reproducir en 3D", command=self.tocar)
        self.b_tocar.pack(fill="x")
        ttk.Button(der, text="Exportar video…", command=self.exportar).pack(fill="x", pady=(4, 0))
        ttk.Button(der, text="Exportar modelo 3D (.glb)…", command=self.exportar_glb).pack(fill="x", pady=(4, 0))
        ttk.Label(der, text="WebM/MOV con transparencia para editores de video; .glb para Blender y visores 3D.",
                  wraplength=int(300 * self.esc), foreground="#555").pack(anchor="w", pady=(6, 0))

        c = self.lienzo
        c.bind("<Configure>", lambda e: self.redibujar())
        c.bind("<ButtonPress-1>", self._presionar)
        c.bind("<B1-Motion>", self._arrastrar)
        c.bind("<ButtonRelease-1>", self._soltar)
        c.bind("<Motion>", self._pasar)
        c.bind("<MouseWheel>", self._rueda)

    def avisar(self, texto: str) -> None:
        self.estado.configure(text=texto)

    def _params(self, silencioso: bool = False) -> dict | None:
        nombres = list(self.lista.get(0, "end"))
        if not nombres:
            if not silencioso:
                messagebox.showinfo("MenteAnima", "Agrega al menos un movimiento.")
            return None
        try:
            seg = None if self.v_seg.get().strip().lower() in ("", "auto") else float(self.v_seg.get())
            return {"nombres": nombres, "segundos": seg, "altura": float(self.v_alt.get()),
                    "fps": float(self.v_fps.get()), "direccion": self.v_dir.get(), "desplazar": self.v_desp.get(),
                    "lienzo": LIENZOS[self.v_lienzo.get()], "vista": self.v_cam.get(),
                    "orbita": float(self.v_orb.get() or 0), "captura": self.capturas.get(self.v_cap.get())}
        except ValueError:
            if not silencioso:
                messagebox.showerror("MenteAnima", "Revisa los números de segundos, alto, cuadros/s y órbita.")
            return None

    # ---------------------------------------------------------------------------------------- imagen y modelo
    def elegir(self) -> None:
        ruta = filedialog.askopenfilename(title="Hoja o imagen del personaje", filetypes=[
            ("Imágenes", "*.png *.jpg *.jpeg *.webp *.bmp"), ("Todos", "*.*")])
        if ruta:
            self.cargar(ruta)

    def _en_hilo(self, texto: str, funcion, indeterminado: bool = False) -> bool:
        if self.ocupado:
            return False
        self.ocupado = True
        self.avisar(texto)
        if indeterminado:
            self.barra.configure(mode="indeterminate")
            self.barra.start(12)
        else:
            self.barra.configure(mode="determinate", value=0)

        def hilo():
            try:
                funcion()
            except Exception as e:
                self.cola.put(("error", str(e)))
        threading.Thread(target=hilo, daemon=True).start()
        return True

    def cargar(self, ruta: str) -> None:
        self.tocando = False

        def trabajo():
            from mente.vistas import preparar_vistas
            vistas = preparar_vistas(an.leer_imagen(ruta))
            rj = m3.ruta_esqueleto3d(ruta)
            fr = vistas["frente"]
            J2 = an.cargar_esqueleto(rj, fr.bgra.shape[1::-1]) if rj.exists() else an.detectar_esqueleto(fr.alfa)
            self.cola.put(("vistas", ruta, vistas, J2, rj.exists()))
            m = m3.construir(vistas, J2, 0.85, progreso=lambda s: self.cola.put(("aviso", s)))
            self.cola.put(("modelo", m, J2))
        self._en_hilo(f"Leyendo {Path(ruta).name}…", trabajo, indeterminado=True)

    def reconstruir(self) -> None:
        if self.vistas is None:
            return
        J2 = dict(self.J2)
        vistas = self.vistas

        def trabajo():
            m = m3.construir(vistas, J2, 0.85, progreso=lambda s: self.cola.put(("aviso", s)))
            self.cola.put(("modelo", m, J2))
        self._en_hilo("Rehaciendo el modelo 3D con el esqueleto corregido…", trabajo, indeterminado=True)

    def redetectar(self) -> None:
        if self.vistas is None:
            return
        self.J2 = an.detectar_esqueleto(self.vistas["frente"].alfa)
        self.cambiar_modo("esqueleto")
        self.avisar("Esqueleto detectado de nuevo. Pulsa «Rehacer 3D» para aplicarlo.")

    def guardar_esqueleto(self) -> None:
        if self.J2 is None or self.ruta_img is None:
            return
        try:
            an.guardar_esqueleto(m3.ruta_esqueleto3d(self.ruta_img), self.J2,
                                 self.vistas["frente"].bgra.shape[1::-1])
        except OSError:
            pass

    def _listar_capturas(self) -> None:
        self.capturas.update(cp.buscar_capturas(RAIZ))
        self.c_cap.configure(values=[SIN_CAPTURA] + list(self.capturas))

    def abrir_captura(self) -> None:
        ruta = filedialog.askopenfilename(title="Captura de movimiento", filetypes=[
            ("Capturas", "*.bvh *.amc"), ("Todos", "*.*")])
        if ruta:
            self.capturas[Path(ruta).stem] = Path(ruta)
            self.c_cap.configure(values=[SIN_CAPTURA] + list(self.capturas))
            self.v_cap.set(Path(ruta).stem)

    def _captura(self, ruta: Path):
        if ruta not in self._leidas:
            self._leidas[ruta] = cp.leer(ruta)
        return self._leidas[ruta]

    def agregar(self) -> None:
        self.lista.insert("end", self.v_mov.get())
        self._preparar_poses()

    def quitar(self) -> None:
        for i in reversed(self.lista.curselection() or (self.lista.size() - 1,)):
            if i >= 0:
                self.lista.delete(i)
        self._preparar_poses()

    # ---------------------------------------------------------------------------------------- dibujo
    def cambiar_modo(self, modo: str) -> None:
        self.modo = modo
        self.v_modo.set(modo)
        ayudas = {"esqueleto": "Arrastra los puntos (naranja = izquierda de la imagen, azul = derecha)",
                  "3d": "Arrastra para girar la cámara · rueda para acercar", "vistas": "Así se reconoció la hoja"}
        self.ayuda.configure(text=ayudas.get(modo, ""))
        self.redibujar()

    def redibujar(self) -> None:
        c = self.lienzo
        cw, ch = max(10, c.winfo_width()), max(10, c.winfo_height())
        if self.modo == "3d":
            self._dibujar_3d()
            return
        c.delete("all")
        if self.vistas is None:
            c.create_text(cw / 2, ch / 2, fill="#edf2f4", font=("Segoe UI", 13), justify="center",
                          text="Sube la hoja de tu personaje\n(frente, perfil y espalda, de cuerpo entero)\n"
                               "o una sola imagen de frente")
            return
        if self.modo == "vistas":
            from mente.vistas import resumen
            im = Image.fromarray(resumen(self.vistas, 420)[..., ::-1].copy())
            s = min(1.0, (cw - 20) / im.width, (ch - 20) / im.height)
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), LANCZOS)
            self._img_tk = ImageTk.PhotoImage(im)
            c.create_image(cw / 2, ch / 2, image=self._img_tk)
            return
        fr = self.vistas["frente"]
        h, w = fr.bgra.shape[:2]
        self.k = min((cw - 20) / w, (ch - 20) / h)
        self.ox, self.oy = (cw - w * self.k) / 2, (ch - h * self.k) / 2
        im = self._compuesta.resize((max(1, round(w * self.k)), max(1, round(h * self.k))), LANCZOS)
        self._img_tk = ImageTk.PhotoImage(im)
        c.create_image(self.ox, self.oy, image=self._img_tk, anchor="nw")
        g = max(2, round(3 * min(1.5, self.k * 2)))
        for a, b in an.LINEAS:
            lado = "izq" if "izq" in a + b else ("der" if "der" in a + b else "centro")
            c.create_line(*self._a_lienzo(self.J2[a]), *self._a_lienzo(self.J2[b]), fill="white", width=g + 2,
                          tags=(f"l:{a}:{b}",))
            c.create_line(*self._a_lienzo(self.J2[a]), *self._a_lienzo(self.J2[b]), fill=COLOR[lado], width=g,
                          tags=(f"l:{a}:{b}",))
        r = self.RADIO
        for k in an.ARTICULACIONES:
            x, y = self._a_lienzo(self.J2[k])
            lado = k[-3:] if k[-3:] in ("izq", "der") else "centro"
            c.create_oval(x - r, y - r, x + r, y + r, fill=COLOR[lado], outline="white", width=2, tags=("art", f"a:{k}"))

    def _dibujar_3d(self) -> None:
        c = self.lienzo
        cw, ch = max(10, c.winfo_width()), max(10, c.winfo_height())
        if self.m is None:
            c.delete("all")
            c.create_text(cw / 2, ch / 2, fill="#edf2f4", font=("Segoe UI", 13),
                          text="Armando el modelo 3D…" if self.vistas is not None else "Sube una imagen primero")
            return
        if self.gl_de is not self.m:
            from mente import render_gl
            self.gl = render_gl.renderizador(self.m)
            self.gl_de = self.m
        self._mostrar(self._imagen_3d(self.m, self.i, cw, ch, self.gl))

    def _imagen_3d(self, m, i: int, cw: int, ch: int, gl) -> Image.Image:
        """Un cuadro del modelo `m` (pose `i`) sobre los cuadros grises. Sirve en cualquier hilo: `gl` es el
        renderizador de ESE hilo (OpenGL no se comparte entre hilos) o None para el de numpy."""
        poses, desps = self.poses, self.desps
        if poses:
            k = i % len(poses)
            pose, desp = poses[k], desps[k] if k < len(desps) else (0.0, 0.0)
        else:
            pose, desp = mv3.respirar(0.0, m), (0.0, 0.0)
        cam = m3.Camara(azimut=self.az, elevacion=self.el, objetivo=(desp[0], 0.5 * m.altura, desp[1]))
        escala = 0.78 * ch / m.altura * self.zoom
        origen, tam = (-cw / 2, -ch / 2), (cw, ch)
        if gl is not None:
            img = gl.render(pose, cam, escala, origen, tam, desp)
        else:                                   # sin tarjeta gráfica: a media resolución y sin supermuestreo
            s = 0.5
            P = m3.posar3d(m, pose)
            P.V[:, 0] += desp[0]
            P.V[:, 2] += desp[1]
            img = m3.renderizar3d(m, P, cam, escala * s, (origen[0] * s, origen[1] * s),
                                  (int(cw * s), int(ch * s)), ss=1)
            img = np.asarray(Image.fromarray(img).resize((cw, ch)))
        return _sobre_cuadros(img, 16)

    def _mostrar(self, imagen: Image.Image) -> None:
        self._img_tk = ImageTk.PhotoImage(imagen)
        self.lienzo.delete("all")
        self.lienzo.create_image(0, 0, image=self._img_tk, anchor="nw")

    def _bucle_3d(self) -> None:
        """Hilo aparte que dibuja la reproducción a 30 cuadros/s. Antes se dibujaba en el hilo de la ventana
        (~70 ms por cuadro) y los clics no alcanzaban a entrar: no se podía pausar ni cambiar de movimiento."""
        from mente import render_gl
        gl, de = None, None
        while True:
            self._sonando.wait()
            m = self.m
            if m is None:
                time.sleep(0.05)
                continue
            if de is not m:
                gl, de = render_gl.renderizador(m), m
            t = time.time()
            try:
                imagen = self._imagen_3d(m, int((t - self.t0) * 30), *self._tam, gl)
            except Exception:                   # p. ej. el modelo cambió en medio del cuadro: se salta
                time.sleep(0.05)
                continue
            try:
                self._cuadro.get_nowait()       # solo importa el último cuadro
            except queue.Empty:
                pass
            self._cuadro.put(imagen)
            time.sleep(max(0.0, 1 / 30 - (time.time() - t)))

    def _a_lienzo(self, p) -> tuple:
        return self.ox + p[0] * self.k, self.oy + p[1] * self.k

    def _a_imagen(self, x, y) -> tuple:
        h, w = self.vistas["frente"].bgra.shape[:2]
        return (min(w - 1, max(0, (x - self.ox) / self.k)), min(h - 1, max(0, (y - self.oy) / self.k)))

    def _cercana(self, x, y):
        if self.J2 is None or self.modo != "esqueleto":
            return None
        mejor, dist = None, 14.0
        for k in an.ARTICULACIONES:
            px, py = self._a_lienzo(self.J2[k])
            d = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            if d < dist:
                mejor, dist = k, d
        return mejor

    def _presionar(self, e) -> None:
        if self.modo == "3d":
            self.orbitando = (e.x, e.y, self.az, self.el)
            return
        self.arrastre = self._cercana(e.x, e.y)

    def _arrastrar(self, e) -> None:
        if self.modo == "3d" and self.orbitando:
            x0, y0, az, el = self.orbitando
            self.az = az - (e.x - x0) * 0.5
            self.el = max(-30.0, min(60.0, el + (e.y - y0) * 0.3))
            if not self.tocando:
                self._dibujar_3d()
            return
        k = self.arrastre
        if not k:
            return
        self.J2[k] = self._a_imagen(e.x, e.y)
        x, y = self._a_lienzo(self.J2[k])
        r = self.RADIO
        self.lienzo.coords(f"a:{k}", x - r, y - r, x + r, y + r)
        for a, b in an.LINEAS:
            if k in (a, b):
                for item in self.lienzo.find_withtag(f"l:{a}:{b}"):
                    self.lienzo.coords(item, *self._a_lienzo(self.J2[a]), *self._a_lienzo(self.J2[b]))
        self.lienzo.tag_raise("art")
        self.avisar(f"{an.NOMBRES[k]}: ({self.J2[k][0]:.0f}, {self.J2[k][1]:.0f}) — pulsa «Rehacer 3D» al terminar")

    def _soltar(self, e) -> None:
        if self.arrastre:
            self.guardar_esqueleto()
        self.arrastre = None
        self.orbitando = None

    def _pasar(self, e) -> None:
        if self.modo == "3d":
            self.lienzo.configure(cursor="fleur")
            return
        k = self._cercana(e.x, e.y)
        self.lienzo.configure(cursor="fleur" if k else "arrow")
        if k and not self.arrastre:
            self.avisar(f"{an.NOMBRES[k]} — arrástralo a su lugar")

    def _rueda(self, e) -> None:
        if self.modo == "3d":
            self.zoom = max(0.3, min(3.0, self.zoom * (1.1 if e.delta > 0 else 1 / 1.1)))
            if not self.tocando:
                self._dibujar_3d()

    # ---------------------------------------------------------------------------------------- reproducir y exportar
    def _preparar_poses(self) -> None:
        if self.m is None:
            return
        q = self._params(silencioso=True)
        if not q:
            self.poses, self.desps = [], []
            return
        rumbo = mv3.DIRECCIONES.get(q["direccion"], 0.0)
        if q["captura"]:                        # una persona real: la lista de movimientos no se usa
            try:
                cap = self._captura(q["captura"])
            except (OSError, ValueError) as e:
                messagebox.showerror("MenteAnima", f"No pude leer la captura: {e}")
                self.v_cap.set(SIN_CAPTURA)
                return
            self.poses, self.desps = cp.poses_desde_captura(self.m, cap, 30, 0.0, None, q["desplazar"])
            for pose in self.poses:
                pose["rumbo"] = pose.get("rumbo", 0.0) + rumbo
            self.avisar(f"Persona real: {cap.segundos:.1f} s de movimiento ({self.v_cap.get()}).")
        else:
            self.poses, self.desps = mv3.poses_de(self.m, mv3.preparar_pasos(q["nombres"], q["segundos"]), 30,
                                                  rumbo, q["desplazar"])
        self.az = {"frente": 0.0, "tres-cuartos": 30.0, "perfil": 90.0, "espalda": 180.0}[q["vista"]]
        self.i = 0
        if self.modo == "3d" and not self.tocando:
            self._dibujar_3d()

    def tocar(self) -> None:
        if self.m is None:
            messagebox.showinfo("MenteAnima", "Primero sube una imagen y espera a que se arme el modelo 3D.")
            return
        if self.tocando:
            self.tocando = False
            self._sonando.clear()
            self.b_tocar.configure(text="▶  Reproducir en 3D")
            return
        self._preparar_poses()
        if not self.poses:
            return
        self.cambiar_modo("3d")
        self._tam = (max(10, self.lienzo.winfo_width()), max(10, self.lienzo.winfo_height()))
        self.tocando = True
        self.t0 = time.time() - self.i / 30
        self.b_tocar.configure(text="■  Pausar")
        if self._hilo_3d is None:
            self._hilo_3d = threading.Thread(target=self._bucle_3d, daemon=True)
            self._hilo_3d.start()
        self._sonando.set()
        self._paso()

    def _paso(self) -> None:
        """En la ventana solo se muestra el último cuadro que dejó listo el hilo de dibujo."""
        if not self.tocando or self.modo != "3d":
            self.tocando = False
            self._sonando.clear()
            self.b_tocar.configure(text="▶  Reproducir en 3D")
            return
        self._tam = (max(10, self.lienzo.winfo_width()), max(10, self.lienzo.winfo_height()))
        self.i = int((time.time() - self.t0) * 30) % max(1, len(self.poses))   # a velocidad real
        try:
            self._mostrar(self._cuadro.get_nowait())
        except queue.Empty:
            pass
        self.r.after(15, self._paso)

    def exportar(self) -> None:
        if self.m is None:
            messagebox.showinfo("MenteAnima", "Primero sube una imagen y espera a que se arme el modelo 3D.")
            return
        q = self._params()
        if not q:
            return
        carpeta = RAIZ / "salida" / "animaciones"
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = filedialog.asksaveasfilename(
            title="Exportar con fondo transparente", initialdir=str(carpeta),
            initialfile=f"{self.ruta_img.stem}-{Path(q['captura']).stem if q['captura'] else '-'.join(q['nombres'])}",
            defaultextension=".webm",
            filetypes=[("WebM con transparencia", "*.webm"), ("MOV ProRes 4444 con transparencia", "*.mov"),
                       ("GIF animado", "*.gif"), ("PNG animado", "*.apng")])
        if not ruta:
            return
        m = self.m
        cap = self._captura(q["captura"]) if q["captura"] else None

        def trabajo():
            fotos, tam, n = mv3.animar(m, q["nombres"], q["segundos"], q["fps"], q["direccion"], q["desplazar"],
                                       q["altura"], q["lienzo"], q["vista"], q["orbita"], captura=cap)
            an.exportar(fotos, ruta, q["fps"], tam, lambda i: self.cola.put(("progreso", i, n)))
            self.cola.put(("exportado", ruta, f"{tam[0]}x{tam[1]}, {n} fotogramas, fondo transparente"))
        self._en_hilo("Exportando el video…", trabajo)

    def exportar_glb(self) -> None:
        if self.m is None:
            messagebox.showinfo("MenteAnima", "Primero sube una imagen y espera a que se arme el modelo 3D.")
            return
        carpeta = RAIZ / "salida" / "animaciones"
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = filedialog.asksaveasfilename(title="Exportar modelo 3D", initialdir=str(carpeta),
                                            initialfile=self.ruta_img.stem, defaultextension=".glb",
                                            filetypes=[("Modelo glTF binario", "*.glb")])
        if not ruta:
            return
        m = self.m

        def trabajo():
            from mente import glb
            r = glb.exportar_glb(m, ruta, progreso=lambda s: self.cola.put(("aviso", s)))
            self.cola.put(("exportado", str(r), "modelo 3D con esqueleto y una animación por movimiento"))
        self._en_hilo("Exportando el modelo 3D…", trabajo, indeterminado=True)

    # ---------------------------------------------------------------------------------------- mensajes
    def _fin_trabajo(self) -> None:
        self.ocupado = False
        self.barra.stop()
        self.barra.configure(mode="determinate", value=0)

    def _atender(self) -> None:
        try:
            while True:
                msj = self.cola.get_nowait()
                tipo = msj[0]
                if tipo == "aviso":
                    self.avisar(msj[1])
                elif tipo == "vistas":
                    _, ruta, self.vistas, self.J2, guardado = msj
                    self.ruta_img = Path(ruta)
                    self.m = None
                    self._compuesta = _sobre_cuadros(self.vistas["frente"].bgra)
                    reales = [r for r, v in self.vistas.items() if not v.espejo]
                    texto = f"{len(reales)} vista(s) reales: {', '.join(reales)}."
                    if len(reales) == 1:
                        texto += " Con una sola, el 3D es aproximado (espalda en espejo)."
                    self.l_vistas.configure(text=texto)
                    self.cambiar_modo("esqueleto")
                    self.avisar(f"{self.ruta_img.name}: {texto} Armando el modelo 3D…")
                elif tipo == "modelo":
                    self._fin_trabajo()
                    self.m, self.J2 = msj[1], dict(msj[2])
                    self._preparar_poses()
                    self.avisar(f"Modelo 3D listo ({len(self.m.V)} vértices). Abre «Ver en 3D» y arrastra para girarlo.")
                    if self.modo == "3d":
                        self._dibujar_3d()
                elif tipo == "progreso":
                    self.barra.configure(maximum=msj[2], value=msj[1])
                    self.avisar(f"{msj[1]}/{msj[2]} fotogramas…")
                elif tipo == "exportado":
                    self._fin_trabajo()
                    self.avisar(f"Listo: {Path(msj[1]).name} — {msj[2]}.")      # la ruta completa, en el aviso
                    messagebox.showinfo("MenteAnima", f"Guardado en:\n{msj[1]}")
                elif tipo == "error":
                    self._fin_trabajo()
                    self.avisar("Error: " + msj[1])
                    messagebox.showerror("MenteAnima", msj[1])
        except queue.Empty:
            pass
        self.r.after(60, self._atender)


def abrir(imagen: str | None = None) -> None:
    try:                                  # nítida en pantallas con escala de Windows
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    raiz = tk.Tk()
    Ventana(raiz, imagen)
    raiz.mainloop()
