# -*- coding: utf-8 -*-
"""Render 3D por GPU (OpenGL 3.3 con moderngl) para MenteAnima: el mismo resultado que el rasterizador de numpy
de modelo3d.py, cientos de veces más rápido (vista previa que se gira con el ratón, videos en segundos).
Los huesos deforman la malla en el sombreador de vértices; el de fragmentos pinta cada punto con las vistas que
lo miran de frente y que de verdad lo ven (mapa de profundidad de cada vista). Si no hay OpenGL, se usa numpy."""
from __future__ import annotations

import threading

import numpy as np

from mente import modelo3d as m3
from mente.animacion import HUESOS, ORDEN
from mente.vistas import DIRECCION

VERTICES = """
#version 330
in vec3 en_pos; in vec3 en_nor; in vec4 en_j; in vec4 en_w; in float en_g; in vec3 en_c;
uniform mat4 huesos[48];
uniform mat4 clip;
uniform vec3 cam_pos; uniform vec3 cam_dir; uniform int perspectiva;
out vec3 v_reposo; out vec3 v_nreposo; out vec3 v_npose; out vec3 v_cam; flat out float v_tipo; out vec3 v_respaldo;
void main() {
    v_tipo = en_g;
    v_respaldo = en_c;
    mat4 M = en_w.x * huesos[int(en_j.x)] + en_w.y * huesos[int(en_j.y)]
           + en_w.z * huesos[int(en_j.z)] + en_w.w * huesos[int(en_j.w)];
    vec4 p = M * vec4(en_pos, 1.0);
    gl_Position = clip * p;
    v_reposo = en_pos;
    v_nreposo = en_nor;
    v_npose = mat3(M) * en_nor;
    vec3 hacia = perspectiva == 1 ? normalize(cam_pos - p.xyz) : cam_dir;
    v_cam = transpose(mat3(M)) * hacia;          // la cámara vista desde el reposo de este punto
}
"""

FRAGMENTOS = """
#version 330
in vec3 v_reposo; in vec3 v_nreposo; in vec3 v_npose; in vec3 v_cam; flat in float v_tipo; in vec3 v_respaldo;
uniform sampler2D tex0; uniform sampler2D tex1; uniform sampler2D tex2; uniform sampler2D tex3;
uniform sampler2D prof0; uniform sampler2D prof1; uniform sampler2D prof2; uniform sampler2D prof3;
uniform vec3 eje[4]; uniform vec3 dir[4]; uniform vec4 geo[4]; uniform float factor[4];
uniform int nvistas; uniform float tol; uniform vec3 luz_dir; uniform float luz; uniform float nitidez; uniform float nitidez_cabeza; uniform vec3 color_mano; uniform int hay_mano;
uniform int hay_respaldo;
uniform int guante; uniform float x_centro; uniform vec4 g_e[2]; uniform vec4 g_s[2]; uniform vec3 g_rango[2];
uniform vec3 g_lineas[2]; uniform vec3 g_dorso[2]; uniform vec3 g_color[2]; uniform vec3 color_costura;
uniform vec4 g_pliegues[2]; uniform vec4 g_bases[2]; uniform float g_punta[2]; uniform vec3 sombra_guante;
out vec4 color;
vec3 C;

vec4 acum; float pesos;

void muestra(sampler2D t, sampler2D p, int i, vec3 N, int modo) {
    if (i >= nvistas) return;
    float cs = dot(N, dir[i]);
    vec2 uv = vec2(geo[i].x + dot(eje[i], v_reposo) + 0.5, geo[i].y - v_reposo.y + 0.5) / geo[i].zw;
    vec4 c = texture(t, uv);
    vec2 pz = texture(p, uv).rg;                 // r: profundidad en reposo; g: ¿es guante?
    float w;
    if (modo == 0) {
        if (v_tipo > 0.5 && v_tipo < 1.5) {   // cabeza: una sola vista (la de la cámara) pinta toda la cara
            if (cs <= -0.3) return;
            w = (0.3 + max(cs, 0.0)) * exp(nitidez_cabeza * (dot(C, dir[i]) - 1.0));
            if (abs(dir[i].x) > 0.5 && abs(C.z) > abs(C.x))   // perfil: solo el costado (audífonos, oreja)
                w += smoothstep(0.55, 0.8, cs) * (1.0 - smoothstep(0.12, 0.3, abs(N.z)));
            w *= factor[i];
        } else {
            if (cs <= 0.02) return;
            if (dot(v_reposo, dir[i]) < pz.r - tol) return;
            w = cs * cs * exp(nitidez * (dot(C, dir[i]) - 1.0)) * factor[i];
        }
    } else {
        w = pow(max(cs + 0.6, 0.0), 2.0);
        if (modo == 1) {
            if (dot(v_reposo, dir[i]) < pz.r - 4.0 * tol) return;
        }
    }
    if (v_tipo > 1.5 && hay_mano == 1) w *= pz.g;   // la mano solo toma color del guante
    acum += w * c;
    pesos += w * c.a;
}

void main() {
    vec3 N = normalize(v_nreposo);
    C = normalize(v_cam);
    bool propio = false;
    for (int modo = 0; modo < 3; modo++) {
        acum = vec4(0.0); pesos = 0.0;
        muestra(tex0, prof0, 0, N, modo);
        muestra(tex1, prof1, 1, N, modo);
        muestra(tex2, prof2, 2, N, modo);
        muestra(tex3, prof3, 3, N, modo);
        if (pesos > 1e-7) break;
        if (hay_respaldo == 1) { propio = true; break; }   // ningún dibujo lo ve: color propio del modelo
    }
    vec3 rgb = propio ? v_respaldo : acum.rgb / max(acum.a, 1e-4);
    if (!propio && v_tipo > 1.5 && pesos <= 1e-7 && hay_mano == 1) rgb = color_mano;
    float sombra = (1.0 - luz) + luz * clamp(dot(normalize(v_npose), luz_dir), 0.0, 1.0) * 1.25;
    if (guante == 1 && v_tipo > 1.5) {          // 23-oct: el guante propio (como manos3d.color_guante)
        int h = v_reposo.x < x_centro ? 0 : 1;
        float e = dot(g_e[h].xyz, v_reposo) + g_e[h].w;
        float s = dot(g_s[h].xyz, v_reposo) + g_s[h].w;
        float gr = g_rango[h].z;
        float tramo = smoothstep(g_rango[h].x, g_rango[h].x + gr, e) * (1.0 - smoothstep(g_rango[h].y - gr, g_rango[h].y, e));
        float dorso = smoothstep(0.15, 0.4, dot(N, g_dorso[h]));
        float linea = 0.0;
        for (int k = 0; k < 3; k++) linea = max(linea, 1.0 - smoothstep(0.5 * gr, gr, abs(s - g_lineas[h][k])));
        float k = linea * tramo * dorso;
        for (int j = 0; j < 3; j++) {             // pliegues entre los dedos (sp < -1e8: no hay)
            float sp = g_pliegues[h][j]; float b = g_bases[h][j];
            if (sp < -1e8) continue;
            float en_dedo = smoothstep(b, b + gr, e) * (1.0 - smoothstep(g_punta[h] - gr, g_punta[h], e));
            k = max(k, 0.85 * (1.0 - smoothstep(0.35 * gr, 0.8 * gr, abs(s - sp))) * en_dedo);
        }
        float lam = clamp(dot(normalize(v_npose), luz_dir), 0.0, 1.0);
        rgb = mix(g_color[h], color_costura, k) * (sombra_guante + (1.0 - sombra_guante) * lam);
        sombra = 0.68 + 0.36 * lam;
    }
    color = vec4(clamp(rgb * min(sombra, 1.2), 0.0, 1.0), 1.0);
}
"""

EJE = {"frente": (1.0, 0.0, 0.0), "espalda": (-1.0, 0.0, 0.0), "perfil_der": (0.0, 0.0, -1.0),
       "perfil_izq": (0.0, 0.0, 1.0)}
_local = threading.local()


def disponible() -> bool:
    try:
        import moderngl  # noqa: F401
        return True
    except ImportError:
        return False


class RenderGL:
    """Un contexto OpenGL sin ventana con el modelo cargado. Úsalo desde el mismo hilo que lo creó."""

    def __init__(self, m: m3.Modelo, muestras: int = 8):
        import moderngl
        self.m = m
        self.ctx = moderngl.create_standalone_context(require=330)
        self.muestras = min(muestras, self.ctx.max_samples)
        self.prog = self.ctx.program(vertex_shader=VERTICES, fragment_shader=FRAGMENTOS)
        W = m.W
        orden = np.argsort(-W, axis=1)[:, :4]
        w4 = np.take_along_axis(W, orden, 1)
        w4 /= np.maximum(w4.sum(1, keepdims=True), 1e-9)
        cab = (m.grupo == m3.GRUPOS3D.index("cabeza")).astype(np.float64)[:, None]
        if m.es_mano is not None:
            cab[m.es_mano, 0] = 2.0                                                # 0 cuerpo, 1 cabeza, 2 mano
        cv = getattr(m, "color_vertice", None)
        respaldo = (cv[:, ::-1] / 255.0) if cv is not None else np.zeros((len(m.V), 3))      # BGR → RGB
        datos = np.hstack([m.V, m.N, orden.astype(np.float64), w4, cab, respaldo]).astype("f4")
        self.vbo = self.ctx.buffer(datos.tobytes())
        self.ibo = self.ctx.buffer(m.T.astype("i4").tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "3f 3f 4f 4f 1f 3f", "en_pos", "en_nor", "en_j",
                                                      "en_w", "en_g", "en_c")], index_buffer=self.ibo)
        self.prog["hay_respaldo"].value = int(cv is not None)
        self.texturas = []
        roles = [r for r in ("frente", "perfil_der", "espalda", "perfil_izq") if r in m.vistas]
        ejes, dirs, geos, factores = [], [], [], []
        for i, rol in enumerate(roles):
            v = m.vistas[rol]
            a = v.bgra[..., 3:4].astype(np.float32) / 255
            prem = np.dstack([v.bgra[..., 2::-1] * a, a * 255]).astype(np.uint8)      # RGBA premultiplicada
            h, w = prem.shape[:2]
            t = self.ctx.texture((w, h), 4, np.ascontiguousarray(prem).tobytes())
            t.build_mipmaps()
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            guante = m.mascaras_mano.get(rol, np.zeros((h, w), bool)).astype("f4")
            pz = np.dstack([m.profundidad[rol].astype("f4"), guante])
            p = self.ctx.texture((w, h), 2, np.ascontiguousarray(pz, dtype="f4").tobytes(), dtype="f4")
            p.filter = (moderngl.NEAREST, moderngl.NEAREST)
            p.repeat_x = p.repeat_y = False
            self.texturas += [t, p]
            self.prog[f"tex{i}"].value = 2 * i
            self.prog[f"prof{i}"].value = 2 * i + 1
            ejes.append(EJE[rol])
            dirs.append(tuple(DIRECCION[rol]))
            geos.append((v.c, v.suelo, w, h))
            factores.append(0.6 if v.espejo else 1.0)
        while len(ejes) < 4:
            ejes.append((0.0, 0.0, 0.0)); dirs.append((0.0, 0.0, 0.0)); geos.append((0, 0, 1, 1)); factores.append(0)
        self.prog["eje"].value = ejes
        self.prog["dir"].value = dirs
        self.prog["geo"].value = geos
        self.prog["factor"].value = factores
        self.prog["nvistas"].value = len(roles)
        self.prog["tol"].value = m.altura / 110
        self.prog["hay_mano"].value = int(bool(m.mascaras_mano))
        cm = m.color_mano if m.color_mano is not None else np.array([255.0, 255.0, 255.0])
        self.prog["color_mano"].value = tuple(float(x) / 255 for x in cm[::-1])   # BGR → RGB
        self._guante(m)
        self.fbo, self.final, self.tam = None, None, None

    def _guante(self, m: m3.Modelo) -> None:
        """Uniformes del guante propio (los mismos números que manos3d.color_guante)."""
        from mente import manos3d
        g = manos3d.parametros_guante(m) if manos3d.GUANTE_PROPIO and m.es_mano is not None else {}
        self.prog["guante"].value = int(bool(g))
        if not g:
            return
        self.prog["x_centro"].value = float(np.asarray(m.J["cadera"], float)[0])
        por_lado = [g.get("izq") or next(iter(g.values())), g.get("der") or next(iter(g.values()))]
        self.prog["g_e"].value = [tuple(map(float, p["ke"])) + (float(p["e0"]),) for p in por_lado]
        self.prog["g_s"].value = [tuple(map(float, p["ks"])) + (float(p["s0"]),) for p in por_lado]
        self.prog["g_rango"].value = [(p["e_lo"], p["e_hi"], p["grosor"]) for p in por_lado]
        self.prog["g_lineas"].value = [tuple(map(float, p["lineas"])) for p in por_lado]
        self.prog["g_dorso"].value = [tuple(map(float, p["dorso"])) for p in por_lado]
        self.prog["g_color"].value = [tuple(float(x) / 255 for x in p["color"][::-1]) for p in por_lado]
        self.prog["color_costura"].value = tuple(float(x) / 255 for x in manos3d.COLOR_COSTURA[::-1])
        self.prog["sombra_guante"].value = tuple(float(x) for x in manos3d.SOMBRA_GUANTE[::-1])
        pl = [(list(p.get("pliegues", [])) + [(-1e9, 0.0)] * 3)[:3] for p in por_lado]
        self.prog["g_pliegues"].value = [tuple(float(a) for a, _ in x) + (0.0,) for x in pl]
        self.prog["g_bases"].value = [tuple(float(b) for _, b in x) + (0.0,) for x in pl]
        self.prog["g_punta"].value = [float(p.get("e_punta", 0.0)) for p in por_lado]

    def _marcos(self, tam: tuple) -> None:
        if self.tam == tam:
            return
        for x in (self.fbo, self.final):
            if x is not None:
                x.release()
        ancho, alto = tam
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer(tam, 4, samples=self.muestras)],
                                        depth_attachment=self.ctx.depth_renderbuffer(tam, samples=self.muestras))
        self.final = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer(tam, 4)])
        self.tam = tam

    def huesos(self, pose: dict, desp: tuple = (0.0, 0.0)) -> np.ndarray:
        M = m3.matrices3d(self.m, pose)
        G = m3.transformacion_global(self.m, pose, M)
        G = m3._m4(t=[desp[0], 0.0, desp[1]]) @ G
        mats = [G @ M[hb] for hb in m3.huesos_de(self.m)]
        return np.stack(mats + [np.eye(4)] * (48 - len(mats)))

    def clip(self, cam: m3.Camara, escala: float, origen: tuple, tam: tuple) -> np.ndarray:
        """Matriz mundo → recorte que reproduce Camara.proyectar: píxel x = escala·xc/w - origen (centros de
        píxel en coordenadas enteras, como el rasterizador de numpy)."""
        r, u, d = cam.ejes()
        ancho, alto = tam
        ox, oy = origen[0] - 0.5, origen[1] - 0.5
        obj = np.asarray(cam.objetivo, float)
        alt = self.m.altura
        R = np.stack([r, u, d])
        V = np.eye(4)
        V[:3, :3] = R
        if cam.fov > 0:
            dist = 0.5 * alt / np.tan(np.radians(cam.fov) / 2) + 0.5 * alt
            V[:3, 3] = -R @ (obj + dist * d)                       # la cámara mira hacia -z
            k = escala * dist
            near, far = 0.05 * dist, dist + 3 * alt
            P = np.zeros((4, 4))
            P[0, 0], P[0, 2] = 2 * k / ancho, 2 * ox / ancho + 1        # x_clip = (2k/W)·x + (2·ox/W + 1)·z
            P[1, 1], P[1, 2] = 2 * k / alto, -(1 + 2 * oy / alto)
            P[2, 2], P[2, 3] = -(far + near) / (far - near), -2 * far * near / (far - near)
            P[3, 2] = -1
            return P @ V
        V[:3, 3] = -R @ obj
        P = np.eye(4)
        P[0, 0], P[0, 3] = 2 * escala / ancho, -(2 * ox / ancho + 1)
        P[1, 1], P[1, 3] = 2 * escala / alto, 1 + 2 * oy / alto
        P[2, 2] = -1 / (2 * alt)                                    # más cerca → menor profundidad
        return P @ V

    def render(self, pose: dict, cam: m3.Camara, escala: float, origen: tuple, tam: tuple,
               desp: tuple = (0.0, 0.0), luz: float = 0.18) -> np.ndarray:
        """BGRA uint8 (alfa recta), igual que modelo3d.renderizar3d."""
        import moderngl
        self._marcos(tuple(int(x) for x in tam))
        mats = self.huesos(pose, desp)
        self.prog["huesos"].write(mats.transpose(0, 2, 1).astype("f4").tobytes())
        self.prog["clip"].write(self.clip(cam, escala, origen, tam).T.astype("f4").tobytes())
        r, u, d = cam.ejes()
        pos = cam.posicion(self.m.altura)
        self.prog["perspectiva"].value = int(pos is not None)
        self.prog["cam_pos"].value = tuple(float(x) for x in (pos if pos is not None else d))
        self.prog["cam_dir"].value = tuple(float(x) for x in d)
        self.prog["nitidez"].value = float(m3.NITIDEZ)
        self.prog["nitidez_cabeza"].value = float(m3.NITIDEZ_CABEZA)
        L = 0.6 * d + 0.55 * u - 0.35 * r
        self.prog["luz_dir"].value = tuple(L / np.linalg.norm(L))
        self.prog["luz"].value = float(luz)
        for i, t in enumerate(self.texturas):
            t.use(i)
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.front_face = "ccw"
        self.vao.render()
        self.ctx.copy_framebuffer(self.final, self.fbo)
        crudo = self.final.read(components=4, alignment=1)
        import cv2
        img = np.frombuffer(crudo, np.uint8).reshape(tam[1], tam[0], 4)
        # alfa premultiplicada → recta con OpenCV (en numpy con decimales costaba ~120 ms por cuadro a 1300x930 y
        # trababa la vista previa de la ventana; cv2.divide deja 0 donde el alfa es 0)
        bgra = cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_RGBA2BGRA)
        a = cv2.extractChannel(bgra, 3)
        bgr = cv2.divide(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR), cv2.merge([a, a, a]), scale=255.0)
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        cv2.insertChannel(a, out, 3)
        return out

    def liberar(self) -> None:
        self.ctx.release()


def renderizador(m: m3.Modelo) -> RenderGL | None:
    """El renderizador GPU de este hilo para el modelo (None si no hay OpenGL)."""
    if not disponible():
        return None
    r = getattr(_local, "r", None)
    if r is not None and r.m is m:
        return r
    try:
        if r is not None:
            r.liberar()
        _local.r = RenderGL(m)
    except Exception:
        _local.r = None
    return _local.r
