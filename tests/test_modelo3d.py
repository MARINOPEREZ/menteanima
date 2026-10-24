# -*- coding: utf-8 -*-
"""MenteAnima 3D: hoja → vistas → modelo 3D con huesos → movimientos → render → .glb."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mente import modelo3d as m3  # noqa: E402
from mente import movimientos3d as mv3  # noqa: E402
from mente import vistas as vs  # noqa: E402

PIEL, CABEZA, ROPA, BRAZO, PANTALON, ZAPATO = (60, 120, 230), (40, 80, 200), (200, 90, 40), (50, 160, 60), \
    (40, 40, 160), (30, 30, 30)


def _vista(lado: bool, espalda: bool = False) -> np.ndarray:
    """Un muñeco de frente (o de perfil mirando a la izquierda) en BGRA con fondo transparente."""
    img = np.zeros((700, 400, 4), np.uint8)

    def p(fn, *a, color):
        fn(img, *a, (*color, 255), *([-1] if fn in (cv2.circle, cv2.ellipse, cv2.rectangle) else []))
    if not lado:
        cv2.line(img, (175, 370), (160, 500), (*PANTALON, 255), 34)
        cv2.line(img, (160, 500), (150, 630), (*PANTALON, 255), 30)
        cv2.line(img, (225, 370), (240, 500), (*PANTALON, 255), 34)
        cv2.line(img, (240, 500), (250, 630), (*PANTALON, 255), 30)
        cv2.ellipse(img, (140, 642), (32, 16), 0, 0, 360, (*ZAPATO, 255), -1)
        cv2.ellipse(img, (260, 642), (32, 16), 0, 0, 360, (*ZAPATO, 255), -1)
        cv2.rectangle(img, (145, 165), (255, 385), (*ROPA, 255), -1)
        cv2.line(img, (152, 190), (95, 285), (*BRAZO, 255), 26)
        cv2.line(img, (95, 285), (72, 380), (*BRAZO, 255), 24)
        cv2.line(img, (248, 190), (305, 285), (*BRAZO, 255), 26)
        cv2.line(img, (305, 285), (328, 380), (*BRAZO, 255), 24)
        cv2.circle(img, (70, 392), 20, (220, 220, 240, 255), -1)
        cv2.circle(img, (330, 392), 20, (220, 220, 240, 255), -1)
        cv2.circle(img, (200, 105), 62, (*(CABEZA if espalda else PIEL), 255), -1)
        if not espalda:
            cv2.circle(img, (178, 95), 9, (255, 255, 255, 255), -1)
            cv2.circle(img, (222, 95), 9, (255, 255, 255, 255), -1)
    else:
        cv2.line(img, (200, 370), (200, 500), (*PANTALON, 255), 36)
        cv2.line(img, (200, 500), (200, 630), (*PANTALON, 255), 32)
        cv2.ellipse(img, (182, 642), (42, 16), 0, 0, 360, (*ZAPATO, 255), -1)       # la punta del pie adelante
        cv2.rectangle(img, (158, 165), (242, 385), (*ROPA, 255), -1)
        cv2.circle(img, (200, 105), 60, (*PIEL, 255), -1)
        cv2.circle(img, (142, 110), 14, (*PIEL, 255), -1)                           # nariz: mira a la izquierda
        cv2.circle(img, (160, 95), 9, (255, 255, 255, 255), -1)                    # el ojo del perfil
        cv2.line(img, (200, 190), (196, 300), (*BRAZO, 255), 26)
        cv2.line(img, (196, 300), (192, 380), (*BRAZO, 255), 24)
        cv2.ellipse(img, (192, 398), (26, 32), 0, 0, 360, (220, 220, 240, 255), -1)    # guante
        for x in (183, 201):                                                        # pliegues entre los dedos
            cv2.line(img, (x, 408), (x, 428), (40, 40, 40, 255), 3)
    return img


def hoja() -> np.ndarray:
    frente, perfil = _vista(False), _vista(True)
    espalda = _vista(False, espalda=True)[:, ::-1]
    hueco = np.zeros((700, 40, 4), np.uint8)
    return np.ascontiguousarray(np.hstack([frente, hueco, perfil, hueco, espalda, hueco, perfil[:, ::-1]]))


class Modelo3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hoja = hoja()
        cls.vistas = vs.preparar_vistas(cls.hoja, alto=600)
        cls.m = m3.construir(cls.vistas, detalle=0.6)

    def test_separa_y_reconoce_las_cuatro_vistas(self):
        figuras = vs.separar(self.hoja)
        self.assertEqual(len(figuras), 4)
        roles = vs.asignar_roles(figuras)
        self.assertIs(roles["frente"], figuras[0])
        self.assertIs(roles["perfil_der"], figuras[1])            # mira a la izquierda de la imagen
        self.assertIs(roles["espalda"], figuras[2])
        self.assertIs(roles["perfil_izq"], figuras[3])
        self.assertTrue(all(not v.espejo for v in self.vistas.values()))

    def test_una_sola_imagen_inventa_las_demas_en_espejo(self):
        sola = vs.preparar_vistas(_vista(False), alto=400)
        self.assertEqual(set(sola), {"frente", "espalda"})
        self.assertTrue(sola["espalda"].espejo)
        m = m3.construir(sola, detalle=0.5)
        self.assertGreater(len(m.T), 1000)

    def test_mallas_cerradas_con_normales_hacia_afuera(self):
        m = self.m
        self.assertEqual(set(np.unique(m.grupo)), set(range(len(m3.GRUPOS3D))))
        aristas = np.sort(np.vstack([m.T[:, [0, 1]], m.T[:, [1, 2]], m.T[:, [2, 0]]]), axis=1)
        _, cuenta = np.unique(aristas, axis=0, return_counts=True)
        self.assertGreater((cuenta == 2).mean(), 0.99)
        for g in range(len(m3.GRUPOS3D)):
            sel = m.grupo == g
            c = m.V[sel].mean(0)
            self.assertGreater(((m.V[sel] - c) * m.N[sel]).sum(1).mean(), 0, m3.GRUPOS3D[g])

    def _silueta(self, rol: str, az: float) -> tuple:
        v = self.vistas[rol]
        cam = m3.Camara(azimut=az, elevacion=0, fov=0)
        h, w = v.alfa.shape
        P = m3.posar3d(self.m, m3_reposo(self.m))
        r = m3.renderizar3d(self.m, P, cam, 1.0, (-v.c, -v.suelo), (w, h), ss=1, luz=0)
        return r, v

    def test_de_frente_y_de_perfil_coincide_con_el_dibujo(self):
        for rol, az, minimo in (("frente", 0, 0.85), ("perfil_der", 90, 0.8), ("espalda", 180, 0.85)):
            r, v = self._silueta(rol, az)
            a, b = r[..., 3] > 127, v.alfa > 127
            self.assertGreater((a & b).sum() / (a | b).sum(), minimo, rol)
        r, v = self._silueta("frente", 0)
        ys, xs = np.nonzero(np.all(np.abs(v.bgra[..., :3].astype(int) - ROPA) < 8, axis=2) & (r[..., 3] > 250))
        color = r[ys, xs, :3].astype(float).mean(0)
        self.assertLess(np.abs(color - ROPA).max(), 30)

    def test_caminar_con_los_pies_en_el_suelo_y_brincar_despega(self):
        poses, _ = mv3.poses_de(self.m, mv3.preparar_pasos(["caminar"], 1), 15)
        idx = np.concatenate(list(self.m.pies.values()))
        for pose in poses:
            self.assertAlmostEqual(m3.posar3d(self.m, pose, idx).V[:, 1].min(), 0.0, delta=0.01 * self.m.altura)
        poses, _ = mv3.poses_de(self.m, mv3.preparar_pasos(["brincar"]), 15)
        altos = [m3.posar3d(self.m, p, idx).V[:, 1].min() for p in poses]
        self.assertGreater(max(altos), 0.2 * self.m.altura)
        self.assertAlmostEqual(altos[-1], 0.0, delta=0.02 * self.m.altura)

    def test_caminar_hacia_la_derecha_avanza_en_x(self):
        poses, desp = mv3.poses_de(self.m, mv3.preparar_pasos(["caminar"], 2), 15, rumbo=90, desplazar=True)
        self.assertGreater(desp[-1][0], 0.3 * self.m.altura)
        self.assertLess(abs(desp[-1][1]), 0.1 * self.m.altura)

    def test_todos_los_movimientos_y_la_animacion(self):
        fotos, tam, n = mv3.animar(self.m, list(mv3.MOVIMIENTOS), 0.5, fps=6, altura=120, ss=1, gpu=False,
                                   procesos=1)
        fotos = list(fotos)
        self.assertEqual(len(fotos), n)
        self.assertEqual(fotos[0].shape, (tam[1], tam[0], 4))
        self.assertEqual(int(fotos[0][0, 0, 3]), 0)
        self.assertTrue(all((f[..., 3] == 255).any() for f in fotos))

    def test_la_tarjeta_grafica_dibuja_igual_que_numpy(self):
        from mente import render_gl
        if not render_gl.disponible():
            self.skipTest("sin moderngl")
        try:
            gl = render_gl.RenderGL(self.m)
        except Exception as e:                     # sin OpenGL 3.3 en esta máquina
            self.skipTest(f"sin OpenGL: {e}")
        pose = mv3.caminar(0.3, self.m)
        cam = m3.Camara(azimut=35, elevacion=8, objetivo=(0, self.m.altura / 2, 0))
        a = gl.render(pose, cam, 0.5, (-150, -170), (300, 340))
        b = m3.renderizar3d(self.m, m3.posar3d(self.m, pose), cam, 0.5, (-150, -170), (300, 340), ss=2)
        self.assertLess(np.abs(a[..., 3].astype(float) - b[..., 3]).mean(), 3)
        lleno = (a[..., 3] > 250) & (b[..., 3] > 250)
        self.assertLess(np.abs(a[lleno, :3].astype(float) - b[lleno, :3]).mean(), 12)
        gl.liberar()

    def test_exporta_un_glb_valido(self):
        from mente import glb
        with tempfile.TemporaryDirectory() as d:
            ruta = glb.exportar_glb(self.m, Path(d) / "x.glb",
                                    {"caminar": mv3.poses_de(self.m, mv3.preparar_pasos(["caminar"], 1), 10)[0]})
            datos = ruta.read_bytes()
        magia, version, total = struct.unpack("<4sII", datos[:12])
        self.assertEqual((magia, version, total), (b"glTF", 2, len(datos)))
        largo, tipo = struct.unpack("<I4s", datos[12:20])
        js = json.loads(datos[20:20 + largo])
        self.assertEqual(tipo, b"JSON")
        pos = js["accessors"][js["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
        self.assertEqual(pos["count"], 3 * len(self.m.T))
        self.assertEqual([a["name"] for a in js["animations"]], ["caminar"])
        self.assertEqual(len(js["skins"][0]["joints"]), len(self.m.huesos))
        bin_largo = struct.unpack("<I", datos[20 + largo:24 + largo])[0]
        self.assertEqual(bin_largo, js["buffers"][0]["byteLength"])


def _blancos(img: np.ndarray) -> int:
    """Cuántas manchas blancas (ojos) hay en la imagen."""
    m = (img[..., :3] > 240).all(2) & (img[..., 3] > 250)
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return int((st[1:, cv2.CC_STAT_AREA] > 12).sum())


class CaraYManos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vistas = vs.preparar_vistas(hoja(), alto=600)
        cls.m = m3.construir(cls.vistas, detalle=0.6)

    def _cabeza(self, az: float) -> np.ndarray:
        m = self.m
        cab = np.asarray(m.J["cabeza"], float)
        cam = m3.Camara(azimut=az, elevacion=0, fov=0, objetivo=tuple(cab))
        return m3.renderizar3d(m, m3.posar3d(m, {"apoyo": 0.0}), cam, 1.5, (-120, -120), (240, 240), ss=1, luz=0)

    def test_la_cara_no_suma_los_ojos_de_frente_y_de_perfil(self):
        """19-oct: de tres cuartos se veían 4 ojos (los del frente + el del perfil en otro sitio del volumen)."""
        self.assertEqual(_blancos(self._cabeza(0)), 2)
        self.assertEqual(_blancos(self._cabeza(30)), 2)
        self.assertEqual(_blancos(self._cabeza(60)), 1)        # mezclando vistas aquí salían 2
        self.assertEqual(_blancos(self._cabeza(90)), 1)

    def test_detecta_los_dedos_del_guante_y_los_dobla(self):
        self.assertIn("der", self.m.manos)
        self.assertEqual(len(self.m.manos["der"]["falanges"]), 3)
        self.assertIn("muneca_der", self.m.huesos)
        idx = np.flatnonzero(self.m.es_mano)
        quieto = m3.posar3d(self.m, {"apoyo": 0.0}, idx).V
        puno = m3.posar3d(self.m, {"apoyo": 0.0, "dedos_der": (40, 0)}, idx).V
        muneca = m3.posar3d(self.m, {"apoyo": 0.0, "muneca_der": (0, 40, 0)}, idx).V
        self.assertGreater(np.linalg.norm(puno - quieto, axis=1).max(), 3)
        self.assertGreater(np.linalg.norm(muneca - quieto, axis=1).max(), 5)

    def test_el_glb_lleva_munecas_y_falanges(self):
        from mente import glb
        with tempfile.TemporaryDirectory() as d:
            ruta = glb.exportar_glb(self.m, Path(d) / "x.glb",
                                    {"saludar": mv3.poses_de(self.m, mv3.preparar_pasos(["saludar"], 1), 8)[0]})
            datos = ruta.read_bytes()
        largo = struct.unpack("<I", datos[12:16])[0]
        js = json.loads(datos[20:20 + largo])
        nombres = [n["name"] for n in js["nodes"]]
        self.assertIn("muneca_der", nombres)
        self.assertIn("dedo1_der_3", nombres)
        self.assertEqual(len(js["skins"][0]["joints"]), len(self.m.huesos))


class PoseNeutraYTrellis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vistas = vs.preparar_vistas(hoja(), alto=500)
        cls.m = m3.construir(cls.vistas, detalle=0.5)

    def test_brazos_abiertos_se_bajan_para_moverse(self):
        """Una hoja en pose A o T: los movimientos parten con los brazos abajo (no caminan en cruz)."""
        self.assertIn("brazo_der", self.m.neutra)                 # el muñeco de prueba abre los brazos ~30°
        self.assertLess(self.m.neutra["brazo_der"][1], 0)
        J = {k: np.array(v, float) for k, v in self.m.J.items()}
        J["codo_der"] = J["hombro_der"] + np.array([0.0, -100.0, 0.0])
        J["codo_izq"] = J["hombro_izq"] + np.array([0.0, -100.0, 0.0])
        self.assertNotIn("brazo_der", m3.postura_neutra(J))       # brazos colgando: no se tocan
        manos = m3.posar3d(self.m, {}).J
        dibujo = m3.posar3d(self.m, {"dibujo": True}).J
        self.assertLess(abs(manos["mano_der"][0]), abs(dibujo["mano_der"][0]))

    def test_el_codo_es_una_bisagra(self):
        """22-oct (Marino: el brazo «no es un movimiento natural»): el codo solo se dobla, hacia adelante."""
        for lado in ("izq", "der"):
            doblado = m3.posar3d(self.m, {"apoyo": 0, f"antebrazo_{lado}": (90, 0, 0)}).J
            de_lado = m3.posar3d(self.m, {"apoyo": 0, f"antebrazo_{lado}": (90, 40, 0)}).J
            np.testing.assert_allclose(doblado[f"mano_{lado}"], de_lado[f"mano_{lado}"], atol=1e-6)
            f = doblado[f"mano_{lado}"] - doblado[f"codo_{lado}"]
            f /= np.linalg.norm(f)
            self.assertGreater(f[2], 0.9)                 # con la pose A bajada, antes también se abría de lado
            self.assertLess(abs(f[0]), 0.15)
            tope = m3.posar3d(self.m, {"apoyo": 0, f"antebrazo_{lado}": (200, 0, 0)}).J
            limite = m3.posar3d(self.m, {"apoyo": 0, f"antebrazo_{lado}": (150, 0, 0)}).J
            np.testing.assert_allclose(tope[f"mano_{lado}"], limite[f"mano_{lado}"], atol=1e-6)

    def test_el_hombro_rota_sobre_el_eje_del_brazo(self):
        for lado, s in (("izq", -1), ("der", 1)):
            codo = {"apoyo": 0, f"antebrazo_{lado}": (90, 0, 0)}
            quieto = m3.posar3d(self.m, codo).J
            adentro = m3.posar3d(self.m, dict(codo, **{f"brazo_{lado}": (0, 0, 60)})).J
            np.testing.assert_allclose(quieto[f"codo_{lado}"], adentro[f"codo_{lado}"], atol=1e-6)
            self.assertLess(s * adentro[f"mano_{lado}"][0], s * quieto[f"mano_{lado}"][0] - 0.05 * self.m.altura)

    def test_la_mano_en_la_cintura_abre_el_codo_hacia_afuera(self):
        J = m3.posar3d(self.m, mv3.senalar(1.2, self.m)).J
        self.assertLess(J["codo_izq"][0], J["hombro_izq"][0] - 0.05 * self.m.altura)
        self.assertLess(abs(J["mano_izq"][1] - J["cadera"][1]), 0.12 * self.m.altura)
        self.assertLess(J["mano_izq"][2], J["cadera"][2] + 0.06 * self.m.altura)      # al costado, no adelante

    def test_trellis_en_pose_a_vuelve_a_la_pose_del_dibujo(self):
        from mente import trellis3d as t3
        P = m3.posar3d(self.m, t3.POSE_A)
        otra = P.V @ np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T * 2.0       # otra orientación y escala
        V, W, es_mano, iou = t3.a_reposo(self.m, otra, self.m.T, t3.POSE_A)
        self.assertGreater(iou, 0.85)
        error = np.linalg.norm(V - self.m.V, axis=1)
        self.assertLess(float(np.median(error)), 0.01 * self.m.altura)
        self.assertLess(float(np.percentile(error, 95)), 0.02 * self.m.altura)
        self.assertEqual(W.shape, (len(V), len(m3.huesos_de(self.m))))

    def test_trellis_se_alinea_y_hereda_el_esqueleto(self):
        from mente import glb
        from mente import trellis3d as t3
        with tempfile.TemporaryDirectory() as d:
            ruta = glb.exportar_glb(self.m, Path(d) / "forma.glb", {})
            V0, T, UV, tex = t3.leer_glb(ruta)
            girada = V0 @ np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T * 3.0   # otra orientación y escala
            np.save(Path(d) / "v.npy", girada)
            V, iou = t3.alinear(girada, T, self.m.vistas, self.m.altura)
            self.assertGreater(iou, 0.75)
            self.assertAlmostEqual(float(np.ptp(V[:, 1])), self.m.altura, delta=0.05 * self.m.altura)
            mt = t3.desde_trellis(self.m, ruta)
        self.assertEqual(mt.W.shape, (len(mt.V), len(self.m.huesos)))
        self.assertEqual(mt.origen, "trellis")
        P = m3.posar3d(mt, mv3.caminar(0.25, mt))
        self.assertTrue(np.isfinite(P.V).all())
        img = m3.renderizar3d(mt, P, m3.Camara(azimut=30, objetivo=(0, mt.altura / 2, 0)), 0.4, (-100, -150),
                              (200, 300), ss=1)
        self.assertTrue((img[..., 3] == 255).any())


def m3_reposo(m) -> dict:
    return {"apoyo": 0.0, "dibujo": True}


if __name__ == "__main__":
    unittest.main()
