# -*- coding: utf-8 -*-
"""MenteAnima: recorte, esqueleto, títere, movimientos y exportación con fondo transparente."""
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mente import animacion as an  # noqa: E402
from mente import movimientos as mv  # noqa: E402


def figura() -> tuple:
    """Un muñeco de frente sobre fondo blanco (BGRA) y su silueta real."""
    img = np.full((700, 400, 3), 255, np.uint8)
    m = np.zeros((700, 400), np.uint8)
    for capa, color in ((m, 255), (img, None)):
        def c(col):
            return color if color is not None else col
        cv2.line(capa, (175, 370), (160, 500), c((40, 40, 160)), 34)
        cv2.line(capa, (160, 500), (150, 630), c((40, 40, 160)), 30)
        cv2.line(capa, (225, 370), (240, 500), c((40, 40, 160)), 34)
        cv2.line(capa, (240, 500), (250, 630), c((40, 40, 160)), 30)
        cv2.ellipse(capa, (140, 640), (32, 16), 0, 0, 360, c((30, 30, 30)), -1)
        cv2.ellipse(capa, (260, 640), (32, 16), 0, 0, 360, c((30, 30, 30)), -1)
        cv2.rectangle(capa, (145, 170), (255, 385), c((200, 90, 40)), -1)
        cv2.line(capa, (152, 190), (95, 285), c((50, 160, 60)), 26)
        cv2.line(capa, (95, 285), (72, 380), c((50, 160, 60)), 24)
        cv2.line(capa, (248, 190), (305, 285), c((50, 160, 60)), 26)
        cv2.line(capa, (305, 285), (328, 380), c((50, 160, 60)), 24)
        cv2.circle(capa, (70, 392), 20, c((220, 220, 240)), -1)
        cv2.circle(capa, (330, 392), 20, c((220, 220, 240)), -1)
        cv2.circle(capa, (200, 105), 62, c((60, 120, 230)), -1)
    cv2.circle(img, (178, 95), 9, (255, 255, 255), -1)
    cv2.circle(img, (222, 95), 9, (255, 255, 255), -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA), m > 0


class Animacion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bgra, cls.real = figura()
        cls.alfa = an.recortar(cls.bgra)
        cls.J = an.detectar_esqueleto(cls.alfa)
        cls.p = an.crear_personaje(cls.bgra, cls.alfa, cls.J)

    def test_recorte_separa_la_figura_del_fondo_blanco(self):
        m = self.alfa > 127
        iou = (m & self.real).sum() / (m | self.real).sum()
        self.assertGreater(iou, 0.93)

    def test_recorte_respeta_la_transparencia_del_png(self):
        b = self.bgra.copy()
        b[..., 3] = np.where(self.real, 255, 0)
        self.assertTrue((an.recortar(b) == b[..., 3]).all())

    def test_esqueleto_ordenado_de_la_cabeza_a_los_pies(self):
        J = self.J
        for lado in ("izq", "der"):
            ys = [J["cabeza"][1], J["cuello"][1], J["cadera_" + lado][1], J["rodilla_" + lado][1],
                  J["pie_" + lado][1]]
            self.assertEqual(ys, sorted(ys), lado)
            self.assertLess(J["hombro_" + lado][1], J["codo_" + lado][1])
        self.assertLess(J["mano_izq"][0], J["hombro_izq"][0])
        self.assertGreater(J["mano_der"][0], J["hombro_der"][0])
        self.assertLess(math.dist(J["mano_izq"], (70, 392)), 30)
        self.assertLess(math.dist(J["pie_der"], (255, 635)), 35)
        self.assertLess(abs(J["cuello"][1] - 168), 30)

    def test_esqueleto_guardado_se_reescala_con_la_imagen(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "x.esqueleto.json"
            an.guardar_esqueleto(ruta, self.J, (400, 700))
            J2 = an.cargar_esqueleto(ruta, (800, 1400))
            self.assertAlmostEqual(J2["cabeza"][0], self.J["cabeza"][0] * 2, delta=0.2)

    def test_la_pose_del_dibujo_reproduce_la_imagen(self):
        r = an.renderizar(self.p, an.posar(self.p, an.pose_del_dibujo(self.p)), 1.0, (0, 0), (400, 700), 2)
        a = self.alfa[..., None] / 255.0
        dif = np.abs(r[..., :3] * (r[..., 3:] / 255.0) - self.bgra[..., :3] * a)[self.real]
        self.assertLess(float(dif.mean()), 8.0)

    def test_al_caminar_un_pie_siempre_toca_el_suelo(self):
        poses = mv.poses_de(self.p, mv.preparar_pasos(["caminar"], 1), fps=20)
        idx = {c.nombre: i for i, c in enumerate(self.p.capas)}
        for P in poses:
            bajo = max(P.capas[idx[c]][v, 1].max() for c, v in self.p.pies.values())
            self.assertAlmostEqual(bajo, self.p.suelo, delta=1.5)

    def test_brincar_despega_y_vuelve_al_suelo(self):
        poses = mv.poses_de(self.p, mv.preparar_pasos(["brincar"]), fps=20)
        idx = {c.nombre: i for i, c in enumerate(self.p.capas)}
        bajo = [max(P.capas[idx[c]][v, 1].max() for c, v in self.p.pies.values()) for P in poses]
        self.assertLess(min(bajo), self.p.suelo - 0.2 * self.p.altura)
        self.assertAlmostEqual(bajo[-1], self.p.suelo, delta=3)

    def test_todos_los_movimientos_y_la_secuencia(self):
        pasos = mv.preparar_pasos(list(mv.MOVIMIENTOS), 1)
        poses = mv.poses_de(self.p, pasos, fps=10)
        self.assertEqual(len(poses), round(sum(d for _, d in pasos) * 10))
        for P in poses:
            self.assertTrue(all(np.isfinite(v).all() for v in P.capas))

    def test_desplazarse_avanza_hacia_donde_mira(self):
        for d in (1, -1):
            poses = mv.poses_de(self.p, mv.preparar_pasos(["caminar"], 2), fps=20, direccion=d, desplazar=True)
            x0 = np.vstack(poses[0].capas)[:, 0].mean()
            x1 = np.vstack(poses[-1].capas)[:, 0].mean()
            self.assertGreater((x1 - x0) * d, 0.3 * self.p.altura)

    def test_espejo_y_sinonimos(self):
        pose = mv.caminar(0.3, self.p)
        self.assertEqual(mv.espejo(mv.espejo(pose)), pose)
        self.assertEqual(mv.nombre_movimiento("Saltar"), "brincar")
        with self.assertRaises(ValueError):
            mv.nombre_movimiento("volar")

    def test_exporta_con_fondo_transparente(self):
        fotos, tam, n = mv.animar(self.p, ["saludar"], 1, fps=8, altura=120, ss=1)
        fotos = list(fotos)
        self.assertEqual(len(fotos), n)
        self.assertEqual(fotos[0][0, 0, 3], 0)                     # esquina transparente
        self.assertTrue((fotos[0][..., 3] == 255).any())
        formatos = [".gif", ".apng", ""] + ([".webm", ".mov"] if shutil.which("ffmpeg") else [])
        with tempfile.TemporaryDirectory() as d:
            for ext in formatos:
                ruta = an.exportar(iter(fotos), Path(d) / f"anim{ext}", 8, tam)
                self.assertTrue(ruta.exists())
                if ext == "":
                    self.assertEqual(len(list(ruta.glob("*.png"))), n)
            with self.assertRaises(ValueError):
                an.exportar(iter(fotos), Path(d) / "anim.mp4", 8, tam)


if __name__ == "__main__":
    unittest.main()
