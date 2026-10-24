# -*- coding: utf-8 -*-
"""Guante propio (23-oct-2026): la mano se pinta con su material, no proyectando el dibujo (manchas al girar)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from mente import manos3d  # noqa: E402
from mente import modelo3d as m3  # noqa: E402
from mente import vistas as vs  # noqa: E402
from tests.test_modelo3d import hoja  # noqa: E402


class GuantePropio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = m3.construir(vs.preparar_vistas(hoja(), alto=600), detalle=0.6)
        cls.idx = np.flatnonzero(cls.m.es_mano) if cls.m.es_mano is not None else np.array([], int)

    def setUp(self):
        self.assertTrue(len(self.idx), "el muñeco de prueba trae guantes")
        manos3d.GUANTE_PROPIO = True
        self.m.guante = None

    def test_parametros_por_mano(self):
        g = manos3d.parametros_guante(self.m)
        self.assertEqual(set(g), set(self.m.manos))
        for p in g.values():
            self.assertLess(p["e_lo"], p["e_hi"])
            self.assertEqual(len(p["lineas"]), 3)
            self.assertGreater(p["color"].mean(), 180)          # el guante del muñeco es casi blanco

    def test_la_mano_no_cambia_de_color_con_la_camara(self):
        """Con el dibujo proyectado, la mano tomaba colores distintos según desde dónde se mirara."""
        m, idx = self.m, self.idx
        P, N = m.V[idx], m.N[idx]
        mano = np.ones(len(idx), bool)
        colores = []
        for az in (0.0, 70.0, 180.0):
            cam = m3.Camara(azimut=az, elevacion=5.0)
            C = np.tile(cam.hacia_camara(P[:1], m.altura)[0], (len(idx), 1))
            colores.append(m3.colorear(m, P, N, C, None, mano)[:, :3])
        np.testing.assert_allclose(colores[0], colores[1], atol=1e-3)
        np.testing.assert_allclose(colores[0], colores[2], atol=1e-3)

    def test_costuras_solo_en_el_dorso(self):
        m, idx = self.m, self.idx
        g = manos3d.parametros_guante(m)
        col = manos3d.color_guante(m, m.V[idx], m.N[idx])
        oscuros = col.mean(1) < 120
        lados = np.where(m.V[idx, 0] < m.J["cadera"][0], "izq", "der")
        dorso = np.array([m.N[i] @ g[l]["dorso"] for i, l in zip(idx, lados)])
        if oscuros.any():
            self.assertGreater(float(np.mean(dorso[oscuros] > 0.1)), 0.6)   # las costuras, del lado del dorso

    def test_sin_guante_propio_vuelve_al_dibujo(self):
        manos3d.GUANTE_PROPIO = False
        m, idx = self.m, self.idx
        mano = np.ones(len(idx), bool)
        sin = m3.colorear(m, m.V[idx], m.N[idx], None, None, mano)
        manos3d.GUANTE_PROPIO = True
        con = m3.colorear(m, m.V[idx], m.N[idx], None, None, mano)
        self.assertFalse(np.allclose(sin, con))

    def tearDown(self):
        manos3d.GUANTE_PROPIO = True


if __name__ == "__main__":
    unittest.main()
