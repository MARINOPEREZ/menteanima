# -*- coding: utf-8 -*-
"""La ventana de MenteAnima (23-oct-2026, al grabarla para el #6): botones que no se perdían clics y una vista previa
que no trababa la ventana."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from mente import animador_ventana as av  # noqa: E402


class SobreCuadros(unittest.TestCase):
    def test_igual_a_la_mezcla_con_decimales(self):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, (90, 130, 4)).astype(np.uint8)
        a = img[..., 3:4].astype(np.float32) / 255
        iy, ix = np.indices((90, 130))
        cuadros = ((((iy // 16) + (ix // 16)) % 2) * 45 + 185).astype(np.float32)[..., None]
        esperado = img[..., 2::-1].astype(np.float32) * a + cuadros * (1 - a)
        hecho = np.asarray(av._sobre_cuadros(img, 16)).astype(np.float32)
        self.assertLessEqual(np.abs(hecho - esperado).max(), 1.5)       # redondeo, nada más


class SoltarBoton(unittest.TestCase):
    """Con grabadores de pantalla o escritorio remoto llega un «salió» falso entre pulsar y soltar, y Tk perdía el
    clic (no se podía pausar la reproducción)."""

    def setUp(self):
        try:
            import tkinter as tk
            from tkinter import ttk
            self.raiz = tk.Tk()
        except Exception as e:                      # sin pantalla
            self.skipTest(f"sin Tk: {e}")
        self.raiz.geometry("200x80+4000+4000")
        self.veces = []
        self.boton = ttk.Button(self.raiz, text="Pausar", command=lambda: self.veces.append(1))
        self.boton.pack()
        self.raiz.update()

    def tearDown(self):
        self.raiz.destroy()

    def _soltar(self, x):
        class E:
            pass
        e = E()
        e.widget, e.x, e.y = self.boton, x, 5
        self.boton.state(["pressed", "!active"])     # el «salió» falso le quitó «active»
        return av._soltar_boton(e)

    def test_dentro_del_boton_si_actua(self):
        self.assertEqual(self._soltar(5), "break")
        self.assertEqual(self.veces, [1])
        self.assertFalse(self.boton.instate(["pressed"]))

    def test_fuera_del_boton_no(self):
        self._soltar(-4)
        self.assertEqual(self.veces, [])


if __name__ == "__main__":
    unittest.main()
