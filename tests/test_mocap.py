# -*- coding: utf-8 -*-
"""Movimientos de captura real (BVH y ASF/AMC de CMU) → poses del modelo 3D (23-oct-2026)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from mente import mocap as cp  # noqa: E402
from mente import modelo3d as m3  # noqa: E402
from mente import vistas as vs  # noqa: E402
from tests.test_modelo3d import hoja  # noqa: E402

BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0 10 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 10 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.0333333
0 0 0 0 0 0 0 0 0
1 2 3 90 0 0 0 0 0
"""

ASF = """:version 1.10
:name prueba
:units
  mass 1.0
  length 0.45
  angle deg
:root
   order TX TY TZ RX RY RZ
   axis XYZ
   position 0 0 0
   orientation 0 0 0
:bonedata
  begin
     id 1
     name lfemur
     direction 0 -1 0
     length 5
     axis 0 0 0  XYZ
    dof rx
    limits (-160.0 20.0)
  end
:hierarchy
  begin
    root lfemur
  end
"""

AMC = """:FULLY-SPECIFIED
:DEGREES
1
root 0 0 0 0 0 0
lfemur 0
2
root 0 0 0 0 0 0
lfemur 90
"""

NOMBRES = {"cadera_izq": "RightUpLeg", "rodilla_izq": "RightLeg", "pie_izq": "RightFoot",
           "cadera_der": "LeftUpLeg", "rodilla_der": "LeftLeg", "pie_der": "LeftFoot",
           "hombro_izq": "RightArm", "codo_izq": "RightForeArm", "mano_izq": "RightHand",
           "hombro_der": "LeftArm", "codo_der": "LeftForeArm", "mano_der": "LeftHand",
           "cadera": "Hips", "cuello": "Neck", "cabeza": "Head_fin"}


def _angulo(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


class Lectores(unittest.TestCase):
    def test_bvh_cinematica_directa(self):
        cap = cp.leer_bvh("prueba.bvh", BVH)
        self.assertAlmostEqual(cap.fps, 30.0, places=2)
        self.assertEqual(cap.cuadros, 2)
        np.testing.assert_allclose(cap.posiciones["Spine"][0], [0, 10, 0], atol=1e-9)
        np.testing.assert_allclose(cap.posiciones["Spine_fin"][0], [0, 20, 0], atol=1e-9)
        # la cadera gira 90° en Z y se mueve (1, 2, 3): (0, 10, 0) → (−10, 0, 0)
        np.testing.assert_allclose(cap.posiciones["Spine"][1], [1 - 10, 2, 3], atol=1e-9)
        np.testing.assert_allclose(cap.posiciones["Spine_fin"][1], [1 - 20, 2, 3], atol=1e-9)

    def test_asf_amc_de_cmu(self):
        cap = cp.leer_asf_amc("prueba.asf", "prueba_01.amc", ASF, AMC)
        self.assertEqual(cap.cuadros, 2)
        self.assertEqual(cap.fps, 120.0)
        np.testing.assert_allclose(cap.posiciones["lfemur"][0], [0, -5, 0], atol=1e-9)
        np.testing.assert_allclose(cap.posiciones["lfemur"][1], [0, 0, -5], atol=1e-9)   # rx 90: hacia −Z


class Retarget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = m3.construir(vs.preparar_vistas(hoja(), alto=600), detalle=0.6)

    def _captura(self, J: dict, cuadros: int = 3, escala: float = 1.0, giro: float = 0.0) -> cp.Captura:
        """Una captura BVH con las articulaciones dadas (en el espacio del modelo), girada y escalada."""
        c, s = np.cos(np.radians(giro)), np.sin(np.radians(giro))
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pos = {NOMBRES[k]: np.tile(np.asarray(v, float) @ R.T * escala, (cuadros, 1)) for k, v in J.items()}
        return cp.Captura("sintetica", 30.0, pos)

    def _posar(self, cap, **kw):
        poses, desp = cp.poses_desde_captura(self.m, cap, **kw)
        return poses, [m3.posar3d(self.m, p) for p in poses]

    def test_la_pose_de_reposo_no_mueve_nada(self):
        J = {k: np.asarray(v, float) for k, v in self.m.J.items()}
        _, P = self._posar(self._captura(J))
        Jp = P[0].J
        for k in ("mano_izq", "mano_der", "pie_izq", "cabeza", "codo_der"):
            np.testing.assert_allclose(Jp[k] - Jp["cadera"], J[k] - J["cadera"], atol=0.5, err_msg=k)

    def test_el_brazo_derecho_del_personaje_al_frente(self):
        """«izq» del modelo = izquierda de la imagen = brazo derecho del personaje (RightArm en la captura)."""
        J = {k: np.asarray(v, float) for k, v in self.m.J.items()}
        L1 = np.linalg.norm(J["codo_izq"] - J["hombro_izq"])
        L2 = np.linalg.norm(J["mano_izq"] - J["codo_izq"])
        J["codo_izq"] = J["hombro_izq"] + [0, 0, L1]          # +Z: hacia la cámara (adelante del personaje)
        J["mano_izq"] = J["codo_izq"] + [0, 0, L2]
        for escala, giro in ((1.0, 0.0), (0.01, 90.0), (3.0, -140.0)):   # cualquier tamaño y hacia donde mire
            _, P = self._posar(self._captura(J, escala=escala, giro=giro))
            Jp = P[0].J
            self.assertLess(_angulo(Jp["codo_izq"] - Jp["hombro_izq"], [0, 0, 1]), 4, (escala, giro))
            self.assertLess(_angulo(Jp["mano_izq"] - Jp["codo_izq"], [0, 0, 1]), 4, (escala, giro))
            self.assertLess(_angulo(Jp["codo_der"] - Jp["hombro_der"],
                                    self.m.J["codo_der"] - self.m.J["hombro_der"]), 4)   # el otro, quieto

    def test_el_salto_sale_de_la_altura_de_los_pies(self):
        J = {k: np.asarray(v, float) for k, v in self.m.J.items()}
        cap = self._captura(J, cuadros=2)
        alto = float(J["cabeza"][1] - min(J["pie_izq"][1], J["pie_der"][1]))
        for k in cap.posiciones:
            cap.posiciones[k][1, 1] += 0.2 * alto              # el cuadro 2, todo el cuerpo arriba
        poses, _ = cp.poses_desde_captura(self.m, cap)
        self.assertAlmostEqual(poses[0]["y"], 0.0, places=3)
        self.assertAlmostEqual(poses[1]["y"], 0.2, delta=0.02)

    def test_si_la_persona_da_la_vuelta_la_mascota_sigue_de_frente(self):
        """Las caminatas de CMU van y vuelven: sin --con-giro, el pecho mira siempre a la cámara."""
        J = {k: np.asarray(v, float) for k, v in self.m.J.items()}
        cuadros = 60
        pos = {}
        for k, v in J.items():
            filas = []
            for f in range(cuadros):
                a = np.radians(180.0 * f / (cuadros - 1))          # media vuelta en 2 s
                c, s_ = np.cos(a), np.sin(a)
                d = v - J["cadera"]
                filas.append(J["cadera"] + [c * d[0] + s_ * d[2], d[1], -s_ * d[0] + c * d[2]])
            pos[NOMBRES[k]] = np.array(filas)
        cap = cp.Captura("vuelta", 30.0, pos)
        for girar, esperado in ((False, 0.0), (True, 180.0)):
            poses, _ = cp.poses_desde_captura(self.m, cap, girar=girar)
            Jp = m3.posar3d(self.m, poses[-1]).J
            hombros = Jp["hombro_der"] - Jp["hombro_izq"]
            rumbo = abs(np.degrees(np.arctan2(hombros[2], hombros[0])))
            self.assertLess(abs(rumbo - esperado), 25, girar)

    def test_remuestrea_a_los_cuadros_del_video(self):
        J = {k: np.asarray(v, float) for k, v in self.m.J.items()}
        cap = self._captura(J, cuadros=240)
        cap.fps = 120.0
        poses, desp = cp.poses_desde_captura(self.m, cap, fps=30.0)
        self.assertEqual(len(poses), 60)                       # 2 s a 30 cuadros por segundo
        self.assertEqual(len(desp), 60)


class CapturasCMU(unittest.TestCase):
    """23-oct (#6, el programa público): las capturas no van en el repositorio; se bajan de CMU y se comprueban."""

    def test_catalogo_con_huellas_y_urls(self):
        for archivo, (que, huella) in cp.CMU.items():
            self.assertRegex(huella, r"^[0-9a-f]{64}$", archivo)
            self.assertTrue(que)
        self.assertEqual(cp._url_cmu("141_16.amc", "https"), "https://mocap.cs.cmu.edu/subjects/141/141_16.amc")
        self.assertEqual(cp._url_cmu("14.asf", "http"), "http://mocap.cs.cmu.edu/subjects/14/14.asf")

    def test_no_vuelve_a_bajar_lo_que_ya_esta_y_lo_encuentra(self):
        import hashlib
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            datos = b"# AMC de prueba\n"
            original = cp.CMU
            cp.CMU = {"141_16.amc": ("saludar", hashlib.sha256(datos).hexdigest())}
            try:
                carpeta = Path(d) / "capturas" / "cmu"
                carpeta.mkdir(parents=True)
                (carpeta / "141_16.amc").write_bytes(datos)
                self.assertEqual(cp.descargar_cmu(carpeta), [carpeta / "141_16.amc"])     # sin internet
                (carpeta / "propia.bvh").write_text("HIERARCHY", encoding="utf-8")
                self.assertEqual(cp.buscar_capturas(d), {"saludar · CMU 141_16": carpeta / "141_16.amc",
                                                         "propia": carpeta / "propia.bvh"})
            finally:
                cp.CMU = original


if __name__ == "__main__":
    unittest.main()
