# MenteAnima 3D — anima tu dibujo con IA, gratis

De la hoja de tu personaje (frente, perfil y espalda) a un personaje 3D que camina, saluda, baila o salta, con
**fondo transparente** (WebM, MOV, GIF) o como **modelo 3D** (.glb) con esqueleto y animaciones. Puede copiar los
movimientos de **personas reales** (capturas de Carnegie Mellon). Todo corre en tu computador.

Lo creamos con Claude Code para el canal [MenteXYZ](https://www.youtube.com/@MenteXYZ).

**▶ Mira cómo lo hicimos:** [Animar dibujos con IA: crea tu propio programa paso a paso](https://youtu.be/kxiTpOouA48) (lección 6 del
[curso gratis](https://www.youtube.com/playlist?list=PLHPY78Hq8y6I)).

## 1. Descárgalo (no necesitas cuenta de GitHub)

Arriba en esta página: botón verde **Code** → **Download ZIP**. Descomprime el ZIP en una carpeta.

> ¿Qué es GitHub? El sitio donde los programadores guardan su código con todo su historial (usa Git, que recuerda
> cada versión de cada archivo). Para descargar no hace falta registrarse.

## 2. Instálalo (Windows)

1. Instala **Python 3.11 o más nuevo** desde [python.org](https://www.python.org/downloads/) y marca
   «Add python.exe to PATH».
2. Para exportar en .webm o .mov, instala **ffmpeg**: abre una terminal y escribe `winget install Gyan.FFmpeg`.
3. En la carpeta del programa, abre una terminal y escribe:

```
pip install -r requirements.txt
```

## 3. Úsalo

```
python menteanima.py
```

Se abre la ventana: **Subir imagen…** → revisa el esqueleto (arrastra un punto si quedó mal) → **Ver en 3D** →
elige movimientos o una **persona real** → **Reproducir en 3D** → **Exportar video…**

- ¿Sin dibujo propio? Prueba con el personaje de ejemplo: `python menteanima.py ventana ejemplo/personaje_de_prueba.png`
- Movimientos de personas reales (Carnegie Mellon, gratis, 8 movimientos): `python menteanima.py capturas`
- Sin ventana: `python menteanima.py animar hoja.png -m caminar,saludar --vista frente -o camina.webm`
- Con una persona real: `python menteanima.py animar hoja.png --captura capturas/cmu/141_16.amc -o saluda.webm`
- Modelo 3D para Blender: `python menteanima.py modelo hoja.png -o personaje.glb`
- Todos los movimientos y opciones: `python menteanima.py movimientos` · `python menteanima.py -h`

## Tu hoja de personaje

Cuatro vistas del mismo personaje, del mismo tamaño y paradas sobre el mismo suelo, con fondo transparente o liso.
Si no la tienes, este es el prompt que te recomendamos para pedírsela a ChatGPT junto con la imagen de tu personaje:

> Hoja de personaje del mismo personaje de la imagen adjunta, en pose A: brazos rectos separados del cuerpo a 45°, piernas un poco abiertas, manos abiertas con los dedos separados, mirando al frente. Cuatro vistas en una fila, del mismo tamaño y alineadas al mismo suelo: frente, perfil derecho, espalda y perfil izquierdo. Fondo transparente, sin sombra ni reflejo en el piso, sin brillo alrededor de la figura, iluminación neutra y pareja.

Con una sola imagen de frente también funciona: el 3D sale aproximado (la espalda es un espejo del frente).

## Cómo funciona

1. Separa la hoja en sus vistas y las alinea.
2. Pone un esqueleto de 15 articulaciones en el frente (lo corriges en la ventana) y le da profundidad con los perfiles.
3. Arma el volumen de cada parte con el ancho del frente y la profundidad del perfil, y lo pinta con las 4 vistas.
4. Lo mueve con huesos 3D: movimientos hechos a mano o la **dirección de cada hueso** de una persona real capturada.
5. Lo dibuja con la tarjeta gráfica (OpenGL) o, si no hay, con numpy.

Pruebas: `python -m unittest discover -s tests`

## Privacidad y seguridad

- Tus dibujos no salen de tu computador. Solo usan internet `capturas` (baja los archivos de CMU y comprueba cada
  uno con su huella SHA-256) y la opción `trellis` (envía las vistas a la demo pública de TRELLIS en Hugging Face).
- No pide cuentas ni contraseñas.

## Créditos y licencias

- Código: licencia MIT (archivo `LICENSE`). Hecho por MenteXYZ con Claude Code.
- Capturas de movimiento: [CMU Graphics Lab Motion Capture Database](https://mocap.cs.cmu.edu/) — no vienen en
  este repositorio; según su página, son libres para todos los usos y se pueden usar en productos, pero no se pueden
  revender los archivos. Si las usas, CMU pide este agradecimiento: «The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.»
- Opcional: [TRELLIS](https://github.com/microsoft/TRELLIS) (Microsoft, MIT) y [Blender](https://www.blender.org/).
- El nombre, el logo y la mascota de MenteXYZ no están incluidos ni cubiertos por la licencia.
