# Laboratorio integrador: K-Means y aprendizaje por refuerzo

## 1. Archivos

Conserve estos **cuatro archivos juntos, en una misma carpeta**:

| Archivo | Uso |
|---|---|
| `Taller_Estudiantes_KMeans_PPO.ipynb` | Actividad, código y preguntas para responder. |
| `penguins_size.csv` | Datos de los pingüinos (columnas originales del dataset Kaggle `penguins_size`). |
| `apoyo_visual.py` | Gráficos y controles interactivos. **No debe editarlo.** |
| `README.md` | Estas instrucciones. |

Abra el `.ipynb` desde Visual Studio Code con la extensión **Jupyter**. El notebook tiene dos secciones: K-Means y PPO/CartPole. Cada actividad solicita cambios pequeños de código, salvo las visualizaciones que se ejecutan directamente. Escriba las respuestas debajo de las preguntas en celdas Markdown.

## 2. Preparar Python en Windows

**Recomendado: Python 3.12 y un entorno aislado.** Esto evita mezclar bibliotecas antiguas de OpenCV con NumPy 2, que fue la causa del error de compatibilidad en equipos con instalaciones anteriores.

Abra una terminal de VS Code **en la carpeta del taller** y ejecute, en orden:

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
.\.venv\Scripts\python.exe -m pip install "numpy==1.26.4" "opencv-python-headless==4.10.0.84" "pandas==2.2.3" "scikit-learn==1.5.2" "matplotlib==3.9.4" "gymnasium==1.1.1" "stable-baselines3==2.7.0" "ipywidgets==8.1.5" "ipykernel==6.29.5"
```

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name taller-ml --display-name "Python (Taller ML)"
```

**En el notebook:** haga clic en **Seleccionar kernel** (esquina superior derecha) y elija **Python (Taller ML)**. Reinicie el kernel si cambió de entorno después de iniciar el notebook.

> Estos comandos se copian en la **terminal**, no en una celda de Python. No hace falta activar el entorno ni utilizar un archivo `.bat`.

Si el equipo ya tiene un entorno compatible con todas las dependencias, puede utilizarlo; no es necesario instalar dos veces.

## 3. Desarrollo

1. Comience por la sección **A. K-Means**. Complete las líneas `# Complete:` y ejecute las celdas en orden.
2. En la preparación de datos, elija una estrategia para los valores faltantes y observe el efecto en el número de registros. Pruebe la otra estrategia si desea comparar.
3. Mueva el control **Iteración** de K-Means. Observe la asignación de colores y el desplazamiento de los centroides. Después responda las preguntas A.
4. Continúe con **B. CartPole / PPO**. Observe primero las acciones y recompensas de un paso, luego entrene en dos etapas.
5. Use los controles **Etapa**, **Caso inicial**, **Paso** y **Reproducir** para examinar el agente **antes, parcialmente y al finalizar**. Compare esto con la tabla de recompensas de varios episodios.
6. Responda las preguntas B y C y guarde el notebook.

## 4. Conceptos que deben reconocerse

- **K-Means:** datos numéricos preparados → escalado → `fit` → `predict` → silhouette / interpretación.
- **PPO:** entorno y observación → acción → recompensa → `learn` → `predict` → recompensa media por episodio.
- **Recompensa media:** promedio de varias partidas independientes, **no** el resultado de la animación de una única partida.
- **Iteración de PPO:** `n_steps` indica pasos que se recolectan por actualización y `n_epochs` indica pasadas de optimización sobre esos datos; **ninguno** equivale directamente al número de episodios.

## 5. Problemas frecuentes

**Error NumPy 2 / OpenCV:** revise que el kernel activo sea **Python (Taller ML)**, no el Python global de su equipo. Desde ese kernel, puede comprobar:

```python
import numpy, cv2
print(numpy.__version__, cv2.__version__)
```

**No aparece el control interactivo:** confirme que tiene `ipywidgets` en el kernel seleccionado, reinicie el kernel y vuelva a ejecutar el notebook desde el principio.

**No se encuentra `penguins_size.csv` o `apoyo_visual.py`:** confirme que los cuatro archivos estén en una misma carpeta y que el notebook se ejecute desde ella.

**El resultado final de PPO es diferente al de otro equipo:** el aprendizaje y las evaluaciones pueden variar. Compare **promedios de varios episodios** y explique sus observaciones; no se exige una recompensa final exacta.

## 6. Entrega

Entregue **solo el notebook** ejecutado, con gráficos, resultados y respuestas. No es necesario entregar ni modificar `apoyo_visual.py`.
