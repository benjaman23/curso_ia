"""Visualizaciones interactivas del taller de K-Means y CartPole.

Las operaciones de preparación, entrenamiento, predicción y evaluación
se realizan en el notebook; aquí solo se construyen visualizaciones.
"""

from copy import deepcopy
from html import escape
from math import cos, sin, pi

import numpy as np
import matplotlib.pyplot as plt


def mostrar_clusters(X_original, etiquetas, centros_originales, columnas):
    """Dibuja datos y centroides en las unidades originales."""
    grupos = np.asarray(etiquetas, dtype=int)
    centros = np.asarray(centros_originales)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    palette = plt.get_cmap("tab10")
    for g in sorted(set(grupos)):
        mask = grupos == g
        ax.scatter(
            X_original.iloc[mask, 0], X_original.iloc[mask, 1],
            s=21, color=palette(int(g) % 10), alpha=0.67, label=f"Cluster {g}",
        )
    ax.scatter(centros[:, 0], centros[:, 1], marker="X", s=240,
               color="#191919", edgecolor="white", linewidth=1.5,
               zorder=5, label="Centroides")
    for g, (x, y) in enumerate(centros):
        ax.annotate(str(g), (x, y), xytext=(8, 7), textcoords="offset points",
                    fontweight="bold", fontsize=10)
    ax.set_xlabel(columnas[0] + " (mm)")
    ax.set_ylabel(columnas[1] + " (g)")
    ax.set_title("K-Means: registros y centroides finales")
    ax.grid(alpha=0.13)
    ax.legend(loc="best", ncol=2, fontsize=9)
    fig.tight_layout()
    plt.show()


def explorar_centroides(X_escalado, escalador, k=3, semilla=15):
    """Simula iteraciones K-Means sobre los mismos datos, con otro inicio.

    Esto es una visualización didáctica del ciclo asignar/recalcular;
    no reconstruye la inicialización de KMeans de scikit-learn.
    """
    import ipywidgets as widgets
    from IPython.display import display

    datos = np.asarray(X_escalado, dtype=float)
    if datos.ndim != 2 or datos.shape[1] != 2 or not np.isfinite(datos).all():
        raise ValueError("Se requieren dos características numéricas, ya preparadas.")
    k = int(k)
    if k < 2 or k >= len(datos):
        raise ValueError("K debe ser al menos 2 y menor al número de registros.")
    rng = np.random.default_rng(semilla)
    centros = datos[rng.choice(len(datos), size=k, replace=False)].copy()
    historia = [(centros.copy(), None)]
    for _ in range(12):
        distancias = np.linalg.norm(datos[:, None, :] - centros[None, :, :], axis=2)
        etiquetas = np.argmin(distancias, axis=1)
        nuevos = np.array([
            datos[etiquetas == i].mean(axis=0) if np.any(etiquetas == i)
            else centros[i]
            for i in range(k)
        ])
        historia.append((nuevos.copy(), etiquetas.copy()))
        if np.allclose(centros, nuevos, rtol=0, atol=1e-6):
            break
        centros = nuevos

    originales = escalador.inverse_transform(datos)
    centros_hist = [escalador.inverse_transform(c) for c, _ in historia]
    xlo, xhi = float(originales[:, 0].min()), float(originales[:, 0].max())
    ylo, yhi = float(originales[:, 1].min()), float(originales[:, 1].max())
    xpad, ypad = max((xhi - xlo) * 0.10, 1), max((yhi - ylo) * 0.10, 1)
    xlo, xhi = xlo - xpad, xhi + xpad
    ylo, yhi = ylo - ypad, yhi + ypad
    def xy(x, y):
        return 60 + 620 * (float(x) - xlo) / (xhi-xlo), 305 - 260 * (float(y)-ylo) / (yhi-ylo)
    colors = ["#2369be", "#d17a32", "#309679", "#9157a3", "#d8586a", "#8e8a23"]

    etapa = widgets.IntSlider(value=0, min=0, max=len(historia)-1, step=1,
                              description="Iteración", continuous_update=False,
                              layout=widgets.Layout(width="98%"))
    panel = widgets.HTML()

    def render(change=None):
        paso = etapa.value
        centros, etiquetas = historia[paso]
        pieces = [
            '<div style="font-family:system-ui;color:#303847">',
            f'<b>{"Centroides iniciales" if paso==0 else f"Iteración {paso}: asignar y recalcular"}</b>',
            '<div style="font-size:12px;color:#627083">Los puntos cambian de color según el centroide cercano; los centros se desplazan.</div>',
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 735 346" style="width:100%;max-width:735px;background:#fcfcfd;border:1px solid #d5dae2;border-radius:9px">',
            '<line x1="60" y1="305" x2="690" y2="305" stroke="#858e9d"/>',
            '<line x1="60" y1="24" x2="60" y2="305" stroke="#858e9d"/>',
        ]
        for index, (x, y) in enumerate(originales):
            cx, cy = xy(x, y)
            col = colors[int(etiquetas[index]) % len(colors)] if etiquetas is not None else "#9ba5b3"
            pieces.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3.0" fill="{col}" fill-opacity=".68"/>')
        for j, (x, y) in enumerate(centros_hist[paso]):
            cx, cy = xy(x, y)
            pieces.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" fill="#ffffff" stroke="#222" stroke-width="2.5"/>')
            pieces.append(f'<text x="{cx:.1f}" y="{cy+4:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="#222">{j}</text>')
        pieces += ['<text x="370" y="336" text-anchor="middle" font-size="12" fill="#444">Largo del pico (mm)</text>',
                   '<text x="18" y="160" text-anchor="middle" transform="rotate(-90 18 160)" font-size="12" fill="#444">Masa corporal (g)</text>',
                   '</svg>',
                   '<div style="font-size:12px;color:#627083">Visualización ilustrativa con una inicialización propia; no representa necesariamente los mismos centros iniciales del modelo entrenado.</div>',
                   '</div>']
        panel.value = "".join(pieces)

    etapa.observe(render, names="value")
    render()
    display(widgets.VBox([etapa, panel]))


def guardar_politica(modelo):
    """Guarda una copia en memoria de los pesos, solo para mostrar etapas."""
    return deepcopy(modelo.policy.state_dict())


def explorar_cartpole(modelo, etapas, semilla=17):
    '''CartPole en vivo: PPO controla el carro y el estudiante aplica perturbaciones.

    La política activa se copia del modelo y usa los pesos guardados en cada etapa.
    El entorno avanza solo cuando se reproduce o se pulsa «Un paso».
    Los empujones alteran el estado físico; las acciones siguientes las decide PPO.
    '''
    import gymnasium as gym
    import ipywidgets as widgets
    from IPython.display import display

    if not etapas:
        raise ValueError("Se necesita al menos una política guardada.")

    # Se conserva intacto el agente entrenado en el notebook.
    politica = deepcopy(modelo.policy)
    politica.set_training_mode(False)
    nombres = list(etapas)

    etapa = widgets.Dropdown(options=nombres, value=nombres[0], description="Política:",
                             style={"description_width": "70px"},
                             layout=widgets.Layout(width="310px"))
    caso = widgets.IntSlider(value=int(semilla), min=0, max=100, step=1,
                             description="Semilla:", continuous_update=False,
                             style={"description_width": "70px"},
                             layout=widgets.Layout(width="310px"))
    intensidad = widgets.Dropdown(options=[("Suave", 0.45), ("Medio", 0.9), ("Fuerte", 1.35)],
                                  value=0.9, description="Empujón:",
                                  style={"description_width": "75px"},
                                  layout=widgets.Layout(width="200px"))

    reproducir = widgets.Play(value=0, min=0, max=500, step=1, interval=120,
                              description="Iniciar / pausar")
    siguiente = widgets.Button(description="Un paso", icon="step-forward",
                               button_style="", layout=widgets.Layout(width="105px"))
    reinicio = widgets.Button(description="Reiniciar", icon="refresh",
                              layout=widgets.Layout(width="110px"))
    izquierda = widgets.Button(description="Empujar a la izquierda", icon="arrow-left",
                               button_style="warning", layout=widgets.Layout(width="190px"))
    derecha = widgets.Button(description="Empujar a la derecha", icon="arrow-right",
                             button_style="warning", layout=widgets.Layout(width="190px"))
    panel = widgets.HTML()

    datos = {"env": None, "estado": None, "paso": 0, "recompensa": 0.0,
             "accion": None, "vivo": False, "final": "", "impulso": "Ninguno",
             "historia": [], "eventos": []}

    def dibujar():
        obs = np.asarray(datos["estado"], dtype=float)
        pos, velocidad, angulo, giro = [float(z) for z in obs]
        # Dibujo SVG simple del estado real del entorno, sin trayectorias precalculadas.
        cx = 360 + np.clip(pos, -2.4, 2.4) * 116
        bx = cx + sin(angulo) * 135
        by = 152 - cos(angulo) * 135
        accion = ("← Izquierda" if datos["accion"] == 0 else
                  "Derecha →" if datos["accion"] == 1 else "Todavía no actúa")
        estado_txt = datos["final"] if not datos["vivo"] else "Controlando en vivo"
        estado_col = "#b91c1c" if not datos["vivo"] else "#197a55"
        deg = angulo * 180 / pi

        # Evolución angular para distinguir recuperación de una barra que sigue cayendo.
        hist = datos["historia"][-110:]
        primero = hist[0][0]
        ultimo = hist[-1][0]
        span = max(ultimo-primero, 1)
        def px(p):
            return 42 + 638*(p-primero)/span
        def py(a):
            return 60 - np.clip(a, -0.28, 0.28)*110
        poly = " ".join(f"{px(p):.1f},{py(a):.1f}" for p,a in hist)
        marks = "".join(
            f'<line x1="{px(p):.1f}" y1="18" x2="{px(p):.1f}" y2="100" stroke="#d97706" stroke-width="2" opacity=".9"/>'
            for p in datos["eventos"] if primero <= p <= ultimo
        )
        panel.value = f'''<div style="font-family:system-ui,sans-serif;max-width:780px;color:#253448">
<div style="display:flex;gap:8px;flex-wrap:wrap;margin:8px 0">
 <div style="background:#eaf1fa;border-radius:7px;padding:8px 12px">Paso: <b>{datos['paso']}</b></div>
 <div style="background:#e7f4ec;border-radius:7px;padding:8px 12px">Recompensa acumulada: <b>{datos['recompensa']:.0f}</b></div>
 <div style="background:#eaf1fa;border-radius:7px;padding:8px 12px">PPO decidió: <b>{accion}</b></div>
</div>
<svg viewBox="0 0 720 229" style="background:#fcfdff;border:1px solid #ccd6e3;border-radius:9px;display:block;width:100%" xmlns="http://www.w3.org/2000/svg">
 <rect x="0" y="190" width="720" height="39" fill="#f0f4f9"/>
 <line x1="58" y1="176" x2="662" y2="176" stroke="#77849a" stroke-width="3"/>
 <line x1="{360-2.4*116:.1f}" y1="167" x2="{360-2.4*116:.1f}" y2="186" stroke="#b91c1c" stroke-width="2"/>
 <line x1="{360+2.4*116:.1f}" y1="167" x2="{360+2.4*116:.1f}" y2="186" stroke="#b91c1c" stroke-width="2"/>
 <line x1="{cx:.1f}" y1="152" x2="{bx:.1f}" y2="{by:.1f}" stroke="#d59046" stroke-width="9" stroke-linecap="round"/>
 <circle cx="{bx:.1f}" cy="{by:.1f}" r="7" fill="#ab642a"/>
 <rect x="{cx-25:.1f}" y="150" width="50" height="23" rx="5" fill="#324964"/>
 <circle cx="{cx-14:.1f}" cy="177" r="6" fill="#17263b"/>
 <circle cx="{cx+14:.1f}" cy="177" r="6" fill="#17263b"/>
 <text x="360" y="213" text-anchor="middle" font-size="12" fill="#6c778b">La política mueve el carro por sí sola</text>
</svg>
<div style="display:flex;flex-wrap:wrap;gap:13px;padding:9px 0;font-size:13px">
 <span>Ángulo: <b>{deg:+.1f}°</b></span>
 <span>Velocidad del carro: <b>{velocidad:+.2f}</b></span>
 <span>Velocidad angular: <b>{giro:+.2f}</b></span>
 <span style="color:{estado_col}"><b>{estado_txt}</b></span>
</div>
<div style="font-size:12px;color:#546175">Última intervención: <b>{datos['impulso']}</b></div>
<div style="font-size:12px;font-weight:600;margin-top:11px">Ángulo de la barra durante los últimos pasos · Las líneas naranjas indican empujones</div>
<svg viewBox="0 0 720 118" style="background:#fcfdff;border:1px solid #ccd6e3;border-radius:8px;display:block;width:100%" xmlns="http://www.w3.org/2000/svg">
 <line x1="42" y1="60" x2="680" y2="60" stroke="#63758d" stroke-dasharray="4 4"/>
 <line x1="42" y1="36" x2="680" y2="36" stroke="#dfa9a9" stroke-dasharray="3 4"/>
 <line x1="42" y1="84" x2="680" y2="84" stroke="#dfa9a9" stroke-dasharray="3 4"/>
 {marks}
 <polyline points="{poly}" fill="none" stroke="#2563a6" stroke-width="2.3" stroke-linejoin="round"/>
 <text x="4" y="17" font-size="10" fill="#4b5563">+15°</text>
 <text x="8" y="63" font-size="10" fill="#4b5563">0°</text>
 <text x="6" y="102" font-size="10" fill="#4b5563">−15°</text>
</svg>
<p style="font-size:12px;color:#59677a;margin:8px 0 0">Pulse ▶ para que el agente actúe automáticamente. Aplique un empujón en movimiento o en pausa; PPO decidirá cómo reaccionar en los pasos siguientes. Si el episodio termina, pulse Reiniciar.</p>
</div>'''

    def un_paso():
        if not datos["vivo"]:
            return
        accion, _ = politica.predict(datos["estado"], deterministic=True)
        accion = int(np.asarray(accion).item())
        obs, premio, terminado, truncado, _ = datos["env"].step(accion)
        datos["estado"] = obs
        datos["paso"] += 1
        datos["recompensa"] += float(premio)
        datos["accion"] = accion
        datos["historia"].append((datos["paso"], float(obs[2])))
        if terminado or truncado:
            datos["vivo"] = False
            datos["final"] = ("Perdió el equilibrio o salió del límite" if terminado
                               else "Terminó el episodio: 500 pasos")
            reproducir.disabled = True
            izquierda.disabled = derecha.disabled = siguiente.disabled = True
        dibujar()

    def empezar(change=None):
        reproducir.disabled = True
        if datos["env"] is not None:
            datos["env"].close()
        politica.load_state_dict(etapas[etapa.value])
        politica.set_training_mode(False)
        env = gym.make("CartPole-v1")
        estado, _ = env.reset(seed=int(caso.value))
        datos.update(env=env, estado=estado, paso=0, recompensa=0.0, accion=None,
                     vivo=True, final="", impulso="Ninguno", historia=[(0, float(estado[2]))],
                     eventos=[])
        reproducir.value = 0
        izquierda.disabled = derecha.disabled = siguiente.disabled = False
        reproducir.disabled = False
        dibujar()

    def aplicar_empujon(sentido):
        if not datos["vivo"]:
            return
        # Empujón externo representado por una variación instantánea de velocidades.
        # Es una perturbación didáctica, no una acción elegida por la política PPO.
        magnitud = float(intensidad.value)
        estado = np.asarray(datos["env"].unwrapped.state, dtype=float).copy()
        estado[1] += sentido*magnitud         # velocidad horizontal del carro
        estado[3] -= sentido*0.9*magnitud     # velocidad angular de la barra
        datos["env"].unwrapped.state = estado
        datos["estado"] = estado.astype(np.float32)
        datos["impulso"] = ("Hacia la derecha" if sentido > 0 else "Hacia la izquierda") + \
                            f" · intensidad {magnitud:.2f} · paso {datos['paso']}"
        datos["eventos"].append(datos["paso"])
        dibujar()

    def al_reproducir(change):
        # Play avanza en la interfaz a intervalos; cada cambio ejecuta un paso real.
        if change["new"] > change["old"] and datos["vivo"]:
            un_paso()

    reproducir.observe(al_reproducir, names="value")
    siguiente.on_click(lambda _: un_paso())
    izquierda.on_click(lambda _: aplicar_empujon(-1))
    derecha.on_click(lambda _: aplicar_empujon(1))
    reinicio.on_click(empezar)
    etapa.observe(empezar, names="value")
    caso.observe(empezar, names="value")
    empezar()
    display(widgets.VBox([
        widgets.HTML("<b>Laboratorio de recuperación del equilibrio</b> — la política actúa sola; usted modifica el entorno."),
        widgets.HBox([etapa, caso]),
        widgets.HBox([reproducir, siguiente, reinicio]),
        widgets.HBox([intensidad, izquierda, derecha]),
        panel,
    ]))


def grafico_recompensas(tabla):
    """Visualiza promedios y dispersión calculados previamente en el notebook."""
    ejes = np.arange(len(tabla))
    medias = tabla["Media"].to_numpy()
    desviaciones = tabla["Desviación"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.errorbar(ejes, medias, yerr=desviaciones, fmt="o-", capsize=5,
                linewidth=2, markersize=7, color="#34699b")
    ax.set_xticks(ejes, tabla["Etapa"].tolist())
    ax.set_ylabel("Recompensa por episodio")
    ax.set_title("Evaluación: promedio ± desviación en varios episodios")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    plt.show()
