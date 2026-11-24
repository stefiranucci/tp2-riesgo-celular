# App de deploy del TP2: encuesta de un alumno -> riesgo de adicción al celular
# Corre la red neuronal entrenada en TP2_Red_Neuronal.ipynb usando solo numpy.
import json
from pathlib import Path

import numpy as np
import streamlit as st

CARPETA = Path(__file__).parent
pesos = dict(np.load(CARPETA / "pesos_red.npz"))
with open(CARPETA / "prepro.json", encoding="utf-8") as f:
    prepro = json.load(f)


def predecir(fila):
    # mismo preprocesamiento que en la notebook: escalado + one-hot (sin la primera categoría)
    num = (np.array([fila[c] for c in prepro["num_feats"]]) - np.array(prepro["media"])) / np.array(prepro["desvio"])
    cat = []
    for col, cats in zip(prepro["cat_feats"], prepro["categorias"]):
        cat += [1.0 if fila[col] == c else 0.0 for c in cats[1:]]
    a = np.concatenate([num, cat])[None, :]
    # la red: capas densas con ReLU y softmax al final
    n_capas = len(pesos) // 2
    for i in range(n_capas):
        a = a @ pesos[f"W{i}"] + pesos[f"b{i}"]
        if i < n_capas - 1:
            a = np.maximum(a, 0)
    a = np.exp(a - a.max())
    return (a / a.sum())[0]


st.set_page_config(page_title="Riesgo de adicción al celular", page_icon="📱")
st.title("📱 Screening de riesgo de adicción al celular")
st.write("Completá la encuesta de un alumno. La red neuronal estima la probabilidad de que su riesgo "
         "sea Bajo, Moderado o Alto y te dice si conviene convocarlo al gabinete.")

with st.form("encuesta"):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Datos del alumno")
        edad = st.slider("Edad", 13, 19, 16)
        grado = st.slider("Año escolar (grado)", 7, 12, 9)
        genero = st.selectbox("Género", ["Female", "Male", "Other"],
                              format_func={"Female": "Femenino", "Male": "Masculino", "Other": "Otro"}.get)
        rendimiento = st.slider("Rendimiento académico (50-100)", 50, 100, 75)
        sueno = st.slider("Horas de sueño", 3.0, 10.0, 6.5, 0.1)
        ejercicio = st.slider("Horas de ejercicio por día", 0.0, 4.0, 1.0, 0.1)
        social = st.slider("Interacciones sociales (0-10)", 0, 10, 5)
        familia = st.slider("Comunicación con la familia (1-10)", 1, 10, 5)
        control = st.radio("¿Tiene control parental?", ["No", "Sí"], horizontal=True)
        ansiedad = st.slider("Ansiedad (1-10)", 1, 10, 5)
        depresion = st.slider("Depresión (1-10)", 1, 10, 5)
        autoestima = st.slider("Autoestima (1-10)", 1, 10, 5)
    with c2:
        st.subheader("Uso del celular")
        redes = st.slider("Horas por día en redes sociales", 0.0, 5.0, 2.5, 0.1)
        juegos = st.slider("Horas por día jugando", 0.0, 4.0, 1.5, 0.1)
        educacion = st.slider("Horas por día en apps educativas", 0.0, 3.0, 1.0, 0.1)
        finde = st.slider("Horas de uso en el fin de semana", 0.0, 14.0, 6.0, 0.1)
        noche = st.slider("Horas de pantalla antes de dormir", 0.0, 2.6, 1.0, 0.1)
        chequeos = st.slider("Veces que mira el celular por día", 20, 150, 80)
        apps = st.slider("Apps que usa por día", 5, 20, 12)
        proposito = st.selectbox("Uso principal", ["Browsing", "Education", "Social Media", "Gaming", "Other"],
                                 format_func={"Browsing": "Navegar", "Education": "Educación", "Social Media": "Redes sociales",
                                              "Gaming": "Juegos", "Other": "Otro"}.get)
    enviar = st.form_submit_button("Calcular riesgo", type="primary")

if enviar:
    deficit = max(0.0, 8 - sueno)
    entret = redes + juegos
    fila = {"Age": edad, "Grade_Num": grado, "Sleep_Hours": sueno, "Academic_Performance": rendimiento,
            "Social_Interactions": social, "Exercise_Hours": ejercicio, "Anxiety_Level": ansiedad,
            "Depression_Level": depresion, "Self_Esteem": autoestima, "Parental_Control": int(control == "Sí"),
            "Screen_Time_Before_Bed": noche, "Phone_Checks_Per_Day": chequeos, "Apps_Used_Daily": apps,
            "Time_on_Social_Media": redes, "Time_on_Gaming": juegos, "Time_on_Education": educacion,
            "Family_Communication": familia, "Weekend_Usage_Hours": finde,
            "Entretenimiento": entret, "Ratio_Educativo": educacion / (entret + educacion + 0.1),
            "Deficit_Sueno": deficit, "Chequeos_por_App": chequeos / apps,
            "Pantalla_Noche_x_Deficit": noche * deficit,
            "Gender": genero, "Phone_Usage_Purpose": proposito}
    p = predecir(fila)
    umbral = prepro["umbral"]

    st.divider()
    if p[2] >= umbral:
        st.error(f"**Convocar al gabinete.** Probabilidad de riesgo Alto: {p[2]:.0%} (umbral {umbral:.0%})")
    else:
        st.success(f"**No hace falta convocarlo por ahora.** Probabilidad de riesgo Alto: {p[2]:.0%} (umbral {umbral:.0%})")
    cols = st.columns(3)
    for col, clase, prob in zip(cols, prepro["clases"], p):
        col.metric(f"Riesgo {clase}", f"{prob:.0%}")
    for clase, prob in zip(prepro["clases"], p):
        st.progress(float(prob), text=f"{clase}: {prob:.0%}")

st.caption("TP2 · Análisis Predictivo Avanzado. Modelo: red neuronal (perceptrón multicapa) entrenada con el "
           "dataset Teen Phone Addiction (Kaggle). Uso académico, no es un diagnóstico.")
