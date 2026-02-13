#  Eye Diseases Classification - Deep Learning Project
Repositorio del trabajo final de la asignatura Aprendizaje Profundo del Grado en Ciencia de Datos (Universitat de València). El proyecto aborda un problema de clasificación de imágenes médicas utilizando técnicas de Deep Learning.

El dataset original esta publicado en Kaggle en este link:

<https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data>

Consiste en un aproximado de 4000 imagenes de retinas clasificadas en 4 categorias: 
- Normal
- Cataratas
- Glaucoma
- Retinopatía diabética

La licencia del dataset es Open Data Commons Open Database License (ODbL) v1.0:

<https://opendatacommons.org/licenses/odbl/1-0/>

Los datos medicos estan anonimizados y proviene de una base pública.  El resulatado del proyecto es educativo, centrado en el aprendizaje de técnicas de Deep Learning, y no pretende desarrollar una herramienta de diagnóstico clínico real.

---

La metodología consistira en entrenar dos modelos de clasificación:
- Un modelo base/baselina (una CNN lo mas simple posible).
- Un modelo complejo.

El modelo baselina irá complejizando progresivamente, mientras que el modelo complejo se irá simplificando, con el objetivo de encontrat dos modelos que tengan resultados similares, y así analizar como afecta la complejidad del modelo al rendimiento.

 Métricas a utilizar

Para evaluar los modelos de clasificación se usarán:
- Accuracy: porcentaje de predicciones correctas.
- F1-score: balance entre precision y recall.
- Confusion Matrix: para analizar errores por clase.


Estructura del repositorio

eye_diseases_classification_trabajo_DL/
│
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ nootbooks/
│   └─ 01_EDA.ipynb
├─ data/
│   ├─ raw/                # datos descargados/descomprimidos (IGNORADOS por git)
│   │   └─ dataset/
│   │        ├─ cataract
│   │        ├─ diabetic_retinopathy
│   │        ├─ glaucoma
│   │        ├─ normal
│   └─ processed/
         └─ model_results.csv          

Los datos no estan en el repositorio. El codigo los descarga al ejecutarlo la primera vez. Usando la API de kaggle.  **Es necesario tener un API de kaggle una para ejecutar la nootbook en Colab**
