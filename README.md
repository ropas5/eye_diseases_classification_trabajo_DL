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


## Resultados:

| Modelo              | Parametros | Epochs | Train Accuracy | Val Accuracy | Test Accuracy | Train F1 | Val F1 | Test F1 |
|---------------------|------------|--------|----------------|--------------|---------------|---------|---------|---------|
| Modelo Simple       | 196612     | 1000   | 0.99           | 0.79         | 0.80          | 0.99    | 0.79    | 0.80    |
| Random Forest       | 47844      | -      | 1              | 0.73         | 0.71          | 1       | 0.76    |  0.75   |
| CNN Básica          | 516        | 140    | 0.47| 0.483         | 0.462          | 0.467    | 0.48    | 0.62    |
| CNN 2xConv16        | 2836       | 110     |0.51            | 0.51         | 0.523         | 0.48    | 0.48    | 0.49    |
| CNN 3xConv16        | 5156       | 160     | 0.762          | 0.749        | 0.718         | 0.745   | 0.735   | 0.698   |
| CNN 3xConv16+Dense16| 5428       | 160    | 0.880          | 0.866        | 0.859         | 0.877   | 0.860   | 0.854   |
| CNN Conv32 + 2xConv16 + Dense16 | 8.180    | 160    | 0.742          |0.737        | 0.718         | 0.726   | 0.724   | 0.703   |
| CNN 2xConv32 + Conv16 + Dense 16 | 15108 | 100 | 0.811|  0.815 | 0.808| 0.793 |  0.798 | 0.789 |
| VGG(Frozzen) + Dense512 + Dense256 | 15109700 | 70 | 0.866 | 0.850 | 0.853 | 0.865 | 0.847 | 0.850 |
| VGG + Dense512 + Dense256 | 15109700 | 40 | 0.999 | 0.906 | 0.914 | 0.999 | 0.906 | 0.913 |
| VGG(Block4) + Dense512 + Dense256 | 8030276 | 40 | 0.999 | 0.906  |0.914 | 0.999 | 0.906 | 0.913 |
| VGG(Block3) + Dense512 + Dense256 | 1999428 | 40 | 0.998 | 0.915 | 0.927 | 0.998 | 0.914 | 0.926 |
| VGG(Block2) + Dense512 + Dense256 | 458,564 | 40 | 0.261 | 0.255| 0.236|0.108 |0.103|0.09|

Estructura del repositorio
```
eye_diseases_classification_trabajo_DL/
│
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ notebooks/
│   ├─ 01_EDA.ipynb
│   ├─ importacion_preprocesado.py
│   ├─ modelo_1.ipynb
│   ├─ modelo_2.ipynb
│   ├─ modelo_3.ipynb
│   ├─ modelo_2_simple.ipynb
│   ├─ modelo_3_simple.ipynb
│   ├─ modelo_4_simple.ipynb
│   ├─ modelo_5_simple.ipynb
│   ├─ modelo_6_simple.ipynb
│   ├─ modelo_vgg.ipynb
│   ├─ modelo_vgg_2.ipynb
│   ├─ modelo_vgg_3.ipynb
│   ├─ modelo_vgg_4.ipynb
│   ├─ modelo_vgg_5.ipynb
│   ├─ modelo_resnet.ipynb
│   ├─ modelo_mas_complejo.ipynb
│   └─ modelo_complex_2.ipynb
├─ models/
│   ├─ simple_cnn.py
│   ├─ simple_2_cnn.py
│   ├─ simple_3_cnn.py
│   ├─ simple_4_cnn.py
│   ├─ simple_5_cnn.py
│   ├─ simple_6_cnn.py
│   ├─ complejo_cnn.py
│   ├─ complejo_2_cnn.py
│   ├─ complejo_vgg_2.py
│   ├─ complejo_vgg_3.py
│   └─ vgg_pesos.weights.h5
├─ data/
│   ├─ raw/        # datos descargados/descomprimidos (IGNORADOS por git)
│   │   └─ dataset/
│   │       ├─ cataract/
│   │       ├─ diabetic_retinopathy/
│   │       ├─ glaucoma/
│   │       └─ normal/
│   └─ processed/
│       └─ model_results.csv        
```

Los datos no estan en el repositorio. El codigo los descarga al ejecutarlo la primera vez. Usando la API de kaggle.  **Es necesario tener un API de kaggle una para ejecutar la nootbook en Colab**
