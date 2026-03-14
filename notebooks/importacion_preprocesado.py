# Este codigo es un ejecutable para automatizar la importacion de los datos en cada notebook
# Es el codigo que se usa en 01_eda.ipynb adaptaba para su uso en los otros notebooks


import os
import zipfile
import numpy as np
from PIL import Image


def descarga_y_carga_de_datos(target_size=(512,512), uploader=None):
    try:
        import google.colab
        IN_COLAB = True
        path = "data/raw"
    except:
        IN_COLAB = False
        path = "../data/raw"

    os.makedirs(path, exist_ok=True)

    dataset_path = os.path.join(path, "dataset")
    zip_file_path = os.path.join(path, "eye-diseases-classification.zip")

    if not os.path.exists(dataset_path):
        print("Dataset no encontrado, descargando y extrayendo...")
        if IN_COLAB:
            kaggle_json_path = "/root/.kaggle/kaggle.json"
            if not os.path.exists(kaggle_json_path):
                # Intentar leer del widget si se pasó
                if uploader and len(uploader.value) > 0:
                    print("Leyendo kaggle.json del widget...")
                    uploaded_file = list(uploader.value.values())[0]
                    content = uploaded_file['content']
                    os.makedirs("/root/.kaggle", exist_ok=True)
                    with open(kaggle_json_path, 'wb') as f:
                        f.write(content)
                    os.system("chmod 600 /root/.kaggle/kaggle.json")
                    print("kaggle.json instalado correctamente.")
                else:
                    # Fallback: diálogo web (solo funciona en Colab web)
                    print("Sube tu kaggle.json")
                    from google.colab import files
                    files.upload()
                    os.makedirs("/root/.kaggle", exist_ok=True)
                    os.system("mv kaggle.json /root/.kaggle/")
                    os.system("chmod 600 /root/.kaggle/kaggle.json")


            os.system(f"kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification -p {path} --force")
            print("Cargando imagenes")
        else:
            print("Ejecutando en local")
            os.system(f"kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification -p {path} --force")

        if os.path.exists(zip_file_path):
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(path)
            print("Dataset extraído en:", dataset_path)
        else:
            raise FileNotFoundError(f"No se encontró el zip descargado: {zip_file_path}")
    else:
        print("Dataset ya existe, solo se van a cargar las imágenes.")

    # --- CARGA DE IMÁGENES ---
    imgs = []
    labels = []
    folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                with Image.open(fpath) as img:
                    img = img.convert("RGB")
                    img = img.resize(target_size)
                    arr = np.array(img)
                imgs.append(arr)
                labels.append(folder)
            except Exception as e:
                print("skip:", fpath, str(e))

    X = np.stack(imgs)
    y = np.array(labels)
    print("X shape:", X.shape, "y shape:", y.shape)

    return X, y


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def preprocesamiento(X, y):
    # Hacemos one hot encoding a la variable objetivo para su uso posterio en los modelos
    le = LabelEncoder()
    y_int = le.fit_transform(y) # Tranformar a enteros

    y_cat = to_categorical(y_int, num_classes=4) # transforma a categoria
    
    # Normalización de los datos
    X_norm = X.astype("float32")/255.0
    
    # Creacion de train(60%), validacion(20%) y test(20%)
    random=42

    X_train_val, X_test, y_train_val, y_test = train_test_split(X_norm, y_cat, test_size=0.2, random_state=random, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=random, shuffle=True) 

    return X_train, X_val, X_test, y_train, y_val, y_test