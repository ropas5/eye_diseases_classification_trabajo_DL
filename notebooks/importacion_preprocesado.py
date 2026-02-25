# Este codigo es un ejecutable para automatizar la importacion de los datos en cada notebook
# Es el codigo que se usa en 01_eda.ipynb adaptaba para su uso en los otros notebooks


import os
import zipfile
import numpy as np
from PIL import Image


def download_and_load_data( target_size=(512,512)):
    """
    Descarga, descomprime y carga imágenes como X (arrays) e y (labels).
    Funciona en Colab y en entorno local.
    """
    # Detectar si estamos en Colab
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
    
    # --- DESCARGA Y EXTRACCIÓN (solo si no existe dataset) ---
    if not os.path.exists(dataset_path):
        print("Dataset no encontrado, descargando y extrayendo...")
        if IN_COLAB:
            print("Ejecutando en Colab")
            from google.colab import files
            kaggle_json_path = "/root/.kaggle/kaggle.json"
            if not os.path.exists(kaggle_json_path):
                print("Sube tu kaggle.json")
                files.upload()  # abre diálogo
                os.makedirs("/root/.kaggle", exist_ok=True)
                os.system("mv kaggle.json /root/.kaggle/")
                os.system("chmod 600 /root/.kaggle/kaggle.json")
            os.system(f"kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification -p {path} --force")
        else:
            print("Ejecutando en local")
            os.system(f"kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification -p {path} --force")
        
        # Descomprimir
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
    
    # Creacion de train y test
    random=42
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_cat, train_size=0.7 ,random_state=random ,shuffle=True)

    return X_train, X_test, y_train, y_test