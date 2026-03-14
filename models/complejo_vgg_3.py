# models/simple_cnn.py
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2, l2

def create_model(input_shape=(128,128,3), num_classes=4, l_rate=0.01, bloque = "block4_pool"):
    # 1. Reconstruir la arquitectura EXACTA con la que guardamos los pesos
    vgg_base = VGG16(
        weights=None,  # entrenado desde 0
        input_shape=input_shape,
        include_top=False  # Sin capa de salida (la agregaremos nosotros)
        )
    
    full_model = Sequential([
        vgg_base,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # 2. Cargar pesos 
    full_model.load_weights('vgg_pesos.weights.h5')
    
    # 3. Ahora recortar la VGG hasta el blocke que queremos
    vgg_reduced = Model(
        inputs=vgg_base.input,
        outputs=vgg_base.get_layer(bloque).output
    )
    
    # 4. Construir modelo final
    model = Sequential([
        vgg_reduced,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    # Los pesos de vgg_reduced ya están cargados porque comparte
    # los mismos objetos de capa que vgg_base
    optimizer = Adam(learning_rate=l_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model