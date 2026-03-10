import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.optimizers import SGD


def create_model():
    model = Sequential([
    # Bloque 1 - Dropout FUERTE
    Conv2D(32, kernel_size=7, activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    Dropout(0.5),  # ← 50% (fuerte)
    MaxPooling2D(),
    
    # Bloque 2 - Dropout FUERTE
    Conv2D(64, kernel_size=5, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),  # ← 50% (fuerte)
    MaxPooling2D(),
    
    # Bloque 3 - Dropout FUERTE
    Conv2D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),  # ← 50% (fuerte)
    MaxPooling2D(),
    
    GlobalAveragePooling2D(),
    
    # Dense 1 - Dropout MUY FUERTE
    Dense(256, kernel_regularizer=l1_l2(0.0001, 0.0001), activation='relu'),
    BatchNormalization(),
    Dropout(0.6),  # ← 60% (muy fuerte)
    
    # Dense 2 - Dropout MUY FUERTE
    Dense(128, kernel_regularizer=l1_l2(0.00005, 0.00005), activation='relu'),
    BatchNormalization(),
    Dropout(0.6),  # ← 60% (muy fuerte)
    
    # Salida
    Dense(4, activation='softmax', kernel_regularizer=l2(0.0001))
    ])
    
    # Compilar con LEARNING RATE MÁS ALTO
    optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)  # ← 0.001 (más alto)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ["accuracy"])
    
    return model