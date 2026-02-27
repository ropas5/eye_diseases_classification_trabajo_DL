# models/simple_cnn.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def create_model(input_shape=(128,128,3), num_classes=4):
    model = Sequential()
    # Capa convolucional simple
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(GlobalAveragePooling2D())

    # Flatten y capa de salida
    model.add(Dense(num_classes, activation='softmax'))

    # Compilación
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model