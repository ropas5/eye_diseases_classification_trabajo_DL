# models/simple_cnn.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2, l2

def create_model(input_shape=(128,128,3), num_classes=4, l_rate=0.01):
    model = Sequential()
    # Capa convolucional simple
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv2D(16, kernel_size=3, activation="relu"))
    model.add(Conv2D(16, kernel_size=3, activation="relu"))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(16, activation="relu"))
    # Capa de salida
    model.add(Dense(num_classes, activation='softmax'))

    # Compilación
    optimizer = Adam(learning_rate= l_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model