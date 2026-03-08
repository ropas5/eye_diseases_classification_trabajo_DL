# models/simple_cnn.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1_l2, l2


def create_model(input_shape=(128,128,3), num_classes=4, l_rate=0.01):
    model = Sequential()
    # Bloque 1
    model.add(Conv2D(32, kernel_size=7, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=2))
    
    # Bloque 2
    model.add(Conv2D(64, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=2))

    # Bloque 3
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=2))

    # Global pooling
    model.add(GlobalAveragePooling2D())

    # Capas densas
    model.add(Dense(1024, kernel_regularizer= l1_l2(0.0001,0.0001), activation= "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(512, kernel_regularizer= l1_l2(0.00005,0.00005), activation= "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(256, kernel_regularizer= l1_l2(0.0001,0.0001), activation= "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Capa de salida
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer = l2(0.0001)))

    # Compilación
    optimizer = SGD(learning_rate = l_rate, momentum= 0.9, nesterov= True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model