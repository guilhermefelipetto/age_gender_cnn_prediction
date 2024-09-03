import os

import cv2
import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split

dataset_path = 'data/UTKFace/'

images = []
ages = []
genders = []

def load_utkface_data(dataset_path):
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            try:
                age, gender, race, _ = filename.split('_')

                img_path = os.path.join(dataset_path, filename)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (128, 128))
                
                image = image / 255.0

                images.append(image)
                ages.append(int(age))
                genders.append(int(gender))
            except Exception as e:
                print(f"Erro ao carregar a imagem {filename}: {e}")

load_utkface_data(dataset_path)

images = np.array(images)
ages = np.array(ages)
genders = np.array(genders)

X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender = train_test_split(
    images, ages, genders, test_size=0.2, random_state=42
)

def create_cnn_model():
    input_layer = Input(shape=(128, 128, 3))

    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    age_output = Dense(1, name='age_output')(x)

    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)

    model = Model(inputs=input_layer, outputs=[age_output, gender_output])
    return model

model = create_cnn_model()
model.compile(optimizer='adam',
              loss={'age_output': 'mse', 'gender_output': 'binary_crossentropy'},
              metrics={'age_output': 'mae', 'gender_output': 'accuracy'})

history = model.fit(X_train, {'age_output': y_train_age, 'gender_output': y_train_gender},
                    validation_data=(X_test, {'age_output': y_test_age, 'gender_output': y_test_gender}),
                    epochs=20, batch_size=32)

loss, age_loss, gender_loss, age_mae, gender_accuracy = model.evaluate(
    X_test, {'age_output': y_test_age, 'gender_output': y_test_gender})
print(f"Mean Absolute Error for Age Prediction: {age_mae}")
print(f"Accuracy for Gender Prediction: {gender_accuracy}")

model.save('modelo_utkface_keras.h5')
print("Modelo salvo como 'modelo_utkface_keras.h5'.")
