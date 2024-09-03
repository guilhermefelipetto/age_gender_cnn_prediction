import cv2
import numpy as np
from keras.models import load_model

model_path = 'modelo_utkface_keras.h5'

model = load_model(model_path, compile=False)
print("Modelo carregado com sucesso!")

model.compile(optimizer='adam',
              loss={'age_output': 'mse', 'gender_output': 'binary_crossentropy'},
              metrics={'age_output': 'mae', 'gender_output': 'accuracy'})

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
    image = cv2.resize(image, (128, 128))
    
    image = image / 255.0
    
    image = np.expand_dims(image, axis=0)
    
    return image

image_path = 'lady.jpg'

input_image = preprocess_image(image_path)

predictions = model.predict(input_image)

predicted_age = predictions[0][0][0]
predicted_gender = predictions[1][0][0]

gender_label = 'Masculino' if predicted_gender < 0.5 else 'Feminino'

print(f"Predição de Idade: {predicted_age:.2f} anos")
print(f"Predição de Gênero: {gender_label} (Probabilidade: {predicted_gender:.2f})")
