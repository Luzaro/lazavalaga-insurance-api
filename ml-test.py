import joblib
import numpy as np
import sklearn

model = joblib.load('./model/model.pkl')
sc_x = joblib.load('./model/scaler_x.pkl')
sc_y = joblib.load('./model/scaler_y.pkl')

age = int(input("Ingrese edad : "))
age_sc = sc_x.transform(np.array([[age]]))

prediction_sc = model.predict(age_sc)
prediction = sc_y.inverse_transform(prediction_sc)
print(f'El el monto asegurado para una persona con  {age} años es de : $ {prediction[0][0]:.2f}') 