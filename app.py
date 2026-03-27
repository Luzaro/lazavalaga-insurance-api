from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from decouple import config
from flask_cors import CORS

import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# =========================
# CONFIGURACIÓN
# =========================
app.config["SQLALCHEMY_DATABASE_URI"] = config("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)

# =========================
# MODELO DE BASE DE DATOS
# =========================
class Insurance(db.Model):
    __tablename__ = "insurance"

    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=True)

    def __init__(self, age, price=None):
        self.age = age
        self.price = price


# =========================
# ESQUEMA DE SERIALIZACIÓN
# =========================
class InsuranceSchema(ma.Schema):
    id = ma.Integer(dump_only=True)
    age = ma.Integer(required=True)
    price = ma.Float()


insurance_schema = InsuranceSchema()
insurances_schema = InsuranceSchema(many=True)

# =========================
# CREAR TABLAS
# =========================
with app.app_context():
    db.create_all()
    print("Tablas en base de datos creadas")

# =========================
# CARGA DEL MODELO ML
# =========================
model = joblib.load("./model/model.pkl")
sc_x = joblib.load("./model/scaler_x.pkl")
sc_y = joblib.load("./model/scaler_y.pkl")


def predict_price(age):
    age_array = np.array([[age]])
    age_sc = sc_x.transform(age_array)
    prediction = model.predict(age_sc)

    prediction = np.array(prediction).reshape(-1, 1)
    prediction_sc = sc_y.inverse_transform(prediction)

    price = round(float(prediction_sc[0][0]), 2)
    return price


# =========================
# RUTAS
# =========================
@app.route("/", methods=["GET"])
def index():
    context = {
        "title": "TRABAJO FINAL MODULO 8",
        "message": "AUTOR: Luis Zavalaga Rodrigo"
    }
    return jsonify(context), 200


@app.route("/insurance_price", methods=["POST"])
def insurance_price():
    data = request.get_json()

    if not data or "age" not in data:
        return jsonify({"message": "Debe enviar el campo 'age'"}), 400

    age = data["age"]
    price = predict_price(age)

    context = {
        "message": "Precio predicho",
        "age": age,
        "insurance_price": price
    }

    return jsonify(context), 200


@app.route("/insurance", methods=["POST"])
def create_insurance():
    data = request.get_json()

    if not data or "age" not in data:
        return jsonify({"message": "Debe enviar el campo 'age'"}), 400

    age = data["age"]
    price = predict_price(age)

    new_data = Insurance(age=age, price=price)
    db.session.add(new_data)
    db.session.commit()

    return jsonify(insurance_schema.dump(new_data)), 201


@app.route("/insurance", methods=["GET"])
def get_all_insurance():
    data = Insurance.query.all()
    return jsonify(insurances_schema.dump(data)), 200


@app.route("/insurance/<int:id>", methods=["GET"])
def get_insurance_by_id(id):
    data = Insurance.query.get(id)

    if not data:
        return jsonify({"message": "Registro no encontrado"}), 404

    return jsonify(insurance_schema.dump(data)), 200


@app.route("/insurance/<int:id>", methods=["PUT"])
def update_insurance(id):
    data = Insurance.query.get(id)

    if not data:
        return jsonify({"message": "Registro no encontrado"}), 404

    body = request.get_json()

    if not body or "age" not in body:
        return jsonify({"message": "Debe enviar el campo 'age'"}), 400

    age = body["age"]
    price = predict_price(age)

    data.age = age
    data.price = price
    db.session.commit()

    return jsonify(insurance_schema.dump(data)), 200


@app.route("/insurance/<int:id>", methods=["DELETE"])
def delete_insurance(id):
    data = Insurance.query.get(id)

    if not data:
        return jsonify({"message": "Registro no encontrado"}), 404

    db.session.delete(data)
    db.session.commit()

    return jsonify({"message": "Registro eliminado correctamente"}), 200


if __name__ == "__main__":
    app.run(debug=True)