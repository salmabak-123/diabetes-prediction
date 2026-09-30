from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Chargement du modele entraine (voir train_model.py)
model = pickle.load(open('model.pkl', 'rb'))

# Ordre des caracteristiques attendu par le modele (doit correspondre
# exactement a l'ordre utilise lors de l'entrainement dans train_model.py)
FEATURES = ["bmi", "s5", "bp", "s4", "s3"]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Recuperation et conversion des valeurs du formulaire, dans le bon ordre
    values = [float(request.form[feature]) for feature in FEATURES]
    X = np.array(values).reshape(1, -1)

    prediction = model.predict(X)[0]

    return render_template(
        'result.html',
        prediction=round(prediction, 1),
        inputs=dict(zip(FEATURES, values))
    )


if __name__ == '__main__':
    app.run(debug=True)
