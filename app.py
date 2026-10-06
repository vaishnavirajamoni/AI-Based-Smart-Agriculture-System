from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained AI model
model = joblib.load("models/crop_model.pkl")


@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/crop-recommendation", methods=["GET", "POST"])
def crop_recommendation():

    prediction = None

    if request.method == "POST":

        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        prediction = model.predict([[
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]])[0]

    return render_template(
        "crop_recommendation.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=False)