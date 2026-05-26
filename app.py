from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import catboost as cb

app = Flask(__name__)

# ---------------- CARBON MODEL ----------------
carbon_model = cb.CatBoostRegressor()
carbon_model.load_model("catboost_carbon_model.cbm")

scalers = joblib.load("scalers.pkl")
feature_names = joblib.load("feature_names.pkl")

# ---------------- AQI MODEL ----------------
aqi_model = joblib.load("best_aqi_model.pkl")
aqi_encoder = joblib.load("aqi_label_encoder.pkl")

try:
    aqi_scaler = joblib.load("aqi_scaler.pkl")
    use_scaler = True
except:
    use_scaler = False


# ---------------- UTIL ----------------
def safe_float(v, d=0):
    try:
        return float(v)
    except:
        return d


# ---------------- CARBON PREPROCESS ----------------
def input_preprocessing(df):
    maps = {
        "Body Type": {'underweight':0,'normal':1,'overweight':2,'obese':3},
        "Sex": {'female':0,'male':1},
        "Diet": {'omnivore':0,'vegetarian':1,'vegan':2,'pescatarian':3},
        "Shower": {'less frequently':0,'daily':1,'twice a day':2,'more frequently':3},
        "Heating": {'electricity':0,'natural gas':1,'coal':2,'wood':3},
        "Transport": {'private':0,'public':1},
        "Vehicle": {'None':0,'petrol':1,'diesel':2,'electric':3},
        "Social": {'never':0,'sometimes':1,'often':2},
        "Flight": {'never':0,'rarely':1,'frequently':2,'very frequently':3},
        "Bag Size": {'small':0,'medium':1,'large':2,'extra large':3},
        "Energy Eff": {'No':0,'Sometimes':1,'Yes':2}
    }
    for col, mp in maps.items():
        df[col] = df[col].map(mp)
    return df.apply(pd.to_numeric)


# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template("index.html", page="home")


# ---------------- CARBON ----------------
@app.route('/carbon', methods=['GET', 'POST'])
def carbon():
    prediction, message = None, ""

    if request.method == 'POST':
        f = request.form

        user_input = {
            "Body Type": f.get("body_type"),
            "Sex": f.get("sex"),
            "Diet": f.get("diet"),
            "Shower": f.get("shower"),
            "Heating": f.get("heating"),
            "Transport": f.get("transport"),
            "Vehicle": f.get("vehicle"),
            "Social": f.get("social"),
            "Flight": f.get("flight"),
            "Bag Size": f.get("bag_size"),
            "Energy Eff": f.get("energy_eff"),
            "Grocery": safe_float(f.get("grocery")),
            "Vehicle Distance": safe_float(f.get("vehicle_distance")),
            "Waste Weekly": safe_float(f.get("waste")),
            "TV Daily Hour": safe_float(f.get("tv")),
            "Internet Daily": safe_float(f.get("internet")),
            "Clothes Monthly": safe_float(f.get("clothes"))
        }

        for c in ["Plastic","Glass","Metal","Paper","Microwave","Oven","Stove","Airfryer","Grill"]:
            user_input[c] = 1 if f.get(c) else 0

        df = pd.DataFrame([user_input])
        df = df[feature_names]
        df = input_preprocessing(df)

        for col in ['Grocery','Vehicle Distance','Waste Weekly','TV Daily Hour','Internet Daily','Clothes Monthly']:
            df[col] = scalers[col].transform(df[[col]])

        prediction = round(float(carbon_model.predict(df)[0]), 2)

        if prediction < 2000:
            message = "🌱 Low Carbon Footprint"
        elif prediction < 4000:
            message = "⚠️ Moderate Carbon Footprint"
        else:
            message = "🔥 High Carbon Footprint"

    return render_template("index.html", page="carbon", prediction=prediction, message=message)


# ---------------- AQI ----------------
@app.route('/aqi', methods=['GET', 'POST'])
def aqi():
    prediction = None

    if request.method == 'POST':
        data = {
            'CO': float(request.form['CO']),
            'HC': float(request.form['HC']),
            'Temp': float(request.form['Temp']),
            'Hum': float(request.form['Hum']),
            'CO2': float(request.form['CO2']),
            'TVOC': float(request.form['TVOC']),
            'DD': float(request.form['DD'])
        }

        df = pd.DataFrame([data])
        if use_scaler:
            df = aqi_scaler.transform(df)

        pred = aqi_model.predict(df)
        prediction = aqi_encoder.inverse_transform(pred)[0]

    return render_template("index.html", page="aqi", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
