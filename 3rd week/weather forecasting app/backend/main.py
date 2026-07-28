from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import WeatherInput, UserRegister, UserLogin
import joblib
import pandas as pd

from database import SessionLocal, engine, Base
from db_models import WeatherRecord, User

# Create Database Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Weather Classification API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Home API
@app.get("/")
def home():
    return {
        "message": "Weather Classification API is Running Successfully!"
    }

@app.post("/register")
def register(user: UserRegister):

    db = SessionLocal()

    existing_user = db.query(User).filter(
        User.username == user.username
    ).first()

    if existing_user:
        db.close()
        return {"message": "Username already exists"}

    new_user = User(
        username=user.username,
        email=user.email,
        password=user.password
    )

    db.add(new_user)
    db.commit()
    db.close()

    return {"message": "Registration Successful"}
@app.post("/login")
def login(user: UserLogin):

    db = SessionLocal()

    existing_user = db.query(User).filter(
        User.username == user.username,
        User.password == user.password
    ).first()

    db.close()

    if existing_user:
        return {
            "message": "Login Successful",
            "username": existing_user.username
        }

    return {
        "message": "Invalid Username or Password"
    }
# Prediction API
@app.post("/predict")
def predict(data: WeatherInput):

    # Select Model
    if data.Model == "logistic":
        model = joblib.load("models/logistic_model.pkl")

    elif data.Model == "svm":
        model = joblib.load("models/svm_model.pkl")

    elif data.Model == "knn":
        model = joblib.load("models/knn_model.pkl")

    else:
        return {"error": "Invalid Model Selected"}

    # Create DataFrame
    input_data = pd.DataFrame([{
        "Temperature": data.Temperature,
        "Humidity": data.Humidity,
        "Wind Speed": data.Wind_Speed,
        "Precipitation (%)": data.Precipitation,
        "Cloud Cover": data.Cloud_Cover,
        "Atmospheric Pressure": data.Atmospheric_Pressure,
        "UV Index": data.UV_Index,
        "Season": data.Season,
        "Visibility (km)": data.Visibility,
        "Location": data.Location
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Save in Database
    db = SessionLocal()

    weather = WeatherRecord(
        Temperature=data.Temperature,
        Humidity=data.Humidity,
        Wind_Speed=data.Wind_Speed,
        Precipitation=data.Precipitation,
        Cloud_Cover=data.Cloud_Cover,
        Atmospheric_Pressure=data.Atmospheric_Pressure,
        UV_Index=data.UV_Index,
        Season=data.Season,
        Visibility=data.Visibility,
        Location=data.Location,
        Model=data.Model,
        Prediction=prediction
    )

    db.add(weather)
    db.commit()
    db.close()

    return {
        "Selected Model": data.Model,
        "Prediction": prediction
    }


# History API
@app.get("/history")
def history():

    db = SessionLocal()

    data = db.query(WeatherRecord).all()

    history_list = []

    for row in data:
        history_list.append({
            "id": row.id,
            "Temperature": row.Temperature,
            "Humidity": row.Humidity,
            "Wind Speed": row.Wind_Speed,
            "Precipitation": row.Precipitation,
            "Cloud Cover": row.Cloud_Cover,
            "Atmospheric Pressure": row.Atmospheric_Pressure,
            "UV Index": row.UV_Index,
            "Season": row.Season,
            "Visibility": row.Visibility,
            "Location": row.Location,
            "Model": row.Model,
            "Prediction": row.Prediction
        })

    db.close()

    return history_list