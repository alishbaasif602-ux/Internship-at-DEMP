from sqlalchemy import Column, Integer, Float, String
from database import Base


# ---------------- Weather Table ----------------
class WeatherRecord(Base):
    __tablename__ = "weather_records"

    id = Column(Integer, primary_key=True, index=True)

    Temperature = Column(Integer)
    Humidity = Column(Integer)
    Wind_Speed = Column(Float)
    Precipitation = Column(Integer)
    Cloud_Cover = Column(String)
    Atmospheric_Pressure = Column(Float)
    UV_Index = Column(Integer)
    Season = Column(String)
    Visibility = Column(Float)
    Location = Column(String)

    Model = Column(String)
    Prediction = Column(String)


# ---------------- User Table ----------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)