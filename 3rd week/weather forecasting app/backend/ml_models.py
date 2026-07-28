from sqlalchemy import Column, Integer, Float, String
from database import Base

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
    Visibility = Column(Float)
    Season = Column(String)
    Location = Column(String)

    Prediction = Column(String)