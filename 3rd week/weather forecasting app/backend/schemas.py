from pydantic import BaseModel

class WeatherInput(BaseModel):
    Temperature: int
    Humidity: int
    Wind_Speed: float
    Precipitation: int
    Cloud_Cover: str
    Atmospheric_Pressure: float
    UV_Index: int
    Season: str
    Visibility: float
    Location: str
    Model: str


class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    
    
class UserLogin(BaseModel):
    username: str
    password: str