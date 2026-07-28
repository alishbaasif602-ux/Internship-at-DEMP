import "./App.css";
import { useState, useEffect } from "react";
import axios from "axios";
import Login from "./Login";

function App() {
  const [formData, setFormData] = useState({
    Temperature: "",
    Humidity: "",
    Wind_Speed: "",
    Precipitation: "",
    Cloud_Cover: "partly cloudy",
    Atmospheric_Pressure: "",
    UV_Index: "",
    Season: "Summer",
    Visibility: "",
    Location: "inland",
    Model: "logistic",
  });

  const [prediction, setPrediction] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [error, setError] = useState("");
  const [user, setUser] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };
  const getWeatherIcon = (weather) => {
    switch (weather) {
      case "Sunny":
        return "☀️";
      case "Rainy":
        return "🌧️";
      case "Cloudy":
        return "☁️";
      case "Snowy":
        return "❄️";
      case "Thunderstorm":
        return "⛈️";
      default:
        return "🌤️";
    }
  };
  const validateInputs = () => {
    if (
      formData.Temperature === "" ||
      formData.Humidity === "" ||
      formData.Wind_Speed === "" ||
      formData.Precipitation === "" ||
      formData.Atmospheric_Pressure === "" ||
      formData.UV_Index === "" ||
      formData.Visibility === ""
    ) {
      setError("Please fill all fields.");
      return false;
    }

    if (
      Number(formData.Humidity) < 0 ||
      Number(formData.Humidity) > 100
    ) {
      setError("Humidity must be between 0 and 100.");
      return false;
    }

    if (
      Number(formData.Precipitation) < 0 ||
      Number(formData.Precipitation) > 100
    ) {
      setError("Precipitation must be between 0 and 100.");
      return false;
    }

    setError("");
    return true;
  };

  const fetchHistory = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:8000/history");
      setHistory(response.data);
    } catch (error) {
      console.log(error);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const predictWeather = async () => {

    if (!validateInputs()) return;

    setLoading(true);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        {
          Temperature: Number(formData.Temperature),
          Humidity: Number(formData.Humidity),
          Wind_Speed: Number(formData.Wind_Speed),
          Precipitation: Number(formData.Precipitation),
          Cloud_Cover: formData.Cloud_Cover,
          Atmospheric_Pressure: Number(formData.Atmospheric_Pressure),
          UV_Index: Number(formData.UV_Index),
          Season: formData.Season,
          Visibility: Number(formData.Visibility),
          Location: formData.Location,
          Model: formData.Model,
        }
      );

      setPrediction(response.data.Prediction);
      setSelectedModel(response.data["Selected Model"]);

      fetchHistory();

    } catch (error) {
      alert("Prediction Failed");
      console.log(error);

    } finally {
      setLoading(false);
    }

  };
  if (!user) {
    return <Login onLogin={setUser} />;
  }
  return (
    <div className={darkMode ? "container dark" : "container"}>

      <h1>🌦 AI Weather Forecasting Dashboard</h1>

      <button
        className="dark-btn"
        onClick={() => setDarkMode(!darkMode)}
      >
        {darkMode ? "☀ Light Mode" : "🌙 Dark Mode"}
      </button>

      <label>🌡 Temperature</label>
      <input
        type="number"
        name="Temperature"
        value={formData.Temperature}
        onChange={handleChange}
      />
      <label>💧 Humidity</label>
      <input
        type="number"
        name="Humidity"
        value={formData.Humidity}
        onChange={handleChange}
      />

      <label>💨 Wind Speed</label>
      <input
        type="number"
        name="Wind_Speed"
        value={formData.Wind_Speed}
        onChange={handleChange}
      />

      <label>🌧 Precipitation (%)</label>
      <input
        type="number"
        name="Precipitation"
        value={formData.Precipitation}
        onChange={handleChange}
      />

      <label>🌡 Atmospheric Pressure</label>
      <input
        type="number"
        name="Atmospheric_Pressure"
        value={formData.Atmospheric_Pressure}
        onChange={handleChange}
      />

      <label>☀ UV Index</label>
      <input
        type="number"
        name="UV_Index"
        value={formData.UV_Index}
        onChange={handleChange}
      />

      <label>👀 Visibility (km)</label>
      <input
        type="number"
        name="Visibility"
        value={formData.Visibility}
        onChange={handleChange}
      />

      <label>☁ Cloud Cover</label>
      <select
        name="Cloud_Cover"
        value={formData.Cloud_Cover}
        onChange={handleChange}
      >
        <option value="clear">☀ Clear</option>
        <option value="partly cloudy">⛅ Partly Cloudy</option>
        <option value="cloudy">☁ Cloudy</option>
        <option value="overcast">🌫 Overcast</option>
      </select>

      <label>🍂 Season</label>
      <select
        name="Season"
        value={formData.Season}
        onChange={handleChange}
      >
        <option value="Spring">🌸 Spring</option>
        <option value="Summer">☀ Summer</option>
        <option value="Autumn">🍂 Autumn</option>
        <option value="Winter">❄ Winter</option>
      </select>

      <label>📍 Location</label>
      <select
        name="Location"
        value={formData.Location}
        onChange={handleChange}
      >
        <option value="inland">🏙 Inland</option>
        <option value="coastal">🌊 Coastal</option>
        <option value="mountain">⛰ Mountain</option>
      </select>

      <label>🤖 Select Model</label>
      <select
        name="Model"
        value={formData.Model}
        onChange={handleChange}
      >
        <option value="logistic">🧠 Logistic Regression</option>
        <option value="svm">⚡ Support Vector Machine</option>
        <option value="knn">📍 K-Nearest Neighbors</option>
      </select>

      {error && (
        <p className="error">{error}</p>
      )}

      <button
        onClick={predictWeather}
        disabled={loading}
      >
        {loading ? "⏳ Predicting..." : "🌤 Predict Weather"}
      </button>

      {prediction && (
        <div className="result">
          <h2>Prediction Result</h2>

          <p>
            <strong>{getWeatherIcon(prediction)} Weather:</strong> {prediction}
          </p>

          <p>
            <strong>🤖 Model:</strong> {selectedModel.toUpperCase()}
          </p>
        </div>
      )}

      <h2 className="history-title">📋 Prediction History</h2>

      <table>
        <thead>
          <tr>
            <th>Temp</th>
            <th>Humidity</th>
            <th>Model</th>
            <th>Prediction</th>
          </tr>
        </thead>

        <tbody>
          {history.map((item) => (
            <tr key={item.id}>
              <td>{item.Temperature}</td>
              <td>{item.Humidity}</td>
              <td>{item.Model}</td>
              <td>{item.Prediction}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <button
        className="logout-btn"
        onClick={() => setUser(null)}
      >
        🚪 Logout
      </button>
      <footer className="footer">
        💖 Developed by <strong>Alishba Asif</strong> | AI Engineer 🤖✨
      </footer>
    </div>
  );
}

export default App;