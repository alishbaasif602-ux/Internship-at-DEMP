<div align="center">

# 🌦️ AI Weather Forecasting Dashboard

### An intelligent, full-stack weather prediction platform powered by Machine Learning

[![React](https://img.shields.io/badge/Frontend-React.js-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-Educational-lightgrey)](#-license)

A modern dashboard that predicts weather conditions from environmental data using multiple Machine Learning models — wrapped in a sleek, glassmorphism-styled, dark-mode-ready interface.

</div>

---

## ✨ Overview

**AI Weather Forecasting Dashboard** is a full-stack web application that combines the power of **React.js**, **FastAPI**, and **Scikit-learn** to deliver real-time, ML-driven weather predictions. Users can register, log in, choose from multiple trained models, and instantly get predictions — all while their prediction history is securely stored and easily reviewable.

Designed with both functionality and aesthetics in mind, the app features a clean, responsive UI with a modern glassmorphism design and full dark mode support.

---

## 🚀 Key Features

| Category | Features |
|---|---|
| 🌤 **Prediction** | Real-time weather prediction from environmental inputs |
| 🤖 **Machine Learning** | Choose between Logistic Regression, SVM, or KNN |
| 🔐 **Authentication** | Secure user registration, login & logout |
| 📊 **History Tracking** | Every prediction automatically logged and viewable |
| 🌙 **Dark Mode** | Seamless light/dark theme toggle |
| ✅ **Validation** | Robust input validation to prevent bad data |
| ⏳ **UX Polish** | Loading spinners, weather icons, responsive layout |
| 💖 **Design** | Modern glassmorphism-inspired UI |

---

## 🛠 Tech Stack

<table>
<tr>
<td valign="top" width="33%">

**Frontend**
- React.js
- Axios
- CSS3

</td>
<td valign="top" width="33%">

**Backend**
- FastAPI
- SQLAlchemy
- Pydantic
- SQLite

</td>
<td valign="top" width="33%">

**Machine Learning**
- Scikit-learn
- Pandas
- NumPy
- Joblib

</td>
</tr>
</table>

---

## 📂 Project Structure

```
Weather-Forecasting-Dashboard/
│
├── frontend/
│   ├── App.jsx
│   ├── Login.jsx
│   ├── App.css
│   └── Login.css
│
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── db_models.py
│   ├── schemas.py
│   └── models/
│       ├── logistic_model.pkl
│       ├── svm_model.pkl
│       └── knn_model.pkl
│
├── weather.db
└── README.md
```

---

## 📊 Input Parameters

The model generates predictions based on the following environmental features:

| Parameter | Parameter | Parameter |
|---|---|---|
| 🌡️ Temperature | 💧 Humidity | 💨 Wind Speed |
| 🌧️ Precipitation | ☁️ Cloud Cover | 🧭 Atmospheric Pressure |
| ☀️ UV Index | 🍂 Season | 👁️ Visibility |
| 📍 Location | | |

---

## 🤖 Machine Learning Models

Users can select from **three trained models** before generating a prediction:

1. **Logistic Regression** — Fast, interpretable baseline classifier
2. **Support Vector Machine (SVM)** — Robust performance on complex boundaries
3. **K-Nearest Neighbors (KNN)** — Simple, effective similarity-based prediction

Each model is pre-trained and served via the FastAPI backend using `joblib`.

---

## 🔐 Authentication

The application includes a complete auth flow:

- **Register** — Create a new account
- **Login** — Secure sign-in for returning users
- **Logout** — End the session safely

All credentials are stored securely in the SQLite database.

---

## 📈 Prediction History

Every prediction made by a user is automatically saved and displayed in a history table, including:

- 🌡️ Temperature
- 💧 Humidity
- 🤖 Selected Model
- ✅ Predicted Weather Condition

This allows users to track and review their past forecasts at any time.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/weather-dashboard.git
cd weather-dashboard
```

### 2️⃣ Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
Backend will start at: `http://127.0.0.1:8000`

### 3️⃣ Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend will start at: `http://localhost:5173`

---

## 📷 Dashboard Highlights

- 🌤 Instant Weather Prediction
- 🔐 Secure User Authentication
- 📊 Full Prediction History
- 🌙 Dark Mode Toggle
- ⏳ Smooth Loading Indicators
- ☁️ Dynamic Weather Icons
- ✅ Real-time Input Validation
- 📱 Fully Responsive Design

---

## 🎯 Roadmap / Future Improvements

- [ ] 🌐 Live Weather API Integration
- [ ] 📊 Interactive Weather Charts & Graphs
- [ ] 🔍 Advanced Search & Filter for History
- [ ] 📤 Export Prediction Reports (PDF/CSV)
- [ ] 📧 Email Notifications
- [ ] 🧠 AI-based Weather Recommendations

---

## 👩‍💻 Developer

**Alishba Asif**
*AI Engineer*

---

## 📜 License

This project is developed for **educational and internship purposes**.

---

<div align="center">

⭐ If you found this project useful, consider giving it a star on GitHub!

</div>
