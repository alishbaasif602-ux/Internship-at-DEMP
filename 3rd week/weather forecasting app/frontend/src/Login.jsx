import { useState } from "react";
import axios from "axios";
import "./Login.css";

function Login({ onLogin }) {
  const [isRegister, setIsRegister] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async () => {
    try {
      if (isRegister) {
        const res = await axios.post(
          "http://127.0.0.1:8000/register",
          formData
        );

        alert(res.data.message);
      } else {
        const res = await axios.post(
          "http://127.0.0.1:8000/login",
          {
            username: formData.username,
            password: formData.password,
          }
        );

        if (res.data.message === "Login Successful") {
          onLogin(res.data.username);
        } else {
          alert(res.data.message);
        }
      }
    } catch (err) {
      alert("Something went wrong!");
      console.log(err);
    }
  };

  return (
    <div className="login-container">

      <h1>🌦 AI Weather Dashboard</h1>

      <h2>
        {isRegister ? "Create Account" : "Login"}
      </h2>

      <input
        type="text"
        placeholder="Username"
        name="username"
        onChange={handleChange}
      />

      {isRegister && (
        <input
          type="email"
          placeholder="Email"
          name="email"
          onChange={handleChange}
        />
      )}
      <div className="password-box">
        <input
          type={showPassword ? "text" : "password"}
          placeholder="Password"
          name="password"
          onChange={handleChange}
        />

        <span
          className="eye-icon"
          onClick={() => setShowPassword(!showPassword)}
        >
          {showPassword ? "🙈" : "👁"}
        </span>
      </div>
      <button onClick={handleSubmit}>
        {isRegister ? "Register" : "Login"}
      </button>

      <p
        onClick={() => setIsRegister(!isRegister)}
        style={{ cursor: "pointer", marginTop: "15px" }}
      >
        {isRegister
          ? "Already have an account? Login"
          : "Don't have an account? Register"}
      </p>

    </div>
  );
}

export default Login;