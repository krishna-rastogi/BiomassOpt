# 🌱 BiomassOpt – Biomass Supply Chain Optimization Platform

**BiomassOpt** is a web-based platform that integrates **three Machine Learning models** to analyze and optimize biomass supply chain parameters such as cost, logistics, and energy yield.  
It provides an intuitive frontend for users and a powerful Flask-based backend for real-time ML inference.

---

## 🚀 Features

- 🔹 **Three integrated ML models** for biomass optimization.
- 🔹 **Interactive web interface** built with HTML, CSS, and JavaScript.
- 🔹 **Flask backend** serving model predictions via REST APIs.
- 🔹 **Modular code structure** (frontend, backend, data, notebooks).
- 🔹 **Deployable on Render or any cloud platform** easily.
- 🔹 Supports **CSV data uploads** and result visualization.

---

## 🧠 Tech Stack

| Component | Technology Used |
|------------|----------------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python (Flask) |
| ML Libraries | Scikit-learn / TensorFlow / NumPy / Pandas |
| Model Serialization | Joblib / Pickle |
| Deployment | Render / Gunicorn |
| Visualization | Matplotlib / Plotly (if used) |

---

## 📁 Project Structure
```
BiomassOpt/
│
├── backend/ # Flask backend
│ ├── app.py # Main Flask entrypoint
│ ├── models/ # Stored ML models (.pkl / .h5)
│ ├── static/ # Static files (CSS, JS)
│ └── templates/ # HTML templates
│
├── frontend/ # Frontend source (if separate)
│
├── notebooks/ # Jupyter notebooks for model training
│
├── data/ # Sample datasets
│
└── README.md # Project documentation
```


---

## ⚙️ Installation & Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/krishna-rastogi/BiomassOpt.git
   cd BiomassOpt/backend
   ```
   
2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate       # (Mac/Linux)
   venv\Scripts\activate          # (Windows)
   ```
   
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**
   ```bash
   python app.py
   ```
5. Open your browser at http://127.0.0.1:5000/  🚀
   
---

🧑‍💻 Author
**Krishna Rastogi**
🌐 GitHub Profile: https://github.com/krishna-rastogi/

If you like this project, ⭐ star the repo to support future improvements!
