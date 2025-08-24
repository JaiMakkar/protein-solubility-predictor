# 🧪 Protein Solubility Predictor

A Machine Learning powered web app (built with **Streamlit**) that predicts the solubility of proteins based on molecular features.  
This project was developed to demonstrate how AI/ML can assist in **bioinformatics** and **drug discovery**.

---

## 🚀 Features
- Upload your own **protein dataset** (CSV format).
- Train models using **Scikit-learn** & **XGBoost**.
- Handles class imbalance with **SMOTE**.
- Interactive **UI with Streamlit**.
- Visualizations powered by **Plotly**.
- Real-time prediction of protein solubility.

---

## 🖥️ Tech Stack
- **Frontend/UI** → Streamlit  
- **Backend/ML** → Python, Scikit-learn, XGBoost, Imbalanced-learn  
- **Data Visualization** → Plotly, Pandas  
- **Deployment** → Streamlit Cloud  

---

## 📂 Project Structure
```
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
└── .venv               # Virtual environment (not pushed to GitHub)
```

---

## 📊 Example Input
Users can input molecular features such as:
- **Molecular Weight**
- **Isoelectric Point (pH)**
- **Hydrophobicity Index**
- **Aromaticity**
- **Charge Density**

And get a **solubility prediction** instantly.

---

## ⚡ Quick Start

### 1️⃣ Clone the repo
```bash
git clone https://github.com/JaiMakkar/protein-solubility-predictor.git
cd protein-solubility-predictor
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app
```bash
streamlit run app.py
```

---

## 🌐 Deployment
The app is deployed on **Streamlit Community Cloud**:  
👉 https://protein-solubility-predictor.streamlit.app

---

## 📌 Future Improvements
- Add more advanced models (e.g., Deep Learning).  
- Expand dataset for real protein sequences.  
- Improve UI/UX with custom styling.  

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License
This project is licensed under the **MIT License** – feel free to use and modify it.

---

⭐ If you like this project, consider giving it a star on GitHub!
