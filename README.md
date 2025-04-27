🩺 Disease Prediction App

A beautiful, health-themed Streamlit web app that predicts the presence of Parkinson's, Kidney, and Liver diseases based on patient input parameters. Powered by machine learning models trained on real medical datasets.



🚀 Features

-> 🧠 Parkinson's Disease Prediction
-> 🩸 Kidney Disease Prediction
-> 🏥 Liver Disease Prediction
-> 🎨 Elegant Light Theme with health-oriented colors
-> 🖋️ User-friendly interface for seamless data input
-> 📈 Accurate ML models for fast and reliable predictions


🛠️ Tech Stack

- Python 3.9+
- Streamlit
- Scikit-learn
- NumPy
- Joblib


📂 Project Structure:

Multiple_Disease_Predictor/
|
├── data/
│   ├── parkinsons.csv
│   ├── kidney.csv
│   └── liver.csv
|
├── models/
│   ├── parkinsons_model.pkl
│   ├── kidney_model.pkl
│   └── liver_model.pkl
|
├── scripts/
│   ├── train_model.py
│   └── preprocessing.py
|
├── app.py
├── requirements.txt
└── README.md


🧠 Model Details
-The models were trained using key clinical features selected for each disease.
-Preprocessing steps like scaling, encoding, and cleaning were applied before training.
-Models saved using Joblib for fast loading and prediction.

📋 Notes
-Important: Make sure the models/ folder with the .pkl files is available alongside app.py.
-This project is built for educational and demonstration purposes only.
-It should not be used as a substitute for professional medical diagnosis.