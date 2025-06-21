🪙 Gold Price Prediction 📈

This project predicts the price of gold (per gram in INR) using the USD to INR exchange rate and historical gold prices. It combines machine learning with a simple user interface to make real-time predictions easy and interactive.

🚀 Demo
👉 Try the Gradio app locally by running main.py or the final code block to launch the UI.

🧠 Project Features

📥 Data Collection using yfinance for USD/INR rates and web scraping for gold rates

🧼 Preprocessing: Cleansing, renaming, and transforming numeric data

📊 Exploratory Data Analysis (EDA): Boxplots, regression plots

🤖 Modeling: Linear Regression with hyperparameter tuning

📦 MLOps: Model serialization using pickle

🖥️ Gradio UI for interactive prediction

🗂️ Project Structure

    Gold-Price-Prediction/
    ├── model.pkl              # Saved ML model
    ├── scaler.pkl             # Scaler used for prediction
    ├── Gold vs USDINR.csv     # Main dataset
    ├── gold_price_predict.py  # Main script for training + UI
    └── README.md              # Project documentation

📉 Dataset

Gold vs USDINR.csv: Contains historical data for gold prices and USD/INR exchange rates.

yfinance is used to fetch updated USDINR data weekly from Jan 2024 to Dec 2025.

🔧 Installation
Clone the repository
git clone https://github.com/Kprakash9426/Gold-Price-Prediction.git
cd Gold-Price-Prediction

Install dependencies
pip install -r requirements.txt

If requirements.txt is missing, install manually:
pip install yfinance gradio pandas numpy matplotlib seaborn scikit-learn

🧪 How It Works
1. Data Preprocessing

Loads and cleans Goldrate and USD_INR

Converts gold rate from string (₹) to numeric

Handles outliers and scaling

2. Model Training

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

Trained using StandardScaler

Evaluated with mean_squared_error

3. Hyperparameter Tuning

RandomizedSearchCV(..., param_distributions=param_space, ...)

4. Model Saving

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

💻 Gradio Interface

import gradio as gr

def calculate_gold_rate(usd_inr):
    scaled_input = scaler.transform(np.array(usd_inr).reshape(1, -1))
    return model.predict(scaled_input)[0][0].round(2)

demo = gr.Interface(
    fn=calculate_gold_rate,
    inputs=gr.Number(label="USD to INR"),
    outputs=gr.Number(label="Predicted Gold Rate (₹/g)"),
    title="How much 1g now"
)
demo.launch()
📈 Sample Output
Input: USD/INR = 86.58

Predicted Gold Price: ₹[8402.91]

📌 Future Improvements
🧠 Use XGBoost or LSTM for better accuracy

🌐 Add real-time gold price fetching from an API

📱 Deploy to web (e.g., with Hugging Face Spaces)

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.
