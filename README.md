# 🚲 Bicycle Price Prediction with Neural Network

This project is a beginner-friendly machine learning regression example that predicts bicycle prices based on two numerical features.

It was built using **Python, TensorFlow, Scikit-learn, Pandas, and Seaborn** and demonstrates the full ML workflow:
data loading → preprocessing → train/test split → scaling → model training → evaluation → visualization → prediction.

---

## 📊 Project Goal

The goal of this project is to learn and demonstrate how a neural network can model the relationship between product features and price.

We train a regression model to predict the **price of a bicycle** using:

- `BisikletOzellik1`
- `BisikletOzellik2`

---

## 🧠 Model Architecture

The neural network is built with TensorFlow/Keras:

- Input layer (2 features)
- 3 hidden layers (ReLU activation)
- Output layer (1 neuron – price prediction)

Loss function: **Mean Squared Error (MSE)**  
Optimizer: **Adam**

---

## ⚙️ Data Preprocessing

- Train/test split: **67% train / 33% test**
- Feature scaling: **MinMaxScaler**
- Model trained for **500 epochs**

---

## 📈 Evaluation Metrics

The model is evaluated using:

- Train Loss
- Test Loss
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Scatter plot: *Actual vs Predicted Prices*

These metrics help us understand how well the model generalizes to unseen data.

---

## 📉 Training Visualization

A loss curve is plotted to show how the model improves during training.

A scatter plot compares:

- Real prices
- Predicted prices

A good model should show points close to the diagonal line.

---

## 🔮 Interactive Prediction

After training, the user can enter new feature values to predict a bicycle price.

⚠️ Important:  
The model was trained on feature values roughly between **1745 and 1754**.  
For meaningful predictions, you should enter values within this range.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 📚 Project Type

This is an **educational machine learning project** created to practice:

- Regression with neural networks
- Data preprocessing
- Model evaluation
- Visualization
- GitHub project structure

---

## 🚀 Future Improvements

- Add more input features
- Try Linear Regression and compare results
- Hyperparameter tuning
- Save and load trained model
- Add real-world dataset
