🚲 Bicycle Price Prediction with TensorFlow

This project is an educational machine learning application that predicts bicycle prices using a feedforward neural network built with TensorFlow/Keras.
The model is trained on two numerical features and learns the relationship between these features and the target variable (price).

The workflow includes data loading, exploratory visualization, train–test splitting, feature scaling with MinMaxScaler, neural network training, and model evaluation using MAE, MSE, and RMSE metrics.
Training loss is visualized to observe the learning process, and a scatter plot of actual vs predicted values is used to assess model performance.

The neural network consists of three hidden layers with ReLU activation and an output layer for regression.
The model is optimized with Adam and trained for 500 epochs.

After training, users can enter new feature values to obtain a real-time price prediction.
For reliable results, input values should be within the range 1740–1760, since the model was trained on data in this interval and does not extrapolate well beyond it.

This project was created to practice:

Neural network regression

TensorFlow/Keras workflow

Data preprocessing and scaling

Model evaluation and visualization

Building a complete end-to-end ML pipeline

Future improvements may include model saving/loading, hyperparameter tuning, additional features, and comparison with classical regression models.
