# Laptop-Price-Prediction-Using-ML-and-Deep-Learning


# Laptop Price Prediction Using ML and Deep Learning

This repository contains a comprehensive notebook that applies both classical machine learning models and deep learning techniques to predict laptop prices based on their specifications. The target variable is `buynow_price`, and the goal is to minimize Root Mean Squared Error (RMSE).

## Project Structure

- **Data Exploration and Preprocessing**: Feature cleaning, encoding, and normalization
- **Multicollinearity Check**: VIF-based feature pruning to address correlated inputs
- **Train/Validation/Test Split**: Ensures unbiased model evaluation
- **Classical Machine Learning**: Includes:
  - Linear Regression
  - Decision Trees
  - Random Forest
  - Extra Trees
  - Gradient Boosting
  - Voting Regressor (ensemble)
- **Deep Learning**: A small dense neural network trained with custom RMSE loss and learning rate scheduling
- **Evaluation**: RMSE used as the core evaluation metric across all models

## Technologies Used

- Python 3
- Scikit-learn
- TensorFlow / Keras
- Pandas, NumPy
- Seaborn, Matplotlib

## Results

- Voting Regressor and a tuned Deep Neural Network (DNN) showed the best RMSE performance
- Models were trained on scaled features using `MaxAbsScaler` for optimal performance on sparse-like inputs


## Note

This project was designed for educational purposes as part of a machine learning assignment. It emphasizes good practices in model evaluation, overfitting control, and error minimization.

##  Author

- Created by Shiva Sriram
  
