# Credit Card Fraud Detection

Custom Multi-Layer Perceptron implementation from scratch to detect fraudulent credit card transactions on a highly imbalanced dataset of 13M+ transactions.

## Dataset
- **Size:** 13M+ transactions (2010-2019)
- **Features:** 19 features including transaction amount, merchant details, card info, and transaction type
- **Target:** Binary classification (Fraud: Yes/No)
- **Challenge:** Extreme class imbalance (~0.01% fraud cases)

## Approach
1. **Data Preprocessing:** Merged transaction and card data, one-hot encoded categorical features, handled missing values
2. **Class Balancing:** Strategic undersampling + SMOTE to create balanced training set (2,000 samples per class)
3. **Model:** Custom MLP with 2 hidden layers (30 neurons each), ReLU activation, SGD optimizer
4. **Training:** 1,000 epochs, batch size 32, learning rate 0.001, binary cross-entropy loss

## Model Architecture
```
Input (19 features) → Hidden (30) → Hidden (30) → Output (1)
Activation: ReLU → ReLU → Sigmoid
```

## Results
| Metric    | Training | Validation |
|-----------|----------|------------|
| Accuracy  | 72.18%   | 65.75%     |
| Precision | 74.04%   | 69.81%     |
| Recall    | 68.30%   | 55.50%     |
| F1-Score  | 71.05%   | 61.84%     |

## Key Features
- Built neural network from scratch (no TensorFlow/PyTorch)
- Memory-efficient processing for 13M+ rows
- Custom SGD implementation with mini-batch training
- Addressed extreme class imbalance with SMOTE

## Technologies
Python, NumPy, pandas, scikit-learn, imbalanced-learn

## Usage
```python
# Train model
mlp = MLP_SGD(hidden_layer_sizes=(30, 30), learning_rate=0.001, n_epochs=1000)
mlp.fit(X_train, y_train)

# Predict
predictions = mlp.predict(X_test)
```