# **Logistic Regression from Scratch**

This project implements **Binary** and **Polynomial Logistic Regression** completely from scratch using only **NumPy**, without relying on any machine learning frameworks. The primary goal is to gain a **deep understanding of how logistic regression works internally** — from mathematical formulation to gradient descent optimization.

---

## **📘 Project Overview**

This notebook builds a logistic regression model step-by-step:

1. **Generates synthetic data** using random weights, bias, and Gaussian noise.
2. **Applies the sigmoid function** to map outputs into the probability range (0,1).
3. **Implements gradient descent manually** to optimize parameters.
4. **Extends to polynomial logistic regression** to handle non-linear decision boundaries.
5. **Visualizes data and model predictions** using Matplotlib and Seaborn.

---

## **🧠 Key Concepts Learned**

| Concept                             | Description                                                            |
| ----------------------------------- | ---------------------------------------------------------------------- |
| **Sigmoid Function**                | Converts linear predictions into probabilities between 0 and 1.        |
| **Log-Loss (Binary Cross Entropy)** | The loss function used to measure prediction accuracy.                 |
| **Gradient Descent**                | Optimization algorithm used to minimize log-loss by adjusting weights. |
| **Polynomial Features**             | Extension of logistic regression to capture non-linear patterns.       |
| **Decision Boundary**               | The separation line/curve learned by the model to distinguish classes. |

---

## **🧩 Technical Highlights**

* **Language:** Python
* **Libraries Used:** `NumPy`, `Matplotlib`, `Seaborn`, `Pandas` (for data organization)
* **No external ML libraries (like Scikit-learn or TensorFlow) used.**
* **Fully manual implementation** of:

  * Weight initialization
  * Sigmoid computation
  * Gradient and bias updates
  * Polynomial feature expansion

---

## **⚙️ Implementation Details**

**Core Formula:**
[
y_{pred} = \sigma(Wx + b)
]
where
(\sigma(z) = \frac{1}{1 + e^{-z}})

**Gradient Updates:**
[
W = W - \alpha \cdot \frac{\partial L}{\partial W}
]
[
b = b - \alpha \cdot \frac{\partial L}{\partial b}
]

---

## **📊 Results**

* Successfully demonstrates **binary classification** on synthetic data.
* Extends to **polynomial logistic regression**, illustrating limitations and improvements visually.
* Helps in understanding **why more advanced architectures like LSTM are needed** for capturing complex relationships.

---

## **🎯 Learning Outcome**

This project served as a foundation for understanding:

* How machine learning models *learn from data* without libraries.
* The mathematics behind **gradient descent and logistic regression**.
* How non-linear patterns require **feature transformation or deeper architectures**.

> This project directly motivated the exploration of **Recurrent Neural Networks (RNN)** and **LSTMs** to understand how models handle sequential dependencies and memory.

---

## **📁 File Structure**

```
├── logistic_regression.ipynb     # Implementation of binary & polynomial logistic regression
└── README.md                     # Project documentation
```

---

## **🚀 Future Improvements**

* Add **multi-class classification** support.
* Introduce **regularization** (L1/L2) for better generalization.
* Implement **Mini-Batch Gradient Descent** for efficiency.
* Compare performance with **Scikit-learn** logistic regression for benchmarking.

---

## **👨‍💻 Author**

**Prakhar Pathak**
*Machine Learning Enthusiast | Building ML Models from Scratch*
📫 [Personal Website](https://phantomsynth.com/) | [LinkedIn](https://www.linkedin.com/in/prince-pandey-4a58031ba)

---

**⭐ If you found this project insightful, consider starring the repo!**
