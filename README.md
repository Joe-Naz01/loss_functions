# Loss Functions — Understanding Model Optimization

**Problem.** Different learning objectives use distinct loss functions that shape how models learn and generalize. This notebook demonstrates and visualizes key loss functions used in classification and regression.

**Approach.**
- Implemented **Logistic Loss**, **Hinge Loss**, and **Squared Error** manually and with `scikit-learn`.
- Visualized how each loss penalizes misclassification or residual errors.
- Compared gradient behaviors and optimization landscapes.
- Discussed the effect of smoothness and margins on convergence.
- Linked loss selection to model families (e.g., Logistic → LogisticRegression, Hinge → SVM).

**Results (qualitative).**
- Logistic loss provides smoother gradients for probabilistic outputs.
- Hinge loss enforces a margin, making it suitable for SVMs.
- Squared error can struggle with classification due to non-convexity in discrete outputs.

**What I Learned.**
- How loss curvature affects gradient descent stability.
- Why logistic and hinge losses differ in robustness to outliers.
- How visualization clarifies optimization trade-offs.

## Quick Start
```bash
git clone https://github.com/Joe-Naz01/loss_functions.git
cd loss_functions

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
