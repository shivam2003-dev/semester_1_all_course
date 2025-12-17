---
layout: course
title: "Machine Learning Fundamentals"
short_description: "Core ML concepts, supervised and unsupervised learning, model evaluation and selection"
description: "Master the fundamental concepts and algorithms of Machine Learning"
credits: 4
level: "Intermediate"
instructor: "Faculty"
topics:
  - Supervised Learning
  - Unsupervised Learning
  - Model Evaluation
  - Feature Engineering
  - Hyperparameter Tuning
  - Ensemble Methods
github_repo: "https://github.com/shivam2003-dev/semester_1_all_course"
---

## 📝 Course Overview

This course covers the fundamental concepts of machine learning, from basic supervised learning algorithms to advanced techniques like ensemble methods. You'll learn how to build, train, and evaluate ML models.

---

## 📚 Course Content

### Module 1: Supervised Learning - Regression

#### Key Topics:
- **Linear Regression**: Foundation of supervised learning
- **Multiple Linear Regression**: Handling multiple features
- **Polynomial Regression**: Non-linear relationships
- **Regularization**: L1 (Lasso) and L2 (Ridge)
- **Evaluation Metrics**: MSE, RMSE, R² Score

<div class="alert alert-note">
  <h4>📌 Note: Linear Regression Foundation</h4>
  <p>Linear regression is often the first ML algorithm learned. Despite its simplicity, it's powerful for understanding core ML concepts like loss functions, optimization, and evaluation.</p>
</div>

**Key Equations:**
```
Linear Regression:
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

Loss Function (Mean Squared Error):
MSE = (1/n) Σ(yᵢ - ŷᵢ)²

R² Score:
R² = 1 - (SS_res / SS_tot) where 0 ≤ R² ≤ 1

L2 Regularization (Ridge):
Loss = MSE + λ × Σ wᵢ²

L1 Regularization (Lasso):
Loss = MSE + λ × Σ |wᵢ|
```

<div class="alert alert-warning">
  <h4>⚠️ Warning: Multicollinearity</h4>
  <p>When features are highly correlated, regression models become unstable. Use regularization or feature selection to handle multicollinearity.</p>
</div>

<div class="alert alert-tip">
  <h4>💡 Industry Tip: Feature Scaling</h4>
  <p>Always normalize or standardize features before applying regression, especially when using regularization. Features on different scales can bias the model towards features with larger values.</p>
</div>

---

### Module 2: Supervised Learning - Classification

#### Key Topics:
- **Logistic Regression**: Binary classification
- **Multi-class Classification**: One-vs-Rest, Softmax
- **Decision Trees**: Interpretable models
- **Support Vector Machines (SVM)**: Maximum margin classifiers
- **Naive Bayes**: Probabilistic classifier
- **K-Nearest Neighbors (KNN)**: Instance-based learning

<div class="alert alert-note">
  <h4>📌 Note: Classification vs Regression</h4>
  <p>Classification predicts discrete categories while regression predicts continuous values. Despite similar algorithms (like logistic regression), they solve different problem types.</p>
</div>

**Key Concepts:**
```
Logistic Function (Sigmoid):
σ(z) = 1 / (1 + e^(-z))

Cross-Entropy Loss:
Loss = -[y × log(ŷ) + (1-y) × log(1-ŷ)]

SVM Decision Boundary:
min ||w||² + C × Σ ξᵢ

KNN Prediction:
ŷ = majority class among k nearest neighbors
```

<div class="alert alert-danger">
  <h4>🔴 Common Mistake: Decision Boundary</h4>
  <p>Linear classifiers like logistic regression can only learn linear decision boundaries. Non-linear problems require non-linear models like trees, SVMs with kernel tricks, or neural networks.</p>
</div>

<div class="alert alert-success">
  <h4>✅ Exam Tip: Confusion Matrix</h4>
  <p>Memorize the confusion matrix! Know how to calculate precision, recall, F1-score, and accuracy. These are tested frequently and are crucial for model evaluation.</p>
</div>

**Confusion Matrix Metrics:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)  [How many predicted positives are correct?]
Recall = TP / (TP + FN)     [How many actual positives are found?]
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

### Module 3: Unsupervised Learning

#### Key Topics:
- **Clustering**: K-Means, Hierarchical Clustering
- **Dimensionality Reduction**: PCA, t-SNE
- **Anomaly Detection**: Isolation Forest, LOF
- **Association Rules**: Market basket analysis
- **Density Estimation**: Gaussian Mixture Models

<div class="alert alert-note">
  <h4>📌 Note: Unsupervised Learning Challenges</h4>
  <p>Without labels, it's hard to evaluate unsupervised models. Common approaches include silhouette score, Davies-Bouldin index, and domain expert validation.</p>
</div>

**Key Algorithms:**
```
K-Means Objective:
min Σ Σ ||xᵢ - μₖ||²

PCA (Principal Component Analysis):
- Find directions of maximum variance
- Reduces dimensionality while preserving information

Gaussian Mixture Model:
p(x) = Σ πₖ × N(x|μₖ, Σₖ)
```

<div class="alert alert-warning">
  <h4>⚠️ Warning: K-Means Limitations</h4>
  <p>K-Means struggles with non-convex clusters and requires knowing k in advance. Consider alternatives like DBSCAN for more complex cluster shapes.</p>
</div>

<div class="alert alert-tip">
  <h4>💡 Industry Tip: Feature Reduction</h4>
  <p>In production, PCA and other dimensionality reduction techniques help with: (1) reducing storage, (2) speeding up models, (3) removing noise, and (4) enabling visualization.</p>
</div>

---

### Module 4: Model Evaluation & Selection

#### Key Topics:
- **Train/Test Split**: Avoiding overfitting
- **Cross-Validation**: k-fold, stratified
- **Overfitting vs Underfitting**: Bias-variance tradeoff
- **ROC Curves and AUC**: Classification evaluation
- **Learning Curves**: Diagnosing model performance

<div class="alert alert-note">
  <h4>📌 Note: The Bias-Variance Tradeoff</h4>
  <p>Bias is underfitting (model too simple), variance is overfitting (model too complex). The goal is to find the sweet spot. Use regularization and cross-validation to manage this.</p>
</div>

**Evaluation Techniques:**
```
Cross-Validation Principle:
Split data into k folds, train on k-1 folds, test on 1

Learning Curve Interpretation:
- High bias: high train error = underfitting (get more complex model)
- High variance: gap between train and test = overfitting (get more data or regularize)

ROC-AUC:
- ROC plots True Positive Rate vs False Positive Rate
- AUC = Area Under Curve (higher is better, 1.0 is perfect, 0.5 is random)
```

<div class="alert alert-success">
  <h4>✅ Exam Tip: Model Selection</h4>
  <p>Know when to use which technique: Use cross-validation for small datasets, use hold-out test set for large datasets. Always stratify in classification to maintain class distribution.</p>
</div>

<div class="alert alert-warning">
  <h4>⚠️ Warning: Data Leakage</h4>
  <p>Never fit preprocessing (scaling, encoding) on the entire dataset before splitting. Always fit on training data only, then apply to test data. Data leakage inflates performance estimates.</p>
</div>

---

### Module 5: Feature Engineering

#### Key Topics:
- **Feature Selection**: Choosing relevant features
- **Feature Extraction**: Creating new features
- **Encoding**: Categorical to numerical
- **Scaling/Normalization**: Standardizing features
- **Handling Missing Data**: Imputation strategies

<div class="alert alert-note">
  <h4>📌 Note: Feature Engineering is Critical</h4>
  <p>70% of machine learning success comes from good features. A simple model with great features often outperforms complex models with poor features.</p>
</div>

**Key Techniques:**
```
Feature Selection Methods:
1. Filter methods: correlation, chi-square, mutual information
2. Wrapper methods: recursive feature elimination, forward selection
3. Embedded methods: L1/L2 regularization, tree importance

Feature Scaling:
- Standardization: (x - μ) / σ (zero mean, unit variance)
- Min-Max Scaling: (x - min) / (max - min) (0 to 1 range)

Missing Data Imputation:
- Mean/Median/Mode imputation
- Forward/Backward fill (time series)
- K-NN imputation
- Model-based imputation
```

<div class="alert alert-danger">
  <h4>🔴 Common Mistake: Feature Leakage</h4>
  <p>Don't use future information when training. For example, don't use the target variable's statistics to engineer features, or future data points in time series.</p>
</div>

<div class="alert alert-tip">
  <h4>💡 Industry Tip: Domain Knowledge</h4>
  <p>The best features often come from domain expertise. Work with subject matter experts to create features that capture domain-specific insights. This often beats automated feature selection.</p>
</div>

---

### Module 6: Ensemble Methods

#### Key Topics:
- **Bagging**: Bootstrap Aggregating
- **Boosting**: Gradient Boosting, AdaBoost, XGBoost
- **Stacking**: Combining multiple models
- **Random Forests**: Ensemble of trees
- **Voting Classifiers**: Hard and soft voting

<div class="alert alert-note">
  <h4>📌 Note: Ensemble Power</h4>
  <p>Ensembles combine multiple models to reduce overfitting and improve generalization. The key is diversity: each model should make different errors.</p>
</div>

**Key Ensemble Methods:**
```
Bagging:
- Train multiple models on random subsets (with replacement)
- Average predictions (regression) or vote (classification)

Boosting:
- Train models sequentially, each correcting previous mistakes
- Assign higher weights to misclassified samples
- Final prediction = weighted sum of all models

Random Forest:
- Bagging + feature randomness
- Each tree uses random subset of features
- More diversity = better generalization
```

<div class="alert alert-success">
  <h4>✅ Exam Tip: Ensemble Advantages</h4>
  <p>Know the advantages: reduced variance (bagging), reduced bias (boosting), robustness, and strong generalization. Understand why ensembles work better than single models.</p>
</div>

<div class="alert alert-warning">
  <h4>⚠️ Warning: Computational Cost</h4>
  <p>Ensembles are slower to train and predict than single models. In production, consider this tradeoff. Sometimes a simpler model is better if performance difference is minimal.</p>
</div>

---

## 🎯 Learning Outcomes

By the end of this course, you should be able to:

- ✅ Implement and evaluate regression and classification models
- ✅ Choose appropriate evaluation metrics for different problem types
- ✅ Apply cross-validation and diagnose model performance
- ✅ Engineer and select relevant features
- ✅ Build ensemble models and understand their advantages
- ✅ Recognize and avoid overfitting, underfitting, and data leakage

---

## ⚡ Exam Tips

<div class="card">
  <h4>🎓 Key Concepts to Master</h4>
  <ul>
    <li><strong>Confusion Matrix</strong>: Precision, Recall, F1-Score</li>
    <li><strong>Bias-Variance Tradeoff</strong>: What causes overfitting/underfitting?</li>
    <li><strong>Model Selection</strong>: When to use which algorithm?</li>
    <li><strong>Cross-Validation</strong>: Why and how to use it</li>
    <li><strong>Regularization</strong>: L1 vs L2 and when to use each</li>
    <li><strong>Feature Engineering</strong>: How to improve models</li>
  </ul>
</div>

<div class="alert alert-warning">
  <h4>⚠️ Common Exam Mistakes</h4>
  <ul>
    <li>Confusing precision and recall</li>
    <li>Not checking for data leakage before evaluation</li>
    <li>Forgetting to normalize features before scaling-sensitive algorithms</li>
    <li>Not stratifying splits in imbalanced classification</li>
    <li>Overfitting to the test set by trying too many models</li>
  </ul>
</div>

---

## 💼 Industry Applications

### E-Commerce
Classification models predict customer churn. Regression models forecast sales and demand.

### Healthcare
Classification models diagnose diseases. Ensemble methods combine multiple diagnostic signals.

### Finance
Fraud detection uses classification and anomaly detection. Lending decisions use ensemble models.

### Recommendation Systems
Clustering and similarity metrics group users and items. Ensemble methods combine multiple recommendation signals.

---

## 🔗 External Resources

### Online Courses
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [Kaggle Learn - Machine Learning](https://www.kaggle.com/learn/machine-learning)
- [Fast.ai - Practical Deep Learning](https://course.fast.ai/)

### Libraries & Tools
- **Scikit-learn**: Classical ML in Python
- **XGBoost**: Gradient boosting library
- **LightGBM**: Fast gradient boosting
- **CatBoost**: Handles categorical features well

### Important Papers
- [Random Forests - Leo Breiman](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [Gradient Boosting Machines - Friedman](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)

### Cheatsheets
- [Scikit-learn Cheatsheet](https://scikit-learn.org/)
- [Algorithm Selection Flowchart](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

---

## 📋 Quick Algorithm Selection Guide

**When to use:**
- **Linear Regression**: Continuous predictions, interpretability needed
- **Logistic Regression**: Binary classification, interpretability needed
- **Decision Trees**: Non-linear relationships, interpretability
- **Random Forest**: High accuracy needed, moderate interpretability
- **SVM**: Small-medium datasets, high-dimensional data
- **KNN**: Small datasets, can capture local patterns
- **Naive Bayes**: Text classification, fast training needed
- **K-Means**: Clustering unknown data
- **PCA**: Reduce dimensionality, visualization

---

## 📞 Need Help?

- Practice with Kaggle datasets and competitions
- Review scikit-learn documentation with examples
- Check the [Resources page]({{ site.baseurl }}/resources/) for more materials

**Last Updated**: December 2025
