---
layout: course
title: "Introduction to Statistical Methods"
short_description: "Probability, distributions, hypothesis testing, time series analysis, and statistical inference"
description: "Master statistical methods for data analysis, inference, and forecasting"
credits: 4
level: "Intermediate"
instructor: "Dr Y V K Ravi Kumar"
topics:
  - Probability Theory & Bayes Theorem
  - Probability Distributions
  - Hypothesis Testing & ANOVA
  - Time Series Analysis
  - Regression & Correlation
  - Gaussian Mixture Models
github_repo: "https://github.com/shivam2003-dev/semester_1_all_course"
---

## 📝 Course Overview

This comprehensive course equips you with essential statistical methods for data analysis, inference, and prediction. From fundamental probability concepts to advanced time series forecasting and expectation-maximization algorithms, you'll learn practical statistical techniques used across data science, machine learning, and business analytics.

---

## 📚 Course Content

### Module 1: Basic Probability & Statistics

#### Key Topics:
- **Measures of Central Tendency**: Mean, Median, Mode
- **Measures of Variability**: Variance, Standard Deviation, Range, IQR
- **Outlier Detection**: 5-point summary, Skewness, Kurtosis
- **Probability Basics**: Sample spaces, events, axioms
- **Probability Concepts**: Mutually exclusive and independent events

<div class="alert alert-note">
  <h4>📌 Note: Statistics Foundation</h4>
  <p>Understanding descriptive statistics is essential. Measures of central tendency and variability give us insights into data distribution. These simple concepts are the foundation for all advanced statistical analysis.</p>
</div>

**Key Statistical Measures:**
```
Mean: μ = Σx / n

Variance: σ² = Σ(x - μ)² / n

Standard Deviation: σ = √(σ²)

Coefficient of Variation: CV = σ / μ × 100%

Skewness: Indicates asymmetry of distribution
Positive skew: tail on right
Negative skew: tail on left

Kurtosis: Measures tail heaviness
High kurtosis = heavy tails (outliers likely)
```

<div class="alert alert-success">
  <h4>✅ Exam Tip: 5-Point Summary</h4>
  <p>The 5-point summary (Min, Q1, Median, Q3, Max) is crucial for understanding data distribution and identifying outliers. Use the IQR method: outliers are values beyond Q1-1.5×IQR or Q3+1.5×IQR.</p>
</div>

**Probability Basics:**
```
Axioms of Probability:
1. P(A) ≥ 0 for any event A
2. P(S) = 1 (where S is sample space)
3. For mutually exclusive events: P(A ∪ B) = P(A) + P(B)

Mutually Exclusive: Events cannot occur together
P(A ∩ B) = 0

Independent Events: Occurrence of one doesn't affect other
P(A ∩ B) = P(A) × P(B)
```

<div class="alert alert-warning">
  <h4>⚠️ Warning: Independence vs Exclusivity</h4>
  <p>Students often confuse independent events with mutually exclusive events. They're opposite concepts! Mutually exclusive events cannot occur together, while independent events don't affect each other's probability.</p>
</div>

<div class="alert alert-danger">
  <h4>🔴 Common Mistake: Probability Calculations</h4>
  <p>Remember: P(A) is always between 0 and 1. If your calculation gives probability > 1 or < 0, you've made an error. Also, probabilities of all outcomes must sum to 1.</p>
</div>

<div class="alert alert-tip">
  <h4>💡 Industry Tip: Data Profiling</h4>
  <p>In production systems, continuous data profiling is critical. Generate statistics regularly to detect data quality issues, anomalies, and distribution shifts. Use statistical checks to trigger alerts.</p>
</div>

---

### Module 2: Conditional Probability & Bayes Theorem

#### Key Topics:
- **Conditional Probability**: Probability given prior information
- **Total Probability**: Decomposing complex probabilities
- **Bayes Theorem**: Updating beliefs with new evidence
- **Naive Bayes**: Probabilistic classification algorithm
- **Applications**: Medical diagnosis, spam detection, classification

<div class="alert alert-note">
  <h4>📌 Note: Conditional Probability is Fundamental</h4>
  <p>Most real-world problems involve conditional probabilities. "What's the probability given that..." appears constantly. Understanding this concept unlocks Bayes Theorem and probabilistic reasoning.</p>
</div>

**Conditional Probability:**
```
P(A|B) = P(A ∩ B) / P(B)  [Probability of A given B occurred]

For Independent Events:
P(A|B) = P(A)  [B doesn't affect A's probability]

Total Probability Law:
P(A) = Σ P(A|Bᵢ) × P(Bᵢ)
```

<div class="alert alert-success">
  <h4>✅ Exam Tip: Bayes Theorem</h4>
  <p>Know Bayes Theorem by heart: P(A|B) = P(B|A) × P(A) / P(B). Understand it conceptually: it updates prior beliefs P(A) with new evidence B to give posterior probability P(A|B).</p>
</div>

**Bayes Theorem - The Heart of Statistical Inference:**
```
Bayes Theorem:
P(A|B) = P(B|A) × P(A) / P(B)

Components:
- P(A): Prior (what we believed before new evidence)
- P(B|A): Likelihood (probability of evidence given A)
- P(B): Evidence (total probability of observing B)
- P(A|B): Posterior (updated belief after seeing evidence)

Example - Medical Testing:
Disease prevalence (prior): P(Disease) = 0.01
Test accuracy (likelihood): P(Positive|Disease) = 0.95
False positive rate: P(Positive|No Disease) = 0.05

After positive test:
P(Disease|Positive) = 0.95 × 0.01 / [0.95 × 0.01 + 0.05 × 0.99]
                    = 0.0095 / 0.0590
                    ≈ 0.161 (16.1%, not 95%!)
```

<div class="alert alert-warning">
  <h4>⚠️ Warning: Base Rate Fallacy</h4>
  <p>The base rate (prior probability) is crucial! Ignoring it is a common error in medical diagnosis and other domains. Even with high test accuracy, if disease is rare, a positive test might still have low probability of disease.</p>
</div>

**Naive Bayes Classification:**
```
Assumes features are conditionally independent given class label

P(Class|Features) = P(Features|Class) × P(Class) / P(Features)

≈ P(x₁|Class) × P(x₂|Class) × ... × P(xₙ|Class) × P(Class)

Despite "naive" assumption, works surprisingly well in practice!
```

<div class="alert alert-tip">
  <h4>💡 Industry Tip: Bayesian Methods</h4>
  <p>Bayesian inference is becoming standard in industry. Tools like Bayesian optimization, Bayesian networks, and probabilistic programming (PyMC, Stan) help quantify uncertainty - crucial for decision-making under uncertainty.</p>
</div>

---

### Module 3: Probability Distributions

#### Key Topics:
- **Random Variables**: Discrete and continuous
- **Discrete Distributions**: Bernoulli, Binomial, Poisson
- **Continuous Distributions**: Normal, t, F, Chi-Square
- **Transformation of Variables**: Function of random variables
- **Joint Distributions**: Correlation and covariance

<div class="alert alert-note">
  <h4>📌 Note: Distribution Selection Matters</h4>
  <p>Choosing the right distribution is critical for statistical modeling. Different distributions suit different phenomena. Understanding their characteristics helps model selection and inference.</p>
</div>

**Discrete Distributions:**
```
Bernoulli Distribution (single trial, success/failure)
P(X=k) = p^k × (1-p)^(1-k)  for k ∈ {0,1}
Mean = p, Variance = p(1-p)

Binomial Distribution (n independent Bernoulli trials)
P(X=k) = C(n,k) × p^k × (1-p)^(n-k)
Mean = np, Variance = np(1-p)

Poisson Distribution (count of events in fixed interval)
P(X=k) = (e^(-λ) × λ^k) / k!
Mean = λ, Variance = λ
Used for: page hits, accidents, radioactive decay
```

<div class="alert alert-success">
  <h4>✅ Exam Tip: Distribution Properties</h4>
  <p>Know the mean and variance of each distribution. For binomial: mean = np, var = np(1-p). For Poisson: mean = var = λ. These appear in hypothesis tests and confidence intervals.</p>
</div>

**Continuous Distributions:**
```
Normal (Gaussian) Distribution
f(x) = (1/(σ√(2π))) × e^(-(x-μ)²/(2σ²))
Defined by: mean μ and standard deviation σ
68-95-99.7 rule: 68% within 1σ, 95% within 2σ, 99.7% within 3σ

Standardized Normal (Z-distribution)
μ = 0, σ = 1
Z = (X - μ) / σ

t-Distribution
Similar to normal but with heavier tails
Used when σ is unknown (more common in practice)
Degrees of freedom parameter

Chi-Square Distribution
Used for variance tests and goodness-of-fit tests
Defined by degrees of freedom

F-Distribution
Used for ANOVA and comparing variances
Ratio of two chi-square distributions
```

<div class="alert alert-warning">
  <h4>⚠️ Warning: Normal Distribution Assumption</h4>
  <p>Many statistical tests assume normality. Always check this assumption! Use Q-Q plots or Shapiro-Wilk test. If data is non-normal, use non-parametric alternatives or transform the data.</p>
</div>

**Transformations of Random Variables:**
```
If Y = g(X), find distribution of Y

Example: If X ~ N(0,1), then Y = X² ~ χ²(1)

For continuous: fᵧ(y) = fₓ(g⁻¹(y)) × |dg⁻¹/dy|

Understanding transformations helps with:
- Model building
- Change of variables in integration
- Understanding derived statistics
```

<div class="alert alert-danger">
  <h4>🔴 Common Mistake: Covariance and Correlation</h4>
  <p>Covariance depends on units (hard to interpret), correlation is unitless (-1 to 1). Use correlation to measure strength of linear relationship. Remember: correlation ≠ causation!</p>
</div>

**Covariance and Correlation:**
```
Covariance: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]
- Positive: variables move together
- Negative: variables move opposite
- But scale-dependent (hard to interpret)

Correlation: ρ = Cov(X,Y) / (σₓ × σᵧ)
- Ranges from -1 to 1
- 0: no linear relationship
- 1: perfect positive relationship
- -1: perfect negative relationship
```

<div class="alert alert-tip">
  <h4>💡 Industry Tip: Distribution Fitting</h4>
  <p>In production, fit appropriate distributions to your data. Use goodness-of-fit tests (Kolmogorov-Smirnov, Anderson-Darling) to validate. This helps with confidence intervals, anomaly detection, and forecasting.</p>
</div>

---

### Module 4: Hypothesis Testing

#### Key Topics:
- **Sampling & Sampling Distributions**: Central Limit Theorem
- **Point & Interval Estimation**: Confidence intervals
- **Hypothesis Testing**: Test statistics, p-values, significance levels
- **Tests for Means**: One sample, two sample, matched pairs
- **Tests for Proportions**: Binomial tests
- **ANOVA**: Single factor and two-factor
- **Maximum Likelihood Estimation**: Parameter estimation

<div class="alert alert-note">
  <h4>📌 Note: Central Limit Theorem is Powerful</h4>
  <p>The Central Limit Theorem states that sample means are approximately normally distributed regardless of original distribution. This enables hypothesis testing even when population distribution is unknown!</p>
</div>

**Central Limit Theorem:**
```
If X₁, X₂, ..., Xₙ are i.i.d. with mean μ and variance σ²
Then: X̄ ~ N(μ, σ²/n)  as n → ∞

Key insight:
- Sample mean is normally distributed
- This holds even if population isn't normal
- Larger n → smaller variance of X̄
- Enables hypothesis testing for any distribution
```

<div class="alert alert-success">
  <h4>✅ Exam Tip: Hypothesis Testing Framework</h4>
  <p>Follow the framework: 1) State H₀ and H₁ 2) Choose α (significance level) 3) Calculate test statistic 4) Find p-value 5) Make decision 6) Interpret in context. Practice all types: mean, proportion, ANOVA.</p>
</div>

**Hypothesis Testing Steps:**
```
1. Formulate Hypotheses
   H₀: Null hypothesis (status quo)
   H₁: Alternative hypothesis (what we're testing)

2. Choose Significance Level α (typically 0.05)

3. Collect Data and Calculate Test Statistic
   t = (X̄ - μ₀) / (s/√n)  [for means]

4. Find p-value
   p-value = probability of observing data if H₀ true

5. Make Decision
   If p-value < α: Reject H₀ (significant result)
   If p-value ≥ α: Fail to reject H₀ (no significant result)

6. Interpret Results
   Include effect size and confidence intervals
```

**Confidence Intervals:**
```
General form: Point Estimate ± (Critical Value × Standard Error)

Confidence Interval for Mean (σ unknown):
X̄ ± t*(α/2, n-1) × (s/√n)

Confidence Interval for Proportion:
p̂ ± z*(α/2) × √(p̂(1-p̂)/n)

Interpretation:
"We're 95% confident the true parameter lies in this interval"
NOT "95% probability parameter is in interval" (it's fixed!)
```

<div class="alert alert-warning">
  <h4>⚠️ Warning: P-value Misinterpretation</h4>
  <p>p-value is NOT probability that H₀ is true! It's the probability of data given H₀. Small p-value suggests data is unlikely if H₀ true, so we reject H₀. Don't confuse with effect size or practical significance.</p>
</div>

**ANOVA (Analysis of Variance):**
```
One-Factor ANOVA:
Tests if means differ across k groups

Test Statistic: F = MS_Between / MS_Within

If F is large → groups have different means

Two-Factor ANOVA:
Tests effect of two factors and their interaction

Null Hypotheses:
- H₀(A): Factor A has no effect
- H₀(B): Factor B has no effect
- H₀(AB): No interaction between factors
```

<div class="alert alert-danger">
  <h4>🔴 Common Mistake: Multiple Testing</h4>
  <p>If you run many tests, some will be significant by chance alone! Use Bonferroni correction: divide α by number of tests. Or use ANOVA instead of multiple t-tests.</p>
</div>

**Maximum Likelihood Estimation (MLE):**
```
Find parameter values that maximize likelihood of observed data

Likelihood: L(θ|data) = ∏ f(xᵢ|θ)

Log-Likelihood: ℓ(θ) = Σ log f(xᵢ|θ)

MLE: θ̂ = argmax ℓ(θ)

Often solved by: ∂ℓ/∂θ = 0

Properties:
- Asymptotically normal
- Efficient (reaches Cramér-Rao lower bound)
- Invariant under transformations
```

<div class="alert alert-tip">
  <h4>💡 Industry Tip: A/B Testing</h4>
  <p>A/B testing uses hypothesis testing principles to make business decisions. Understand power analysis to choose sample size, sequential testing to stop early, and multiple comparison corrections.</p>
</div>

---

### Module 5: Prediction & Forecasting

#### Key Topics:
- **Correlation & Regression**: Linear relationships
- **Simple & Multiple Regression**: Model fitting and evaluation
- **Time Series Components**: Trend, seasonality, noise
- **MA & Weighted MA**: Moving average models
- **AR, ARMA, ARIMA**: Autoregressive models
- **SARIMA, SARIMAX**: Seasonal variants
- **Exponential Smoothing**: Forecasting techniques

<div class="alert alert-note">
  <h4>📌 Note: Regression is Fundamental</h4>
  <p>Regression is the most widely used statistical technique. From simple linear regression to complex time series models, regression concepts appear everywhere in data science and statistics.</p>
</div>

**Regression Fundamentals:**
```
Simple Linear Regression:
ŷ = β₀ + β₁x

Estimates:
β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
β₀ = ȳ - β₁x̄

R² (Coefficient of Determination):
R² = 1 - (SS_res / SS_tot)
Fraction of variance explained by model

Multiple Regression:
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ

Matrix form: ŷ = Xβ
β = (X'X)⁻¹ X'y  (normal equations)
```

<div class="alert alert-success">
  <h4>✅ Exam Tip: Regression Output Interpretation</h4>
  <p>Understand regression tables: coefficients, standard errors, t-statistics, p-values. Know what each means. Practice interpreting confidence intervals for coefficients and predictions.</p>
</div>

**Time Series Analysis:**
```
Components:
- Trend: Long-term direction
- Seasonality: Regular patterns (daily, monthly, yearly)
- Noise: Random fluctuations

Decomposition:
Series = Trend × Seasonality × Noise  (multiplicative)
Or
Series = Trend + Seasonality + Noise  (additive)

Moving Average (MA):
ŷₜ = (yₜ + yₜ₋₁ + ... + yₜ₋ₖ₊₁) / k
Smooths data, removes short-term fluctuations

Weighted Moving Average:
ŷₜ = w₁yₜ + w₂yₜ₋₁ + ... + wₖyₜ₋ₖ₊₁
Recent observations weighted more heavily
```

<div class="alert alert-warning">
  <h4>⚠️ Warning: Stationarity Assumption</h4>
  <p>Most time series models assume stationarity (mean, variance constant over time). Check this! If non-stationary, difference the series or use ARIMA. Ignoring non-stationarity invalidates analysis.</p>
</div>

**ARIMA Models:**
```
AR (AutoRegressive): Use past values
yₜ = φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φₚyₜ₋ₚ + εₜ

MA (Moving Average): Use past errors
yₜ = εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θqεₜ₋q

ARIMA(p,d,q):
p = AR order, d = differencing, q = MA order

Example: ARIMA(1,1,1)
- Difference once (d=1)
- Use 1 past value (p=1)
- Use 1 past error (q=1)

SARIMA(p,d,q)(P,D,Q,s):
Adds seasonal components
s = seasonal period (12 for monthly data, etc.)
```

<div class="alert alert-danger">
  <h4>🔴 Common Mistake: Over-differencing</h4>
  <p>Differencing removes trend and seasonality but can introduce artificial patterns. Check ACF/PACF plots carefully. Usually, first differencing is enough; rarely need more than 2.</p>
</div>

**Exponential Smoothing:**
```
Simple Exponential Smoothing (SES):
ŷₜ₊₁ = αyₜ + (1-α)ŷₜ

α = smoothing coefficient (0 < α < 1)
High α: more weight to recent observations
Low α: more weight to history

Holt-Winters (with trend and seasonality):
Combines level, trend, and seasonal components
Can be additive or multiplicative
```

<div class="alert alert-tip">
  <h4>💡 Industry Tip: Forecasting in Practice</h4>
  <p>Ensemble forecasting (combining multiple models) often outperforms single models. Use cross-validation for time series (don't mix future and past). In production, retrain models regularly as new data arrives.</p>
</div>

<div class="alert alert-note">
  <h4>📌 Note: VAR Models</h4>
  <p>Vector Autoregression (VAR) and VARMAX extend ARIMA to multivariate setting. Useful when forecasting multiple related time series simultaneously.</p>
</div>

---

### Module 6: Gaussian Mixture Models & Expectation Maximization

#### Key Topics:
- **Gaussian Mixture Models (GMM)**: Probabilistic clustering
- **Expectation-Maximization (EM) Algorithm**: Parameter estimation
- **Soft vs Hard Clustering**: Probabilistic assignments
- **Model Selection**: Choosing number of components
- **Applications**: Clustering, anomaly detection, density estimation

<div class="alert alert-note">
  <h4>📌 Note: GMM as Probabilistic Alternative to K-Means</h4>
  <p>While K-means assigns points to clusters (hard assignment), GMM assigns probabilities (soft assignment). This provides more flexibility and uncertainty quantification. GMM is also a generative model.</p>
</div>

**Gaussian Mixture Model:**
```
Assume data comes from mixture of k Gaussian distributions

Model:
p(x) = Σ πₖ × N(x|μₖ, Σₖ)

where:
- πₖ: mixing coefficient (sum to 1)
- N(x|μₖ, Σₖ): Gaussian with mean μₖ and covariance Σₖ
- k: component/cluster index

Interpretation:
Each data point has probability of belonging to each component
Soft clustering (vs hard clustering in K-means)
```

<div class="alert alert-success">
  <h4>✅ Exam Tip: EM Algorithm Steps</h4>
  <p>Know the EM algorithm: E-step computes responsibilities (probabilities), M-step updates parameters maximizing expected likelihood. Repeat until convergence. Understand both steps conceptually.</p>
</div>

**Expectation-Maximization (EM) Algorithm:**
```
Goal: Estimate parameters θ = {π, μ, Σ}

E-Step (Expectation):
Compute responsibility (posterior probability) of each component for each data point
rₖₙ = (πₖ × N(xₙ|μₖ, Σₖ)) / Σⱼ(πⱼ × N(xₙ|μⱼ, Σⱼ))

M-Step (Maximization):
Update parameters using responsibilities as weights

Nₖ = Σₙ rₖₙ (effective number of points in cluster k)

πₖ^(new) = Nₖ / N

μₖ^(new) = (1/Nₖ) × Σₙ rₖₙ × xₙ

Σₖ^(new) = (1/Nₖ) × Σₙ rₖₙ × (xₙ - μₖ)(xₙ - μₖ)ᵀ

Repeat E and M steps until convergence
```

<div class="alert alert-warning">
  <h4>⚠️ Warning: Local Optima</h4>
  <p>EM can get stuck in local optima. Run multiple times with different initializations and choose best result. Use BIC or AIC to choose number of components.</p>
</div>

**Model Selection for GMM:**
```
Use Information Criteria:

AIC = -2 log(L) + 2k  (k = number of parameters)

BIC = -2 log(L) + k log(n)  (penalizes complexity more than AIC)

Silhouette Score: For clustering quality

Bayesian Model Comparison: Compare marginal likelihoods

Lower AIC/BIC = better model

Typical process:
1. Try k = 1, 2, 3, ..., k_max
2. Compute AIC/BIC for each
3. Choose k with lowest score
```

<div class="alert alert-danger">
  <h4>🔴 Common Mistake: Singular Covariance</h4>
  <p>If cluster has too few points, covariance becomes singular (non-invertible). Use regularization (add small diagonal term) or tie covariances. In software, use regularization parameter.</p>
</div>

**Applications:**
```
Clustering:
- Soft clustering with uncertainty
- Anomaly detection: points with low probability in all components

Density Estimation:
- Estimate probability distribution
- Generate new samples from model

Data Imputation:
- Use model to fill missing values
- More sophisticated than simple mean imputation
```

<div class="alert alert-tip">
  <h4>💡 Industry Tip: EM Beyond GMM</h4>
  <p>EM algorithm extends beyond GMM! Used in Hidden Markov Models, missing data imputation, and many other problems. Understanding EM deeply enables solving complex estimation problems.</p>
</div>

---

## 🎯 Learning Outcomes

By the end of this course, you should be able to:

- ✅ Understand and apply probability fundamentals and Bayes Theorem
- ✅ Select appropriate probability distributions for data
- ✅ Perform hypothesis testing and construct confidence intervals
- ✅ Analyze and forecast time series data
- ✅ Build regression models for prediction
- ✅ Apply Gaussian Mixture Models and EM algorithm
- ✅ Interpret statistical results and draw valid conclusions

---

## 📋 Lab Work Schedule

| Lab | Objective | Topics |
|-----|-----------|--------|
| **Lab 1** | Statistical Data Display & Summary | Descriptive Statistics, Data Visualization |
| **Lab 2** | Bayes Theorem & Naive Bayes | Conditional Probability, Classification |
| **Lab 3** | Probability Distributions & Sampling | Distribution Fitting, Random Sampling |
| **Lab 4** | ANOVA Analysis | Hypothesis Testing, Multi-group Comparison |
| **Lab 5** | Regression Analysis | Linear Regression, Model Evaluation |
| **Lab 6** | Time Series Forecasting | AR, MA, ARIMA, SARIMA Models |

---

## ⚡ Exam Tips

<div class="alert alert-success">
  <h4>✅ Mid-Semester Focus (Sessions 1-8)</h4>
  <ul>
    <li><strong>Probability Fundamentals</strong>: Axioms, conditional probability, Bayes</li>
    <li><strong>Distributions</strong>: Properties, mean, variance, when to use each</li>
    <li><strong>Central Limit Theorem</strong>: Why it's important, implications</li>
    <li><strong>Sampling Distributions</strong>: How sample statistics vary</li>
  </ul>
</div>

<div class="alert alert-success">
  <h4>✅ Comprehensive Exam Focus (All Sessions)</h4>
  <ul>
    <li><strong>Hypothesis Testing Framework</strong>: Steps, p-values, errors (Type I, II)</li>
    <li><strong>ANOVA</strong>: One-factor, two-factor, interpretation</li>
    <li><strong>Time Series</strong>: Components, stationarity, ARIMA, forecasting</li>
    <li><strong>GMM & EM</strong>: Algorithm steps, model selection</li>
    <li><strong>Integration</strong>: Apply multiple concepts to real problems</li>
  </ul>
</div>

<div class="alert alert-warning">
  <h4>⚠️ Common Exam Mistakes</h4>
  <ul>
    <li>Confusing independent and mutually exclusive events</li>
    <li>Misinterpreting p-values and confidence intervals</li>
    <li>Assuming normality without checking</li>
    <li>Forgetting to check stationarity before time series modeling</li>
    <li>Over-interpreting regression results (causation vs correlation)</li>
    <li>Not checking assumptions of statistical tests</li>
    <li>Base rate fallacy in Bayes' theorem applications</li>
  </ul>
</div>

---

## 💼 Industry Applications

### Finance & Risk Management
Hypothesis testing for trading strategies, Monte Carlo simulations using distributions, time series forecasting for markets.

### Quality Control
ANOVA for process comparison, statistical process control using hypothesis tests.

### Marketing & A/B Testing
Hypothesis testing for campaign effectiveness, Bayesian methods for personalization, regression for demand estimation.

### Healthcare & Pharmaceuticals
Clinical trials use hypothesis testing, survival analysis uses time series concepts, mixture models for disease subtypes.

### Manufacturing & Operations
Quality assurance through ANOVA, forecasting demand using ARIMA, GMM for anomaly detection.

### Data Science & ML
Statistical validation of model results, confidence intervals for predictions, time series for forecasting, GMM for clustering.

---

## 🔗 External Resources

### Textbooks
- **Statistics for Data Scientists** - Maurits Kaptein et al, Springer 2022
- **Probability and Statistics for Engineering** - Jay L Devore, Cengage Learning
- **Introduction to Time Series and Forecasting** - Brockwell & Davis, Springer

### Online Courses
- [Stanford StatLearning](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning) - Free course
- [Coursera Statistics with R](https://www.coursera.org/specializations/statistics)
- [MIT OpenCourseWare Statistics](https://ocw.mit.edu/search/?q=Statistics)

### Tools & Libraries
- **Python**: StatsModels, SciPy, Scikit-learn, PyMC
- **R**: ggplot2, tidyverse, forecast, mixtools
- **ARIMA**: `statsmodels.tsa.arima`, `auto.arima` in R
- **GMM**: `sklearn.mixture.GaussianMixture`, `mclust` in R

### Important Papers & References
- [Time Series Analysis - ARIMA Models](https://arxiv.org/abs/1704.01745)
- [Expectation-Maximization Algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
- [Gaussian Mixture Models](https://en.wikipedia.org/wiki/Mixture_model)

### Cheatsheets
- [Probability Cheatsheet](https://static1.squarespace.com/static/54bf3241e4b0f0d81bf7ff36/t/55e9494fe4b011aed10e1190/1441352015658/probability_cheatsheet.pdf)
- [Statistical Tests Decision Tree](https://www.scribd.com/doc/14524423/Selecting-the-Right-Statistical-Test)
- [Time Series Cheatsheet](https://www.kaggle.com/code/hrishikeshg/time-series-cheatsheet)

---

## 📊 Quick Reference - Distribution Selection

**For Counting Problems:**
- Bernoulli: Single trial, success/fail
- Binomial: Fixed number of independent trials
- Poisson: Count of events in fixed interval (rare events)

**For Continuous Data:**
- Normal: Symmetric, common (CLT)
- t-distribution: When variance unknown, small samples
- Chi-Square: For variance tests, goodness-of-fit

**For Comparison Tests:**
- Use t-test for means (if normal, or large n)
- Use ANOVA for multiple groups
- Use Mann-Whitney U for non-normal data

---

## 📞 Need Help?

- Review lab materials and solutions
- Practice with dataset examples
- Work through past exam problems
- Consult textbook references for deep concepts
- Check the [Resources page]({{ site.baseurl }}/resources/) for more materials

**Last Updated**: December 2025
