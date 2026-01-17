## Q1. When to use / when not to use Logistic Regression

**When to use:**
- For binary classification when the relationship between features and the log-odds is approximately linear.
- When you need fast training, and a strong baseline model.
- When interpretability is important, since coefficients directly show feature impact.
- When well-calibrated probability estimates are required (e.g., risk scoring, decision thresholds).

**When not to use:**
- When the true decision boundary is highly non-linear and complex.
- When strong feature interactions dominate and are hard to capture with linear terms.
- When high model capacity is required without heavy feature engineering.

## Q2. Why Logistic Regression is preferred when calibrated probabilities are required?

Logistic Regression is preferred when calibrated probabilities are required because it is directly optimized to output true probabilities:

        P(y=1|x)

using a log-likelihood objective, which encourages probabilistic correctness. So if LR says 0.7, it is mathematically trained to make that number mean "70% chance of being positive". That’s not true for many other models.

Models like SVM optimize margins, not probabilities, and tree models produce piecewise constant, often overconfident estimates, so both typically need explicit calibration methods to produce reliable probabilities.

## Q3. Explain LR's model capacity

Logistic Regression has limited model capacity since it learns only linear decision boundaries. If the task requires modeling complex non-linear relationships and interactions without manual feature engineering, higher-capacity models like tree ensembles or neural networks are more appropriate.

## Q4. Core intuition of Logistic Regression

Logistic Regression learns a linear combination of features that represents the log-odds of the positive class, resulting in sigmoid function to get a probability. So it finds a linear decision boundary in feature space and interprets the distance from that boundary as confidence

## Q5. Loss function used in Logistic Regression

Logistic Regression uses binary cross-entropy (log loss), which penalizes confident wrong predictions heavily and rewards confident correct ones. It directly measures how close the predicted probabilities are to the true labels.

## Q6. Optimization

Training Logistic Regression is a convex optimization problem, so it has a single global minimum. In practice it is solved using gradient-based methods like Gradient Descent, Stochastic Gradient Descent, or second-order methods such as Newton’s method and LBFGS for faster convergence.

## Q7. Assumptions of Logistic Regression

1. Logistic Rgression assumes log-odds of the postive class are a linear function of the features
2. Multicollinearity mainly impacts coefficient stability and interpretability, though the decision boundary usually remains similar
3. Features should be on a comparable scale for stable optimization and fair regularization
4. Extreme outliers should be handled as they can dominate the loss

## Q8. Where does Logistic Regression lie on the bias–variance spectrum? Why?

Logistic Regression is a high-bias, low-variance model because it has limited capacity due to its linear decision boundary. It cannot model complex non-linear relationships, so it tends to underfit when the true pattern is complex. With regularization, variance is further reduced at the cost of slightly higher bias.

## Q9. Key Hyperparameters of Logistic Regression

The key hyperparameter are:
1. The regularization strength (C = 1/λ), which controls the bias–variance tradeoff:
    - smaller C means stronger regularization and higher bias
    - larger C means weaker regularization and higher variance

2. The penalty type (L1, L2, ElasticNet) which controls sparsity and feature selection
    - L1 (Lasso) drives many coefficients exactly to zero, so it performs implicit feature selection and produces sparse models
    - L2 (Ridge) shrinks coefficients smoothly without setting them to zero, which improves stability when features are correlated
    - ElasticNet combines both, giving a balance between sparsity and stability

3. In practice, the solver and class weights are also important for convergence and handling class imbalance
    - The solver determines how the optimization is performed and affects speed, stability, and which penalties are supported. For example, LBFGS is fast and stable for L2 regularization, liblinear supports L1 but is slower on large datasets, and saga scales well to large data and supports ElasticNet
    - Class weights are used to handle class imbalance by giving higher importance to the minority class in the loss function, so the model does not get biased toward predicting the majority class

## Q10. How L2 results in more stability?

The way L1 creates sparsity is what causes instability
- With L1, when two features are highly correlated, the model arbitrarily picks one and drives the other to zero. So, small changes in data -> different feature gets selected. 
- With L2, correlated features share weight, coefficients shrink smoothly instead of being zeroed. So, small data changes -> small coefficient changes -> Much more stable solution

## Q11. Data Requirements for Logistic Regression

- Feature scaling is needed as if features are on very different scales, large-scale features dominate the gradients and regularization, making optimization unstable and causing the model to penalize some weights more than others unfairly. Scaling puts all features on a comparable range so training is stable and regularization works correctly
- Logistic Regression requires numerical input features, so categorical variables must be encoded
- Missing values must be handled because the model cannot operate on NaNs as operations like `wTx` and gradient computations break if any `x` is NaN
- It also needs sufficient data in each class, and either balanced classes or proper class weighting
- Extreme outliers should be treated since they can dominate the loss

## Q12. Which models can work on NaNs? 

Some tree-based models can handle NaNs natively: Decision Trees (certain implementations), XGBoost, LightGBM, CatBoost. These treat NaNs as a separate "branch" during splitting, so the algorithm learns whether missing values should go left or right and missingness itself becomes informative.

Logistic Regression cannot handle NaNs because its prediction and gradient computations require all feature values to be numeric and defined.

## Q13. Suitable Metrics for Logistic Regression

- AUC is used because it measures how well the model ranks positive samples above negative ones. It evaluates ranking ability, not just final class predictions, and works well even when classes are imbalanced
- Log-loss is used when we care about the quality of predicted probabilities, since it penalizes confident wrong predictions heavily
- For decision making at a fixed threshold, Precision, Recall, and F1-score are also important

## Q14. Difference b/w ROC and AUC

- ROC curve is used to analyze the trade-off between TPR and FPR across different thresholds, which helps in selecting an operating threshold based on business constraints
- AUC summarizes the ROC curve into a single number and is mainly used to compare and select models based on their ranking ability, independent of any specific threshold

## Q15. Failure Modes of Logistic Regression

- Logistic Regression fails when the true decision boundary is highly non-linear or depends on complex feature interactions
- It performs poorly on heavily imbalanced data without class weighting, as it becomes biased toward the majority class
- Extreme outliers can dominate the loss and distort the decision boundary
- It also suffers when features are highly correlated, poorly scaled, or incorrectly encoded

## Q16. What to try next when Logistic Regression underperforms

1. First, I'll try to optimize LR
    - I would try feature engineering like interaction terms or non-linear transformations
    - I would also tune regularization strength and penalty type
    - If the data is imbalanced, I would use class weights or resampling
2. If performance is still limited, I would move to higher-capacity models such as tree-based methods for non-linear patterns, or kernel SVMs when the dataset is small and non-linearity is strong

## Q17. Should multicollinearity be handled before trying Logistic Regression?

Multicollinearity does not usually change the decision boundary much, but it makes coefficients unstable and feature importance unreliable, so it should be handled when interpretability or stability matters.

So, if LR is used only for prediction and heavily regularized, we can sometimes ignore it, but if LR is used for explainability, risk modeling, then you must handle multicollinearity as in risk modeling, coefficients and probabilities are directly used for business and regulatory decisions. Multicollinearity makes coefficients unstable and explanations unreliable, which is unacceptable even if predictive performance is unchanged.


