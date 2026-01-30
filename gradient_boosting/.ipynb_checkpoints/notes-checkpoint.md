## Q1. When to use / when not to use Gradient Boosting

**When to use:**
- Use for tabular data with complex non-linearities and interactions when you need strong predictive performance and are willing to tune hyperparameters
- It's especially effective when reducing bias is the main goal

**When not to use:**
- Avoid on very noisy or very small datasets, where it can overfit
- When the data is extremely high-dimensional and sparse
- When model size or inference latency is critical because predictions require traversing many trees
- When full model interpretability is required

## Q2. Core Intuition

Gradient Boosting is a sequential additive model where each new shallow tree is trained to correct the errors of the current ensemble by fitting the negative gradient of the loss (pseudo-residuals). By adding weak learners step by step, it progressively reduces bias. The learning rate controls how much each tree contributes, trading faster bias reduction for overfitting risk

## Q3. Loss / Split Criterion in Gradient Boosting

Gradient Boosting optimizes a single global loss function in a stage-wise manner. At each iteration, it adds a new shallow tree that updates the model in the direction that minimizes the overall loss:
- Classification: Log-loss (binary cross-entropy)
- Regression: Squared error, absolute error, etc

At each iteration, the new tree is trained on the negative gradient of the loss with respect to the current predictions, often called pseudo-residuals

Although the global objective is loss minimization, the individual trees are still built greedily, similar to standard decision trees:
- For classification:
    - Trees act as regression trees on gradients
    - Splits minimize squared error of pseudo-residuals
- For regression:
    - Trees are fitted on residuals
    - Split chosen to minimize squared error of residuals

## Q4. Key difference in the Loss / Split Criterion b/w RF and GB

**Random Forest:**
- Does not optimize a single global loss function
- Each tree is trained independently on a bootstrap sample
- Splits inside each tree are chosen greedily to maximize local impurity reduction

**Gradient Boosting:**
- Optimizes a single global loss function in a stage-wise manner
- Trees are trained sequentially, where each new tree corrects the errors of the current ensemble
- Each tree is fitted to the negative gradient of the loss (pseudo-residuals)
- Splits are greedy, but guided by the loss gradient, not the raw labels

## Q5. Optimization view of Gradient Boosting

**Gradient:**
- In Gradient Boosting, gradient refers to the gradient of the loss function with respect to the model's current predictions
    - The loss tells us how wrong the current predictions are
    - The gradient tells us how to change the predictions to reduce that loss

So at each step, the gradient answers: "In which direction should I change my predictions to reduce error the fastest?". This is why it's called Gradient Boosting

**Pseudo-residuals:**
- Pseudo-residuals are the negative gradients of the loss function with respect to the current predictions

        rᵢ = -dL(yᵢ,yᵢ^)/dyᵢ^
  
- For squared error loss, pseudo-residuals reduce to simple residuals:

        rᵢ = yᵢ - yᵢ^

- For log-loss, pseudo-residuals are not raw errors, instead they come from the gradient of log-loss

Trees do not fit the labels again. Instead, they fit pseudo-residuals because the goal is to correct what the current ensemble is getting wrong, not to relearn the target from scratch. Each tree learns how predictions should change to reduce the overall loss

## Q6. How does Gradient Boosting work step-by-step?

1. Initialization
    - Start with a simple initial model that makes the same prediction for all samples
    - This is usually a constant that minimizes the loss:
        - For regression -> mean of the target
        - For classification -> log-odds of the positive class

This serves as the starting point for optimization

2. Iterative correction (Boosting steps)

   For each boosting iteration:
   - Compute the pseudo-residuals, which are the negative gradients of the loss with respect to the current predictions
   - Train a shallow decision tree to predict these pseudo-residuals
   - Scale the tree's predictions by the learning rate
   - Add the scaled tree to the existing model to update predictions

Each new tree corrects the mistakes made by the current ensemble

3. Final prediction aggregation
    - The final model is the sum of all trees, including the initial model and all subsequent correction trees
    - For regression -> predictions are summed directly
    - For classification -> summed scores are passed through a link function (e.g., sigmoid) to obtain probabilities

## Q7. Assumptions

1. Gradient Boosting makes very few assumptions about the data. It does not assume linearity, normality, or feature independence, and is insensitive to feature scaling
2. However, it assumes reasonably clean, representative (i.i.d.) data, since boosting sequentially fits errors and can overfit noise
3. Multicollinearity mainly affects feature importance rather than predictive performance
4. While Gradient Boosting is more stable than a single decision tree, it still assumes that class imbalance is handled explicitly. Otherwise, the loss function becomes dominated by the majority class, and the model may focus on optimizing majority-class performance at the expense of minority-class recall

## Q8. Explain the i.i.d. assumption held by the Gradient Boosting

The i.i.d. assumption means that training samples are independent of each other and are drawn from the same underlying data distribution
1. Independent:
    - Each data point should not depend on another. Example where this breaks:
        - Time series (today depends on yesterday)
        - User sessions (multiple rows from same user)
        - Grouped data (multiple records from same hospital, same customer, etc.)
    - If independence is violated, Gradient Boosting can overfit patterns that don't generalize, because it repeatedly corrects correlated errors
2. Identically distributed:
    - All samples should come from the same underlying data distribution which means:
        - Train and test data should represent the same population
        - No major data drift
        - No covariate shift
    - If this is violated, the model may minimize training loss very well but fail badly in production, since boosting aggressively fits training patterns

## Q9. Bias/variance

Gradient Boosting primarily reduces bias by sequentially adding shallow trees that correct previous errors. Because trees are added in a correlated, sequential manner, it can have high variance and overfit, especially with many iterations or noisy data. The learning rate and early stopping help control variance, but unlike Random Forest, variance is not inherently reduced

## Q10. Why Gradient Boosting reduces bias but can increase variance

Gradient Boosting reduces bias by sequentially adding shallow trees that correct the errors of the current ensemble, allowing it to model increasingly complex patterns. However, because trees are added sequentially and are highly correlated, the model can overfit and exhibit high variance, especially on noisy data. The learning rate controls how much each tree contributes, slowing down learning to reduce overfitting and help manage variance

## Q11. Key hyperparameters

The key hyperparameters of Gradient Boosting control bias-variance tradeoff:
1. n_estimators: number of trees; More trees improve fit but can increase variance if not regularized
2. learning_rate: scales each tree's contribution; smaller values slow learning and reduce overfitting, requiring more trees
3. subsample: row sampling; values < 1 introduce randomness, reduce variance, and improve generalization
4. early_stopping: stops training when validation loss stops improving, preventing overfitting
5. Tree complexity (max_depth, min_samples_leaf, min_samples_split): controls individual tree bias; shallower trees increase bias but reduce variance

## Q12. Role of learning rate

The learning rate controls how much each new tree contributes to the overall model
- A large learning rate makes big updates:
    - Faster training
    - Higher risk of overfitting
- A small learning rate makes small, incremental updates:
    - Slower training
    - Better generalization
    - Requires more trees

Hence, the `learning rate` acts as a shrinkage parameter that slows down boosting, improving generalization at the cost of requiring more trees

## Q13. Data Requirements for Gradient Boosting

- Gradient Boosting does not require feature scaling since splits depend on feature ordering, not magnitude
- Classical sklearn implementations require missing values to be imputed, while XGBoost, LightGBM, and CatBoost handle NaNs natively
- Categorical features must be encoded for sklearn, whereas CatBoost and LightGBM support native handling
- Gradient Boosting works best on reasonably large, clean datasets. It is more sensitive to noise and extreme outliers than Random Forests because errors are corrected sequentially
- Because Gradient Boosting optimizes a global loss sequentially, class imbalance must be handled explicitly or the model will bias toward the majority class

## Q14. Suitable Metrics for Gradient Boosting

- For classification:
    - Suitable metrics include Precision, Recall, and F1-score depending on business costs, especially under class imbalance
    - ROC-AUC is used when ranking quality matters independent of a threshold
    - PR-AUC is preferred for rare-event problems
    - If it outputs probabilities, Log-loss is used to evaluate probability calibration
- For regression:
    - MAE, MSE, and RMSE are used to measure prediction error magnitude, with RMSE penalizing large errors more

## Q15. Failure Modes of Gradient Boosting

- Gradient Boosting performs poorly on very high-dimensional sparse data, noisy datasets
- It also struggles with heavily imbalanced datasets without class weighting or resampling
- In regression, it cannot extrapolate beyond the range of the training targets
- Sensitive to poorly tuned hyperparameters; bad learning rate / depth can severely hurt performance
- With many trees, it can also become memory-heavy and slow at inference, making it unsuitable for low-latency or resource-constrained systems

## Q16. What to try next if Gradient Boosting underperforms

1. First, I would tune learning rate, number of trees, tree depth, subsampling, and class weights, and verify feature quality
2. If performance is still limited, I would move to optimized boosting libraries like XGBoost, LightGBM, or CatBoost for better regularization and efficiency
3. If non-tree models are needed, I would try kernel SVMs for small datasets with strong non-linearity, and neural networks for very large and highly complex problems