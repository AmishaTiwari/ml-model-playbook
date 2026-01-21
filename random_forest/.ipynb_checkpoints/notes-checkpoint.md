## Q1. When to use / when not use Random Forests

**When to use:**
- Data has non-linear relationships and feature interactions, and we want strong performance on tabular data with minimal feature engineering
- When we want a model that works well without feature scaling and can handle mixed data types
- It is a strong default model when single trees overfit because ensembling many trees reduces variance, making it robust to noise and improving generalization

**When not to use:**
- When the data is extremely high-dimensional and sparse
- When model size or inference latency is critical because predictions require traversing many trees
- When full model interpretability is required, since Random Forests are much less transparent than single trees

## Q2. Core Intuition

Random Forest builds many decision trees on bootstrapped samples of the data and uses random subsets of features at each split. Each tree is a high-variance, low-bias model, and averaging their predictions reduces variance while keeping bias roughly the same, resulting in a more stable and better-generalizing model

## Q3. Loss / Split Criterion in Random Forests

Random Forest does not optimize a single global loss. It trains multiple decision trees independently on bootstrapped samples of the data. Each tree uses a greedy algorithm to choose splits that maximize impurity reduction (for classification) or variance reduction (for regression). The forest then aggregates their predictions by averaging or majority voting

## Q4. Assumptions

1. Random Forest makes very few assumptions about the data. It does not assume linearity, normality, or feature independence, and is insensitive to feature scaling
2. It assumes samples are independent and identically distributed
3. Its effectiveness relies on trees being diverse and weakly correlated, which is achieved through bootstrapping and random feature selection
4. While it is more stable than a single tree, it still requires handling class imbalance explicitly, and multicollinearity mainly affects feature importance rather than prediction accuracy

## Q5. Explain i.i.d. assumption held by Random Forest

Independent and identically distributed (i.i.d.) means:
1. Independent:
    - Each data point should not depend on another. Example where this breaks:
        - Time series (today depends on yesterday)
        - User sessions (multiple rows from same user)
        - Grouped data (multiple records from same hospital, same customer, etc.)
    - If independence is violated, the model may show optimistic validation performance but generalizes poorly in deployment
2. Identically distributed:
    - All samples should come from the same underlying data distribution which means:
        - Train and test data should represent the same population
        - No major data drift
        - No covariate shift
    - If violated, the model works fine training, but fails badly in production

This assumption is required for bootstrapping to be valid as bootstrapping randomly resamples the dataset assuming each point is an independent draw, and if samples are dependent, bootstrapping breaks statistically

## Q6. Bias/variance

Random Forest has low bias and much lower variance than a single Decision Tree. The bias stays low because each tree is a low-bias model. The variance is reduced because averaging many decorrelated trees cancels out individual overfittings, leading to more stable and better-generalizing predictions

## Q7. Key hyperparameters

The key hyperparameters of Random Forest control variance reduction and individual tree complexity:
1. n_estimators: number of trees. More trees reduce variance and stabilize predictions, at the cost of higher computation
2. bootstrap / max_samples: control row sampling. Bootstrapping creates diverse trees and is critical for variance reduction
3. max_features: number of features considered at each split. Lower values increase randomness and reduce correlation between trees, lowering variance but slightly increasing bias
4. Tree-level parameters like `max_depth`, `min_samples_leaf`, and `min_samples_split` control overfitting within each tree
5. class_weight: handles class imbalance by penalizing mistakes on the minority class more heavily

## Q8. Data Requirements for Random Forest

- Random Forest does not require feature scaling since splits depend on feature ordering, not magnitude
- Some implementations like XGBoost, LightGBM, and CatBoost can handle missing values natively by learning split directions for NaNs, whereas scikit-learn requires imputation
- It works well with numerical and categorical data, though categorical variables must be encoded in scikit-learn, while CatBoost and LightGBM support native categorical handling
- Random Forest needs a reasonably large dataset so that bootstrapping creates diverse trees
- It is more robust to outliers than linear models, but extreme outliers can still affect split quality
- Class imbalance must still be handled using class weights or resampling, since RF can otherwise become biased toward the majority class

## Q9. Decision Trees and Random Forests are said to be robust to outliers. Then when do you actually handle outliers, and how do you decide whether they are extreme or not?

For classification trees/RFs, I usually keep outliers unless they are errors, because splits depend on order, not magnitude.If outliers represent real but rare behavior, I usually keep them. If they are due to data issues like wrong units, typos, or impossible values, I remove or fix them

For regression trees/RFs, I usually handle outliers, because trees predict by averaging and extreme values can skew predictions. This is why winsorization or clipping is often used for regression trees

Extreme outliers can be detected using:
- Domain rules (e.g., negative age)
- Statistical checks like IQR, percentile cuts (e.g., above 99.9th percentile)
- Visual tools like boxplots and histograms

## Q10. Suitable Metrics for Random Forest

- For classification:
    - Suitable metrics include Precision, Recall, and F1-score depending on business costs, 
	especially under class imbalance
    - AUC is used when we care about ranking quality independent of a threshold
    - If it outputs probabilities, Log-loss is used to evaluate probability calibration
- For regression:
    - Common metrics are MSE, RMSE, and MAE, which measure prediction error magnitude

## Q11. Failure Modes of Random Forest

- Random Forest performs poorly on very high-dimensional and sparse data where meaningful splits are hard to find
- It also struggles with heavily imbalanced datasets without class weighting or resampling
- In regression, it cannot extrapolate beyond the range of the training targets
- Additionally, Random Forests can become heavy in memory and slow at inference when many trees are used, which makes them unsuitable for low-latency or resource-constrained environments

## Q12. What to try next

1. First, I would tune Random Forest hyperparameters like number of trees, max depth, max features, and min samples per leaf. I would also check class imbalance handling and feature quality
2. If RF still underperforms, the next step is usually Gradient Boosting models like XGBoost, LightGBM, or CatBoost, which can capture more complex patterns and reduce bias
3. If non-tree models are needed, I would try kernel SVMs for small datasets with strong non-linearity, and neural networks for very large and highly complex problems

## Q13. What is Out-of-Bag (OOB) validation in Random Forest?

Out-of-Bag validation is an internal validation method in Random Forest that uses the samples not selected in a tree’s bootstrap sample as its validation set. Since each tree is trained on about 63% of the data, the remaining 37% can be used to estimate generalization error without a separate validation split

## Q14. Why is OOB validation useful?

OOB validation provides an unbiased estimate of model performance without needing a separate validation set or cross-validation. It saves data, reduces training cost, and is especially useful when data is limited

However, OOB works well when trees are sufficiently uncorrelated, otherwise its estimate can be slightly optimistic

## Q15. Why OOB works well when trees are sufficiently uncorrelated?

OOB validation assumes trees are weakly correlated. If trees are highly correlated, then even the OOB trees behave similarly to the trees that saw the sample, which makes the OOB error slightly optimistic compared to true generalization error

Random Forest usually avoids correlation and keep OOB reliable by using
- Bootstrapping (different samples per tree)
- Random feature selection at each split

But if
- `max_features` is too large (almost all features used)
- Trees are very deep and similar
- Dataset is small

then trees become more correlated leading to the OOB error slightly optimistic compared to true generalization error

## Q16. How is OOB error calculated in Random Forest?

- Each tree in a Random Forest is trained on a bootstrap sample, so about 37% of the data is left out for that tree and becomes its Out-of-Bag (OOB) set
- For each data point x_i, we collect predictions only from the trees that did not see x_i during training. These trees form the OOB ensemble for that point
- The predictions are then aggregated:
    - For classification: majority voting
    - For regression: average prediction
- The error is computed for each point using its OOB prediction, and the final OOB error is obtained by averaging over all data points 