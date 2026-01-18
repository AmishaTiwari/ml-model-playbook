## Q1. When to use / when not to use Decision Trees

**When to use:**
- Data has non-linear relationships and feature interactions
- Model interpretability is required
- When we want a model that works well without feature scaling and can handle mixed data types

**When not to use:**
- When the data is very high-dimensional and sparse
- When strong generalization and stability are required, since trees easily overfit and are sensitive to small data changes

## Q2. Why trees easily overfit and are sensitive to small data changes?

Because trees make hard, greedy splits. At each node:
- The tree picks the best feature and threshold based on current data which might result in perfectly fitting training data
- A tiny change in data can change which split looks "best" which changes the entire subtree structure making the model highly sensitive to data and therefore high variance

## Q3. Core intuition of Decision Tree

A Decision Tree recursively partitions the feature space into regions by making greedy splits that maximize impurity or variance reduction. Each split makes the data more homogeneous, and predictions are made based on the majority class or mean value in each final region.

## Q4. Loss function and Optimization used in Decision Tree

- Decision Trees do not optimize a global loss, instead they use a greedy strategy
- At each node, they choose the split that maximizes impurity reduction
	- For classification, common criteria are Information Gain (entropy reduction) and Gini impurity reduction
	- For regression, the split is chosen to minimize MSE or equivalently maximize variance reduction

## Q5. Assumptions of the Decision Tree

1. Decision Trees make very few assumptions about the data. They do not assume linearity, normality, or feature independence, and can model complex non-linear relationships and interactions
2. They are insensitive to feature scaling as well
3. However, they are sensitive to data imbalance, which can bias splits toward the majority class, and multicollinearity mainly affects feature importance rather than the decision structure

## Q6. What are parametric assumptions?

Parametric assumption means you assume a fixed functional form for the relationship between input and output. Example:
- Logistic Regression assumes log-odds is linear in features

        log(p/1-p) = wTx

- Linear Regression assumes output is linear in features

        y = wTx + b

However, DTs make no such parametric assumptions. They don't assume linearity, normality, any equation form, and instead discover structure purely from data splits.

So, Parametric models assume a predefined mathematical form for the relationship between features and target. Decision Trees are non-parametric, and they do not assume any specific functional form and learn structure directly from data

## Q7. Where does Decision Tree lie on the bias–variance spectrum? Why?

Decision Tree is a low-bias, high-variance model because trees make hard, greedy splits. At each node:
- The tree picks the best feature and threshold based on current data which allows it to overfit the training data
- A tiny change in data can change which split looks "best" which changes the entire subtree structure resulting in high variance model

## Q8. Key Hyperparameters of a Decision Tree

The key hyperparameters are:
1. max_depth: Maximum depth of tree (Primary control knob)
    - ↓ depth → higher bias, lower variance (simpler tree) (inc underfitting)
    - ↑ depth → lower bias, higher variance (complex, memorizes) (inc overfitting)
2. min_samples_leaf: Min samples in a leaf (Prevents outlier memorization)
    - ↑ value → higher bias, lower variance (forces smoothing) 
    - ↓ value → lower bias, higher variance (allows tiny leaves)
3. min_samples_split: Min samples to split a node (Stops splits on small noisy nodes)
    - ↑ value → higher bias, lower variance 
    - ↓ value → lower bias, higher variance 
4. ccp_alpha: Cost-complexity pruning strength (post-pruning control)
    - ↑ alpha → higher bias, lower variance (prunes more)
    - ↓ alpha → lower bias, higher variance
5. min_impurity_decrease: Required impurity reduction to split
    - ↑ → fewer splits → higher bias, lower variance
    - ↓ → more splits → lower bias, higher variance
6. class_weight: Handles class imbalance by penalizing mistakes on the minority class more heavily

## Q9. Data Requirements for Decision Trees

- Decision Trees do not require feature scaling since they split based on ordering, not magnitude
- Some tree implementations like XGBoost, LightGBM, and CatBoost can handle missing values natively by learning split directions for NaNs, but scikit-learn trees require missing values to be imputed
- They work well with mixed feature types (numerical + categorical after encoding) and can model non-linear relationships and interactions naturally. Categorical features must be encoded for sklearn, while CatBoost and LightGBM support native categorical handling
- Trees need sufficient data per leaf to avoid overfitting, and class imbalance should be handled using class weights or resampling
- For classification trees, outliers usually have limited impact since splits depend on order, but for regression trees, outliers must be treated because they can heavily skew mean predictions and split decisions

## Q10. Suitable Metrics for Decision Trees

- For classification trees:
    - Suitable metrics include Precision, Recall, and F1-score depending on business costs, especially under class imbalance
    - AUC is used when we care about ranking quality independent of a threshold
    - If the tree outputs probabilities, Log-loss is used to evaluate probability calibration
- For regression trees:
    - Common metrics are MSE, RMSE, and MAE, which measure prediction error magnitude

## Q11. Failure Modes of Decision Trees

- Decision Trees fail when the data is high-dimensional and sparse, where meaningful splits are hard to find
- They perform poorly on heavily imbalanced data without class weighting
- They also overfit easily on small or noisy datasets because they are high-variance models
- Small changes in data can lead to completely different tree structures, making them unstable

## Q12. What to try next when a Decision Tree underperforms

1. First, I would tune tree hyperparameters like max depth, min samples per leaf, and pruning strength to control overfitting
2. If a single tree still underperforms, I would move to tree ensembles like Random Forests or Gradient Boosting, which reduce variance and give much better generalization
3. If non-tree models are needed, I would try kernel SVMs for small datasets with strong non-linearity, and neural networks for very large and highly complex problems