# ML Model Playbook

This repository is a structured playbook for building, understanding, and applying core Machine Learning models and concepts.

It contains:

1. Model implementations:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - XGBoost

2. Core ML concepts:
   - Evaluation metrics
   - Feature engineering
   - ML pipelines and data leakage prevention

Repository structure:

Both model folders and ML concept folders contain:
- A Jupyter notebook with minimal, clean experiments
- A `notes.md` file with concise explanations:
  - When to use a method
  - Why it works
  - Assumptions and limitations
  - Failure modes and tradeoffs

In addition, the repository also contains:
- models/
    - Saved trained model artifacts (`.pkl` files) for demonstration and reproducibility
    - These simulate real-world model persistence and reuse
- data/
    - Datasets used across experiments
    - Shared across notebooks to enable consistent comparison between models and techniques

The goal of this repository is to:
- Build strong modeling intuition
- Practice industry-grade ML workflows
- Demonstrate correct experiment design and evaluation
- Serve as a reference for real-world ML problem solving