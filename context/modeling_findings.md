# Modeling Findings — Outcome Prediction Signal Audit

## Summary

The current dataset is not suitable for reliable clinical outcome prediction.
The MLOps pipeline can still be developed end-to-end, but model predictions
must be treated as a technical demo artifact, not clinically meaningful output.

## Benchmark Results

Training benchmark:

```bash
uv run python src/training/train.py
```

Configuration:

- Models: LightGBM, XGBoost, CatBoost
- Tuning: Optuna, 100 trials per model
- Selection metric: validation F1-score macro
- Final test evaluation: held-out test split only after model selection

Best model from the 100-trial run:

| Metric | Validation | Test |
|---|---:|---:|
| F1-score macro | 0.3668 | 0.3139 |
| Accuracy | 0.3727 | 0.3152 |
| AUC-ROC OvR | 0.5095 | 0.4994 |

Per-model validation F1-score macro:

| Model | Best validation F1 macro |
|---|---:|
| LightGBM | 0.3668 |
| XGBoost | 0.3639 |
| CatBoost | 0.3564 |

These values are near random-baseline performance for a balanced 3-class
classification problem.

## Overfitting Evidence

The best model can nearly memorize the training data but does not generalize:

| Split | F1-score macro | Accuracy | AUC-ROC OvR |
|---|---:|---:|---:|
| Train | 0.9844 | 0.9844 | 0.9993 |
| Validation | 0.3668 | 0.3727 | 0.5095 |
| Test | 0.3139 | 0.3152 | 0.4994 |

This indicates that additional hyperparameter search is unlikely to solve the
problem. More trials mainly fit validation noise.

## Statistical Signal Checks

Chi-square tests between clinically important categorical predictors and
`Outcome` found no statistically significant association:

| Feature | p-value | Interpretation |
|---|---:|---|
| Meropenem | 0.3592 | No signal |
| Colistin | 0.9604 | No signal |
| Resistance_Genes | 0.9099 | No signal |
| Specimen_Type | 0.7503 | No signal |

Feature-target mutual information was also approximately zero across all
available model features.

## Conclusion

The target variable `Outcome` appears effectively independent of the available
features. This may be because the dataset is synthetic or randomly generated.

The project should continue as an MLOps demonstration:

- Keep the ingestion, preprocessing, DVC, training, MLflow, serving, and
  monitoring pipeline.
- Do not claim clinical usefulness for the trained model.
- Document model limitations clearly in project docs and serving API metadata.
- Treat the registered model as a pipeline artifact used to demonstrate
  reproducible ML operations, not as a decision-support model.
