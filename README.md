# TARP ML Module: Intervention-Outcome Classifier

Machine-learning and control-layer prototype for the **Risk-Constrained Emergency Control System Using Reversible Micro-Interventions**.

The classifier predicts whether a candidate micro-intervention will be **Protective, Harmful, Ineffective or Inconsistent** *before* it is physically tested. It is the first safety layer only. Escalation to a full action still requires the measured response to confirm protection.

## Run
```bash
pip install -r requirements.txt
mkdir -p data outputs
python train_eval.py      # generates data, trains 3 models, evaluates, saves figures + model
python predict.py         # screens one example candidate with the saved model
```

## Files
| File | Purpose |
|---|---|
| `simulate.py` | Four-zone, source-informed dataset generator (5,000 records) and reference-label rules |
| `train_eval.py` | Validation checks, stratified 80:20 split, leakage exclusion, 3 classifiers, safety-gate tuning, safety-first model selection, control-layer analysis, figures |
| `train_eval_utils.py` | Safety-gated decision rule shared by training and inference |
| `predict.py` | Screens a single candidate: hard protected-zone constraint, then gated model decision |
| `outputs/` | `model_comparison.csv`, `results_summary.json`, `feature_importance.csv`, `run_log.txt`, figures, `selected_model.pkl` |

## Method
- **Labels:** Inconsistent if sensor fault or command achievement < 45%; else Harmful if protected-zone rise > 6, or airflow toward the zone and rise > 1; else Protective if fall > 5; else Ineffective.
- **Leakage prevention:** `dC_protected`, `command_achievement_pct` and `outcome` are never model inputs.
- **Models:** Random Forest (balanced), Gradient Boosting, Logistic Regression (balanced, scaled). Same split for all.
- **Safety gate:** a candidate is labelled Protective only if P(Protective) >= tau. tau is tuned with 5-fold out-of-fold predictions on the *training* set to keep the false-safe rate <= 1%. The test set is never used for tuning.
- **Selection:** lowest test false-safe rate, then highest Harmful recall, then weighted F1.
- **Control layer:** predicted-class screening, early harmful detection below a 10-unit absolute threshold, rollback success (achievement >= 45% and delay <= 300 ms), outcome-based trust update (x1.04 / x0.95 / x0.85 / x0.59, clipped to 0-1).

## Scope
Results come from a controlled simulation with prototype thresholds. They validate the software decision logic, not a physical deployment. Seed = 42; every run is reproducible.
