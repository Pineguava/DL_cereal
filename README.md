# Cereal Rating Prediction with TensorFlow & Keras

A deep learning regression project that predicts breakfast cereal ratings from nutritional information and product metadata using TensorFlow / Keras, built on the `cereal.csv` dataset.

## Dataset

- **Source:** `cereal.csv` (`dl-course-data` / *80 Cereals* on Kaggle)
- **Size:** ~77 cereals, 16 columns
- **Target:** `rating` (continuous Consumer Reports rating score)
- **Features:**
  - **Numeric:** `calories`, `protein`, `fat`, `sodium`, `fiber`, `carbo`, `sugars`, `potass`, `vitamins`, `shelf`, `weight`, `cups`
  - **Categorical:** `mfr` (manufacturer), `type` (Cold `C` / Hot `H`)

## Methodology


1. **Train/Test Split:** 80/20 split (`random_state=42`) performed *before* fitting preprocessors to prevent data leakage.
2. **Preprocessing:**
   - Numeric features: Median imputation (`SimpleImputer`) + standardization (`StandardScaler`).
   - Categorical features (`mfr`, `type`): One-hot encoding (`OneHotEncoder` / `pd.get_dummies`).
3. **Model Architecture:** Compact Multi-Layer Perceptron (MLP) designed for small tabular datasets:
   - `Input` layer
   - `Dense(64, activation="relu")` + `Dropout(0.2)`
   - `Dense(32, activation="relu")`
   - `Dense(1)` (linear output for regression, bias initialized to training target mean)
5. **Training:**
   - Optimizer: `Adam(learning_rate=0.01)`
   - Loss: `mae` (Mean Absolute Error)
   - Metric: `mse`
   - Regularization: `EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)`

## Evaluation Metrics

Because `rating` is a continuous target (regression), performance is evaluated using:

- **MAE (Mean Absolute Error):** Average absolute difference in rating points.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily.
- **$R^2$ Score:** Proportion of variance in `rating` explained by the model.


### Results *(update with your run output)*

| Metric | Score |
| :--- | :--- |
| Test MAE | `3.302` |
| Test RMSE | `4.226` |
| Test $R^2$ | `0.919` |
