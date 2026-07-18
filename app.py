import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MAX_PLOT_ROWS = 5000
MAX_SHAP_ROWS = 250

st.set_page_config(
    page_title="WISE",
    page_icon="🧭",
    layout="wide",
)

st.title("WISE")
st.subheader("Workflow for Interpretable Scientific Evaluation")
st.write(
    "Upload a dataset and describe your scientific goal. WISE examines the "
    "data structure, visualizes key patterns, compares transparent and nonlinear "
    "models, and recommends an interpretable scientific workflow."
)


@st.cache_data
def make_demo_data(n_rows: int = 720) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.date_range("2022-01-01", periods=n_rows, freq="D")
    latitude = 37.0 + rng.uniform(0, 3.5, n_rows)
    longitude = -79.5 + rng.uniform(0, 3.5, n_rows)
    temperature = 19 + 10 * np.sin(np.arange(n_rows) * 2 * np.pi / 365) + rng.normal(0, 2, n_rows)
    precipitation = rng.gamma(2.0, 2.0, n_rows)
    soil_moisture = np.clip(0.28 + 0.018 * precipitation - 0.004 * temperature + rng.normal(0, 0.035, n_rows), 0.05, 0.65)
    vegetation_index = np.clip(
        0.45 + 0.025 * temperature + 0.55 * soil_moisture + rng.normal(0, 0.08, n_rows),
        0.05,
        0.95,
    )
    elevation = 80 + 90 * (latitude - latitude.min()) + rng.normal(0, 35, n_rows)
    management = rng.integers(0, 3, n_rows)
    yield_value = (
        3.2
        + 4.0 * vegetation_index
        + 2.3 * soil_moisture
        - 0.014 * np.maximum(temperature - 29, 0) ** 2
        - 0.0018 * elevation
        + 0.25 * management
        + rng.normal(0, 0.38, n_rows)
    )
    return pd.DataFrame(
        {
            "date": dates,
            "latitude": latitude,
            "longitude": longitude,
            "temperature": temperature,
            "precipitation": precipitation,
            "soil_moisture": soil_moisture,
            "vegetation_index": vegetation_index,
            "elevation": elevation,
            "management_code": management,
            "yield": yield_value,
        }
    )


def infer_task_type(y: pd.Series) -> str:
    non_missing = y.dropna()
    if pd.api.types.is_numeric_dtype(non_missing):
        unique_count = non_missing.nunique()
        threshold = max(12, min(30, int(len(non_missing) * 0.05)))
        return "Regression" if unique_count > threshold else "Classification"
    return "Classification"


def id_like_columns(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    suggestions = []
    for col in df.columns:
        if col in excluded:
            continue
        name = col.lower()
        ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        if (
            name == "id"
            or name.endswith("_id")
            or name.startswith("id_")
            or "identifier" in name
            or (ratio > 0.95 and not pd.api.types.is_numeric_dtype(df[col]))
        ):
            suggestions.append(col)
    return suggestions


def complexity_assessment(n_samples: int, n_features: int, task_type: str) -> tuple[str, str]:
    ratio = n_samples / max(n_features, 1)
    if n_samples < 200 or ratio < 5:
        level = "Low"
        guidance = (
            "Use transparent or strongly regularized models first. Complex deep-learning "
            "models are not justified without additional data or a specialized data modality."
        )
    elif n_samples < 2000 or ratio < 20:
        level = "Low to moderate"
        guidance = (
            "Compare a transparent baseline with restrained nonlinear models such as "
            "Random Forest, gradient boosting, or Explainable Boosting Machines."
        )
    elif n_samples < 10000:
        level = "Moderate"
        guidance = (
            "Moderately flexible models are reasonable, but validation structure and "
            "interpretability should still determine whether added complexity is worthwhile."
        )
    else:
        level = "Moderate to high"
        guidance = (
            "Higher-capacity models may be considered when supported by validation results, "
            "data modality, and the scientific objective—not sample size alone."
        )
    if task_type == "Classification":
        guidance += " Check class balance before interpreting accuracy."
    return level, guidance


def select_top_features(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task_type: str,
    max_features: int,
) -> tuple[list[str], pd.Series, str]:
    X = model_df[feature_cols]
    y = model_df[target_col]

    if task_type == "Regression":
        correlations = {}
        for col in feature_cols:
            valid = pd.concat(
                [
                    pd.to_numeric(X[col], errors="coerce"),
                    pd.to_numeric(y, errors="coerce"),
                ],
                axis=1,
            ).dropna()
            if len(valid) >= 3 and valid.iloc[:, 0].nunique() > 1:
                correlations[col] = valid.iloc[:, 0].corr(valid.iloc[:, 1])
            else:
                correlations[col] = np.nan
        scores = pd.Series(correlations, dtype=float).abs().fillna(0).sort_values(ascending=False)
        label = "Absolute correlation with target"
    else:
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)
        encoded_y = LabelEncoder().fit_transform(y.astype(str))
        mi = mutual_info_classif(X_imp, encoded_y, random_state=RANDOM_STATE)
        scores = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
        label = "Mutual information with target"

    selected = scores.head(max_features).index.tolist()
    return selected, scores, label


def prepare_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task_type: str,
    datetime_col: str | None,
    group_col: str | None,
):
    required = feature_cols + [target_col]
    for optional_col in [datetime_col, group_col]:
        if optional_col and optional_col not in required:
            required.append(optional_col)

    work = df[required].copy()
    work = work.dropna(subset=[target_col])

    if task_type == "Regression":
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
        work = work.dropna(subset=[target_col])
    else:
        work[target_col] = work[target_col].astype(str)

    X = work[feature_cols]
    y = work[target_col]

    if datetime_col:
        parsed_dates = pd.to_datetime(work[datetime_col], errors="coerce")
        valid = parsed_dates.notna()
        work = work.loc[valid].copy()
        work["_wise_date"] = parsed_dates.loc[valid]
        work = work.sort_values("_wise_date")
        X = work[feature_cols]
        y = work[target_col]
        cut = int(len(work) * 0.8)
        cut = min(max(cut, 2), len(work) - 1)
        train_idx = np.arange(cut)
        test_idx = np.arange(cut, len(work))
        strategy = "Temporal holdout: earliest 80% for training, latest 20% for testing"
        return (
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
            strategy,
        )

    if group_col and work[group_col].nunique(dropna=True) >= 3:
        groups = work[group_col].fillna("__MISSING_GROUP__").astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        strategy = f"Group holdout by '{group_col}': groups do not overlap between train and test"
        return (
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
            strategy,
        )

    stratify = None
    if task_type == "Classification":
        counts = y.value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    strategy = "Random 80/20 holdout"
    return X_train, X_test, y_train, y_test, strategy


def fit_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
):
    if task_type == "Regression":
        baseline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
        forest = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=350,
                        min_samples_leaf=max(1, int(round(len(X_train) * 0.01))),
                        max_features=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    else:
        baseline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        forest = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        min_samples_leaf=max(1, int(round(len(X_train) * 0.01))),
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    baseline.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    baseline_pred = baseline.predict(X_test)
    forest_pred = forest.predict(X_test)

    if task_type == "Regression":
        baseline_metrics = {
            "R²": r2_score(y_test, baseline_pred),
            "RMSE": root_mean_squared_error(y_test, baseline_pred),
            "MAE": mean_absolute_error(y_test, baseline_pred),
        }
        forest_metrics = {
            "R²": r2_score(y_test, forest_pred),
            "RMSE": root_mean_squared_error(y_test, forest_pred),
            "MAE": mean_absolute_error(y_test, forest_pred),
        }
        scoring = "r2"
    else:
        baseline_metrics = {
            "Accuracy": accuracy_score(y_test, baseline_pred),
            "F1 weighted": f1_score(y_test, baseline_pred, average="weighted"),
        }
        forest_metrics = {
            "Accuracy": accuracy_score(y_test, forest_pred),
            "F1 weighted": f1_score(y_test, forest_pred, average="weighted"),
        }
        if y_test.nunique() == 2:
            try:
                classes = list(baseline.named_steps["model"].classes_)
                positive_class = classes[1]
                baseline_prob = baseline.predict_proba(X_test)[:, 1]
                forest_prob = forest.predict_proba(X_test)[:, 1]
                binary_y = (y_test == positive_class).astype(int)
                baseline_metrics["ROC-AUC"] = roc_auc_score(binary_y, baseline_prob)
                forest_metrics["ROC-AUC"] = roc_auc_score(binary_y, forest_prob)
            except Exception:
                pass
        scoring = "f1_weighted"

    importance = permutation_importance(
        forest,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_df = (
        pd.DataFrame(
            {
                "Feature": X_test.columns,
                "Importance": importance.importances_mean,
                "Std": importance.importances_std,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "baseline": baseline,
        "forest": forest,
        "baseline_pred": baseline_pred,
        "forest_pred": forest_pred,
        "baseline_metrics": baseline_metrics,
        "forest_metrics": forest_metrics,
        "importance": importance_df,
    }


def calculate_shap(
    forest_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    task_type: str,
):
    imputer = forest_pipeline.named_steps["imputer"]
    model = forest_pipeline.named_steps["model"]

    X_train_imp = pd.DataFrame(
        imputer.transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    if len(X_test_imp) > MAX_SHAP_ROWS:
        X_sample = X_test_imp.sample(MAX_SHAP_ROWS, random_state=RANDOM_STATE)
    else:
        X_sample = X_test_imp.copy()

    background = X_train_imp.sample(
        min(200, len(X_train_imp)),
        random_state=RANDOM_STATE,
    )

    explainer = shap.TreeExplainer(model, data=background)
    explanation = explainer(X_sample)

    if task_type == "Classification" and explanation.values.ndim == 3:
        class_index = 1 if explanation.values.shape[2] > 1 else 0
        base_values = explanation.base_values
        if np.ndim(base_values) == 2:
            base_values = base_values[:, class_index]
        explanation = shap.Explanation(
            values=explanation.values[:, :, class_index],
            base_values=base_values,
            data=explanation.data,
            feature_names=list(X_sample.columns),
        )

    mean_abs = pd.Series(
        np.abs(explanation.values).mean(axis=0),
        index=X_sample.columns,
    ).sort_values(ascending=False)

    return explanation, mean_abs


def target_plot(df: pd.DataFrame, target_col: str, task_type: str):
    plot_df = df[[target_col]].dropna().copy()
    if len(plot_df) > MAX_PLOT_ROWS:
        plot_df = plot_df.sample(MAX_PLOT_ROWS, random_state=RANDOM_STATE)

    if task_type == "Regression":
        fig = px.histogram(
            plot_df,
            x=target_col,
            marginal="box",
            nbins=35,
            title=f"Distribution of {target_col}",
        )
    else:
        counts = plot_df[target_col].astype(str).value_counts().reset_index()
        counts.columns = [target_col, "Count"]
        fig = px.bar(
            counts,
            x=target_col,
            y="Count",
            title=f"Class distribution of {target_col}",
        )
    fig.update_layout(height=480)
    return fig


def temporal_plot(df: pd.DataFrame, target_col: str, datetime_col: str, task_type: str):
    temp = df[[datetime_col, target_col]].copy()
    temp[datetime_col] = pd.to_datetime(temp[datetime_col], errors="coerce")
    temp = temp.dropna()
    if temp.empty:
        return None

    span_days = max((temp[datetime_col].max() - temp[datetime_col].min()).days, 1)
    if span_days > 730:
        frequency, label = "YS", "Yearly"
    elif span_days > 120:
        frequency, label = "MS", "Monthly"
    elif span_days > 45:
        frequency, label = "W", "Weekly"
    else:
        frequency, label = "D", "Daily"

    temp = temp.set_index(datetime_col)

    if task_type == "Regression":
        result = temp[target_col].resample(frequency).mean().dropna().reset_index()
        fig = px.line(
            result,
            x=datetime_col,
            y=target_col,
            markers=True,
            title=f"{label} mean {target_col}",
        )
    else:
        temp[target_col] = temp[target_col].astype(str)
        result = (
            temp.groupby([pd.Grouper(freq=frequency), target_col])
            .size()
            .rename("Count")
            .reset_index()
        )
        fig = px.line(
            result,
            x=datetime_col,
            y="Count",
            color=target_col,
            markers=True,
            title=f"{label} class counts",
        )
    fig.update_layout(height=480)
    return fig


def geospatial_plot(
    df: pd.DataFrame,
    target_col: str,
    latitude_col: str,
    longitude_col: str,
    task_type: str,
    datetime_col: str | None,
):
    cols = [latitude_col, longitude_col, target_col]
    if datetime_col:
        cols.append(datetime_col)

    geo = df[cols].copy()
    geo[latitude_col] = pd.to_numeric(geo[latitude_col], errors="coerce")
    geo[longitude_col] = pd.to_numeric(geo[longitude_col], errors="coerce")
    geo = geo.dropna(subset=[latitude_col, longitude_col, target_col])

    if geo.empty:
        return None, geo, "No valid coordinates"

    if datetime_col and task_type == "Regression":
        geo[target_col] = pd.to_numeric(geo[target_col], errors="coerce")
        geo = (
            geo.groupby([latitude_col, longitude_col], as_index=False)[target_col]
            .mean()
            .dropna()
        )
        title = f"Average {target_col} by location"
    elif datetime_col:
        geo[datetime_col] = pd.to_datetime(geo[datetime_col], errors="coerce")
        geo = geo.sort_values(datetime_col).drop_duplicates(
            [latitude_col, longitude_col],
            keep="last",
        )
        title = f"Latest {target_col} by location"
    else:
        title = f"Spatial distribution of {target_col}"

    if len(geo) > MAX_PLOT_ROWS:
        geo = geo.sample(MAX_PLOT_ROWS, random_state=RANDOM_STATE)

    fig = px.scatter_map(
        geo,
        lat=latitude_col,
        lon=longitude_col,
        color=target_col,
        hover_data=[target_col],
        zoom=4,
        height=520,
        map_style="open-street-map",
        title=title,
    )
    return fig, geo, title


def regression_scatter(y_true, y_pred, title: str):
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={"x": "Observed", "y": "Predicted"},
        title=title,
        trendline="ols",
    )
    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="1:1 line",
        )
    )
    fig.update_layout(height=480)
    return fig


def classification_matrix(y_true, y_pred, title: str):
    labels = sorted(pd.Series(y_true).astype(str).unique().tolist())
    matrix = confusion_matrix(
        pd.Series(y_true).astype(str),
        pd.Series(y_pred).astype(str),
        labels=labels,
    )
    fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=True,
        labels={"x": "Predicted", "y": "Observed", "color": "Count"},
        title=title,
        aspect="auto",
    )
    fig.update_layout(height=480)
    return fig


def recommendation_text(task_type: str, baseline_metrics: dict, forest_metrics: dict, complexity: str):
    if task_type == "Regression":
        improvement = forest_metrics["R²"] - baseline_metrics["R²"]
        if improvement >= 0.05:
            primary = "Random Forest"
            reason = (
                f"It improved test R² by {improvement:.3f} over the transparent linear baseline, "
                "suggesting meaningful nonlinear structure."
            )
        else:
            primary = "Multiple Linear Regression"
            reason = (
                f"Random Forest improved test R² by only {improvement:.3f}. The simpler model is "
                "currently preferable unless nonlinear behavior is scientifically important."
            )
    else:
        improvement = forest_metrics["F1 weighted"] - baseline_metrics["F1 weighted"]
        if improvement >= 0.03:
            primary = "Random Forest"
            reason = (
                f"It improved weighted F1 by {improvement:.3f} over logistic regression, "
                "suggesting useful nonlinear decision structure."
            )
        else:
            primary = "Logistic Regression"
            reason = (
                f"Random Forest improved weighted F1 by only {improvement:.3f}. The transparent "
                "baseline is currently the more defensible choice."
            )

    return primary, (
        f"**Recommended primary model: {primary}.** {reason} "
        f"The assessed model-complexity level is **{complexity}**. "
        "Use permutation importance and SHAP to explain predictive behavior, but do not interpret "
        "feature contributions as causal effects."
    )


def starter_code(
    task_type: str,
    target_col: str,
    selected_features: list[str],
    validation_strategy: str,
) -> str:
    feature_text = ",\n    ".join(repr(feature) for feature in selected_features)
    if task_type == "Regression":
        baseline_import = "from sklearn.linear_model import LinearRegression"
        forest_import = "from sklearn.ensemble import RandomForestRegressor"
        baseline_model = "LinearRegression()"
        forest_model = (
            "RandomForestRegressor(n_estimators=350, min_samples_leaf=2, "
            "random_state=42, n_jobs=-1)"
        )
        metric_import = "from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error"
        metric_lines = 'print("R2:", r2_score(y_test, predictions))\nprint("RMSE:", root_mean_squared_error(y_test, predictions))\nprint("MAE:", mean_absolute_error(y_test, predictions))'
    else:
        baseline_import = "from sklearn.linear_model import LogisticRegression"
        forest_import = "from sklearn.ensemble import RandomForestClassifier"
        baseline_model = "LogisticRegression(max_iter=3000, class_weight='balanced')"
        forest_model = (
            "RandomForestClassifier(n_estimators=350, min_samples_leaf=2, "
            "class_weight='balanced', random_state=42, n_jobs=-1)"
        )
        metric_import = "from sklearn.metrics import accuracy_score, f1_score"
        metric_lines = 'print("Accuracy:", accuracy_score(y_test, predictions))\nprint("Weighted F1:", f1_score(y_test, predictions, average="weighted"))'

    return f"""import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
{baseline_import}
{forest_import}
{metric_import}

df = pd.read_csv("your_dataset.csv")

features = [
    {feature_text}
]
target = {target_col!r}

X = df[features]
y = df[target]

# WISE validation recommendation:
# {validation_strategy}
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", {baseline_model}),
])

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", {forest_model}),
])

baseline.fit(X_train, y_train)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
{metric_lines}

importance = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42
)
importance_table = (
    pd.DataFrame({{
        "feature": features,
        "importance": importance.importances_mean,
    }})
    .sort_values("importance", ascending=False)
)
print(importance_table)
"""


def presentation_figure(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    datetime_col: str | None,
    latitude_col: str | None,
    longitude_col: str | None,
    selected_features: list[str],
    correlation_matrix: pd.DataFrame,
    y_test: pd.Series,
    results: dict,
    shap_importance: pd.Series,
    complexity: str,
    validation_strategy: str,
    recommendation: str,
):
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        f"WISE Scientific Model Assessment: {target_col}",
        fontsize=20,
        y=0.99,
    )

    axes[0, 0].axis("off")
    missing_total = int(df.isna().sum().sum())
    summary = (
        f"DATASET SUMMARY\n\n"
        f"Samples: {len(df):,}\n"
        f"Variables: {len(df.columns):,}\n"
        f"Selected predictors: {len(selected_features)}\n"
        f"Missing values: {missing_total:,}\n"
        f"Task: {task_type}\n"
        f"Complexity: {complexity}\n\n"
        f"Validation:\n{validation_strategy}"
    )
    axes[0, 0].text(0.02, 0.98, summary, va="top", fontsize=12, wrap=True)

    target_values = df[target_col].dropna()
    if task_type == "Regression":
        axes[0, 1].hist(pd.to_numeric(target_values, errors="coerce").dropna(), bins=30)
        axes[0, 1].set_xlabel(target_col)
        axes[0, 1].set_ylabel("Count")
    else:
        target_values.astype(str).value_counts().plot(kind="bar", ax=axes[0, 1])
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Target distribution")

    context_drawn = False
    if datetime_col:
        temp = df[[datetime_col, target_col]].copy()
        temp[datetime_col] = pd.to_datetime(temp[datetime_col], errors="coerce")
        temp = temp.dropna()
        if not temp.empty and task_type == "Regression":
            temp[target_col] = pd.to_numeric(temp[target_col], errors="coerce")
            temp = temp.dropna().set_index(datetime_col)[target_col].resample("MS").mean().dropna()
            axes[0, 2].plot(temp.index, temp.values)
            axes[0, 2].tick_params(axis="x", rotation=30)
            axes[0, 2].set_title("Temporal trend")
            axes[0, 2].set_ylabel(target_col)
            context_drawn = True

    if not context_drawn and latitude_col and longitude_col:
        geo = df[[latitude_col, longitude_col, target_col]].copy()
        geo[latitude_col] = pd.to_numeric(geo[latitude_col], errors="coerce")
        geo[longitude_col] = pd.to_numeric(geo[longitude_col], errors="coerce")
        geo = geo.dropna()
        if not geo.empty:
            color_values = (
                pd.to_numeric(geo[target_col], errors="coerce")
                if task_type == "Regression"
                else pd.Categorical(geo[target_col].astype(str)).codes
            )
            axes[0, 2].scatter(
                geo[longitude_col],
                geo[latitude_col],
                c=color_values,
                s=12,
                alpha=0.7,
            )
            axes[0, 2].set_xlabel("Longitude")
            axes[0, 2].set_ylabel("Latitude")
            axes[0, 2].set_title("Spatial pattern")
            context_drawn = True

    if not context_drawn and selected_features:
        top_feature = selected_features[0]
        if task_type == "Regression":
            axes[0, 2].scatter(df[top_feature], df[target_col], s=12, alpha=0.7)
            axes[0, 2].set_ylabel(target_col)
        axes[0, 2].set_xlabel(top_feature)
        axes[0, 2].set_title("Top feature relationship")

    if not correlation_matrix.empty:
        im = axes[1, 0].imshow(correlation_matrix.values, aspect="auto", vmin=-1, vmax=1)
        axes[1, 0].set_xticks(range(len(correlation_matrix.columns)))
        axes[1, 0].set_yticks(range(len(correlation_matrix.index)))
        axes[1, 0].set_xticklabels(correlation_matrix.columns, rotation=90, fontsize=8)
        axes[1, 0].set_yticklabels(correlation_matrix.index, fontsize=8)
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    axes[1, 0].set_title("Feature correlation")

    if task_type == "Regression":
        axes[1, 1].scatter(y_test, results["baseline_pred"], s=18, alpha=0.75)
        low = min(float(np.min(y_test)), float(np.min(results["baseline_pred"])))
        high = max(float(np.max(y_test)), float(np.max(results["baseline_pred"])))
        axes[1, 1].plot([low, high], [low, high])
        axes[1, 1].set_xlabel("Observed")
        axes[1, 1].set_ylabel("Predicted")
        axes[1, 1].set_title(f"Linear baseline: R²={results['baseline_metrics']['R²']:.2f}")
    else:
        labels = sorted(pd.Series(y_test).astype(str).unique())
        matrix = confusion_matrix(
            pd.Series(y_test).astype(str),
            pd.Series(results["baseline_pred"]).astype(str),
            labels=labels,
        )
        axes[1, 1].imshow(matrix, aspect="auto")
        axes[1, 1].set_xticks(range(len(labels)), labels, rotation=45)
        axes[1, 1].set_yticks(range(len(labels)), labels)
        axes[1, 1].set_xlabel("Predicted")
        axes[1, 1].set_ylabel("Observed")
        axes[1, 1].set_title(
            f"Logistic baseline: F1={results['baseline_metrics']['F1 weighted']:.2f}"
        )

    if task_type == "Regression":
        axes[1, 2].scatter(y_test, results["forest_pred"], s=18, alpha=0.75)
        low = min(float(np.min(y_test)), float(np.min(results["forest_pred"])))
        high = max(float(np.max(y_test)), float(np.max(results["forest_pred"])))
        axes[1, 2].plot([low, high], [low, high])
        axes[1, 2].set_xlabel("Observed")
        axes[1, 2].set_ylabel("Predicted")
        axes[1, 2].set_title(f"Random Forest: R²={results['forest_metrics']['R²']:.2f}")
    else:
        labels = sorted(pd.Series(y_test).astype(str).unique())
        matrix = confusion_matrix(
            pd.Series(y_test).astype(str),
            pd.Series(results["forest_pred"]).astype(str),
            labels=labels,
        )
        axes[1, 2].imshow(matrix, aspect="auto")
        axes[1, 2].set_xticks(range(len(labels)), labels, rotation=45)
        axes[1, 2].set_yticks(range(len(labels)), labels)
        axes[1, 2].set_xlabel("Predicted")
        axes[1, 2].set_ylabel("Observed")
        axes[1, 2].set_title(
            f"Random Forest: F1={results['forest_metrics']['F1 weighted']:.2f}"
        )

    importance = results["importance"].head(10).sort_values("Importance")
    axes[2, 0].barh(importance["Feature"], importance["Importance"])
    axes[2, 0].set_title("Permutation importance")
    axes[2, 0].set_xlabel("Performance decrease after permutation")

    shap_top = shap_importance.head(10).sort_values()
    axes[2, 1].barh(shap_top.index, shap_top.values)
    axes[2, 1].set_title("Mean |SHAP| contribution")
    axes[2, 1].set_xlabel("Mean absolute SHAP value")

    axes[2, 2].axis("off")
    recommendation_clean = recommendation.replace("**", "")
    note = (
        f"WISE RECOMMENDATION\n\n"
        f"{recommendation_clean}\n\n"
        "Interpretation note:\n"
        "Importance and SHAP explain model behavior, not causal effects."
    )
    axes[2, 2].text(0.02, 0.98, note, va="top", fontsize=11, wrap=True)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=220, bbox_inches="tight")
    output.seek(0)
    return fig, output.getvalue()


with st.sidebar:
    st.header("Analysis setup")
    use_demo = st.toggle(
        "Use built-in demo dataset",
        value=False,
        help="A regression dataset with temporal and geospatial variables.",
    )

uploaded_file = None
if use_demo:
    df = make_demo_data()
    st.info("Using the built-in scientific demo dataset.")
else:
    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded_file is None:
        st.caption(
            "Upload a CSV or enable the built-in demo dataset to begin. "
            "The demo is useful for testing all WISE visualizations."
        )
        st.stop()
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()

if df.empty or len(df.columns) < 2:
    st.error("The dataset must contain at least one target and one predictor column.")
    st.stop()

st.subheader("Research question")
default_goal = (
    "Predict crop yield and understand which environmental variables explain its variability."
    if use_demo
    else ""
)
research_goal = st.text_area(
    "What do you want to learn, predict, or explain?",
    value=default_goal,
    placeholder=(
        "Example: I want to predict crop yield and understand which environmental "
        "variables control yield variability."
    ),
)

columns = list(df.columns)
default_target = "yield" if use_demo and "yield" in columns else columns[-1]
target_index = columns.index(default_target)

setup_col1, setup_col2, setup_col3 = st.columns(3)

with setup_col1:
    target_col = st.selectbox("Target variable", columns, index=target_index)
    inferred = infer_task_type(df[target_col])
    task_choice = st.selectbox(
        "Task type",
        ["Auto", "Regression", "Classification"],
        index=0,
    )
    task_type = inferred if task_choice == "Auto" else task_choice

with setup_col2:
    optional_columns = ["None"] + columns
    default_date_index = (
        optional_columns.index("date")
        if use_demo and "date" in optional_columns
        else 0
    )
    datetime_choice = st.selectbox(
        "Datetime column (optional)",
        optional_columns,
        index=default_date_index,
    )
    datetime_col = None if datetime_choice == "None" else datetime_choice

    group_choice = st.selectbox(
        "Grouping column (optional)",
        optional_columns,
        index=0,
        help="Examples: farm, participant, site, hospital, or watershed.",
    )
    group_col = None if group_choice == "None" else group_choice

with setup_col3:
    default_lat_index = (
        optional_columns.index("latitude")
        if use_demo and "latitude" in optional_columns
        else 0
    )
    default_lon_index = (
        optional_columns.index("longitude")
        if use_demo and "longitude" in optional_columns
        else 0
    )
    latitude_choice = st.selectbox(
        "Latitude column (optional)",
        optional_columns,
        index=default_lat_index,
    )
    longitude_choice = st.selectbox(
        "Longitude column (optional)",
        optional_columns,
        index=default_lon_index,
    )
    latitude_col = None if latitude_choice == "None" else latitude_choice
    longitude_col = None if longitude_choice == "None" else longitude_choice

metadata_cols = {
    col
    for col in [target_col, datetime_col, group_col, latitude_col, longitude_col]
    if col
}
suggested_ids = id_like_columns(df, metadata_cols)

exclude_cols = st.multiselect(
    "Additional columns to exclude from modeling",
    options=[col for col in columns if col != target_col],
    default=suggested_ids,
    help="Identifiers, free text, and known leakage variables should usually be excluded.",
)

numeric_features = [
    col
    for col in columns
    if col not in metadata_cols
    and col not in exclude_cols
    and pd.api.types.is_numeric_dtype(df[col])
    and df[col].nunique(dropna=True) > 1
]

if target_col in numeric_features:
    numeric_features.remove(target_col)

if not numeric_features:
    st.error(
        "No usable numeric predictors remain. WISE's current MVP models numeric "
        "features; retain at least one numeric predictor."
    )
    st.stop()

max_available = max(1, min(20, len(numeric_features)))
default_max = min(10, max_available)
max_features = st.slider(
    "Maximum number of predictors for the initial models",
    min_value=1,
    max_value=max_available,
    value=default_max,
)

if task_type == "Regression" and not pd.api.types.is_numeric_dtype(df[target_col]):
    st.error("Regression requires a numeric target variable.")
    st.stop()

run_analysis = st.button(
    "Run WISE analysis",
    type="primary",
    use_container_width=True,
)

if run_analysis:
    st.session_state["wise_run"] = True

if not st.session_state.get("wise_run", False):
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.stop()

with st.spinner("WISE is profiling the data and evaluating candidate models..."):
    model_df = df.copy()
    model_df = model_df.dropna(subset=[target_col])

    selected_features, association_scores, association_label = select_top_features(
        model_df,
        numeric_features,
        target_col,
        task_type,
        max_features,
    )

    if not selected_features:
        st.error("WISE could not identify usable predictor variables.")
        st.stop()

    X_train, X_test, y_train, y_test, validation_strategy = prepare_split(
        model_df,
        selected_features,
        target_col,
        task_type,
        datetime_col,
        group_col,
    )

    if len(X_train) < 10 or len(X_test) < 2:
        st.error("The selected validation structure leaves too few observations for modeling.")
        st.stop()

    results = fit_models(
        X_train,
        X_test,
        y_train,
        y_test,
        task_type,
    )

    try:
        shap_explanation, shap_importance = calculate_shap(
            results["forest"],
            X_train,
            X_test,
            task_type,
        )
        shap_error = None
    except Exception as exc:
        shap_explanation = None
        shap_importance = pd.Series(dtype=float)
        shap_error = str(exc)

n_samples = len(model_df)
n_features_total = len(numeric_features)
complexity, complexity_guidance = complexity_assessment(
    n_samples,
    n_features_total,
    task_type,
)
primary_model, recommendation = recommendation_text(
    task_type,
    results["baseline_metrics"],
    results["forest_metrics"],
    complexity,
)

corr_columns = selected_features.copy()
if task_type == "Regression":
    corr_columns = corr_columns + [target_col]
correlation_matrix = (
    model_df[corr_columns]
    .apply(pd.to_numeric, errors="coerce")
    .corr()
    .round(2)
)

tabs = st.tabs(
    [
        "1. Data overview",
        "2. Target insights",
        "3. Feature relationships",
        "4. Model comparison",
        "5. Explainability",
        "6. WISE recommendation",
        "7. Presentation figure",
    ]
)

with tabs[0]:
    st.header("Dataset and complexity assessment")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Samples", f"{len(df):,}")
    metric_cols[1].metric("Variables", f"{len(df.columns):,}")
    metric_cols[2].metric("Numeric predictors", f"{len(numeric_features):,}")
    metric_cols[3].metric("Selected predictors", f"{len(selected_features):,}")
    metric_cols[4].metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    metric_cols[5].metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.info(
        f"**Suggested model complexity: {complexity}.** {complexity_guidance}"
    )

    profile = pd.DataFrame(
        {
            "Variable": df.columns,
            "Data type": df.dtypes.astype(str).values,
            "Unique values": [df[col].nunique(dropna=True) for col in df.columns],
            "Missing": [df[col].isna().sum() for col in df.columns],
            "Missing percent": [
                round(df[col].isna().mean() * 100, 2) for col in df.columns
            ],
        }
    )
    st.dataframe(profile, use_container_width=True, height=420)

    st.subheader("WISE validation check")
    st.success(validation_strategy)
    if datetime_col:
        st.warning(
            "A random split was not used because temporal ordering can leak future "
            "information into training."
        )
    elif group_col:
        st.warning(
            f"Observations were separated by '{group_col}' so related groups do not "
            "appear in both training and testing."
        )
    else:
        st.caption(
            "No temporal or grouping column was selected. WISE used a random holdout. "
            "Add a grouping or datetime column when observations are dependent."
        )

    if suggested_ids:
        st.warning(
            "Possible identifier columns detected: " + ", ".join(suggested_ids)
        )

with tabs[1]:
    st.header("Target behavior")
    st.plotly_chart(
        target_plot(model_df, target_col, task_type),
        use_container_width=True,
    )

    if datetime_col:
        time_fig = temporal_plot(model_df, target_col, datetime_col, task_type)
        if time_fig is not None:
            st.plotly_chart(time_fig, use_container_width=True)
        else:
            st.warning("The selected datetime column could not be parsed.")

    if latitude_col and longitude_col:
        try:
            map_fig, _, map_title = geospatial_plot(
                model_df,
                target_col,
                latitude_col,
                longitude_col,
                task_type,
                datetime_col,
            )
            if map_fig is not None:
                st.plotly_chart(map_fig, use_container_width=True)
                st.caption(
                    f"{map_title}. When temporal records repeat at a location, WISE "
                    "shows the temporal average for regression or the latest class."
                )
        except Exception as exc:
            st.warning(f"The geospatial map could not be created: {exc}")

    st.info(
        "**WISE note:** Examine the target distribution and temporal or spatial "
        "structure before selecting a model. Skew, imbalance, trends, and clustering "
        "can affect both validation and interpretation."
    )

with tabs[2]:
    st.header("Feature relationships")

    heatmap = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        title="Correlation heatmap",
    )
    heatmap.update_layout(height=650)
    st.plotly_chart(heatmap, use_container_width=True)

    association_table = (
        association_scores.rename(association_label)
        .reset_index()
        .rename(columns={"index": "Feature"})
    )
    association_fig = px.bar(
        association_table.head(15).sort_values(association_label),
        x=association_label,
        y="Feature",
        orientation="h",
        title=f"Top feature–target associations: {association_label}",
    )
    association_fig.update_layout(height=520)
    st.plotly_chart(association_fig, use_container_width=True)

    st.dataframe(
        association_table.head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "**WISE note:** Correlation is useful for screening but can miss nonlinear "
        "relationships. Highly correlated predictors can also distribute or destabilize "
        "importance across variables."
    )

with tabs[3]:
    st.header("Transparent baseline versus nonlinear model")

    baseline_name = (
        "Multiple Linear Regression"
        if task_type == "Regression"
        else "Logistic Regression"
    )
    comparison = pd.DataFrame(
        [
            {"Model": baseline_name, **results["baseline_metrics"]},
            {"Model": "Random Forest", **results["forest_metrics"]},
        ]
    )
    st.dataframe(
        comparison.style.format(precision=3),
        use_container_width=True,
        hide_index=True,
    )

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        if task_type == "Regression":
            st.plotly_chart(
                regression_scatter(
                    y_test,
                    results["baseline_pred"],
                    f"{baseline_name}: observed versus predicted",
                ),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                classification_matrix(
                    y_test,
                    results["baseline_pred"],
                    f"{baseline_name}: confusion matrix",
                ),
                use_container_width=True,
            )

    with plot_col2:
        if task_type == "Regression":
            st.plotly_chart(
                regression_scatter(
                    y_test,
                    results["forest_pred"],
                    "Random Forest: observed versus predicted",
                ),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                classification_matrix(
                    y_test,
                    results["forest_pred"],
                    "Random Forest: confusion matrix",
                ),
                use_container_width=True,
            )

    st.info(
        f"**WISE note:** {baseline_name} is the transparent scientific reference. "
        "Random Forest should be preferred only when its improvement is meaningful "
        "under the selected validation strategy."
    )

with tabs[4]:
    st.header("Model contribution and SHAP analysis")

    importance_top = results["importance"].head(15).sort_values("Importance")
    importance_fig = px.bar(
        importance_top,
        x="Importance",
        y="Feature",
        error_x="Std",
        orientation="h",
        title="Random Forest permutation importance",
    )
    importance_fig.update_layout(height=560)
    st.plotly_chart(importance_fig, use_container_width=True)

    st.caption(
        "Permutation importance measures how much held-out predictive performance "
        "decreases when a feature is randomly shuffled."
    )

    if shap_explanation is not None:
        st.subheader("SHAP summary")
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(
            shap_explanation,
            max_display=min(15, len(selected_features)),
            show=False,
        )
        shap_fig = plt.gcf()
        st.pyplot(shap_fig, use_container_width=True)
        plt.close(shap_fig)

        shap_table = (
            shap_importance.rename("Mean absolute SHAP")
            .reset_index()
            .rename(columns={"index": "Feature"})
        )
        st.dataframe(
            shap_table.head(20),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(f"SHAP analysis could not be completed: {shap_error}")

    st.warning(
        "**Scientific interpretation:** Permutation importance and SHAP explain "
        "predictive model behavior. They do not prove that a variable causally "
        "controls the target."
    )

with tabs[5]:
    st.header("WISE model decision")
    st.success(recommendation)

    st.subheader("Selected predictors")
    st.write(", ".join(selected_features))

    st.subheader("Recommended workflow")
    workflow = [
        f"Use **{baseline_name}** as the transparent baseline.",
        "Retain the selected validation structure when comparing models.",
        f"Use **{primary_model}** as the current primary model.",
        "Report held-out performance rather than training performance.",
        "Use permutation importance and SHAP for predictive interpretation.",
        "Treat model explanations as associations unless a causal design is used.",
    ]
    for index, item in enumerate(workflow, start=1):
        st.markdown(f"{index}. {item}")

    generated_code = starter_code(
        task_type,
        target_col,
        selected_features,
        validation_strategy,
    )
    st.subheader("Editable starter code")
    st.code(generated_code, language="python")
    st.download_button(
        "Download starter analysis code",
        data=generated_code,
        file_name="wise_starter_analysis.py",
        mime="text/x-python",
        use_container_width=True,
    )

    if research_goal.strip():
        st.subheader("Research goal supplied by the user")
        st.write(research_goal)

with tabs[6]:
    st.header("Presentation-ready summary")

    if shap_explanation is None:
        st.warning(
            "The presentation figure requires SHAP results. Resolve the SHAP warning "
            "in the Explainability tab and rerun the analysis."
        )
    else:
        presentation_fig, presentation_png = presentation_figure(
            df=model_df,
            target_col=target_col,
            task_type=task_type,
            datetime_col=datetime_col,
            latitude_col=latitude_col,
            longitude_col=longitude_col,
            selected_features=selected_features,
            correlation_matrix=correlation_matrix,
            y_test=y_test,
            results=results,
            shap_importance=shap_importance,
            complexity=complexity,
            validation_strategy=validation_strategy,
            recommendation=recommendation,
        )
        st.pyplot(presentation_fig, use_container_width=True)
        plt.close(presentation_fig)

        st.download_button(
            "Download presentation figure (PNG)",
            data=presentation_png,
            file_name="WISE_presentation_summary.png",
            mime="image/png",
            type="primary",
            use_container_width=True,
        )

        st.caption(
            "The single figure combines dataset statistics, target behavior, context, "
            "feature relationships, baseline and Random Forest performance, permutation "
            "importance, SHAP contribution, and the final WISE recommendation."
        )
