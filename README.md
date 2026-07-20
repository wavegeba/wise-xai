# WISE-XAI

## Workflow for Interpretable Scientific Evaluation

**WISE-XAI helps students and scientists choose the right AI model for their data and research goals—not simply the newest or most accurate model.**

WISE-XAI is an educational model-selection and scientific-evaluation platform. It diagnoses the structure of a dataset, recommends an appropriate validation strategy, compares transparent and nonlinear models, explains predictive behavior, and generates reproducible scientific outputs.

> Start with the data and the scientific question—not with the newest model.

## Project Overview

Machine-learning, deep-learning, and advanced AI models are increasingly accessible. However, users often select models mainly because they are popular or report high accuracy, without fully considering:

- Sample size and feature count
- Temporal, spatial, or grouped dependence
- Validation leakage
- Scientific applicability
- Model interpretability
- Reliability and reproducibility
- The actual research objective

WISE-XAI guides users through a structured workflow:

1. Dataset and complexity assessment
2. Target visualization
3. Feature-relationship analysis
4. Validation design
5. Model comparison
6. Permutation importance and SHAP
7. Scientific recommendation and reproducible outputs

The current prototype demonstrates this workflow using tabular regression and classification, transparent linear or logistic baselines, Random Forest, temporal or grouped validation, and explainability analysis.

WISE-XAI is designed as an extensible, model-agnostic framework that can later compare broader statistical, machine-learning, deep-learning, domain-specific, and emerging AI models.

## Live Demo

**Streamlit application:**  
`ADD_YOUR_PUBLIC_STREAMLIT_URL_HERE`

**GitHub repository:**  
`ADD_YOUR_PUBLIC_GITHUB_REPOSITORY_URL_HERE`

No login is required. Judges can test the complete workflow using the built-in demonstration dataset.

## Core Features

### Data Diagnosis

WISE-XAI examines:

- Sample size
- Number of variables
- Numeric predictors
- Missing values
- Duplicate records
- Possible identifier variables
- Sample-to-feature ratio
- Temporal or grouped structure
- Potential validation risks

### Target Insights

Depending on the available variables, WISE-XAI displays:

- Target or class distribution
- Temporal trend
- Spatial pattern

### Feature Relationships

WISE-XAI provides:

- Lower-triangle correlation heatmap
- Feature–target association ranking
- Correlation-based screening for regression
- Mutual-information screening for classification

### Validation Design

The current prototype supports:

- Random 80/20 holdout
- Group-based holdout
- Temporal holdout using the earliest 80% for training and latest 20% for testing

The broader framework is designed to support:

- Stratified cross-validation
- Repeated cross-validation
- Grouped cross-validation
- Leave-one-group-out validation
- Nested cross-validation
- Time-series cross-validation
- Spatial-block validation
- Spatiotemporal validation

### Model Comparison

The current prototype compares:

- Multiple Linear Regression vs. Random Forest Regression
- Logistic Regression vs. Random Forest Classification

WISE-XAI does not automatically prefer the more complex model. Additional complexity must produce a meaningful improvement under a scientifically valid evaluation strategy.

The framework can be extended to include:

- Generalized linear models
- Gradient-boosting models
- Explainable Boosting Machines
- Support Vector Machines
- Neural networks
- CNNs and recurrent models
- Transformers
- Domain-specific and emerging AI models

### Explainability

WISE-XAI currently uses:

- Transparent baseline-model interpretation
- Permutation importance
- SHAP

Model explanations describe predictive behavior. They do not establish causal relationships.

### Reproducible Outputs

WISE-XAI generates:

- Recommended primary model
- Model-selection rationale
- Validation guidance
- Selected predictors
- Held-out performance metrics
- Explainability results
- Editable starter Python code
- Presentation-ready scientific summary figure

## Prototype Scope

This repository contains an **initial working prototype** of the broader WISE-XAI framework.

The prototype currently focuses on:

- Tabular regression and classification
- Numeric predictor variables
- Linear or logistic baselines
- Random Forest comparison
- Random, grouped, and temporal holdouts
- Correlation and mutual-information analysis
- Permutation importance
- SHAP
- GPT-5.6 scientific interpretation
- Starter-code generation
- Presentation-ready output

The limited model set was selected intentionally to demonstrate the complete decision workflow clearly and reproducibly. The WISE-XAI concept and architecture are not limited to these models.

## How GPT-5.6 Was Used

GPT-5.6 powers the **scientific interpretation layer**.

The deterministic Python workflow performs:

- Dataset profiling
- Feature screening
- Validation splitting
- Model training
- Metric calculation
- Permutation importance
- SHAP analysis

GPT-5.6 receives only summarized diagnostics and model results—not the complete uploaded dataset—and translates them into concise educational guidance covering:

- Recommended model
- Model-selection rationale
- Validation considerations
- Interpretation
- Limitations
- Recommended next steps

GPT-5.6 does not replace the numerical analysis. It makes the deterministic results easier for students and scientists to understand.

## How Codex Was Used

Codex supported development, review, debugging, and refinement of the WISE-XAI application.

It was used to improve:

- Streamlit application structure
- Dataset-profiling workflow
- Model-comparison logic
- Temporal and grouped validation logic
- Correlation and visualization code
- SHAP integration
- GPT-5.6 API integration
- Error handling
- Syntax validation
- Code clarity and maintainability

Codex was also used to inspect code changes, identify potential failures, and validate the application after updates.

## Technology Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- Plotly
- Matplotlib
- SHAP
- OpenAI API
- GPT-5.6
- Codex
- GitHub

## How to Run

### 1. Clone the repository

```bash
git clone ADD_YOUR_PUBLIC_GITHUB_REPOSITORY_URL_HERE
cd wise-xai
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the OpenAI API key

Set the key as an environment variable.

On macOS or Linux:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

Do not commit an API key to GitHub.

The deterministic WISE-XAI analysis can run without sending the complete dataset to GPT-5.6. The GPT-5.6 interpretation button requires a valid API key.

### 5. Start the application

```bash
python -m streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Testing Instructions

### Fastest Test: Built-in Demo

1. Open the Streamlit application.
2. Enable **Use built-in demo dataset**.
3. Confirm the target variable is `yield`.
4. Keep the task type as `Auto` or select `Regression`.
5. Keep `date` as the datetime column.
6. Keep the latitude and longitude columns selected.
7. Click **Run WISE analysis**.
8. Review the seven tabs:
   - Data overview
   - Target insights
   - Feature relationships
   - Model comparison
   - Explainability
   - WISE recommendation
   - Presentation figure
9. In **WISE recommendation**, click **Generate GPT-5.6 interpretation**.
10. In **Presentation figure**, review or download the final PNG summary.

### Expected Demonstration Result

The built-in temporal dataset compares Multiple Linear Regression with Random Forest.

In the current demonstration, the linear model performs better on the held-out temporal test set. WISE-XAI therefore recommends the simpler and more interpretable model rather than automatically choosing Random Forest.

### Test With Your Own Dataset

Upload a CSV containing:

- One target variable
- At least one numeric predictor
- Optional date or time column
- Optional grouping column
- Optional latitude and longitude columns

Then select the relevant fields and run the analysis.

## Repository Structure

```text
wise-xai/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── LICENSE
```

## Scientific Principles

WISE-XAI follows several core principles:

1. Validation must reflect the structure of the data.
2. Training performance is not evidence of generalization.
3. A transparent baseline should be established before adding complexity.
4. Higher model capacity must be justified by held-out results.
5. Correlation and feature importance do not establish causality.
6. Explainable AI begins with research design—not only with a SHAP plot.
7. The best model is the one most appropriate for the data and scientific claim.

## What We Learned

Model selection is not simply an accuracy-optimization problem. It is a scientific-design decision.

A simpler model with realistic validation and understandable behavior may provide stronger scientific evidence than a highly complex model with slightly better—or even worse—performance.

Explainability should begin before training by considering the research objective, dataset structure, validation design, baseline model, and justification for complexity.

## What’s Next

Planned development includes:

- Automated comparison across broader ML and DL model libraries
- Integration of recent and domain-specific models
- Advanced and nested cross-validation
- Hyperparameter optimization within valid training folds
- Spatial and spatiotemporal validation
- Image, text, time-series, and multimodal workflows
- CNN, recurrent, transformer, and foundation-model support
- Forecasting and uncertainty estimation
- Explanation-stability analysis
- Model-complexity and compute-cost comparison
- Exportable model-decision reports
- Notebook and research-repository integration
- Classroom modules for responsible AI education

> **WISE-XAI does not reject advanced AI. It helps users determine when advanced AI is scientifically justified.**

## License

This project is released under the MIT License.
