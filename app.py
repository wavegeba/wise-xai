import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="WISE",
    page_icon="🧠",
    layout="wide",
)

st.title("WISE")
st.subheader("Workflow for Interpretable Scientific Evaluation")

st.write(
    "Upload a dataset and describe your scientific goal. "
    "WISE will assess your data and recommend suitable models, "
    "validation strategies, explainability tools, and starter code."
)

research_goal = st.text_area(
    "What do you want to learn or predict?",
    placeholder=(
        "Example: I want to predict crop yield and understand "
        "which environmental variables control yield variability."
    ),
)

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"],
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Samples", len(df))
    col2.metric("Variables", len(df.columns))
    col3.metric("Missing values", int(df.isna().sum().sum()))

    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Variable assessment")

    summary = pd.DataFrame({
        "Variable": df.columns,
        "Data type": df.dtypes.astype(str).values,
        "Unique values": [df[column].nunique() for column in df.columns],
        "Missing values": [df[column].isna().sum() for column in df.columns],
        "Missing percent": [
            round(df[column].isna().mean() * 100, 2)
            for column in df.columns
        ],
    })

    st.dataframe(summary, use_container_width=True)

    target = st.selectbox(
        "Select the target variable",
        options=[""] + list(df.columns),
    )

    if target:
        st.subheader("Initial task assessment")

        unique_count = df[target].nunique()

        if pd.api.types.is_numeric_dtype(df[target]) and unique_count > 10:
            task_type = "Regression"
        else:
            task_type = "Classification"

        st.write(f"**Likely task type:** {task_type}")
        st.write(f"**Target variable:** {target}")

        if st.button("Generate WISE recommendations"):
            st.info(
                "The AI-guided model, validation, explainability, "
                "and code recommendations will appear here."
            )
