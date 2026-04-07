import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import random


random.seed(42)
np.random.seed(42)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (roc_auc_score, f1_score,
                             precision_recall_curve, roc_curve,
                             auc, precision_score, recall_score, accuracy_score)

st.set_page_config(page_title="IncomeScope AI", layout="wide")

st.title("IncomeScope AI – Income Prediction Dashboard")
st.markdown("Interactive analysis, visualization, model evaluation, prediction, and chatbot for the Adult Income dataset.")


# Sidebar Upload

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    
    # Basic Cleaning
    
    df = df.replace("?", np.nan)

    for col in ["workclass", "occupation", "native_country"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    if "income" in df.columns:
        # Clean spaces and possible trailing dots in the target column
        df["income"] = (
            df["income"]
            .astype(str)
            .str.strip()
            .str.replace('.', '', regex=False)
        )

        # Convert the target into binary classes
        df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

        # Remove rows where the target could not be mapped
        df = df.dropna(subset=["income"])
        df["income"] = df["income"].astype(int)

    df = df.rename(columns={
        'hours.per.week': 'hours_per_week_raw',
        'capital.gain': 'capital_gain',
        'capital.loss': 'capital_loss',
        'native.country': 'native_country',
        'education.num': 'education_num',
        'marital.status': 'marital_status'
    })

    
    # Dataset Preview
    
    st.header("Dataset Preview")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(df.head(20), use_container_width=True)

    with col2:
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    
    # Interactive Filtering
    
    st.header("Interactive Data Explorer")

    selected_columns = st.multiselect(
        "Choose columns to display",
        df.columns.tolist(),
        default=df.columns.tolist()[:5]
    )

    if selected_columns:
        st.dataframe(df[selected_columns], use_container_width=True)

    
    # Numerical Summary
    
    st.header("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    
    # Interactive Visualizations
    
    st.header("Interactive Visualizations")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Correlation",
        "Distribution",
        "Categorical Analysis",
        "Target Analysis"
    ])

    with tab1:
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        selected_num = st.selectbox("Select numerical column", numeric_cols)
        fig = px.histogram(
            df,
            x=selected_num,
            nbins=30,
            marginal="box",
            title=f"Distribution of {selected_num}"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if categorical_cols:
            selected_cat = st.selectbox("Select categorical column", categorical_cols)
            fig = px.histogram(
                df,
                x=selected_cat,
                color="income" if "income" in df.columns else None,
                barmode="group",
                title=f"{selected_cat} vs Income"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if "income" in df.columns:
            fig = px.pie(
                names=df["income"].map({0: "<=50K", 1: ">50K"}),
                title="Income Class Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # Model Section
    # =========================

    def perform_feature_engineering(df):
        df = df.copy()

        # 1. Age Binning
        df['age'] = pd.cut(df['age'], bins=[0, 25, 50, 100],
                        labels=['Young', 'Adult', 'Old']).astype(str)

        # 2. Capital Net Calculation & Binning
        # We do the math first, then the binning
        df['capital_net'] = df['capital_gain'] - df['capital_loss']
        df['capital_net'] = pd.cut(df['capital_net'], bins=[-5000, 5000, 100000],
                                labels=['Minor', 'Major']).astype(str)

        # 3. Hours Per Week Binning
        df['hours_per_week'] = pd.cut(df['hours_per_week_raw'], bins=[0, 30, 40, 100],
                                        labels=['Lesser Hours', 'Normal Hours', 'Extra Hours']).astype(str)

        # 4. Race Grouping
        other_races = ['Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
        df['race'] = df['race'].replace(other_races, 'Other')

        # 5. Education Grouping
        low_grades = ['11th', '9th', '7th-8th', '5th-6th', '10th', '1st-4th', 'Preschool', '12th']
        df['education'] = df['education'].replace(low_grades, 'School')

        # 6. THE PRO TIP: Native Country
        # Always keeps 'United-States', everything else becomes 'Other'
        df['native_country'] = df['native_country'].apply(lambda x: x if x == 'United-States' else 'Other')

        # 7. Final Column Cleanup
        cols_to_drop = ['capital_gain', 'capital_loss', 'hours_per_week_raw', 'fnlwgt']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        return df

    # Create the pipeline step
    feature_transformer = FunctionTransformer(perform_feature_engineering)

    class FeatureBinarizer(BaseEstimator, TransformerMixin):
        def __init__(self, mappings=None, unknown_value=0):
            self.mappings = mappings or {
                'sex': {'Male': 1, 'Female': 0},
                'native_country': {'United-States': 1, 'Other': 0},
                'capital_net': {'Minor': 1, 'Major': 0},
                'race': {'White': 1, 'Other': 0}
            }
            self.unknown_value = unknown_value

        def fit(self, X, y=None):
            self.n_features_in_ = X.shape[1]
            return self

        def transform(self, X):
            X = X.copy()
            for col, mapping in self.mappings.items():
                if col in X.columns:
                    # map known categories and set unknown values safely
                    X[col] = X[col].map(lambda v: mapping.get(v, self.unknown_value))
                    X[col] = X[col].fillna(self.unknown_value)
            return X

        def get_feature_names_out(self, input_features=None):
            return input_features
    

    st.header("Random Forest Model Training & Evaluation")

    df_transformed = perform_feature_engineering(df)
    X = df_transformed.drop("income", axis=1)
    y = df_transformed["income"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    cat_col = X_train.select_dtypes(exclude=['number']).columns
    bin_cols = ['sex', 'native_country', 'capital_net', 'race']
    num_col = []
    multi_cat_cols = cat_col.difference(bin_cols)
    assert 'income' not in cat_col, "income should not be in features for preprocessing"
    bin_pipeline = Pipeline([
        ('binarizer', FeatureBinarizer())
    ])
    multi_cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    scaler = Pipeline([
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('bin', bin_pipeline, bin_cols),
            ('multi_cat', multi_cat_pipeline, multi_cat_cols),
            ('num', scaler, num_col)
        ]
    )
    rfc_param_grid = {
        'clf__n_estimators': [100, 300],
        'clf__max_depth': [None, 15, 30],
        'clf__min_samples_split': [2, 10],
        'clf__min_samples_leaf': [1, 4],
        'clf__max_features': ['sqrt'],
        'clf__criterion': ['gini', 'entropy']
    }
    rfc_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(sampling_strategy=0.4, random_state=42)),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rfc_grid_search = GridSearchCV(
        estimator=rfc_pipeline,
        param_grid=rfc_param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )
        
    with st.spinner("Training model..."):
        @st.cache_resource
        def load_and_train_rfc_model():
            try:
                rfc_grid_search.fit(X_train, y_train)

                best_rfc_model = rfc_grid_search.best_estimator_
            except Exception as e:
                st.error(f"Error during Random Forest model training: {e}")
                st.stop()
                best_rfc_model = None

            if best_rfc_model is None:
                st.error("Random Forest model training failed. Please check the data and configuration.")
                st.stop()
            return best_rfc_model
        
        best_rfc_model = load_and_train_rfc_model()
        y_pred_rfc = best_rfc_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_rfc)
        prec = precision_score(y_test, y_pred_rfc)
        rec = recall_score(y_test, y_pred_rfc)
        f1 = f1_score(y_test, y_pred_rfc)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1 Score", f"{f1:.3f}")
        try:
            feature_names = best_rfc_model.named_steps['preprocessor'].get_feature_names_out()
        except Exception:
            feature_names = [f'feature_{i}' for i in range(len(best_rfc_model.named_steps['clf'].feature_importances_))]
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': np.round(best_rfc_model.named_steps['clf'].feature_importances_, 3)
        })
        top_10_importances = importances.sort_values('importance', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        top_10_importances.sort_values('importance', ascending=True).plot.barh(
            x='feature', 
            y='importance', 
            ax=ax
        )
        ax.set_title("Top 10 Feature Importances from Pipeline")
        st.pyplot(fig)
        st.dataframe(top_10_importances)

    # XGBoost model training and evaluation can be added here following a similar pattern if desired

    xgb_param_grid = {
        'clf__n_estimators': [500, 700],
        'clf__learning_rate': [0.01, 0.05],
        'clf__max_depth': [3, 5],
        'clf__subsample': [0.7, 0.8],
        'clf__colsample_bytree': [0.7, 0.8],
        'clf__min_child_weight': [1, 2]
    }

    xgb_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(sampling_strategy=0.4, random_state=42)),
        ('clf', xgb.XGBClassifier(random_state=42, eval_metric='logloss'))
    ])

    xgb_grid_search = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=xgb_param_grid,
        cv=cv,
        scoring='recall',
        n_jobs=-1,
        verbose=2
    )

    st.header("XGBoost Model Training & Evaluation")
    
    with st.spinner("Training model..."):
        @st.cache_resource
        def load_and_train_xgb_model():
            try:
                xgb_grid_search.fit(X_train, y_train)

                best_xgb_model = xgb_grid_search.best_estimator_
            except Exception as e:
                st.error(f"Error during XGBoost model training: {e}")
                st.stop()
                best_xgb_model = None

            if best_xgb_model is None:
                st.error("XGBoost model training failed. Please check the data and configuration.")
                st.stop()
            return best_xgb_model

        best_xgb_model = load_and_train_xgb_model()
        y_pred_xgb = best_xgb_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred_xgb)
        prec = precision_score(y_test, y_pred_xgb)
        rec = recall_score(y_test, y_pred_xgb)
        f1 = f1_score(y_test, y_pred_xgb)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1 Score", f"{f1:.3f}")

        try:
            feature_names = best_xgb_model.named_steps['preprocessor'].get_feature_names_out()
        except Exception:
            feature_names = [f'feature_{i}' for i in range(len(best_xgb_model.named_steps['clf'].feature_importances_))]

        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': np.round(best_xgb_model.named_steps['clf'].feature_importances_, 3)
        })

        top_10_importances = importances.sort_values('importance', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        top_10_importances.sort_values('importance', ascending=True).plot.barh(
            x='feature', 
            y='importance', 
            ax=ax
        )
        ax.set_title("Top 10 Feature Importances from Pipeline")

        st.pyplot(fig)

        st.dataframe(top_10_importances)




    # Model Comparison Table
    comparison_data = {
        'Model': ['RandomForest', 'XGBoost'],
        'Test AUC': [roc_auc_score(y_test, best_rfc_model.predict_proba(X_test)[:, 1]),
                    roc_auc_score(y_test, best_xgb_model.predict_proba(X_test)[:, 1])],
        'Test F1 Macro': [f1_score(y_test, y_pred_rfc, average='macro'),
                        f1_score(y_test, y_pred_xgb, average='macro')],
        'Test F1 Micro': [f1_score(y_test, y_pred_rfc, average='micro'),
                        f1_score(y_test, y_pred_xgb, average='micro')]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df)
    # Numeric curves and AUC
    rfc_scores = best_rfc_model.predict_proba(X_test)[:, 1]
    xgb_scores = best_xgb_model.predict_proba(X_test)[:, 1]
    rfc_pr_prec, rfc_pr_rec, _ = precision_recall_curve(y_test, rfc_scores)
    xgb_pr_prec, xgb_pr_rec, _ = precision_recall_curve(y_test, xgb_scores)
    rfc_roc_fpr, rfc_roc_tpr, _ = roc_curve(y_test, rfc_scores)
    xgb_roc_fpr, xgb_roc_tpr, _ = roc_curve(y_test, xgb_scores)
    rfc_pr_auc = auc(rfc_pr_rec, rfc_pr_prec)
    xgb_pr_auc = auc(xgb_pr_rec, xgb_pr_prec)
    rfc_roc_auc = auc(rfc_roc_fpr, rfc_roc_tpr)
    xgb_roc_auc = auc(xgb_roc_fpr, xgb_roc_tpr)
    # Create the figure object explicitly
    fig = plt.figure(figsize=(12, 5))
    # Precision-Recall curve
    plt.subplot(1, 2, 1)
    plt.plot(rfc_pr_rec, rfc_pr_prec, label=f'RFC PR AUC={rfc_pr_auc:.3f}')
    plt.plot(xgb_pr_rec, xgb_pr_prec, label=f'XGB PR AUC={xgb_pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    # ROC curve
    plt.subplot(1, 2, 2)
    plt.plot(rfc_roc_fpr, rfc_roc_tpr, label=f'RFC ROC AUC={rfc_roc_auc:.3f}')
    plt.plot(xgb_roc_fpr, xgb_roc_tpr, label=f'XGB ROC AUC={xgb_roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    # Use Streamlit to display the plot
    st.pyplot(fig)
    # =========================
    # Prediction Section
    # =========================
    st.header("Try a Prediction")
    st.markdown("Edit the sample values below and predict income level.")
    sample = X.iloc[0].copy()
    user_input = {}
    cols = st.columns(2)
    for i, col in enumerate(X.columns):
        with cols[i % 2]:
            # Use proper dtype detection instead of checking only for object
            if pd.api.types.is_numeric_dtype(X[col]):
                default_value = pd.to_numeric(sample[col], errors="coerce")
                if pd.isna(default_value):
                    default_value = 0.0
                user_input[col] = st.number_input(
                    col,
                    value=float(default_value),
                    key=col
                )
            else:
                options = sorted(X[col].dropna().astype(str).unique())
                default_option = str(sample[col]) if str(sample[col]) in options else options[0]
                user_input[col] = st.selectbox(
                    col,
                    options,
                    index=options.index(default_option),
                    key=col
                )
    if st.button("Predict Income"):
        input_df = pd.DataFrame([user_input])
        prediction = best_xgb_model.predict(input_df)[0]
        probability = best_xgb_model.predict_proba(input_df)[0]
        if prediction == 1:
            st.success(f"Predicted Income: >50K")
        else:
            st.warning(f"Predicted Income: <=50K")
        st.info(f"Confidence: {np.max(probability)*100:.2f}%")

    # =========================
    # Simple Chatbot
    # =========================
    st.header("Dataset Assistant Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask something about the dataset or model...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        prompt_lower = prompt.lower()

        if "rows" in prompt_lower:
            response = f"The dataset contains {df.shape[0]} rows."
        elif "columns" in prompt_lower:
            response = f"The dataset contains {df.shape[1]} columns."
        elif "missing" in prompt_lower:
            response = f"There are {df.isnull().sum().sum()} missing values in the dataset."
        elif "best model" in prompt_lower:
            response = "The Random Forest model is currently being used because it gives strong classification performance on this dataset."
        elif "accuracy" in prompt_lower:
            response = f"The model accuracy is approximately {acc:.3f}."
        else:
            response = "I can answer questions about the dataset size, columns, missing values, model accuracy, and prediction workflow."

        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)

else:
    st.info("Please upload your Adult Income CSV file from the sidebar to begin.")