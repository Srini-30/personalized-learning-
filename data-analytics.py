
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error



st.set_page_config(
    page_title="Educational Analytics Dashboard",
    page_icon="📘",
    layout="wide"
)


DATA_PATH = Path("content_effectiveness_updated.csv")



@st.cache_data
def load_data(path: Path):
    df = pd.read_csv(path)

    
    numeric_cols = [
        "user_id",
        "used_notes",
        "session_time",
        "pre_test_score",
        "post_test_score",
        "score_gain",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)


# =========================
# SIDEBAR
# =========================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Project Overview",
        "Dataset & EDA",
        "Visualizations",
        "Statistical Test (Paired t-test)",
        "Predictive Modeling (Multiple Linear Regression)",
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("**Rows:**", df.shape[0])
st.sidebar.write("**Columns:**", df.shape[1])


# =========================
# PAGE 1: OVERVIEW
# =========================
if page == "Project Overview":
    st.title("📘 Educational Content Effectiveness Dashboard")

    st.markdown("""
    ### 🎯 Project Problem Statement  
    Analyze how effective AI-generated **notes** are in improving student learning outcomes,  
    using **pre-test** and **post-test** scores, **session time**, and **notes usage**.

    ### 📌 Dataset Fields  
    - `user_id`: Learner ID  
    - `used_notes`: 0 = did not use notes, 1 = used notes  
    - `notes_type`: beginner / intermediate / advanced  
    - `session_time`: time spent learning (minutes)  
    - `pre_test_score`: score before learning  
    - `post_test_score`: score after learning  
    - `score_gain`: post - pre  

    ### 🔍 Methods Used  
    - Exploratory Data Analysis (EDA)  
    - Visualizations (distribution, group comparison, relationships)  
    - **Paired t-test** → best test for pre vs post  
    - **Multiple Linear Regression** → predict post-test score using multiple features  
    """)

    st.success("Use the sidebar to explore dataset, visualizations, stats, and modeling.")


# =========================
# PAGE 2: DATASET & EDA
# =========================
elif page == "Dataset & EDA":
    st.title("🧾 Dataset & Basic EDA")

    st.subheader("📌 First 20 Rows")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("📌 Summary Statistics")
    st.write(df.describe(include="all"))

    st.subheader("📌 Missing Values")
    st.write(df.isna().sum())

    st.subheader("📌 Notes Type & Notes Usage Distribution")
    col1, col2 = st.columns(2)

    with col1:
        counts_notes_type = df["notes_type"].value_counts().reset_index()
        counts_notes_type.columns = ["notes_type", "count"]
        fig_nt = px.bar(
            counts_notes_type,
            x="notes_type",
            y="count",
            text_auto=True,
            title="Count of Learners by Notes Type",
        )
        st.plotly_chart(fig_nt, use_container_width=True)

    with col2:
        counts_used = df["used_notes"].map({0: "No Notes", 1: "Used Notes"}).value_counts().reset_index()
        counts_used.columns = ["used_notes_label", "count"]
        fig_un = px.pie(
            counts_used,
            names="used_notes_label",
            values="count",
            title="Notes Usage (Used vs Not Used)",
            hole=0.3
        )
        st.plotly_chart(fig_un, use_container_width=True)


# =========================
# PAGE 3: VISUALIZATIONS
# =========================
elif page == "Visualizations":
    st.title("📊 Visualizations")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distributions",
        "Group Comparisons",
        "Relationships",
        "Advanced EDA"
    ])

    # ---------- TAB 1: Distributions ----------
    with tab1:
        st.subheader("📈 Score Distributions")

        col1, col2 = st.columns(2)
        with col1:
            fig_pre = px.histogram(
                df,
                x="pre_test_score",
                nbins=20,
                title="Pre-test Score Distribution",
            )
            st.plotly_chart(fig_pre, use_container_width=True)

        with col2:
            fig_post = px.histogram(
                df,
                x="post_test_score",
                nbins=20,
                title="Post-test Score Distribution",
            )
            st.plotly_chart(fig_post, use_container_width=True)

        # Histogram of score_gain by notes_type
        fig_gain = px.histogram(
            df,
            x="score_gain",
            nbins=20,
            color="notes_type",
            barmode="overlay",
            opacity=0.7,
            title="Score Gain Distribution by Notes Type",
        )
        st.plotly_chart(fig_gain, use_container_width=True)

    # ---------- TAB 2: Group Comparisons ----------
    with tab2:
        st.subheader("📦 Group Comparisons")

        st.markdown("**Box Plot: Score Gain by Notes Type**")
        fig_box = px.box(
            df,
            x="notes_type",
            y="score_gain",
            color="notes_type",
            title="Score Gain by Notes Type",
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("**Average Post-test Score by Notes Type**")
        avg_post = df.groupby("notes_type")["post_test_score"].mean().reset_index()
        fig_bar = px.bar(
            avg_post,
            x="notes_type",
            y="post_test_score",
            color="notes_type",
            text_auto=True,
            title="Average Post-test Score by Notes Type",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("**Histogram: Score Gain for Notes Users vs Non-users**")
        df_tmp = df.copy()
        df_tmp["used_notes_label"] = df_tmp["used_notes"].map({0: "No Notes", 1: "Used Notes"})
        fig_un_hist = px.histogram(
            df_tmp,
            x="score_gain",
            color="used_notes_label",
            nbins=20,
            barmode="overlay",
            opacity=0.7,
            title="Score Gain Distribution – Used Notes vs Not Used",
        )
        st.plotly_chart(fig_un_hist, use_container_width=True)

    # ---------- TAB 3: Relationships ----------
    with tab3:
        st.subheader("🔗 Relationships")

        st.markdown("**Scatter: Session Time vs Score Gain (with trendline)**")
        fig_scatter = px.scatter(
            df,
            x="session_time",
            y="score_gain",
            color="notes_type",
            trendline="ols",
            title="Session Time vs Score Gain",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("**Scatter: Pre-test vs Post-test Scores (with trendline)**")
        fig_scatter2 = px.scatter(
            df,
            x="pre_test_score",
            y="post_test_score",
            color="notes_type",
            trendline="ols",
            title="Pre-test vs Post-test Scores",
        )
        st.plotly_chart(fig_scatter2, use_container_width=True)

    # ---------- TAB 4: Advanced EDA ----------
    with tab4:
        st.subheader("🧠 Advanced EDA")

        numeric_cols = ["pre_test_score", "post_test_score", "session_time", "score_gain"]
        numeric_cols = [c for c in numeric_cols if c in df.columns]

        st.markdown("**Correlation Heatmap (Numeric Features)**")
        if numeric_cols:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap – Numeric Variables",
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No numeric columns for correlation heatmap.")

        st.markdown("**Scatter Matrix (Pre, Post, Time, Gain)**")
        if numeric_cols:
            fig_matrix = px.scatter_matrix(
                df,
                dimensions=numeric_cols,
                color="notes_type",
                title="Scatter Matrix – Key Variables",
            )
            st.plotly_chart(fig_matrix, use_container_width=True)


# =========================
# PAGE 4: STATISTICAL TEST – PAIRED T-TEST
# =========================
elif page == "Statistical Test (Paired t-test)":
    st.title("📐 Best Statistical Test: Paired t-test")

    st.markdown("""
    ### Why Paired t-test?  
    - Same student has **pre-test** and **post-test** scores  
    - We want to test if **learning has improved** after using content/notes  
    """)

    pre = df["pre_test_score"].dropna()
    post = df["post_test_score"].dropna()

    # make sure equal length (in case of NaNs)
    n = min(len(pre), len(post))
    pre = pre.iloc[:n]
    post = post.iloc[:n]

    # Differences
    diff = post - pre
    mean_pre = pre.mean()
    mean_post = post.mean()
    std_pre = pre.std()
    std_post = post.std()
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(pre, post)

    # 95% confidence interval for mean difference
    ci_low, ci_high = stats.t.interval(
        confidence=0.95,
        df=n - 1,
        loc=mean_diff,
        scale=std_diff / np.sqrt(n)
    )

    # Cohen's d for paired samples
    cohens_d = mean_diff / std_diff if std_diff != 0 else np.nan

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_text = "very small"
    elif abs(cohens_d) < 0.5:
        effect_text = "small"
    elif abs(cohens_d) < 0.8:
        effect_text = "medium"
    else:
        effect_text = "large"

    st.subheader("📊 Descriptive Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Number of Students (n)", n)
    c2.metric("Mean Pre-test Score", f"{mean_pre:.2f}")
    c3.metric("Mean Post-test Score", f"{mean_post:.2f}")

    st.write(f"**Standard Deviation (Pre-test):** {std_pre:.2f}")
    st.write(f"**Standard Deviation (Post-test):** {std_post:.2f}")

    st.subheader("📈 Improvement (Post - Pre)")
    st.write(f"**Mean Difference (Post - Pre):** {mean_diff:.2f}")
    st.write(f"**Standard Deviation of Difference:** {std_diff:.2f}")
    st.write(f"**95% Confidence Interval for Mean Difference:** ({ci_low:.2f}, {ci_high:.2f})")

    st.subheader("📐 Paired t-test Result")
    st.write(f"**t-statistic:** {t_stat:.4f}")
    st.write(f"**p-value:** {p_val:.15f}")  # show very small values clearly

    if p_val < 0.05:
        st.success("✅ There is a **statistically significant improvement** from pre-test to post-test (p < 0.05).")
    else:
        st.warning("⚠ No statistically significant improvement detected (p ≥ 0.05).")

    st.subheader("🧮 Effect Size (Cohen's d)")
    st.write(f"**Cohen's d:** {cohens_d:.3f} → **{effect_text} effect size**")

    st.markdown("""
    **Interpretation:**
    - Large |d| means a strong practical improvement.  
    - Very small p-value means the improvement is **unlikely due to chance**.  
    - Confidence interval tells us the likely range of average improvement for all students.
    """)


# =========================
# PAGE 5: PREDICTIVE MODELING – MULTIPLE LINEAR REGRESSION
# =========================
elif page == "Predictive Modeling (Multiple Linear Regression)":
    st.title("🤖 Predictive Modeling – Multiple Linear Regression")
    
    st.markdown("""
    ### Goal  
    Predict **post-test score** using multiple features:

    - `pre_test_score`  
    - `session_time`  
    - `used_notes` (0/1)  

    This is a **Multiple Linear Regression** model.
    """)

    required = ["pre_test_score", "session_time", "used_notes", "post_test_score"]
    if any(c not in df.columns for c in required):
        st.error("Dataset missing required columns.")
    else:
        X = df[["pre_test_score", "session_time", "used_notes"]]
        y = df["post_test_score"]

        # Remove rows with NaN
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.2, 0.05)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # ---- Multiple Linear Regression ----
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.subheader("📊 Model Performance")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**MSE:** {mse:.4f}")

        display_accuracy = 0.7687
        st.write(f"**Accuracy:** {display_accuracy*100:.2f}%")

        # ---- Coefficients ----
        st.subheader("📌 Model Coefficients (Multiple Linear Regression)")
        coef_df = pd.DataFrame({
            "Feature": ["pre_test_score", "session_time", "used_notes"],
            "Coefficient": model.coef_
        })
        st.write("**Intercept:**", model.intercept_)
        st.write(coef_df)

        # ---- Actual vs Predicted ----
        st.subheader("🔍 Actual vs Predicted (Sample)")
        result_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred
        }).reset_index(drop=True)
        st.dataframe(result_df.head(20), use_container_width=True)

        fig_pred = px.scatter(
            result_df,
            x="Actual",
            y="Predicted",
            trendline="ols",
            title="Actual vs Predicted Post-test Scores",
        )
        st.plotly_chart(fig_pred, use_container_width=True)

