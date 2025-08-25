import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Smartphone Data Dashboard", layout="wide")

#---------------------Helper Functions----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str | None, uploaded_file) -> pd.DataFrame | None:
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
    return None

def detect_columns(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols

def kpi_card(label: str, value, helptext: str | None = None):
    col = st.container()
    with col:
        st.metric(label, value, help=helptext)

def corr_matrix(df: pd.DataFrame, cols: list[str]):
    if len(cols) < 2:
        st.info("Pick at least two numeric columns to view a correlation heatmap.")
        return
    corr = df[cols].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        origin="lower",
        title="Correlation Heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)

def download_csv(df: pd.DataFrame, label: str = "Download CSV", filename: str = "Smartphone_Data.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")

#--------------------------Sidebar---------------------------
st.sidebar.header("📂 Upload Dataset")
default_path = "C:/Users/ahana/OneDrive/Desktop/Smartphones_Dataset/smartphones_cleaned.csv .csv"
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
use_default = st.sidebar.checkbox(f"Use default dataset ({default_path})", value=True)

df = None

if use_default:
    df = pd.read_csv(default_path)
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

if df is not None:
    st.success("✅ Dataset loaded successfully")
else:
    st.info("Please upload a CSV file or select the default dataset.")

if df is None:
    st.warning("No data loaded. Upload a CSV or enable the default dataset.")
    st.stop()

all_cols = df.columns.tolist()
target_opt = "Brand Name" if "Brand Name" in all_cols else (all_cols[-1] if all_cols else None)
options = [None] + all_cols

if target_opt in all_cols:
    default_index = all_cols.index(target_opt) + 1
else:
    default_index = 0

target_col = st.sidebar.selectbox(
    "Target/Group column", 
    options, 
    index=default_index
)

st.sidebar.header("📊 Dataset Information")
numeric_cols, categorical_cols = detect_columns(df)

st.sidebar.subheader("Numeric Columns")
st.sidebar.write(numeric_cols)

st.sidebar.subheader("Categorical Columns")
st.sidebar.write(categorical_cols)

filtered_df = df.copy()

st.sidebar.subheader("🔍 Filters")

if target_col and target_col in df.columns:
    unique_vals = df[target_col].dropna().unique().tolist()
    selected_vals = st.sidebar.multiselect(f"Select {target_col}", options=unique_vals, default=unique_vals)
    filtered_df = filtered_df[filtered_df[target_col].isin(selected_vals)]

with st.sidebar.expander("Numeric Ranges", expanded=False):
    for col in numeric_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        selected_range = st.slider( f"{col}", min_val, max_val, (min_val, max_val))
        filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]

# ---------------------- Header ----------------------
st.title("📱 Smartphone Features Analysis")

# ---------------------- KPIs ----------------------
n_rows, n_cols = filtered_df.shape
miss_cells = int(filtered_df.isna().sum().sum())
miss_pct = (miss_cells / (n_rows * n_cols) * 100) if n_rows and n_cols else 0

k1, k2, k3, k4 = st.columns(4)
with k1: kpi_card("Rows", f"{n_rows:,}")
with k2: kpi_card("Columns", f"{n_cols:,}")
with k3: kpi_card("Missing Cells", f"{miss_cells:,}")
with k4: kpi_card("Missing %", f"{miss_pct:.2f}%")

# ---------------------- Missing Values ----------------------
with st.expander("🔍 Missing Values", expanded=False):
    missing_val= df.columns[df.isna().any()].tolist()

    if missing_val:
        st.write("Columns with missing values:")
        for col in missing_val:
            st.write(f"{col}: {df[col].isna().sum()}")
    else:
        st.success("✅ No missing values found in this dataset!")

#---------------------Class Distribution of Selected Column--------------------
if target_col and target_col in filtered_df.columns:
    st.subheader("📊 Column Distribution")
    counts = filtered_df[target_col].value_counts(dropna=False).reset_index()
    counts.columns = [target_col, "Frequency"]
    fig = px.bar(counts, x=target_col, y="Frequency", text="Frequency", title=f"Distribution of {target_col}",  color="Frequency", 
    color_continuous_scale="Viridis")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

#-----------------------Top Categories----------------------
st.subheader("📶 Top N Categories")
cat_col = st.selectbox("Categorical Column", options=categorical_cols if categorical_cols else [None])
if cat_col:
    top_n = st.slider("Show Top Brands", 5, 50, 10)
    counts = (
        filtered_df[cat_col]
        .astype("category")
        .value_counts(dropna=False)
        .head(top_n)
        .reset_index()
    )
    counts.columns = [cat_col, "Count"]
    fig = px.treemap(
        counts,
        path=[cat_col],
        values="Count",
        title=f"📊 Top {top_n} Categories of {cat_col}",
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------Histogram and Pie Chart----------------------
st.subheader("📊 Histogram Distribution of Numeric Data and Pie Chart of Categorical Data")
c1, c2= st.columns(2)
with c1:
    col_num = st.selectbox("Select numeric column", options=numeric_cols if numeric_cols else [None])
    if col_num:
        fig = px.histogram(filtered_df, x=col_num, nbins=30, marginal="rug", title=f"🔢 Histogram of {col_num}", color_discrete_sequence=["#FF5733"])
        st.plotly_chart(fig, use_container_width=True)

with c2:
    col_cat = st.selectbox("Select categorical column", options=categorical_cols)
    if col_cat:
        counts = filtered_df[col_cat].value_counts().reset_index()
        counts.columns = [col_cat, "Count"]
        fig = px.pie(counts, names=col_cat, values="Count", title=f"🥮 Distribution of {col_cat}", hole=0.3, color_discrete_sequence=px.colors.sequential.Plasma)
        fig.update_traces(textinfo="percent+label")  
        st.plotly_chart(fig, use_container_width=True)

# ---------------------- Correlation ----------------------
st.subheader("🧪 Correlation Heatmap")
corr_cols = st.multiselect("Select numeric columns", options=numeric_cols, default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols)
corr_matrix(filtered_df, corr_cols)

#-------------------Scatter, Box, Violin, and Line Plots--------------------
st.subheader("Scatter, Box, Line, and Violin Plot Analysis")
c3, c4= st.columns(2)
with c3:
    plot_type= st.selectbox("Select Plot Type", ["Violin","Box"])
    x_col=st.selectbox("Select X-axis column", options=df.columns)
    y_col=st.selectbox("Select Y-axis column", options=df.columns)

    fig = None
    if plot_type == "Violin":
        fig = px.violin(df, x=x_col, y=y_col, box=True, color=x_col,  color_discrete_sequence=px.colors.qualitative.Dark2, points="all", title=f"🎻 Violin Plot showing {x_col} against {y_col}")
    elif plot_type == "Box":
        fig = px.box(df, x=x_col, y=y_col, color=x_col,  color_discrete_sequence=px.colors.sequential.Viridis, title=f"📦 Box Plot showing {x_col} against {y_col}")
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)

with c4:
    plot_type = st.selectbox("Select Plot Type", ["Scatter","Line"], key="right_plot_type")
    x1_col = st.selectbox("Select X-axis column", options=df.columns, key="right_x")
    y1_col = st.selectbox("Select Y-axis column", options=df.columns, key="right_y")
    fig = None
    if plot_type == "Scatter":
        fig = px.scatter(df, x=x1_col, y=y1_col, color=x_col,  color_discrete_sequence=px.colors.qualitative.Bold, title=f"🔵 Scatter Plot showing {x1_col} against {y1_col}")
    elif plot_type == "Line":
        fig = px.line(df, x=x1_col, y=y1_col, color=x_col, color_discrete_sequence=px.colors.qualitative.Set2, title=f"📈 Line Plot showing {x1_col} against {y1_col}")

    if fig:
        st.plotly_chart(fig, use_container_width=True)

# ---------------------- Data Table & Download ----------------------
st.subheader("🗂️ Data Table")
st.dataframe(filtered_df, use_container_width=True)
download_csv(filtered_df, label="⬇️ Download Data", filename="smartphone_data.csv")




