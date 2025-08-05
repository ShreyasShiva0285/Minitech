import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------- Styling and Theme Toggle --------------------
st.set_page_config(layout="wide")

theme = st.sidebar.selectbox("🎨 Select Theme", ["Light", "Dark", "Modern Blue"])

if theme == "Light":
    bg_color = "#ffffff"
    text_color = "#1f2e3d"
    plot_bg = "#f9f9f9"
    primary_color = "#4a90e2"
    highlight_color = "#d0e8ff"  # Light blue

elif theme == "Dark":
    bg_color = "#1e1e1e"
    text_color = "#f5f5f5"
    plot_bg = "#2c2c2c"
    primary_color = "#bb86fc"
    highlight_color = "#333333"  # Dark gray

elif theme == "Modern Blue":
    bg_color = "#0f172a"
    text_color = "#e2e8f0"
    plot_bg = "#1e293b"
    primary_color = "#38bdf8"
    highlight_color = "#1e40af"  # Blue


st.markdown(f"""
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{
            font-family: 'Open Sans', sans-serif;
            font-size: 16px;
            background-color: {bg_color};
            color: {text_color};
        }}
        h1, h2, h3, h4 {{
            color: {text_color};
            font-weight: 600;
        }}
        .stMetric {{
            background-color: {plot_bg};
            padding: 0.5rem;
            border-radius: 0.5rem;
            border-left: 5px solid {primary_color};
        }}
        .stButton>button:hover {{
            background-color: {highlight_color};
            color: white;
        }}
        @media (max-width: 768px) {{
            html, body, [class*="css"] {{
                font-size: 14px;
            }}
        }}
    </style>
""", unsafe_allow_html=True)


# Inject CSS
st.markdown(f"""
    <style>
        .main {{
            background: linear-gradient(to bottom right, #ffffff, #e0f2ff);
            color: {text_color};
        }}

        div[data-testid="stMetric"] {{
            background-color: {plot_bg};
            padding: 1rem;
            border-radius: 0.75rem;
            border-left: 6px solid {primary_color};
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }}

        /* Make the metric label and value color consistent */
        div[data-testid="stMetric"] > label, 
        div[data-testid="stMetric"] > div {{
            color: {text_color} !important;
        }}

        /* Optional: style metric columns if needed */
        .block-container {{
            padding-top: 2rem;
        }}
    </style>
""", unsafe_allow_html=True)

import streamlit as st

# Inject custom CSS for table headers
st.markdown("""
    <style>
    /* General table styling */
    thead tr th {
        font-family: 'Segoe UI', sans-serif;
        font-size: 16px;
        padding: 12px 8px;
        text-align: left;
        vertical-align: middle;
    }

    /* Light background → dark text */
    thead tr th:has(div[data-testid="stMarkdownContainer"]:not([style*="background-color: #"])),
    thead tr th[style*="background-color: white"],
    thead tr th[style*="background-color: #f"],
    thead tr th[style*="background-color: rgb(255"],
    thead tr th[style*="background-color: rgba(255"] {
        color: black !important;
    }

    /* Dark background → white text */
    thead tr th[style*="background-color: black"],
    thead tr th[style*="background-color: #000"],
    thead tr th[style*="background-color: #2c3e50"],
    thead tr th[style*="background-color: rgb(0"],
    thead tr th[style*="background-color: rgba(0"] {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------- Data Loader --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("updated_sales_purchase_data.csv")
    df['sales_Invoice Date'] = pd.to_datetime(df['sales_Invoice Date'], dayfirst=True, errors='coerce')
    df['Purchase Invoice Date'] = pd.to_datetime(df['Purchase Invoice Date'], dayfirst=True, errors='coerce')
    df['sales_Year'] = df['sales_Invoice Date'].dt.year

    num_cols = [
        'sales_Grand Amount', 'Purchase Grand Amount',
        'sales_Tax Amount CGST', 'sales_Tax Amount SGST', 'sales_Tax Amount IGST',
        'Purchase Tax Amount CGST', 'Purchase Tax Amount SGST', 'Purchase Tax Amount IGST'
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Net Profit'] = df['sales_Grand Amount'] - df['Purchase Grand Amount']
    return df

df = load_data()

# -------------------- Sidebar Navigation --------------------
st.sidebar.title("🔍 Minitech Engineering Pvt Ltd...")
tabs = ["📋 Overview Of the Company", "📊 Summary Of Sales and Revenue", "📈 Trends & customers Data", "🧾 Tax Summary", "💹 Profitability"]
selected_tab = st.sidebar.radio("Go to", tabs)

years = sorted(pd.concat([
    df['sales_Invoice Date'].dropna().dt.year,
    df['Purchase Invoice Date'].dropna().dt.year
]).dropna().unique())

selected_year = st.sidebar.selectbox("📅 Select Year", years)

df_year = df[
    (df['sales_Invoice Date'].dt.year == selected_year) |
    (df['Purchase Invoice Date'].dt.year == selected_year)
]

# -------------------- Plotly Layout Theme --------------------
def plotly_layout(title):
    return {
        "title": {"text": title, "x": 0.5, "font": {"size": 20}},
        "paper_bgcolor": plot_bg,
        "plot_bgcolor": plot_bg,
        "font": {"family": "Open Sans", "size": 14, "color": text_color},
        "margin": {"l": 40, "r": 20, "t": 60, "b": 40},
        "xaxis": {"title_font": {"size": 14}, "tickangle": -45, "color": text_color},
        "yaxis": {"title_font": {"size": 14}, "color": text_color},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
    }

# -------------------- [REMAINDER OF YOUR CODE UNCHANGED] --------------------

# -------------------- Overview Tab --------------------
if selected_tab == "📋 Overview Of the Company":
    st.title("📋 Company Dashboard Overview")
    st.markdown("Welcome to the business intelligence dashboard. Use the sidebar to explore insights.")

    st.title(f" Executive Overview – {selected_year}")

    total_sales = df_year['sales_Grand Amount'].sum()
    total_purchases = df_year['Purchase Grand Amount'].sum()
    gst_out = df_year[['sales_Tax Amount CGST', 'sales_Tax Amount SGST', 'sales_Tax Amount IGST']].sum().sum()
    gst_in = df_year[['Purchase Tax Amount CGST', 'Purchase Tax Amount SGST', 'Purchase Tax Amount IGST']].sum().sum()
    net_profit = total_sales - total_purchases - gst_out

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Total Sales", f"₹{total_sales:,.2f}")
    col2.metric("📉 Total Purchases", f"₹{total_purchases:,.2f}")
    col3.metric("💰 Net Profit", f"₹{net_profit:,.2f}")

    col4, col5 = st.columns(2)
    col4.metric("🧾 GST Collected (Out)", f"₹{gst_out:,.2f}")
    col5.metric("🧾 GST Paid (In)", f"₹{gst_in:,.2f}")

    st.markdown("---")

    gross_margin = (total_sales - total_purchases) / total_sales * 100 if total_sales else 0
    profit_margin = net_profit / total_sales * 100 if total_sales else 0

    monthly_sales = (
        df_year.dropna(subset=['sales_Invoice Date'])
        .groupby(df_year['sales_Invoice Date'].dt.to_period("M"))['sales_Grand Amount']
        .sum().reset_index()
    )
    monthly_sales['sales_Invoice Date'] = monthly_sales['sales_Invoice Date'].dt.to_timestamp()
    monthly_sales['Month_Num'] = np.arange(len(monthly_sales))

    if len(monthly_sales) >= 2:
        X = monthly_sales[['Month_Num']]
        y = monthly_sales['sales_Grand Amount']
        model = LinearRegression().fit(X, y)
        next_month_num = [[X['Month_Num'].max() + 1]]
        forecast_value = model.predict(next_month_num)[0]
        forecast_value_display = f"₹{forecast_value:,.2f}"
    else:
        forecast_value_display = "Not enough data"

    col6, col_forecast, col7 = st.columns(3)
    col6.metric("📊 Gross Margin", f"{gross_margin:.2f}%")
    col_forecast.metric("📅 Next Month Forecast", forecast_value_display)
    col7.metric("💼 Net Profit Margin", f"{profit_margin:.2f}%")

# -------------------- Summary Tab --------------------
elif selected_tab == "📊 Summary Of Sales and Revenue":
    st.title("📊 Sales and Revenue Summary")

    total_revenue = df_year['sales_Grand Amount'].sum()
    st.metric("Total Revenue", f"₹{total_revenue:,.2f}")

    gst_paid = df_year['sales_Tax Amount CGST'].sum() + df_year['sales_Tax Amount SGST'].sum()
    igst_paid = df_year['sales_Tax Amount IGST'].sum()

    st.metric("GST Paid", f"₹{gst_paid:,.2f}")
    st.metric("IGST Paid", f"₹{igst_paid:,.2f}")

    st.subheader("🏆 Top 5 Clients by Sales")
    if 'sales_Customer Name' in df_year.columns:
        top_clients = df_year.groupby("sales_Customer Name")['sales_Grand Amount'].sum().nlargest(5).reset_index()
        st.table(top_clients.rename(columns={"sales_Customer Name": "Client", "sales_Grand Amount": "Total Sales"}))
    else:
        st.warning("⚠️ 'sales_Customer Name' column missing.")

# -------------------- Trends Tab --------------------
elif selected_tab == "📈 Trends & customers Data":
    st.title("📈 Sales Trends & Customer Insights")
    st.subheader(" Monthly Sales Trend")
    sales_trend = df_year.dropna(subset=['sales_Invoice Date']).groupby(
        df_year['sales_Invoice Date'].dt.to_period("M")
    )['sales_Grand Amount'].sum().reset_index()
    sales_trend['sales_Invoice Date'] = sales_trend['sales_Invoice Date'].dt.to_timestamp()

    fig1 = px.line(sales_trend, x='sales_Invoice Date', y='sales_Grand Amount', markers=True)
    fig1.update_layout(**plotly_layout("Monthly Sales"))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("👤 Top 5 Customers by Revenue")
    top_customers = df_year.groupby("sales_Customer Name")["sales_Grand Amount"].sum().nlargest(5).reset_index()
    st.table(top_customers.rename(columns={"sales_Customer Name": "Customer", "sales_Grand Amount": "Total Revenue"}))

    st.subheader("🧾 Most Frequent Invoice Clients")
    frequent_clients = df_year["sales_Customer Name"].value_counts().head(5).reset_index()
    frequent_clients.columns = ["Customer", "No. of Invoices"]
    st.table(frequent_clients)

    st.subheader("📊 Monthly Purchases Trend")
    purchase_trend = df_year.dropna(subset=['Purchase Invoice Date']).groupby(
        df_year['Purchase Invoice Date'].dt.to_period("M")
    )['Purchase Grand Amount'].sum().reset_index()
    purchase_trend['Purchase Invoice Date'] = purchase_trend['Purchase Invoice Date'].dt.to_timestamp()

    fig2 = px.line(purchase_trend, x='Purchase Invoice Date', y='Purchase Grand Amount', markers=True)
    fig2.update_layout(**plotly_layout("Monthly Purchases"))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🧍‍♂️ Top 5 Vendors by Spend")
    top_vendors = df_year.groupby("Purchase Customer Name")["Purchase Grand Amount"].sum().nlargest(5).reset_index()
    st.table(top_vendors.rename(columns={"Purchase Customer Name": "Vendor", "Purchase Grand Amount": "Total Spend"}))

    st.subheader("🧾 Frequent Vendors")
    frequent_vendors = df_year["Purchase Customer Name"].value_counts().head(5).reset_index()
    frequent_vendors.columns = ["Vendor", "No. of Purchases"]
    st.table(frequent_vendors)

# -------------------- Tax Summary --------------------
elif selected_tab == "🧾 Tax Summary":
    st.title("🧾 GST Summary & Breakdown")
    
    for col in [
        'sales_Tax Amount CGST', 'sales_Tax Amount SGST', 'sales_Tax Amount IGST',
        'Purchase Tax Amount CGST', 'Purchase Tax Amount SGST', 'Purchase Tax Amount IGST'
    ]:
        if col not in df_year.columns:
            df_year[col] = 0

    gst_breakdown = {
        'CGST Out': df_year['sales_Tax Amount CGST'].sum(),
        'SGST Out': df_year['sales_Tax Amount SGST'].sum(),
        'IGST Out': df_year['sales_Tax Amount IGST'].sum(),
        'CGST In': df_year['Purchase Tax Amount CGST'].sum(),
        'SGST In': df_year['Purchase Tax Amount SGST'].sum(),
        'IGST In': df_year['Purchase Tax Amount IGST'].sum(),
    }

    net_gst_df = pd.DataFrame({
        'GST Type': ['CGST', 'SGST', 'IGST'],
        'Outward GST': [gst_breakdown['CGST Out'], gst_breakdown['SGST Out'], gst_breakdown['IGST Out']],
        'Input Credit': [gst_breakdown['CGST In'], gst_breakdown['SGST In'], gst_breakdown['IGST In']],
        'Net Payable': [
            gst_breakdown['CGST Out'] - gst_breakdown['CGST In'],
            gst_breakdown['SGST Out'] - gst_breakdown['SGST In'],
            gst_breakdown['IGST Out'] - gst_breakdown['IGST In'],
        ]
    })

    st.subheader("🔍 Net GST Payable / Receivable")
    st.dataframe(net_gst_df.style.format({
        'Outward GST': "₹{:,.2f}",
        'Input Credit': "₹{:,.2f}",
        'Net Payable': "₹{:,.2f}"
    }))

    # Pie Chart for Net GST Payable
    st.subheader("📊 Net GST Payable Distribution")

    fig_pie = px.pie(
        net_gst_df,
        names='GST Type',
        values='Net Payable',
        title='Net GST Payable by Type',
        hole=0.4  # donut-style
    )

    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(
        showlegend=True,
        font=dict(color=text_color),
        paper_bgcolor=bg_color,
        plot_bgcolor=plot_bg,
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    # Clients and Vendors same as previous version (unchanged)...

# -------------------- Profitability --------------------
elif selected_tab == "💹 Profitability":
    st.title("💹 Profitability")
    st.title(f"Profit & Loss Overview - {selected_year}")

    total_sales = df_year['sales_Grand Amount'].sum()
    total_purchases = df_year['Purchase Grand Amount'].sum()
    gst_out = df_year[['sales_Tax Amount CGST', 'sales_Tax Amount SGST', 'sales_Tax Amount IGST']].sum().sum()
    net_profit = total_sales - total_purchases - gst_out

    st.subheader("📊 Profit Composition Waterfall Chart")
    fig_waterfall = go.Figure(go.Waterfall(
        name="Profit Flow",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Total Sales", "(-) Purchases", "(-) GST Paid", "Net Profit"],
        y=[total_sales, -total_purchases, -gst_out, net_profit],
        text=[f"₹{total_sales:,.2f}", f"-₹{total_purchases:,.2f}", f"-₹{gst_out:,.2f}", f"₹{net_profit:,.2f}"],
        connector={"line": {"color": "gray"}}
    ))
    fig_waterfall.update_layout(**plotly_layout("Profit Composition Waterfall"))
    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.subheader("📅 Quarterly Profit Trend")
    df_year['Quarter'] = df_year['sales_Invoice Date'].dt.to_period("Q").astype(str)
    quarterly_profit = df_year.groupby('Quarter').agg({
        'sales_Grand Amount': 'sum',
        'Purchase Grand Amount': 'sum'
    }).reset_index()
    quarterly_profit['Net Profit'] = quarterly_profit['sales_Grand Amount'] - quarterly_profit['Purchase Grand Amount']

    fig_bar = px.bar(quarterly_profit, x='Quarter', y='Net Profit', title='Quarterly Net Profit', text='Net Profit')
    fig_bar.update_traces(texttemplate='₹%{text:,.2f}', textposition='outside')
    fig_bar.update_layout(**plotly_layout("Quarterly Net Profit"))
    st.plotly_chart(fig_bar, use_container_width=True)
