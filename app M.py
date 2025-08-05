import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np


# -------------------- Styling --------------------
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Open Sans', sans-serif;
            font-size: 16px;
        }
        @media (max-width: 768px) {
            html, body, [class*="css"] {
                font-size: 14px;
            }
        }
        h1, h2, h3, h4 {
            color: #1f2e3d;
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

# -------------------- Plotly Theme --------------------
def plotly_layout(title):
    return {
        "title": {"text": title, "x": 0.5},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Open Sans", "size": 14},
        "margin": {"l": 40, "r": 20, "t": 60, "b": 40}
    }

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
