
import streamlit as st
import pandas as pd
import plotly.express as px

# Load your cleaned data
@st.cache_data
def load_data():
    df = pd.read_csv("updated_sales_purchase_data.csv")

    # Clean numeric fields
    df["sales_Grand Amount"] = pd.to_numeric(df["sales_Grand Amount"], errors="coerce").fillna(0)
    df["Purchase Grand Amount"] = pd.to_numeric(df["Purchase Grand Amount"], errors="coerce").fillna(0)

    # Clean date and extract year
    df['sales_Invoice Date'] = pd.to_datetime(df['sales_Invoice Date'], errors='coerce')
    df['sales_Year'] = df['sales_Invoice Date'].dt.year

    # Calculate Net Profit
    df['Net Profit'] = df["sales_Grand Amount"] - df["Purchase Grand Amount"]
    return df

df = load_data()

# Sidebar
st.sidebar.title("🔍 Dashboard Navigation")
tabs = ["📊 Summary", "📈 Trends", "🧾 Tax Summary", "👥 People", "📋 Invoices"]
selected_tab = st.sidebar.radio("Go to", tabs)

# Time filter
years = sorted(df['sales_Year'].dropna().unique())
selected_year = st.sidebar.selectbox("📅 Select Year", years)

df_year = df[df['sales_Year'] == selected_year]

# TAB 1 - Summary
if selected_tab == "📊 Summary":
    st.title(f"📊 Business Summary - {selected_year}")

    total_sales = df_year['sales_Grand Amount'].sum()
    total_purchase = df_year['Purchase Grand Amount'].sum()
    net_profit = df_year['Net Profit'].sum()

    gst_out = df_year[['sales_Tax Amount CGST', 'sales_Tax Amount SGST', 'sales_Tax Amount IGST']].apply(pd.to_numeric, errors='coerce').sum().sum()
    gst_in = df_year[['Purchase Tax Amount CGST', 'Purchase Tax Amount SGST', 'Purchase Tax Amount IGST']].apply(pd.to_numeric, errors='coerce').sum().sum()
    gst_liability = gst_out - gst_in

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Revenue", f"₹{total_sales:,.2f}")
    col2.metric("💸 Total Purchase", f"₹{total_purchase:,.2f}")
    col3.metric("📈 Net Profit", f"₹{net_profit:,.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("🧾 GST Output", f"₹{gst_out:,.2f}")
    col5.metric("📥 GST Input", f"₹{gst_in:,.2f}")
    col6.metric("⚖️ GST Payable", f"₹{gst_liability:,.2f}")

    col7, col8 = st.columns(2)
    col7.metric("👥 Unique Customers", df_year['sales_Customer Name'].nunique())
    col8.metric("🏢 Unique Vendors", df_year['Purchase Customer Name'].nunique())

# TAB 2 - Trends
elif selected_tab == "📈 Trends":
    st.title(f"📈 Trends - {selected_year}")

    sales_trend = df_year.groupby(df_year['sales_Invoice Date'].dt.to_period("M"))['sales_Grand Amount'].sum().reset_index()
    sales_trend['sales_Invoice Date'] = sales_trend['sales_Invoice Date'].dt.to_timestamp()

    purchase_trend = df_year.groupby(df_year['Purchase Invoice Date'].dt.to_period("M"))['Purchase Grand Amount'].sum().reset_index()
    purchase_trend['Purchase Invoice Date'] = purchase_trend['Purchase Invoice Date'].dt.to_timestamp()

    st.subheader("🟩 Monthly Sales")
    fig1 = px.line(sales_trend, x='sales_Invoice Date', y='sales_Grand Amount', markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🟥 Monthly Purchases")
    fig2 = px.line(purchase_trend, x='Purchase Invoice Date', y='Purchase Grand Amount', markers=True)
    st.plotly_chart(fig2, use_container_width=True)

# TAB 3 - Tax Summary
elif selected_tab == "🧾 Tax Summary":
    st.title(f"🧾 GST Breakdown - {selected_year}")

    gst_breakdown = {
        'CGST Out': df_year['sales_Tax Amount CGST'].sum(),
        'SGST Out': df_year['sales_Tax Amount SGST'].sum(),
        'IGST Out': df_year['sales_Tax Amount IGST'].sum(),
        'CGST In': df_year['Purchase Tax Amount CGST'].sum(),
        'SGST In': df_year['Purchase Tax Amount SGST'].sum(),
        'IGST In': df_year['Purchase Tax Amount IGST'].sum(),
    }

    gst_df = pd.DataFrame.from_dict(gst_breakdown, orient='index', columns=['Amount'])
    st.bar_chart(gst_df)

    st.dataframe(gst_df.style.format("₹{:,.2f}"))

# TAB 4 - People Overview
elif selected_tab == "👥 People":
    st.title(f"👥 Customer & Vendor Overview - {selected_year}")

    top_clients = df_year.groupby('sales_Customer Name')['sales_Grand Amount'].sum().sort_values(ascending=False).head(5)
    top_vendors = df_year.groupby('Purchase Customer Name')['Purchase Grand Amount'].sum().sort_values(ascending=False).head(5)

    st.subheader("🏆 Top 5 Clients by Sales")
    st.dataframe(top_clients.reset_index().rename(columns={'sales_Grand Amount': 'Amount'}).style.format("₹{:,.2f}"))

    st.subheader("📦 Top 5 Vendors by Spend")
    st.dataframe(top_vendors.reset_index().rename(columns={'Purchase Grand Amount': 'Amount'}).style.format("₹{:,.2f}"))

# TAB 5 - Invoices
elif selected_tab == "📋 Invoices":
    st.title(f"📋 All Invoices - {selected_year}")

    st.subheader("🧾 Sales Invoices")
    st.dataframe(df_year[['sales_Invoice Number', 'sales_Invoice Date', 'sales_Customer Name', 'sales_Grand Amount']].sort_values(by='sales_Invoice Date').reset_index(drop=True))

    st.subheader("📥 Purchase Invoices")
    st.dataframe(df_year[['Purchase Invoice Number', 'Purchase Invoice Date', 'Purchase Customer Name', 'Purchase Grand Amount']].sort_values(by='Purchase Invoice Date').reset_index(drop=True))

    st.download_button("⬇️ Download Sales Invoices", df_year.to_csv(index=False), "sales_data.csv", "text/csv")
