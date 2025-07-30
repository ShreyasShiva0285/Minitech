import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load your cleaned data
@st.cache_data
def load_data():
    df = pd.read_csv("updated_sales_purchase_data.csv")

    # Convert dates
    df['sales_Invoice Date'] = pd.to_datetime(df['sales_Invoice Date'], dayfirst=True, errors='coerce')
    df['Purchase Invoice Date'] = pd.to_datetime(df['Purchase Invoice Date'], dayfirst=True, errors='coerce')
    df['sales_Year'] = df['sales_Invoice Date'].dt.year

    # Convert monetary and tax fields to numeric
    num_cols = [
        'sales_Grand Amount', 'Purchase Grand Amount',
        'sales_Tax Amount CGST', 'sales_Tax Amount SGST', 'sales_Tax Amount IGST',
        'Purchase Tax Amount CGST', 'Purchase Tax Amount SGST', 'Purchase Tax Amount IGST'
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute Net Profit
    df['Net Profit'] = df['sales_Grand Amount'] - df['Purchase Grand Amount']

    return df

df = load_data()

# Sidebar
st.sidebar.title("🔍 Dashboard Navigation")
tabs = ["📊 Summary", "📈 Trends", "🧾 Tax Summary", "💹 Profitability"]
selected_tab = st.sidebar.radio("Go to", tabs)

# Time filter - Apply to both sales and purchase years
all_years = pd.concat([
    df['sales_Invoice Date'].dropna().dt.year,
    df['Purchase Invoice Date'].dropna().dt.year
]).dropna().unique()

years = sorted(all_years)
selected_year = st.sidebar.selectbox("📅 Select Year", years)

# Filter data for selected year for both sales and purchases
df_year = df[
    (df['sales_Invoice Date'].dt.year == selected_year) |
    (df['Purchase Invoice Date'].dt.year == selected_year)
]

# 🔍 TAB 1 - Summary
if selected_tab == "📊 Summary":
    st.subheader(f"📊 Summary - {selected_year}")

    # Total Revenue
    total_revenue = df_year['sales_Grand Amount'].sum()
    st.metric("Total Revenue", f"₹{total_revenue:,.2f}")

    # GST Paid = CGST + SGST
    gst_paid = df_year['sales_Tax Amount CGST'].sum() + df_year['sales_Tax Amount SGST'].sum()
    st.metric("GST Paid", f"₹{gst_paid:,.2f}")

    # IGST Paid
    igst_paid = df_year['sales_Tax Amount IGST'].sum()
    st.metric("IGST Paid", f"₹{igst_paid:,.2f}")

    # Top 5 Clients by Sales
    st.subheader("🏆 Top 5 Clients by Sales")
    if 'sales_Customer Name' in df_year.columns:
        top_clients = df_year.groupby("sales_Customer Name")['sales_Grand Amount'].sum().nlargest(5).reset_index()
        st.table(top_clients.rename(columns={
            "sales_Customer Name": "Client",
            "sales_Grand Amount": "Total Sales"
        }))
    else:
        st.warning("⚠️ 'sales_Customer Name' column not found.")


# TAB 2 - Trends
elif selected_tab == "📈 Trends":
    st.title(f"📈 Trends - {selected_year}")

    # --- SALES PERFORMANCE ---

    st.subheader("📈 Monthly Sales Trend")
    sales_trend = df_year.dropna(subset=['sales_Invoice Date']).groupby(
        df_year['sales_Invoice Date'].dt.to_period("M")
    )['sales_Grand Amount'].sum().reset_index()
    sales_trend['sales_Invoice Date'] = sales_trend['sales_Invoice Date'].dt.to_timestamp()

    fig1 = px.line(sales_trend, x='sales_Invoice Date', y='sales_Grand Amount', markers=True,
                   title="Monthly Sales")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("👤 Top 5 Customers by Revenue")
    top_customers = (
        df_year.groupby("sales_Customer Name")["sales_Grand Amount"]
        .sum().nlargest(5).reset_index()
        .rename(columns={"sales_Customer Name": "Customer", "sales_Grand Amount": "Total Revenue"})
    )
    st.table(top_customers)

    st.subheader("🧾 Most Frequent Invoice Clients")
    frequent_clients = (
        df_year["sales_Customer Name"]
        .value_counts().head(5).reset_index()
        .rename(columns={"index": "Customer", "sales_Customer Name": "No. of Invoices"})
    )
    st.table(frequent_clients)

    # --- PURCHASE MONITORING ---

    st.subheader("📊 Monthly Purchases Trend")
    purchase_trend = df_year.dropna(subset=['Purchase Invoice Date']).groupby(
        df_year['Purchase Invoice Date'].dt.to_period("M")
    )['Purchase Grand Amount'].sum().reset_index()
    purchase_trend['Purchase Invoice Date'] = purchase_trend['Purchase Invoice Date'].dt.to_timestamp()

    fig2 = px.line(purchase_trend, x='Purchase Invoice Date', y='Purchase Grand Amount', markers=True,
                   title="Monthly Purchases")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🧍‍♂️ Top 5 Vendors by Spend")
    top_vendors = (
        df_year.groupby("Purchase Customer Name")["Purchase Grand Amount"]
        .sum().nlargest(5).reset_index()
        .rename(columns={"Purchase Customer Name": "Vendor", "Purchase Grand Amount": "Total Spend"})
    )
    st.table(top_vendors)

    st.subheader("🧾 Frequent Vendors")
    frequent_vendors = (
        df_year["Purchase Customer Name"]
        .value_counts().head(5).reset_index()
        .rename(columns={"index": "Vendor", "Purchase Customer Name": "No. of Purchases"})
    )
    st.table(frequent_vendors)


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

    # ✅ Pie Chart instead of bar
    gst_df_reset = gst_df.reset_index().rename(columns={'index': 'GST Type'})
    fig_pie = px.pie(
        gst_df_reset,
        names='GST Type',
        values='Amount',
        title='GST In vs Out Distribution',
        hole=0.4  # Optional donut chart
    )
    fig_pie.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

    # Table below the pie chart
    st.dataframe(gst_df.style.format("₹{:,.2f}"))


# TAB 4 – Profitability Overview
elif selected_tab == "💹 Profitability":
    st.title(f"💹 Profitability Overview - {selected_year}")

    # Define required values
    total_sales = df_year['sales_Grand Amount'].sum()
    total_purchases = df_year['Purchase Grand Amount'].sum()
    gst_out = (
        df_year['sales_Tax Amount CGST'].sum() +
        df_year['sales_Tax Amount SGST'].sum() +
        df_year['sales_Tax Amount IGST'].sum()
    )
    net_profit = total_sales - total_purchases - gst_out

    # Waterfall Chart
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
fig_waterfall.update_layout(title="Profit Composition Waterfall", waterfallgap=0.5)

st.plotly_chart(fig_waterfall, use_container_width=True)

# Monthly Profit Chart (Quarterly works better for large data)
st.subheader("📅 Quarterly Profit Trend")

df_year['Quarter'] = df_year['sales_Invoice Date'].dt.to_period("Q").astype(str)
quarterly_profit = df_year.groupby('Quarter').agg({
'sales_Grand Amount': 'sum',
'Purchase Grand Amount': 'sum'
    }).reset_index()
quarterly_profit['Net Profit'] = quarterly_profit['sales_Grand Amount'] - quarterly_profit['Purchase Grand Amount']

fig_bar = px.bar(
        quarterly_profit,
        x='Quarter',
        y='Net Profit',
        title='Quarterly Net Profit',
        text='Net Profit'
    )
fig_bar.update_traces(texttemplate='₹%{text:,.2f}', textposition='outside')
fig_bar.update_layout(yaxis_title="₹", xaxis_title="Quarter", uniformtext_minsize=8, uniformtext_mode='hide')
st.plotly_chart(fig_bar, use_container_width=True)


    #st.download_button("⬇️ Download Sales Invoices", df_year.to_csv(index=False), "sales_data.csv", "text/csv")

