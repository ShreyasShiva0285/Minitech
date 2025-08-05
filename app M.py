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
st.sidebar.title("🔍 Minitech Engineering Pvt Ltd...")
tabs = ["📋 Overview Of the Company", "📊 Summary Of Sales and Revenue", "📈 Trends & customers Data", "🧾 Tax Summary", "💹 Profitability"]
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

# 📋 Overview Tab
if selected_tab == "📋 Overview Of the Company":
    st.title("📋 Company Dashboard Overview")
    st.markdown("""
    Welcome to the business intelligence dashboard.  
    Use the sidebar to navigate through Sales, Trends, Tax Summary, and Profitability.
    """)
    st.title(f" Executive Overview of the Firm – {selected_year}")

    # Metrics Summary
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

    # Profit Margins
    gross_margin = (total_sales - total_purchases) / total_sales * 100 if total_sales else 0
    profit_margin = net_profit / total_sales * 100 if total_sales else 0

    # Forecasting Next Month Sales using Linear Regression
    from sklearn.linear_model import LinearRegression
    import numpy as np

    monthly_sales = (
        df_year.dropna(subset=['sales_Invoice Date'])
        .groupby(df_year['sales_Invoice Date'].dt.to_period("M"))['sales_Grand Amount']
        .sum()
        .reset_index()
    )
    monthly_sales['sales_Invoice Date'] = monthly_sales['sales_Invoice Date'].dt.to_timestamp()
    monthly_sales['Month_Num'] = np.arange(len(monthly_sales))

    if len(monthly_sales) >= 2:  # at least 2 points to fit a line
        X = monthly_sales[['Month_Num']]
        y = monthly_sales['sales_Grand Amount']

        model = LinearRegression()
        model.fit(X, y)

        next_month_num = [[X['Month_Num'].max() + 1]]
        forecast_value = model.predict(next_month_num)[0]
        forecast_value_display = f"₹{forecast_value:,.2f}"
    else:
        forecast_value_display = "Not enough data"

    # Display Gross Margin, Forecast, Net Profit Margin
    col6, col_forecast, col7 = st.columns(3)
    col6.metric("📊 Gross Margin", f"{gross_margin:.2f}%")
    col_forecast.metric("📅 Next Month Forecast", forecast_value_display)
    col7.metric("💼 Net Profit Margin", f"{profit_margin:.2f}%")

    st.markdown("This overview summarizes your key financial health indicators for the selected year.")

# 📊 Summary Tab
elif selected_tab == "📊 Summary Of Sales and Revenue":
    st.title("📊 Sales and Revenue Summary")
    st.markdown("Snapshot of total sales, GST paid, and top clients to give a high-level view of business performance.")

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

# 📈 Trends Tab
elif selected_tab == "📈 Trends & customers Data":
    st.title("📈 Sales Trends & Customer Insights")
    st.markdown("Visualize monthly trends and explore top customers and vendors by revenue and frequency.")
    # (your existing logic stays here)
    st.title(f"Company Trends - {selected_year}")

    # --- SALES PERFORMANCE ---
    st.subheader(" Monthly Sales Trend")
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

elif selected_tab == "🧾 Tax Summary":
    st.title("🧾 GST Summary & Breakdown")

    # ✅ Ensure all required GST columns are present
    required_columns = [
        'sales_Tax Amount CGST', 'sales_Tax Amount SGST', 'sales_Tax Amount IGST',
        'Purchase Tax Amount CGST', 'Purchase Tax Amount SGST', 'Purchase Tax Amount IGST'
    ]
    for col in required_columns:
        if col not in df_year.columns:
            df_year[col] = 0

    # ✅ GST breakdown calculations
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
        'Outward GST': [
            gst_breakdown['CGST Out'], gst_breakdown['SGST Out'], gst_breakdown['IGST Out']
        ],
        'Input Credit': [
            gst_breakdown['CGST In'], gst_breakdown['SGST In'], gst_breakdown['IGST In']
        ],
        'Net Payable': [
            gst_breakdown['CGST Out'] - gst_breakdown['CGST In'],
            gst_breakdown['SGST Out'] - gst_breakdown['SGST In'],
            gst_breakdown['IGST Out'] - gst_breakdown['IGST In'],
        ]
    })

    # ✅ Display Net GST Summary
    st.subheader("🔍 Net GST Payable / Receivable")
    st.dataframe(net_gst_df.style.format({
        'Outward GST': "₹{:,.2f}",
        'Input Credit': "₹{:,.2f}",
        'Net Payable': "₹{:,.2f}"
    }))

    # 🏆 Top GST-Contributing Clients
    st.markdown("---")
    st.subheader("🏆 Top GST-Contributing Clients")

    df_year['Client GST Out'] = (
        df_year['sales_Tax Amount CGST'].fillna(0) +
        df_year['sales_Tax Amount SGST'].fillna(0) +
        df_year['sales_Tax Amount IGST'].fillna(0)
    )

    if 'sales_Customer Name' in df_year.columns:
        top_gst_clients = (
            df_year.groupby('sales_Customer Name')['Client GST Out']
            .sum().sort_values(ascending=False).head(10)
        )

        st.bar_chart(top_gst_clients, use_container_width=True)

        # ✅ Correct formatting to avoid .style.format errors
        top_gst_clients_df = top_gst_clients.reset_index()
        top_gst_clients_df.columns = ['Client', 'GST Collected']

        st.dataframe(top_gst_clients_df.style.format({
            'GST Collected': "₹{:,.2f}"
        }))
    else:
        st.warning("⚠️ 'sales_Customer Name' column not found in your data.")

    # 🏢 Top GST-Contributing Vendors
    st.subheader("🏢 Top GST-Contributing Vendors")

    df_year['Vendor GST In'] = (
        df_year['Purchase Tax Amount CGST'].fillna(0) +
        df_year['Purchase Tax Amount SGST'].fillna(0) +
        df_year['Purchase Tax Amount IGST'].fillna(0)
    )

    if 'Purchase Customer Name' in df_year.columns:
        top_gst_vendors = (
            df_year.groupby('Purchase Customer Name')['Vendor GST In']
            .sum().sort_values(ascending=False).head(10)
        )

        st.bar_chart(top_gst_vendors, use_container_width=True)

        # ✅ Correct formatting to avoid .style.format errors
        top_gst_vendors_df = top_gst_vendors.reset_index()
        top_gst_vendors_df.columns = ['Vendor', 'GST Paid']

        st.dataframe(top_gst_vendors_df.style.format({
            'GST Paid': "₹{:,.2f}"
        }))
    else:
        st.warning("⚠️ 'Purchase Customer Name' column not found in your data.")

# 💹 Profitability
elif selected_tab == "💹 Profitability":
    st.title("💹 Profitability Overview")
    st.markdown("Analyze profit composition, quarterly trends, and visualize earnings breakdown.")
    st.title(f"💹 Profit & Loss Overview - {selected_year}")

    total_sales = df_year['sales_Grand Amount'].sum()
    total_purchases = df_year['Purchase Grand Amount'].sum()
    gst_out = (
        df_year['sales_Tax Amount CGST'].sum() +
        df_year['sales_Tax Amount SGST'].sum() +
        df_year['sales_Tax Amount IGST'].sum()
    )
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
    fig_waterfall.update_layout(title="Profit Composition Waterfall", waterfallgap=0.5)
    st.plotly_chart(fig_waterfall, use_container_width=True)

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


