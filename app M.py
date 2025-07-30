
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

# 📅 Sidebar Year Selector (make sure this is placed before the summary block)
df['sales_Invoice Date'] = pd.to_datetime(df['sales_Invoice Date'])
df['sales_Year'] = df['sales_Invoice Date'].dt.year

years = sorted(df['sales_Year'].dropna().unique())
selected_year = st.sidebar.selectbox("📅 Select Year", years)

df_year = df[df['sales_Year'] == selected_year]  # Filter data by selected year


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
