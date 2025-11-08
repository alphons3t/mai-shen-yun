# streamlit run mai-shen-yun.py 
# how to run this.
#   Local URL: http://localhost:8501
#  Network URL: http://10.244.154.4:8501


# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from datetime import datetime

DATA_DIR = Path("data")

@st.cache_data
def load_csv(name):
    # Added error handling for missing files/folders
    return pd.read_csv(DATA_DIR/name)

def preprocess(purchases, sales, recipes, shipments):
    # Data Cleaning and Merging
    sales['date'] = pd.to_datetime(sales['date'])
    purchases['date'] = pd.to_datetime(purchases['date'])
    
    # Shipment processing for tracking
    shipments['purchase_date'] = pd.to_datetime(shipments['purchase_date'])
    shipments['delivery_date'] = pd.to_datetime(shipments['delivery_date'])
    shipments['delay_days'] = (shipments['delivery_date'] - shipments['purchase_date']).dt.days

    recipes = recipes.copy()

    # 1. Calculate Ingredient Usage
    merged = sales.merge(recipes, on='menu_item_id', how='left')
    merged['ingredient_used'] = merged['quantity_sold'] * merged['quantity_per_item']
    
    # Weekly usage for alerts and forecast
    usage_weekly = merged.groupby([pd.Grouper(key='date', freq='W'), 'ingredient_id'])['ingredient_used'].sum().reset_index()
    
    # Monthly usage for seasonal trends analysis
    usage_monthly = merged.groupby([pd.Grouper(key='date', freq='M'), 'ingredient_id'])['ingredient_used'].sum().reset_index()

    # 2. Calculate Total Cost (Cost Optimization)
    # Assumes 'unit_cost' column exists in purchases.csv for spending analysis
    if 'unit_cost' in purchases.columns:
        purchases['total_cost'] = purchases['quantity'] * purchases['unit_cost']
    else:
        # Placeholder cost if 'unit_cost' is missing - important for robustness
        purchases['total_cost'] = purchases['quantity'] * 1 
        st.sidebar.warning("Using quantity as a proxy for cost as 'unit_cost' column was not found in purchases.csv. Results are not actual dollar amounts.")

    
    return usage_weekly, usage_monthly, purchases, shipments

def main():
    st.set_page_config(layout="wide") # Use wide layout for better dashboard feel
    st.title("Mai Shan Yun — Inventory Intelligence")
    st.sidebar.header("Data Configuration")
    st.sidebar.write("Place your CSVs in the data/ folder named: purchases.csv, sales.csv, recipes.csv, shipments.csv")
    if st.sidebar.button("Reload data"):
        st.cache_data.clear()

    # Initialize variables outside of the try block to ensure they are defined
    purchases, sales, recipes, shipments = None, None, None, None
    data_loaded = False

    try:
        purchases = load_csv("purchases.csv")
        sales = load_csv("sales.csv")
        recipes = load_csv("recipes.csv")
        # Load the new shipments data
        shipments = load_csv("shipments.csv")
        data_loaded = True
    except Exception as e:
        # Updated warning message to include all assumed files
        st.warning("Could not load all CSVs. Please ensure data/purchases.csv, data/sales.csv, data/recipes.csv, and data/shipments.csv exist in your data/ folder.")
        st.error(str(e))
        st.stop()

    # Only run preprocess and display the dashboard if all data was loaded successfully
    if data_loaded:
        # Process all data and get the new outputs
        usage_weekly, usage_monthly, purchases_with_cost, shipments_processed = preprocess(purchases, sales, recipes, shipments)
        
        # Align variables for minimal change in existing code logic
        usage = usage_weekly 
        purchases = purchases_with_cost
        shipments = shipments_processed

        tabs = st.tabs(["Overview & Alerts", "Forecasting", "Cost Optimization", "Shipments & Logistics"])

        # -------------------------------------------------------------------------
        # TAB 1: OVERVIEW & ALERTS (Original content + Monthly Trends)
        # -------------------------------------------------------------------------
        with tabs[0]:
            st.header("Real-time Inventory Overview")

            # Sidebar ingredient selection for KPIs
            ingredient_options = usage['ingredient_id'].unique()
            selected_ingredient = st.sidebar.selectbox("Select ingredient for KPIs and usage overview", ingredient_options)

            # KPIs
            total_usage = usage['ingredient_used'].sum()
            ingredient_usage = usage[usage['ingredient_id'] == selected_ingredient]['ingredient_used'].sum()
            total_purchases_q = purchases['quantity'].sum()
            ingredient_purchases_q = purchases[purchases['ingredient_id'] == selected_ingredient]['quantity'].sum()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Usage (All Ingredients)", f"{total_usage:.0f}")
            col2.metric(f"Usage for {selected_ingredient}", f"{ingredient_usage:.0f}")
            col3.metric("Total Purchases Quantity", f"{total_purchases_q:.0f}")
            col4.metric(f"Purchases for {selected_ingredient}", f"{ingredient_purchases_q:.0f}")

            st.subheader("Monthly Usage Trend (Seasonal Analysis)")
            # Aggregate usage monthly (sum across all ingredients)
            df_monthly_plot = usage_monthly.groupby(usage_monthly['date'].dt.to_period('M'))['ingredient_used'].sum().reset_index()
            df_monthly_plot['Month'] = df_monthly_plot['date'].astype(str)
            fig_monthly = px.bar(
                df_monthly_plot,
                x='Month',
                y='ingredient_used',
                title='Total Ingredient Usage by Month (Identifying Seasonal Trends)',
                labels={'Month': 'Month', 'ingredient_used': 'Total Usage Quantity'},
                color_discrete_sequence=px.colors.qualitative.T10 # Nice color scheme
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

            st.subheader("Top ingredients by estimated weekly usage")
            top = usage.groupby('ingredient_id')['ingredient_used'].sum().reset_index().sort_values('ingredient_used', ascending=False).head(10)
            st.dataframe(top, use_container_width=True)
            
            # Interactive timeseries plot for selected ingredient
            st.subheader("Weekly usage trends for selected ingredient")
            ingredient_choice = st.selectbox(
                "Select an ingredient to visualize its detailed weekly trend",
                usage['ingredient_id'].unique(),
                index=0,
                key='weekly_plot_select'
            )
            df_plot = usage[usage['ingredient_id'] == ingredient_choice]
            fig = px.line(
                df_plot,
                x='date',
                y='ingredient_used',
                title=f'Weekly usage trend for {ingredient_choice}',
                labels={'ingredient_used': 'Usage Quantity'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Minimal reorder logic example (Reorder Alerts)
            st.subheader("Actionable Reorder Alerts (Simple)")
            avg_daily = usage.groupby('ingredient_id')['ingredient_used'].mean() / 7.0
            # Calculate current inventory: Purchased - Used
            current_inventory = purchases.groupby('ingredient_id')['quantity'].sum() - usage.groupby('ingredient_id')['ingredient_used'].sum()
            df_alert = pd.DataFrame({
                'avg_daily_usage': avg_daily,
                'current_inventory': current_inventory
            }).fillna(0)
            df_alert['days_left'] = df_alert['current_inventory'] / (df_alert['avg_daily_usage'] + 1e-9)
            
            # Highlight items running low (e.g., < 7 days of stock)
            df_alert = df_alert.reset_index()
            def color_days_left(val):
                color = 'background-color: #f8d7da' if val < 7 else '' # Light red for low stock
                return color
            
            st.dataframe(
                df_alert.sort_values('days_left').head(10), 
                use_container_width=True,
                column_config={
                    "days_left": st.column_config.NumberColumn(
                        "Days Left",
                        format="%.1f",
                        help="Stocking days remaining before running out"
                    )
                }
            )
            st.caption("Alerts are based on average daily usage. Focus on ingredients with < 7 days left.")

        # -------------------------------------------------------------------------
        # TAB 2: FORECAST (Original Content)
        # -------------------------------------------------------------------------
        with tabs[1]:
            st.header("Ingredient Demand Forecasting")
            st.markdown("A simple 4-week moving average is used to predict future ingredient demand based on historical weekly usage.")
            
            forecast_ingredient = st.selectbox(
                "Select ingredient to forecast", 
                usage['ingredient_id'].unique(),
                key='forecast_select'
            )

            df_forecast = usage[usage['ingredient_id'] == forecast_ingredient].sort_values('date')
            
            # Ensure enough data points for a smooth moving average
            window_size = 4
            if len(df_forecast) < window_size:
                st.warning(f"Not enough data points ({len(df_forecast)}) to calculate a {window_size}-week moving average accurately.")

            df_forecast['moving_avg'] = df_forecast['ingredient_used'].rolling(window=window_size, min_periods=1).mean()

            # Simple forecast: project next 4 weeks as last moving average value
            last_date = df_forecast['date'].max()
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=4, freq='W')
            last_moving_avg = df_forecast['moving_avg'].iloc[-1] if not df_forecast.empty else 0
            forecast_values = [last_moving_avg] * 4
            forecast_df = pd.DataFrame({'date': forecast_dates, 'forecasted_usage': forecast_values})

            fig = px.line(
                df_forecast, 
                x='date', 
                y='ingredient_used', 
                title=f'4-Week Usage History and Forecast for {forecast_ingredient}', 
                labels={'ingredient_used': 'Usage Quantity'},
                line_dash_map={'ingredient_used': 'solid'}
            )
            
            # Add the forecast line
            fig.add_scatter(
                x=forecast_df['date'], 
                y=forecast_df['forecasted_usage'], 
                mode='lines+markers', 
                name='Forecast (4W MA)',
                line=dict(dash='dash', color='red')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Forecasted Demand Next 4 Weeks (Units)", f"{sum(forecast_values):.0f}")


        # -------------------------------------------------------------------------
        # TAB 3: COST OPTIMIZATION (New Content)
        # -------------------------------------------------------------------------
        with tabs[2]:
            st.header("Cost Optimization Analysis")
            st.markdown("Identify the ingredients and purchases that drive the highest spending to prioritize cost negotiations and inventory control.")

            st.subheader("Top 10 Cost Drivers by Ingredient")
            cost_drivers = purchases.groupby('ingredient_id')['total_cost'].sum().reset_index().sort_values('total_cost', ascending=False).head(10)
            cost_drivers['Cost %'] = (cost_drivers['total_cost'] / purchases['total_cost'].sum()) * 100
            cost_drivers.rename(columns={'total_cost': 'Total Cost ($)'}, inplace=True)
            
            fig_cost = px.bar(
                cost_drivers,
                x='ingredient_id',
                y='Total Cost ($)',
                color='Cost %',
                title='Ingredients Driving the Highest Total Cost',
                labels={'ingredient_id': 'Ingredient ID'},
                hover_data=['Cost %']
            )
            st.plotly_chart(fig_cost, use_container_width=True)
            st.dataframe(cost_drivers, use_container_width=True, hide_index=True)
            
            st.subheader("Purchase Cost Trends Over Time")
            # Aggregate purchase cost monthly
            cost_monthly = purchases.groupby(purchases['date'].dt.to_period('M'))['total_cost'].sum().reset_index()
            cost_monthly['Month'] = cost_monthly['date'].astype(str)
            
            fig_cost_trend = px.line(
                cost_monthly,
                x='Month',
                y='total_cost',
                title='Total Monthly Purchase Spending',
                labels={'total_cost': 'Total Cost ($)'},
                line_shape='linear'
            )
            st.plotly_chart(fig_cost_trend, use_container_width=True)


        # -------------------------------------------------------------------------
        # TAB 4: SHIPMENTS & LOGISTICS (New Content)
        # -------------------------------------------------------------------------
        with tabs[3]:
            st.header("Shipments and Logistics Intelligence")
            st.markdown("Track supplier performance and logistics efficiency to identify potential supply chain risks.")

            # Metrics for shipment data
            col_s1, col_s2, col_s3 = st.columns(3)
            avg_delay = shipments['delay_days'].mean()
            max_delay = shipments['delay_days'].max()
            total_shipments = shipments.shape[0]
            
            col_s1.metric("Total Shipments Tracked", f"{total_shipments}")
            col_s2.metric("Average Delivery Time", f"{avg_delay:.1f} Days", delta="Lower is better")
            col_s3.metric("Maximum Recorded Delay", f"{max_delay:.0f} Days")
            
            st.subheader("Distribution of Delivery Delays (Days)")
            # Filter out negative delays if any
            delay_data = shipments[shipments['delay_days'] >= 0] 
            if not delay_data.empty:
                fig_delay = px.histogram(
                    delay_data, 
                    x='delay_days', 
                    nbins=20, 
                    title='Frequency of Shipment Delivery Times',
                    labels={'delay_days': 'Delivery Time (Days)', 'count': 'Number of Shipments'},
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig_delay, use_container_width=True)
            else:
                st.info("No valid shipment delay data found to visualize.")
                
            st.subheader("Shipment Frequency Over Time")
            shipment_freq = shipments.groupby(pd.Grouper(key='delivery_date', freq='M')).size().reset_index(name='shipment_count')
            fig_freq = px.line(
                shipment_freq, 
                x='delivery_date', 
                y='shipment_count', 
                title='Monthly Shipment Frequency (Demand & Supply Rhythm)',
                labels={'delivery_date': 'Month', 'shipment_count': 'Number of Shipments Delivered'}
            )
            st.plotly_chart(fig_freq, use_container_width=True)

if __name__ == "__main__":
    main()