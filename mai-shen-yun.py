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
from sklearn.cluster import KMeans

DATA_DIR = Path("data")

@st.cache_data
def load_csv(name):
    # Added error handling for missing files/folders
    return pd.read_csv(DATA_DIR/name)

def preprocess(purchases, sales, recipes, shipments):
    # Data Cleaning and Merging
    sales['date'] = pd.to_datetime(sales['date'])
    purchases['date'] = pd.to_datetime(purchases['date'])
    shipments['purchase_date'] = pd.to_datetime(shipments['purchase_date'])
    shipments['delivery_date'] = pd.to_datetime(shipments['delivery_date'])
    shipments['delay_days'] = (shipments['delivery_date'] - shipments['purchase_date']).dt.days

    merged = sales.merge(recipes, on='menu_item_id', how='left')
    merged['ingredient_used'] = merged['quantity_sold'] * merged['quantity_per_item']
    # Keep ingredient_name in merged if it exists in recipes
    if 'ingredient_name' in recipes.columns:
        merged['ingredient_name'] = merged['ingredient_name']
    elif 'ingredient' in recipes.columns:
        merged['ingredient_name'] = merged['ingredient']

    usage_weekly = merged.groupby([pd.Grouper(key='date', freq='W'), 'ingredient_id', 'ingredient_name'])['ingredient_used'].sum().reset_index()
    usage_monthly = merged.groupby([pd.Grouper(key='date', freq='M'), 'ingredient_id', 'ingredient_name'])['ingredient_used'].sum().reset_index()

    # 2. Calculate Total Cost (Cost Optimization)
    # Assumes 'unit_cost' column exists in purchases.csv for spending analysis
    if 'unit_cost' in purchases.columns:
        purchases['total_cost'] = purchases['quantity'] * purchases['unit_cost']
    else:
        # Placeholder cost if 'unit_cost' is missing - important for robustness
        purchases['total_cost'] = purchases['quantity'] * 1 
        st.sidebar.warning("Using quantity as a proxy for cost as 'unit_cost' column was not found in purchases.csv. Results are not actual dollar amounts.")

    return usage_weekly, usage_monthly, purchases, shipments

def monthly_orders_and_usage(purchases, usage_monthly, months=None):
    purchases = purchases.copy()
    purchases['month'] = purchases['date'].dt.month
    orders_monthly = purchases.groupby(['month', 'ingredient_id'])['quantity'].sum().reset_index()

    usage = usage_monthly.copy()
    usage['month'] = usage['date'].dt.month
    used_monthly = usage.groupby(['month', 'ingredient_id'])['ingredient_used'].sum().reset_index()

    if months:
        orders_monthly = orders_monthly[orders_monthly['month'].isin(months)]
        used_monthly = used_monthly[used_monthly['month'].isin(months)]

    merged = pd.merge(orders_monthly, used_monthly, on=['month', 'ingredient_id'], how='outer').fillna(0)
    merged.rename(columns={'quantity': 'ordered_qty', 'ingredient_used': 'used_qty'}, inplace=True)
    merged['waste'] = (merged['ordered_qty'] - merged['used_qty']).clip(lower=0)
    merged['shortage'] = (merged['used_qty'] - merged['ordered_qty']).clip(lower=0)
    return merged

def suggest_ordering_changes(merged_monthly, consecutive_months=2):
    suggestions = []
    for ingredient, grp in merged_monthly.groupby('ingredient_id'):
        grp = grp.sort_values('month')
        shortage_run = (grp['shortage'] > 0).astype(int)
        waste_run = (grp['waste'] > 0).astype(int)
        if shortage_run.rolling(window=consecutive_months).sum().iloc[-1] >= consecutive_months:
            avg_used = max(1, int(grp['used_qty'].mean()))
            suggestions.append({'ingredient_id': ingredient, 'action': 'increase', 'by_qty': int(0.1 * avg_used)})
        if waste_run.rolling(window=consecutive_months).sum().iloc[-1] >= consecutive_months:
            avg_ordered = max(1, int(grp['ordered_qty'].mean()))
            suggestions.append({'ingredient_id': ingredient, 'action': 'decrease', 'by_qty': int(0.1 * avg_ordered)})
    return pd.DataFrame(suggestions)

def cluster_trends(usage_monthly, n_clusters=4):
    pivot = usage_monthly.pivot_table(index='ingredient_id', columns=usage_monthly['date'].dt.to_period('M'),
                                      values='ingredient_used', aggfunc='sum').fillna(0)
    if pivot.empty:
        return pd.DataFrame(), pd.DataFrame()
    n_clusters = min(n_clusters, max(1, pivot.shape[0]))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(pivot.values)
    pivot['cluster'] = labels
    summary = pivot.groupby('cluster').sum().sum(axis=1).reset_index()
    summary.columns = ['cluster', 'total_usage']
    return pivot.reset_index(), summary

def generate_menu_suggestion(trending_ingredients):
    cooking_methods = ['Fried', 'Tossed', 'Braised', 'Grilled']
    flavor_profiles = ['Sweet', 'Savory', 'Sour', 'Spicy', 'Salty']
    top = trending_ingredients[:6]
    menu_items = []
    for i, ing in enumerate(top):
        method = cooking_methods[i % len(cooking_methods)]
        flavor = flavor_profiles[i % len(flavor_profiles)]
        item = f"Limited-Edition: {ing} {method} — {flavor} glaze"
        menu_items.append(item)
    return menu_items

def main():
    st.set_page_config(layout="wide")
    st.title("Mai Shan Yun — Inventory Intelligence")

    st.sidebar.header("Data Configuration")
    st.sidebar.write("Ensure purchases.csv, sales.csv, recipes.csv, and shipments.csv are in your data/ folder.")
    if st.sidebar.button("Reload data"):
        st.cache_data.clear()

    try:
        purchases = load_csv("purchases.csv")
        sales = load_csv("sales.csv")
        recipes = load_csv("recipes.csv")
        shipments = load_csv("shipments.csv")
    except Exception as e:
        st.warning("Could not load all CSVs. Please ensure all four data files exist.")
        st.error(str(e))
        st.stop()

    usage_weekly, usage_monthly, purchases, shipments = preprocess(purchases, sales, recipes, shipments)
    usage = usage_weekly

    tabs = st.tabs([
        "Overview & Alerts", "Forecasting", "Cost Optimization", 
        "Shipments & Logistics", "Ordering Efficiency", "Trend Analysis & Menu"
    ])

    # ---------------------------------------------------------------------
    # TAB 1: OVERVIEW & ALERTS
    # ---------------------------------------------------------------------
    with tabs[0]:
        st.header("Real-time Inventory Overview")
        if 'ingredient_name' in usage.columns:
            ingredient_options = usage[['ingredient_id', 'ingredient_name']].drop_duplicates()
            ingredient_display = {row.ingredient_name: row.ingredient_id for _, row in ingredient_options.iterrows()}
            selected_name = st.sidebar.selectbox("Select ingredient", list(ingredient_display.keys()))
            selected_ingredient = ingredient_display[selected_name]
        else:
            ingredient_options = usage['ingredient_id'].unique()
            selected_ingredient = st.sidebar.selectbox("Select ingredient", ingredient_options)

        total_usage = usage['ingredient_used'].sum()
        ingredient_usage = usage[usage['ingredient_id'] == selected_ingredient]['ingredient_used'].sum()
        total_purchases_q = purchases['quantity'].sum()
        ingredient_purchases_q = purchases[purchases['ingredient_id'] == selected_ingredient]['quantity'].sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Usage (All)", f"{total_usage:.0f}")
        col2.metric(f"{selected_ingredient} Usage", f"{ingredient_usage:.0f}")
        col3.metric("Total Purchases", f"{total_purchases_q:.0f}")
        col4.metric(f"{selected_ingredient} Purchases", f"{ingredient_purchases_q:.0f}")

        st.subheader("Monthly Usage Trends")
        df_monthly = usage_monthly.groupby(usage_monthly['date'].dt.to_period('M'))['ingredient_used'].sum().reset_index()
        df_monthly['Month'] = df_monthly['date'].astype(str)
        fig_monthly = px.bar(df_monthly, x='Month', y='ingredient_used', title='Total Ingredient Usage by Month')
        st.plotly_chart(fig_monthly, use_container_width=True)

        st.subheader("Reorder Alerts")
        avg_daily = usage.groupby('ingredient_id')['ingredient_used'].mean() / 7.0
        current_inventory = purchases.groupby('ingredient_id')['quantity'].sum() - usage.groupby('ingredient_id')['ingredient_used'].sum()
        df_alert = pd.DataFrame({'avg_daily_usage': avg_daily, 'current_inventory': current_inventory}).fillna(0)
        df_alert['days_left'] = df_alert['current_inventory'] / (df_alert['avg_daily_usage'] + 1e-9)
        st.dataframe(df_alert.sort_values('days_left').head(10), use_container_width=True)

    # ---------------------------------------------------------------------
    # TAB 2: FORECASTING
    # ---------------------------------------------------------------------
    with tabs[1]:
        st.header("Ingredient Demand Forecasting")
        forecast_ingredient = st.selectbox("Select ingredient", usage['ingredient_id'].unique(), key='forecast_select')
        df_forecast = usage[usage['ingredient_id'] == forecast_ingredient].sort_values('date')
        df_forecast['moving_avg'] = df_forecast['ingredient_used'].rolling(4, min_periods=1).mean()
        last_date = df_forecast['date'].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=4, freq='W')
        forecast_values = [df_forecast['moving_avg'].iloc[-1]] * 4
        forecast_df = pd.DataFrame({'date': forecast_dates, 'forecasted_usage': forecast_values})

        fig = px.line(df_forecast, x='date', y='ingredient_used', title=f'Usage Forecast: {forecast_ingredient}')
        fig.add_scatter(x=forecast_df['date'], y=forecast_df['forecasted_usage'], mode='lines+markers', name='Forecast')
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Forecasted Demand (4 Weeks)", f"{sum(forecast_values):.0f}")

    # ---------------------------------------------------------------------
    # TAB 3: COST OPTIMIZATION
    # ---------------------------------------------------------------------
    with tabs[2]:
        st.header("Cost Optimization Analysis")
        cost_drivers = purchases.groupby('ingredient_id')['total_cost'].sum().reset_index().sort_values('total_cost', ascending=False).head(10)
        fig_cost = px.bar(cost_drivers, x='ingredient_id', y='total_cost', title='Top Cost Drivers')
        st.plotly_chart(fig_cost, use_container_width=True)

    # ---------------------------------------------------------------------
    # TAB 4: SHIPMENTS & LOGISTICS
    # ---------------------------------------------------------------------
    with tabs[3]:
        st.header("Shipments and Logistics")
        avg_delay = shipments['delay_days'].mean()
        max_delay = shipments['delay_days'].max()
        total_shipments = shipments.shape[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Shipments", f"{total_shipments}")
        col2.metric("Avg Delivery Time", f"{avg_delay:.1f} days")
        col3.metric("Max Delay", f"{max_delay:.0f} days")

        fig_delay = px.histogram(shipments, x='delay_days', nbins=20, title='Delivery Delay Distribution')
        st.plotly_chart(fig_delay, use_container_width=True)

    # ---------------------------------------------------------------------
    # TAB 5: ORDERING EFFICIENCY
    # ---------------------------------------------------------------------
    with tabs[4]:
        st.header("Ordering Efficiency & Waste Analysis")
        months_input = st.multiselect("Select months", [5,6,7,8,9,10], default=[5,6,7,8,9,10])
        merged_monthly = monthly_orders_and_usage(purchases, usage_monthly, months_input)
        st.subheader("Ordered vs Used Quantities")
        st.dataframe(merged_monthly, use_container_width=True)

        st.subheader("Waste / Shortage Summary")
        summary = merged_monthly.groupby('ingredient_id')[['ordered_qty', 'used_qty', 'waste', 'shortage']].sum().reset_index()
        st.dataframe(summary.sort_values('waste', ascending=False), use_container_width=True)

        st.subheader("Ordering Suggestions")
        suggestions = suggest_ordering_changes(merged_monthly)
        st.dataframe(suggestions if not suggestions.empty else pd.DataFrame([{"Note": "No repeated waste/shortage detected."}]))

    # ---------------------------------------------------------------------
    # TAB 6: TREND ANALYSIS & MENU
    # ---------------------------------------------------------------------
    with tabs[5]:
        st.header("Trend Clustering & Menu Suggestions")
        n_clusters = st.slider("Number of clusters", 2, 8, 4)
        pivot_clusters, cluster_summary = cluster_trends(usage_monthly, n_clusters)
        if pivot_clusters.empty:
            st.info("Not enough data for clustering.")
        else:
            st.subheader("Cluster Summary")
            st.dataframe(cluster_summary, use_container_width=True)
            cluster_choice = st.selectbox("Choose cluster", sorted(pivot_clusters['cluster'].unique()))
            st.dataframe(pivot_clusters[pivot_clusters['cluster'] == cluster_choice], use_container_width=True)

            st.subheader("Menu Suggestions")
            top_trending = pivot_clusters.groupby('ingredient_id').sum().sum(axis=1).sort_values(ascending=False).index.tolist()
            for s in generate_menu_suggestion(top_trending):
                st.write("•", s)

            st.expander("Gemini Prompt").write(
                f"Create 3 limited-edition restaurant menu items combining these trending ingredients: {', '.join(top_trending[:10])}. "
                "For each, include name, description, price, and reason why it fits current trends."
            )

if __name__ == "__main__":
    main()