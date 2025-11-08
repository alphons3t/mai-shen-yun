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
    return pd.read_csv(DATA_DIR/name)

def preprocess(purchases, sales, recipes):
    # Minimal example: aggregate monthly usage via sales+recipes
    sales['date'] = pd.to_datetime(sales['date'])
    purchases['date'] = pd.to_datetime(purchases['date'])
    recipes = recipes.copy()
    merged = sales.merge(recipes, on='menu_item_id', how='left')
    merged['ingredient_used'] = merged['quantity_sold'] * merged['quantity_per_item']
    usage = merged.groupby([pd.Grouper(key='date', freq='W'), 'ingredient_id'])['ingredient_used'].sum().reset_index()
    return usage, purchases

def main():
    st.title("Mai Shan Yun — Inventory Intelligence")
    st.sidebar.header("Data")
    st.sidebar.write("Place your CSVs in the data/ folder named: purchases.csv, sales.csv, recipes.csv")
    if st.sidebar.button("Reload data"):
        st.cache_data.clear()

    try:
        purchases = load_csv("purchases.csv")
        sales = load_csv("sales.csv")
        recipes = load_csv("recipes.csv")
    except Exception as e:
        st.warning("Could not load CSVs. Please ensure data/purchases.csv, data/sales.csv, data/recipes.csv exist.")
        st.error(str(e))
        st.stop()

    usage, purchases = preprocess(purchases, sales, recipes)

    st.header("Top ingredients by estimated weekly usage")
    top = usage.groupby('ingredient_id')['ingredient_used'].sum().reset_index().sort_values('ingredient_used', ascending=False).head(10)
    st.dataframe(top)

    # Simple timeseries plot for top ingredient
    top_ing = top['ingredient_id'].iloc[0]
    df_plot = usage[usage['ingredient_id'] == top_ing]
    fig = px.line(df_plot, x='date', y='ingredient_used', title=f'Weekly usage for {top_ing}')
    st.plotly_chart(fig, use_container_width=True)

    # Minimal reorder logic example
    st.header("Reorder alerts (simple)")
    avg_daily = usage.groupby('ingredient_id')['ingredient_used'].mean() / 7.0
    current_inventory = purchases.groupby('ingredient_id')['quantity'].sum() - usage.groupby('ingredient_id')['ingredient_used'].sum()
    df_alert = pd.DataFrame({
        'avg_daily_usage': avg_daily,
        'current_inventory': current_inventory
    }).fillna(0)
    df_alert['days_left'] = df_alert['current_inventory'] / (df_alert['avg_daily_usage'] + 1e-9)
    st.dataframe(df_alert.sort_values('days_left').head(20))

if __name__ == "__main__":
    main()