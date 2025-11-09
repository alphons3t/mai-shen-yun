# streamlit run mai-shen-yun.py 
# how to run this.
# Local URL: http://localhost:8501
# Network URL: http://10.244.154.4:8501

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from datetime import datetime
from sklearn.cluster import KMeans
import google.generativeai as genai
import os
import plotly.graph_objects as go
import requests
from io import BytesIO
from PIL import Image

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(layout="wide")

# Now safe to use other Streamlit commands
gemini_key = st.secrets.get("GEMINI_API_KEY")
openai_key = st.secrets.get("OPENAI_API_KEY")

if gemini_key:
    genai.configure(api_key=gemini_key)
    try:
        # Force use of a known good model for text generation
        st.session_state.gemini_model = "gemini-1.5-flash"
        st.sidebar.success("✅ Gemini API configured with gemini-1.5-flash")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Gemini API issue: {e}")
else:
    st.sidebar.warning("⚠️ GEMINI_API_KEY not found - menu generation will be basic")

if openai_key:
    st.sidebar.success("✅ OpenAI API configured (for premium images)")
    
# Remove the Gemini test button since we'll use it in the menu generation

DATA_DIR = Path("data")

@st.cache_data
def load_csv(name):
    return pd.read_csv(DATA_DIR/name)

def preprocess(purchases, sales, recipes, shipments):
    sales['date'] = pd.to_datetime(sales['date'])
    purchases['date'] = pd.to_datetime(purchases['date'])
    shipments['purchase_date'] = pd.to_datetime(shipments['purchase_date'])
    shipments['delivery_date'] = pd.to_datetime(shipments['delivery_date'])
    shipments['delay_days'] = (shipments['delivery_date'] - shipments['purchase_date']).dt.days

    merged = sales.merge(recipes, on='menu_item_id', how='left')
    merged['ingredient_used'] = merged['quantity_sold'] * merged['quantity_per_item']

    # Group by ingredient_id only (no ingredient_name in recipes.csv)
    usage_weekly = merged.groupby([pd.Grouper(key='date', freq='W'), 'ingredient_id'])['ingredient_used'].sum().reset_index()
    usage_monthly = merged.groupby([pd.Grouper(key='date', freq='M'), 'ingredient_id'])['ingredient_used'].sum().reset_index()

    # Calculate Total Cost
    if 'unit_cost' in purchases.columns:
        purchases['total_cost'] = purchases['quantity'] * purchases['unit_cost']
    else:
        purchases['total_cost'] = purchases['quantity'] * 1 
        st.sidebar.warning("Using quantity as a proxy for cost as 'unit_cost' column was not found in purchases.csv.")

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
        item = f"Limited-Edition: Ingredient {ing} {method} — {flavor} glaze"
        menu_items.append(item)
    return menu_items

def generate_menu_image(menu_description, openai_api_key):
    """Generate an image for a menu item using DALL-E"""
    try:
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # Create a better prompt for food photography
        prompt = f"Professional food photography of {menu_description}, restaurant quality, beautifully plated, studio lighting, 4k"
        
        data = {
            "model": "dall-e-3",
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": "standard"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            image_url = response.json()['data'][0]['url']
            # Download the image
            img_response = requests.get(image_url)
            img = Image.open(BytesIO(img_response.content))
            return img
        else:
            st.error(f"Image generation failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

def generate_menu_with_gemini(trending_ingredients):
    """Use Gemini to create detailed menu items with descriptions and image prompts"""
    try:
        # Use the model we found during initialization
        if 'gemini_model' not in st.session_state:
            st.error("No Gemini model available. Please check your API key.")
            return None
        
        model_name = st.session_state.gemini_model
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        ingredients_str = ", ".join([str(ing) for ing in trending_ingredients[:6]])
        prompt = f"""You are a creative restaurant chef. Create 3 innovative limited-edition menu items using these trending ingredients: {ingredients_str}

For each menu item, provide:
1. Dish Name (creative and appealing)
2. Description (2-3 sentences, mouth-watering)
3. Price (realistic for upscale dining, $15-$35)
4. Why it's trending (1 sentence)
5. Image Prompt (detailed description for AI image generation - include plating, colors, textures, lighting)

Format your response EXACTLY like this:
---
DISH 1: [Name]
Description: [Description]
Price: $[XX]
Trending Because: [Reason]
Image Prompt: [Detailed visual description]
---
DISH 2: [Name]
...
"""
        response = model.generate_content([prompt])  # use list to ensure v1 API compatibility
        return response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        st.error(f"Gemini generation failed: {e}")
        return None

def generate_pollinations_image(prompt):
    """Generate image using Pollinations.ai (free API)"""
    try:
        # Pollinations.ai free image generation
        encoded_prompt = requests.utils.quote(prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
        
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        return None
    except Exception as e:
        st.error(f"Image generation error: {e}")
        return None

def main():
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
        selected_ingredient = st.sidebar.selectbox("Select ingredient", usage['ingredient_id'].unique(), key='overview_select')

        total_usage = usage['ingredient_used'].sum()
        ingredient_usage = usage[usage['ingredient_id'] == selected_ingredient]['ingredient_used'].sum()
        total_purchases_q = purchases['quantity'].sum()
        ingredient_purchases_q = purchases[purchases['ingredient_id'] == selected_ingredient]['quantity'].sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Usage (All)", f"{total_usage:.0f}")
        col2.metric(f"Ingredient {selected_ingredient} Usage", f"{ingredient_usage:.0f}")
        col3.metric("Total Purchases", f"{total_purchases_q:.0f}")
        col4.metric(f"Ingredient {selected_ingredient} Purchases", f"{ingredient_purchases_q:.0f}")

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

        fig = px.line(df_forecast, x='date', y='ingredient_used', title=f'Usage Forecast: Ingredient {forecast_ingredient}')
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
        months_input = st.multiselect("Select months", list(range(1,13)))
        merged_monthly = monthly_orders_and_usage(purchases, usage_monthly, months_input)
        st.subheader("Ordered vs Used Quantities")
        st.dataframe(merged_monthly, use_container_width=True)

        st.subheader("Waste / Shortage Summary")
        summary = merged_monthly.groupby('ingredient_id')[['ordered_qty', 'used_qty', 'waste', 'shortage']].sum().reset_index()
        st.dataframe(summary.sort_values('waste', ascending=False), use_container_width=True)

        # --- Interactive Surplus/Shortage Visualization ---
        st.subheader("Interactive Surplus/Shortage by Month")

        # Pivot data to resemble standalone figure structure
        diff_df = merged_monthly.copy()
        diff_df['difference'] = diff_df['ordered_qty'] - diff_df['used_qty']
        pivot_df = diff_df.pivot_table(index='ingredient_id', columns='month', values='difference', aggfunc='sum').fillna(0)
        pivot_df['Average'] = pivot_df.mean(axis=1)

        fig = go.Figure()
        months = list(pivot_df.columns)

        for month in months:
            fig.add_trace(go.Bar(
                x=pivot_df.index.astype(str),
                y=pivot_df[month],
                name=str(month),
                visible=True,
                marker_color=['green' if v > 0 else 'red' for v in pivot_df[month]]
            ))

        n = len(months)
        buttons = []
        buttons.append(dict(
            label="Show All Months",
            method="update",
            args=[{"visible": [True] * n},
                  {"title": "All Months: Ingredient Surplus (+) / Shortage (–)"}]
        ))

        for i, month in enumerate(months):
            visibility = [False] * n
            visibility[i] = True
            buttons.append(dict(
                label=str(month),
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Ingredient Surplus (+) / Shortage (–): {month}"}]
            ))

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=1.05,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ],
            title="All Months: Ingredient Surplus (+) / Shortage (–)",
            xaxis_title="Ingredient",
            yaxis_title="Difference (Ordered – Used)",
            xaxis_tickangle=-45,
            template="plotly_white",
            height=600,
            legend_title="Month"
        )

        st.plotly_chart(fig, use_container_width=True)

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
            cluster_choice = st.selectbox("Choose cluster", sorted(pivot_clusters['cluster'].unique()), key='cluster_select')
            st.dataframe(pivot_clusters[pivot_clusters['cluster'] == cluster_choice], use_container_width=True)

            st.subheader("AI-Generated Menu Suggestions")
            top_trending = pivot_clusters.groupby('ingredient_id').sum().sum(axis=1).sort_values(ascending=False).index.tolist()
            
            gemini_key = st.secrets.get("GEMINI_API_KEY")
            
            if st.button("🤖 Generate Menu with Gemini + Images", type="primary"):
                if gemini_key:
                    with st.spinner("Gemini is crafting your menu..."):
                        menu_content = generate_menu_with_gemini(top_trending)
                        if menu_content:
                            st.session_state.menu_content = menu_content
                            
                            # Parse and generate images for each dish
                            dishes = menu_content.split("---")
                            st.session_state.dish_images = {}
                            
                            for idx, dish in enumerate(dishes):
                                if "Image Prompt:" in dish:
                                    # Extract image prompt
                                    prompt_start = dish.find("Image Prompt:") + len("Image Prompt:")
                                    image_prompt = dish[prompt_start:].strip()
                                    
                                    with st.spinner(f"Generating image {idx+1}..."):
                                        img = generate_pollinations_image(image_prompt)
                                        if img:
                                            st.session_state.dish_images[idx] = img
                else:
                    st.error("Please add GEMINI_API_KEY to .streamlit/secrets.toml")
            
            # Display generated content
            if 'menu_content' in st.session_state:
                dishes = st.session_state.menu_content.split("---")
                
                for idx, dish_text in enumerate(dishes):
                    if dish_text.strip() and "DISH" in dish_text:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Show generated image if available
                            if 'dish_images' in st.session_state and idx in st.session_state.dish_images:
                                st.image(st.session_state.dish_images[idx], use_container_width=True)
                            else:
                                st.info("Image generating...")
                        
                        with col2:
                            st.markdown(dish_text.strip())
                        
                        st.markdown("---")
            else:
                # Show basic suggestions as fallback
                st.write("*Click the button above to generate AI-powered menu items with images*")
                st.write("")
                st.write("**Quick Preview (Basic):**")
                for s in generate_menu_suggestion(top_trending):
                    st.write("•", s)

            with st.expander("Gemini Prompt"):
                st.write(
                    f"Create 3 limited-edition restaurant menu items combining these trending ingredients: {', '.join(map(str, top_trending[:10]))}. "
                    "For each, include name, description, price, and reason why it fits current trends."
                )

if __name__ == "__main__":
    main()
