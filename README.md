# Mai Shan Yun - Inventory Intelligence Dashboard

## Overview

Mai Shan Yun is a real-time inventory intelligence dashboard designed for restaurants or food businesses. It helps optimize:

- Ingredient usage forecasting
- Cost analysis and high-spend contributors
- Shipment delay tracking
- Ordering efficiency (waste vs. shortage detection)
- Menu suggestions based on ingredient trends

The dashboard converts raw purchase, sales, recipe, and shipment data into actionable insights and alerts.

---

## Key Features

Overview & Alerts       | Inventory usage, reorder warnings, ingredient-level metrics 

Forecasting             | Moving average prediction for ingredient demand 

Cost Optimization       | Detects top cost drivers and highlights spending patterns 

Shipments & Logistics   | Tracks delivery delays and visualizes distributions 

Ordering Efficiency     | Identifies recurring waste/shortages and suggests optimal ordering 

Trend Analysis & Menu   | Clusters trending ingredients using KMeans and generates menu ideas 

---

## Dataset Requirements

Place the following CSV files in a folder named `data/`:

```
purchases.csv
sales.csv
recipes.csv
shipments.csv
```

### Required Columns

File            | Required Columns
                |
purchases.csv   | ingredient_id, quantity, date, (optional) unit_cost
sales.csv       | menu_item_id, quantity_sold, date
recipes.csv     | menu_item_id, ingredient_id, quantity_per_item, ingredient_name
shipments.csv   | ingredient_id, purchase_date, delivery_date

If `unit_cost` is missing in purchases.csv, the app will use quantity as a temporary cost proxy.

---

## How to Run the App

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run mai-shen-yun.py
```

### 3. Open the dashboard

Check the terminal output for the local URL (usually http://localhost:8501) and open it in a browser.

---

## Example Insights

Reduce waste        | Ingredient "Tomatoes" shows consistent over-ordering; suggest ordering 10% less |
Low stock warning   | Only 2 days of inventory remaining based on average usage |
Shipment issue      | Vendor X averages 4.2 days delay; consider adjusting orders or switching vendors |
Menu strategy       | "Ginger Grilled - Savory glaze" generated from trending ingredient clusters |

---

## Tech Stack

Frontend dashboard  | Streamlit
Data processing     | Pandas / NumPy
Visualization       | Plotly Express
ML clustering       | Scikit-learn (KMeans)

---

## Repository Structure

```
mai-shen-yun.py        # main Streamlit app
/data                  # CSV files go here
README.md              # documentation
```
## Credits

Developed by: Noah Brown, Archelaus Paxon, Alphonse Thomas

