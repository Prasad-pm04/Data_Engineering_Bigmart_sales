import streamlit as st
import pandas as pd
import pickle

# === 1. Load the Model ===
with open("bigmart_best_model.pkl", "rb") as f:
    model, skl_version = pickle.load(f)

st.set_page_config(page_title="BigMart Sales Prediction", layout="wide")

st.title("🛒 BigMart Sales Prediction")
st.write("Enter product and outlet details to predict *Item Outlet Sales*.")

# === 2. Sidebar Input Form ===
st.sidebar.header("Item Information")
item_identifier = st.sidebar.text_input("Item Identifier", "FDA15")
item_weight = st.sidebar.number_input("Item Weight (kg)", 1.0, 50.0, 10.0)
item_fat_content = st.sidebar.selectbox("Item Fat Content", ["Low Fat", "Regular"])
item_visibility = st.sidebar.slider("Item Visibility", 0.0, 0.3, 0.05)
item_type = st.sidebar.selectbox(
    "Item Type",
    ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
     "Baking Goods", "Snack Foods", "Breakfast", "Health and Hygiene",
     "Hard Drinks", "Canned", "Frozen Foods", "Breads"]
)
item_mrp = st.sidebar.number_input("Item MRP (₹)", 1.0, 500.0, 150.0)

st.sidebar.header("Outlet Information")
outlet_identifier = st.sidebar.text_input("Outlet Identifier", "OUT049")
outlet_size = st.sidebar.selectbox("Outlet Size", ["Small", "Medium", "High"])
outlet_location_type = st.sidebar.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
outlet_type = st.sidebar.selectbox(
    "Outlet Type",
    ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"]
)
outlet_age = st.sidebar.slider("Outlet Age (years)", 1, 50, 10)

# === 3. Create Input DataFrame ===
input_dict = {
    "Item_Identifier": [item_identifier],
    "Item_Weight": [item_weight],
    "Item_Fat_Content": [item_fat_content],
    "Item_Visibility": [item_visibility],
    "Item_Type": [item_type],
    "Item_MRP": [item_mrp],
    "Outlet_Identifier": [outlet_identifier],
    "Outlet_Size": [outlet_size],
    "Outlet_Location_Type": [outlet_location_type],
    "Outlet_Type": [outlet_type],
    "Outlet_Age": [outlet_age]
}
input_df = pd.DataFrame(input_dict)

st.subheader("🔍 Input Preview")
st.dataframe(input_df)

# === 4. Predict Sales ===
if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Item Outlet Sales: ₹{prediction:,.2f}")