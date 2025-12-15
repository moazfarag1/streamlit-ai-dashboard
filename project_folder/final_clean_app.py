# final_clean_app.py
# Run: streamlit run final_clean_app.py
# CPU-only, no GPU required

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import json
import warnings
import logging

# Silence all warnings
warnings.filterwarnings("ignore")
logging.getLogger("xgboost").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)

st.set_page_config(page_title="AI Solutions Dashboard", layout="wide")

# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_sentiment_models():
    try:
        sent_model = pickle.load(open('trained_model.sav', 'rb'))
        vect = pickle.load(open('vectorizer.sav', 'rb'))
        print("Model loaded")
        return sent_model, vect
    except:
        return None, None

@st.cache_resource
def load_sales_model():
    try:
        sales_model = xgb.Booster()
        sales_model.load_model('artifacts/xgb_model.json')
        with open('artifacts/label_encoder.json', 'r') as f:
            label_encoder = json.load(f)
        print("Model loaded")
        return sales_model, label_encoder
    except:
        return None, None

@st.cache_resource
def load_advanced_model():
    try:
        with open('sales_predictor_model.json', 'r') as f:
            model_data = json.load(f)
        print("Model loaded")
        return model_data
    except:
        return None

sentiment_model, vectorizer = load_sentiment_models()
sales_model, label_encoder = load_sales_model()
advanced_model = load_advanced_model()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("🎛️ Dashboard")
    st.write("Welcome to the Data Analysis System")
    selected_option = st.radio(
        "Select Service:",
        ['📊 Sentiment Analysis', '📦 Retail Sales Forecasting (Simple)', '💎 Advanced Price Prediction']
    )
    st.markdown("---")
    st.info("Product Intelligence Engine Project: moaz, fady, Maryam")

# =========================================================
# MODULE 1: SENTIMENT ANALYSIS
# =========================================================

if selected_option == '📊 Sentiment Analysis':
    st.header("📂 Customer Sentiment Analysis from Files")
    st.markdown("Upload a CSV file containing customer reviews for bulk analysis.")
    
    if sentiment_model is None or vectorizer is None:
        st.error("Sentiment model not loaded")
        st.stop()
    
    uploaded_file = st.file_uploader("Upload CSV file here", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin-1')
            st.success("File uploaded successfully! ✅")
            st.write("Data Preview:", df.head())
            
            text_column = st.selectbox("Select the column containing the review text:", df.columns)
            
            if st.button('🚀 Start Analysis'):
                with st.spinner('Analyzing customer reviews...'):
                    texts = df[text_column].astype(str)
                    transformed_texts = vectorizer.transform(texts)
                    predictions = sentiment_model.predict(transformed_texts)
                    df['Sentiment_Prediction'] = predictions
                    df['Sentiment_Label'] = df['Sentiment_Prediction'].apply(lambda x: 'Positive 😃' if x == 1 else 'Negative 😡')
                    
                    st.divider()
                    st.subheader("📊 Analysis Report")
                    counts = df['Sentiment_Label'].value_counts()
                    col1, col2 = st.columns(2)
                    col1.metric("Positive Reviews Count", counts.get('Positive 😃', 0))
                    col2.metric("Negative Reviews Count", counts.get('Negative 😡', 0))
                    st.bar_chart(counts)
                    st.write("Data after analysis:")
                    st.dataframe(df[[text_column, 'Sentiment_Label']])
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Final Report (CSV)",
                        data=csv,
                        file_name='sentiment_analysis_results.csv',
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"Error reading the file: {e}")

# =========================================================
# MODULE 2: SIMPLE SALES FORECASTING
# =========================================================

elif selected_option == '📦 Retail Sales Forecasting (Simple)':
    st.header("📦 Retail Sales Forecasting System")
    st.markdown("Predict daily product sales using XGBoost with 4 features.")
    
    if sales_model is None or label_encoder is None:
        st.error("Sales forecasting model not loaded")
        st.stop()
    
    prediction_mode = st.radio(
        "Select Prediction Mode:",
        ['🔍 Manual Prediction', '📂 CSV Batch Prediction'],
        horizontal=True
    )
    st.markdown("---")
    
    if prediction_mode == '🔍 Manual Prediction':
        st.subheader("🔍 Manual Single Product Prediction")
        st.markdown("Enter product details to predict units sold.")
        
        with st.form("sales_form"):
            col1, col2 = st.columns(2)
            with col1:
                lag_7_input = st.number_input("Units sold 7 days ago (lag_7)", min_value=0.0, value=50.0, step=1.0)
                sale_price_input = st.number_input("Sale Price ($)", min_value=0.0, value=29.99, step=0.01)
            with col2:
                is_promo_input = st.selectbox("Promotional Status (is_promo)", options=[0, 1], format_func=lambda x: "❌ No Promotion" if x == 0 else "✅ On Promotion")
                available_products = sorted(label_encoder.keys())
                product_id_input = st.selectbox("Product ID", options=available_products)
            
            submit_button = st.form_submit_button("🔮 Predict Sales", use_container_width=True)
            
            if submit_button:
                try:
                    if product_id_input not in label_encoder:
                        st.error(f"Product ID '{product_id_input}' not found in encoder")
                        st.stop()
                    
                    product_id_encoded = label_encoder[product_id_input]
                    features_df = pd.DataFrame({
                        'lag_7': [np.float32(lag_7_input)],
                        'sale_price': [np.float32(sale_price_input)],
                        'is_promo': [np.int8(is_promo_input)],
                        'product_id_encoded': [np.int32(product_id_encoded)]
                    })
                    dmatrix = xgb.DMatrix(features_df)
                    prediction = sales_model.predict(dmatrix)[0]
                    predicted_units = int(round(prediction))
                    
                    st.success(f"### 📊 Predicted Units Sold: **{predicted_units} units**")
                    st.markdown("---")
                    st.subheader("📋 Prediction Summary")
                    col_sum1, col_sum2 = st.columns(2)
                    with col_sum1:
                        st.metric("Product ID", product_id_input)
                        st.metric("Sales 7 Days Ago", f"{lag_7_input:.0f} units")
                        st.metric("Sale Price", f"${sale_price_input:.2f}")
                    with col_sum2:
                        st.metric("Promotion Status", "✅ Active" if is_promo_input == 1 else "❌ Inactive")
                        st.metric("Raw Prediction", f"{prediction:.2f} units")
                        st.metric("Rounded Prediction", f"{predicted_units} units")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
    
    elif prediction_mode == '📂 CSV Batch Prediction':
        st.subheader("📂 CSV Batch Prediction")
        st.markdown("Upload a CSV file containing multiple products for bulk sales forecasting.")
        
        uploaded_csv = st.file_uploader("Upload CSV file here", type=['csv'], key="sales_csv")
        
        if uploaded_csv is not None:
            try:
                df_sales = pd.read_csv(uploaded_csv, encoding='utf-8')
                st.success("CSV uploaded")
                print("CSV uploaded")
                st.write("**Data Preview:**")
                st.dataframe(df_sales.head(10))
                st.markdown("---")
                
                st.subheader("📋 Map Your Columns")
                col_map1, col_map2 = st.columns(2)
                with col_map1:
                    col_lag7 = st.selectbox("Column for 'lag_7'", options=df_sales.columns)
                    col_price = st.selectbox("Column for 'sale_price'", options=df_sales.columns)
                with col_map2:
                    col_promo = st.selectbox("Column for 'is_promo'", options=df_sales.columns)
                    col_product = st.selectbox("Column for 'product_id'", options=df_sales.columns)
                
                if st.button("🚀 Run Batch Prediction", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        try:
                            df_pred = df_sales.copy()
                            df_pred[col_lag7] = pd.to_numeric(df_pred[col_lag7], errors='coerce')
                            df_pred[col_price] = pd.to_numeric(df_pred[col_price], errors='coerce')
                            df_pred[col_promo] = pd.to_numeric(df_pred[col_promo], errors='coerce')
                            
                            if df_pred[col_lag7].isna().any():
                                st.error("Invalid values in lag_7 column")
                                st.stop()
                            if df_pred[col_price].isna().any():
                                st.error("Invalid values in sale_price column")
                                st.stop()
                            if df_pred[col_promo].isna().any():
                                st.error("Invalid values in is_promo column")
                                st.stop()
                            
                            df_pred['product_id_str'] = df_pred[col_product].astype(str)
                            df_pred['product_encoded'] = df_pred['product_id_str'].map(label_encoder)
                            
                            valid_mask = df_pred['product_encoded'].notna()
                            df_valid = df_pred[valid_mask].copy()
                            
                            if len(df_valid) > 0:
                                batch_features = pd.DataFrame({
                                    'lag_7': df_valid[col_lag7].astype(np.float32),
                                    'sale_price': df_valid[col_price].astype(np.float32),
                                    'is_promo': df_valid[col_promo].astype(np.int8),
                                    'product_id_encoded': df_valid['product_encoded'].astype(np.int32)
                                })
                                dmatrix = xgb.DMatrix(batch_features)
                                predictions = sales_model.predict(dmatrix)
                                df_pred.loc[valid_mask, 'predicted_units_sold'] = predictions
                            
                            df_pred['predicted_units_sold_rounded'] = df_pred['predicted_units_sold'].apply(
                                lambda x: int(round(x)) if pd.notna(x) else np.nan
                            )
                            
                            st.markdown("---")
                            st.subheader("📊 Prediction Results")
                            valid_predictions = df_pred['predicted_units_sold'].dropna()
                            total_rows = len(df_pred)
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            metric_col1.metric("Total Rows", total_rows)
                            metric_col2.metric("Successful", len(valid_predictions))
                            metric_col3.metric("Failed", total_rows - len(valid_predictions))
                            metric_col4.metric("Success Rate", f"{(len(valid_predictions)/total_rows*100):.1f}%")
                            
                            if len(valid_predictions) > 0:
                                stat_col1, stat_col2, stat_col3 = st.columns(3)
                                stat_col1.metric("Mean", f"{valid_predictions.mean():.2f} units")
                                stat_col2.metric("Min", f"{valid_predictions.min():.2f} units")
                                stat_col3.metric("Max", f"{valid_predictions.max():.2f} units")
                            
                            st.markdown("**Results Table:**")
                            display_cols = [col_product, col_lag7, col_price, col_promo, 'predicted_units_sold', 'predicted_units_sold_rounded']
                            st.dataframe(df_pred[display_cols])
                            
                            csv_output = df_pred.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Predictions (CSV)",
                                data=csv_output,
                                file_name='sales_predictions_results.csv',
                                mime='text/csv',
                            )
                            print("Prediction complete")
                        except Exception as e:
                            st.error(f"Batch prediction error: {e}")
            except Exception as e:
                st.error(f"Error reading file: {e}")

# =========================================================
# MODULE 3: ADVANCED PRICE PREDICTION
# =========================================================

elif selected_option == '💎 Advanced Price Prediction':
    st.header("💎 Advanced Price & Sales Prediction")
    st.markdown("Predict product pricing and sales based on detailed attributes.")
    
    if advanced_model is None:
        st.error("Advanced model not loaded")
        st.stop()
    
    weights = advanced_model['weights']
    
    def predict_advanced(category, subcategory, brand, season, material, gender):
        base_price = weights['basePrice']
        base_units = weights['baseUnits']
        predicted_price = base_price
        predicted_units = base_units
        
        if category in weights['categoryWeights']:
            predicted_price += weights['categoryWeights'][category]['priceDiff']
            predicted_units += weights['categoryWeights'][category]['unitsDiff']
        if subcategory in weights['subcategoryWeights']:
            predicted_price += weights['subcategoryWeights'][subcategory]['priceDiff']
            predicted_units += weights['subcategoryWeights'][subcategory]['unitsDiff']
        if brand in weights['brandWeights']:
            predicted_price += weights['brandWeights'][brand]['priceDiff']
            predicted_units += weights['brandWeights'][brand]['unitsDiff']
        if season in weights['seasonWeights']:
            predicted_price += weights['seasonWeights'][season]['priceDiff']
            predicted_units += weights['seasonWeights'][season]['unitsDiff']
        if material in weights['materialWeights']:
            predicted_price += weights['materialWeights'][material]['priceDiff']
            predicted_units += weights['materialWeights'][material]['unitsDiff']
        if gender in weights['genderWeights']:
            predicted_price += weights['genderWeights'][gender]['priceDiff']
            predicted_units += weights['genderWeights'][gender]['unitsDiff']
        
        return max(0, predicted_price), max(0, predicted_units)
    
    prediction_mode = st.radio("Select Prediction Mode:", ['🔍 Single Prediction', '📂 CSV Upload'], horizontal=True)
    st.markdown("---")
    
    if prediction_mode == '🔍 Single Prediction':
        st.subheader("🔍 Single Product Prediction")
        
        with st.form("advanced_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                category = st.selectbox("Category", options=sorted(weights['categoryWeights'].keys()))
                subcategory = st.selectbox("Subcategory", options=sorted(weights['subcategoryWeights'].keys()))
            with col2:
                brand = st.selectbox("Brand", options=sorted(weights['brandWeights'].keys()))
                season = st.selectbox("Season", options=sorted(weights['seasonWeights'].keys()))
            with col3:
                material = st.selectbox("Material", options=sorted(weights['materialWeights'].keys()))
                gender = st.selectbox("Gender", options=sorted(weights['genderWeights'].keys()))
            
            submit_btn = st.form_submit_button("🔮 Predict Price & Sales", use_container_width=True)
            
            if submit_btn:
                try:
                    price, units = predict_advanced(category, subcategory, brand, season, material, gender)
                    st.success("### ✅ Prediction Complete!")
                    result_col1, result_col2 = st.columns(2)
                    result_col1.metric("💰 Predicted Price", f"${price:.2f}")
                    result_col2.metric("📦 Predicted Sales", f"{int(round(units))} units")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
    
    elif prediction_mode == '📂 CSV Upload':
        st.subheader("📂 CSV Batch Prediction")
        
        uploaded_csv = st.file_uploader("Upload CSV file here", type=['csv'], key="advanced_csv")
        
        if uploaded_csv is not None:
            try:
                df_adv = pd.read_csv(uploaded_csv, encoding='utf-8')
                st.success("CSV uploaded")
                print("CSV uploaded")
                st.write("**Data Preview:**")
                st.dataframe(df_adv.head(10))
                st.markdown("---")
                
                st.subheader("📋 Map Your Columns")
                col_map1, col_map2, col_map3 = st.columns(3)
                with col_map1:
                    col_category = st.selectbox("Column for 'Category'", options=df_adv.columns)
                    col_subcategory = st.selectbox("Column for 'Subcategory'", options=df_adv.columns)
                with col_map2:
                    col_brand = st.selectbox("Column for 'Brand'", options=df_adv.columns)
                    col_season = st.selectbox("Column for 'Season'", options=df_adv.columns)
                with col_map3:
                    col_material = st.selectbox("Column for 'Material'", options=df_adv.columns)
                    col_gender = st.selectbox("Column for 'Gender'", options=df_adv.columns)
                
                if st.button("🚀 Run Batch Predictions", use_container_width=True):
                    with st.spinner("Processing batch predictions..."):
                        try:
                            df_adv['cat_str'] = df_adv[col_category].astype(str)
                            df_adv['sub_str'] = df_adv[col_subcategory].astype(str)
                            df_adv['brand_str'] = df_adv[col_brand].astype(str)
                            df_adv['season_str'] = df_adv[col_season].astype(str)
                            df_adv['material_str'] = df_adv[col_material].astype(str)
                            df_adv['gender_str'] = df_adv[col_gender].astype(str)
                            
                            cat_map = {k: v['priceDiff'] for k, v in weights['categoryWeights'].items()}
                            cat_units = {k: v['unitsDiff'] for k, v in weights['categoryWeights'].items()}
                            sub_map = {k: v['priceDiff'] for k, v in weights['subcategoryWeights'].items()}
                            sub_units = {k: v['unitsDiff'] for k, v in weights['subcategoryWeights'].items()}
                            brand_map = {k: v['priceDiff'] for k, v in weights['brandWeights'].items()}
                            brand_units = {k: v['unitsDiff'] for k, v in weights['brandWeights'].items()}
                            season_map = {k: v['priceDiff'] for k, v in weights['seasonWeights'].items()}
                            season_units = {k: v['unitsDiff'] for k, v in weights['seasonWeights'].items()}
                            material_map = {k: v['priceDiff'] for k, v in weights['materialWeights'].items()}
                            material_units = {k: v['unitsDiff'] for k, v in weights['materialWeights'].items()}
                            gender_map = {k: v['priceDiff'] for k, v in weights['genderWeights'].items()}
                            gender_units = {k: v['unitsDiff'] for k, v in weights['genderWeights'].items()}
                            
                            df_adv['predicted_price'] = weights['basePrice']
                            df_adv['predicted_units'] = weights['baseUnits']
                            
                            df_adv['predicted_price'] += df_adv['cat_str'].map(cat_map).fillna(0)
                            df_adv['predicted_units'] += df_adv['cat_str'].map(cat_units).fillna(0)
                            df_adv['predicted_price'] += df_adv['sub_str'].map(sub_map).fillna(0)
                            df_adv['predicted_units'] += df_adv['sub_str'].map(sub_units).fillna(0)
                            df_adv['predicted_price'] += df_adv['brand_str'].map(brand_map).fillna(0)
                            df_adv['predicted_units'] += df_adv['brand_str'].map(brand_units).fillna(0)
                            df_adv['predicted_price'] += df_adv['season_str'].map(season_map).fillna(0)
                            df_adv['predicted_units'] += df_adv['season_str'].map(season_units).fillna(0)
                            df_adv['predicted_price'] += df_adv['material_str'].map(material_map).fillna(0)
                            df_adv['predicted_units'] += df_adv['material_str'].map(material_units).fillna(0)
                            df_adv['predicted_price'] += df_adv['gender_str'].map(gender_map).fillna(0)
                            df_adv['predicted_units'] += df_adv['gender_str'].map(gender_units).fillna(0)
                            
                            df_adv['predicted_price'] = df_adv['predicted_price'].clip(lower=0)
                            df_adv['predicted_units'] = df_adv['predicted_units'].clip(lower=0).round().astype(int)
                            
                            df_adv = df_adv.drop(columns=['cat_str', 'sub_str', 'brand_str', 'season_str', 'material_str', 'gender_str'])
                            
                            st.markdown("---")
                            st.subheader("📊 Prediction Results")
                            valid_prices = df_adv['predicted_price'].dropna()
                            
                            res_col1, res_col2, res_col3 = st.columns(3)
                            res_col1.metric("Total Rows", len(df_adv))
                            res_col2.metric("Successful", len(valid_prices))
                            res_col3.metric("Avg Price", f"${valid_prices.mean():.2f}")
                            
                            st.markdown("**Results Table:**")
                            st.dataframe(df_adv)
                            
                            csv_out = df_adv.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Results (CSV)",
                                data=csv_out,
                                file_name='advanced_predictions_results.csv',
                                mime='text/csv',
                            )
                            print("Prediction complete")
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")