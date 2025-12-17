# final_clean_app.py
# Product Intelligence Engine - University Project
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

st.set_page_config(
    page_title="Product Intelligence Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.title("🎯 Product Intelligence Engine")
    st.markdown("### AI-Powered Business Analytics")
    st.markdown("---")
    
    selected_option = st.radio(
        "Choose Analysis Module:",
        [
            '📊 Sentiment Analysis',
            '📦 Sales Forecasting',
            '💎 Price & Demand Prediction'
        ],
        help="Select the prediction module you want to use"
    )
    
    st.markdown("---")
    st.markdown("**Project Team**")
    st.caption("Moaz • Fady • Maryam")
    st.caption("Business Intelligence & ML Systems")

# =========================================================
# MODULE 1: SENTIMENT ANALYSIS (IMMUTABLE - DO NOT MODIFY)
# =========================================================

if selected_option == '📊 Sentiment Analysis':
    st.header("📊 Customer Sentiment Analysis")
    
    # Module explanation
    with st.expander("ℹ️ About This Module", expanded=False):
        st.markdown("""
        **What it does:** Analyzes customer reviews and classifies them as Positive or Negative using machine learning.
        
        **How it works:** This module uses a trained text classification model that has learned patterns from thousands of customer reviews. It examines the words, phrases, and overall tone to determine sentiment.
        
        **Best for:** Understanding customer feedback at scale, identifying trends in satisfaction, and prioritizing which reviews need attention.
        
        **Limitations:** Works best with English text. Very short reviews (1-2 words) may be less accurate.
        """)
    
    st.markdown("### Upload Your Customer Reviews")
    st.markdown("Upload a CSV file containing customer feedback for automated sentiment analysis.")
    
    if sentiment_model is None or vectorizer is None:
        st.error("⚠️ Sentiment model not loaded. Please ensure trained_model.sav and vectorizer.sav are available.")
        st.stop()
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], help="File should contain a column with review text")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin-1')
            st.success("✅ File uploaded successfully!")
            
            if len(df) == 0:
                st.error("The uploaded file is empty. Please upload a file with data.")
                st.stop()
            
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Showing first 10 of {len(df)} rows")
            
            text_column = st.selectbox(
                "Select the column containing review text:",
                options=df.columns,
                help="Choose which column contains the customer feedback"
            )
            
            if st.button('🚀 Analyze Sentiment', type="primary", use_container_width=True):
                with st.spinner('Analyzing customer reviews...'):
                    texts = df[text_column].astype(str)
                    transformed_texts = vectorizer.transform(texts)
                    predictions = sentiment_model.predict(transformed_texts)
                    df['Sentiment_Prediction'] = predictions
                    df['Sentiment_Label'] = df['Sentiment_Prediction'].apply(
                        lambda x: 'Positive 😃' if x == 1 else 'Negative 😡'
                    )
                    
                    st.divider()
                    st.subheader("📊 Analysis Results")
                    
                    counts = df['Sentiment_Label'].value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Reviews", len(df))
                    col2.metric("Positive Reviews", counts.get('Positive 😃', 0))
                    col3.metric("Negative Reviews", counts.get('Negative 😡', 0))
                    
                    positive_pct = (counts.get('Positive 😃', 0) / len(df) * 100) if len(df) > 0 else 0
                    st.progress(positive_pct / 100)
                    st.caption(f"Positive Sentiment: {positive_pct:.1f}%")
                    
                    st.bar_chart(counts)
                    
                    st.markdown("### 📄 Detailed Results")
                    st.dataframe(
                        df[[text_column, 'Sentiment_Label']],
                        use_container_width=True,
                        height=400
                    )
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name='sentiment_analysis_results.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                    
                    print("Prediction complete")
                    
        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")
            st.info("Please ensure your CSV is properly formatted and contains text data.")

# =========================================================
# MODULE 2: SIMPLE SALES FORECASTING
# =========================================================

elif selected_option == '📦 Sales Forecasting':
    st.header("📦 Daily Sales Forecasting System")
    
    # Module explanation
    with st.expander("ℹ️ About This Module", expanded=False):
        st.markdown("""
        **What it does:** Predicts how many units of a product will sell on a given day based on historical patterns, pricing, and promotional status.
        
        **How it works:** This module uses XGBoost machine learning trained on thousands of historical sales transactions. It identifies patterns between past sales, prices, promotions, and future demand.
        
        **The 4 Key Factors:**
        - **Previous Sales (lag_7):** Units sold 7 days ago - the strongest predictor
        - **Current Price:** Today's selling price affects demand
        - **Promotion Status:** Whether the product is currently on sale
        - **Product Identity:** Each product has unique sales patterns
        
        **Best for:** Daily inventory planning, promotional impact analysis, demand forecasting for known products.
        
        **Limitations:** Requires 7-day sales history. Less accurate for brand new products or during major market disruptions.
        """)
    
    if sales_model is None or label_encoder is None:
        st.error("⚠️ Sales forecasting model not loaded. Please ensure model files exist in artifacts/ folder.")
        st.stop()
    
    st.markdown("### Choose Prediction Mode")
    prediction_mode = st.radio(
        "How would you like to predict?",
        ['🔍 Single Product', '📂 Batch Upload (CSV)'],
        horizontal=True
    )
    st.markdown("---")
    
    if prediction_mode == '🔍 Single Product':
        st.subheader("🔍 Single Product Forecast")
        st.markdown("Enter the details below to predict tomorrow's sales.")
        
        with st.form("sales_form"):
            st.markdown("**📊 Input the 4 Required Features:**")
            
            col1, col2 = st.columns(2)
            with col1:
                lag_7_input = st.number_input(
                    "Units Sold (7 Days Ago)",
                    min_value=0.0,
                    value=50.0,
                    step=1.0,
                    help="How many units of this product sold exactly 7 days ago? This is the most important predictor."
                )
                
                sale_price_input = st.number_input(
                    "Current Sale Price ($)",
                    min_value=0.0,
                    value=29.99,
                    step=0.01,
                    help="Today's selling price for this product"
                )
                
            with col2:
                is_promo_input = st.selectbox(
                    "Promotional Status",
                    options=[0, 1],
                    format_func=lambda x: "❌ No Promotion" if x == 0 else "✅ Active Promotion",
                    help="Is this product currently on sale or promotion?"
                )
                
                available_products = sorted(label_encoder.keys())
                product_id_input = st.selectbox(
                    "Product ID",
                    options=available_products,
                    help="Select the specific product to forecast"
                )
            
            submit_button = st.form_submit_button("🔮 Generate Forecast", type="primary", use_container_width=True)
            
            if submit_button:
                try:
                    if product_id_input not in label_encoder:
                        st.error(f"Product '{product_id_input}' not found in the system.")
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
                    
                    st.success("### ✅ Forecast Complete")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    result_col1.metric("📦 Predicted Sales", f"{predicted_units} units")
                    result_col2.metric("💰 Expected Revenue", f"${predicted_units * sale_price_input:,.2f}")
                    
                    confidence = "High" if lag_7_input > 10 else "Medium" if lag_7_input > 0 else "Low"
                    result_col3.metric("🎯 Confidence", confidence)
                    
                    st.markdown("---")
                    st.markdown("### 📋 Forecast Breakdown")
                    
                    explain_col1, explain_col2 = st.columns(2)
                    
                    with explain_col1:
                        st.markdown("**Input Factors:**")
                        st.write(f"• Product: {product_id_input}")
                        st.write(f"• Sales 7 Days Ago: {lag_7_input:.0f} units")
                        st.write(f"• Current Price: ${sale_price_input:.2f}")
                        st.write(f"• Promotion: {'Yes' if is_promo_input == 1 else 'No'}")
                    
                    with explain_col2:
                        st.markdown("**Prediction Details:**")
                        st.write(f"• Raw Prediction: {prediction:.2f} units")
                        st.write(f"• Rounded Forecast: {predicted_units} units")
                        
                        if predicted_units < lag_7_input * 0.5:
                            st.warning("⚠️ Significant sales decrease predicted")
                        elif predicted_units > lag_7_input * 1.5:
                            st.info("📈 Strong sales increase predicted")
                    
                    with st.expander("🔍 Why This Prediction?"):
                        st.markdown(f"""
                        The model analyzed your inputs and compared them to historical patterns:
                        
                        - Your product sold **{lag_7_input:.0f} units** last week, which is the strongest indicator
                        - At **${sale_price_input:.2f}**, the price is {'higher' if sale_price_input > 30 else 'lower'} than typical range
                        - {'Promotions typically boost sales by 10-30%' if is_promo_input == 1 else 'Without promotion, sales follow normal patterns'}
                        - This specific product has unique demand characteristics learned from history
                        
                        **Confidence Level:** {confidence}
                        {'- High: Strong historical data supports this forecast' if confidence == 'High' else '- Medium: Adequate data, but less certainty' if confidence == 'Medium' else '- Low: Limited historical data, use with caution'}
                        """)
                    
                    print("Prediction complete")
                    
                except Exception as e:
                    st.error(f"⚠️ Forecast error: {e}")
                    st.info("Please verify all inputs are valid.")
    
    elif prediction_mode == '📂 Batch Upload (CSV)':
        st.subheader("📂 Batch Sales Forecasting")
        st.markdown("Upload a CSV file with multiple products for bulk forecasting.")
        
        uploaded_csv = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            key="sales_csv",
            help="CSV should contain columns for lag_7, sale_price, is_promo, and product_id"
        )
        
        if uploaded_csv is not None:
            try:
                df_sales = pd.read_csv(uploaded_csv, encoding='utf-8')
                st.success("✅ CSV uploaded successfully")
                print("CSV uploaded")
                
                if len(df_sales) == 0:
                    st.error("The uploaded file is empty.")
                    st.stop()
                
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df_sales.head(10), use_container_width=True)
                    st.caption(f"Showing first 10 of {len(df_sales)} rows")
                
                st.markdown("---")
                st.subheader("🔗 Map Your Columns")
                st.markdown("Tell us which columns in your CSV correspond to each required feature:")
                
                col_map1, col_map2 = st.columns(2)
                with col_map1:
                    col_lag7 = st.selectbox(
                        "Column for 'lag_7' (units sold 7 days ago)",
                        options=df_sales.columns,
                        help="Numeric column with historical sales"
                    )
                    col_price = st.selectbox(
                        "Column for 'sale_price'",
                        options=df_sales.columns,
                        help="Numeric column with product prices"
                    )
                with col_map2:
                    col_promo = st.selectbox(
                        "Column for 'is_promo' (0 or 1)",
                        options=df_sales.columns,
                        help="Binary: 0 = no promotion, 1 = on promotion"
                    )
                    col_product = st.selectbox(
                        "Column for 'product_id'",
                        options=df_sales.columns,
                        help="Product identifier"
                    )
                
                if st.button("🚀 Generate Batch Forecasts", type="primary", use_container_width=True):
                    with st.spinner("Processing batch forecasts..."):
                        try:
                            df_pred = df_sales.copy()
                            
                            # Validate and convert
                            df_pred[col_lag7] = pd.to_numeric(df_pred[col_lag7], errors='coerce')
                            df_pred[col_price] = pd.to_numeric(df_pred[col_price], errors='coerce')
                            df_pred[col_promo] = pd.to_numeric(df_pred[col_promo], errors='coerce')
                            
                            # Check for invalid data
                            if df_pred[col_lag7].isna().any():
                                st.error(f"⚠️ Invalid values found in '{col_lag7}' column. Must be numeric.")
                                st.stop()
                            if df_pred[col_price].isna().any():
                                st.error(f"⚠️ Invalid values found in '{col_price}' column. Must be numeric.")
                                st.stop()
                            if df_pred[col_promo].isna().any():
                                st.error(f"⚠️ Invalid values found in '{col_promo}' column. Must be numeric.")
                                st.stop()
                            
                            # Vectorized prediction
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
                            st.subheader("📊 Batch Forecast Results")
                            
                            valid_predictions = df_pred['predicted_units_sold'].dropna()
                            total_rows = len(df_pred)
                            successful = len(valid_predictions)
                            failed = total_rows - successful
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            metric_col1.metric("Total Rows", total_rows)
                            metric_col2.metric("✅ Successful", successful)
                            metric_col3.metric("❌ Failed", failed)
                            metric_col4.metric("Success Rate", f"{(successful/total_rows*100):.1f}%")
                            
                            if successful > 0:
                                st.markdown("**Forecast Statistics:**")
                                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                                stat_col1.metric("Mean Forecast", f"{valid_predictions.mean():.1f} units")
                                stat_col2.metric("Min Forecast", f"{valid_predictions.min():.0f} units")
                                stat_col3.metric("Max Forecast", f"{valid_predictions.max():.0f} units")
                                stat_col4.metric("Total Units", f"{valid_predictions.sum():.0f}")
                            
                            if failed > 0:
                                st.warning(f"⚠️ {failed} rows failed. Common reasons: unknown product IDs or missing data.")
                            
                            st.markdown("### 📄 Detailed Results")
                            display_cols = [col_product, col_lag7, col_price, col_promo, 
                                          'predicted_units_sold_rounded']
                            st.dataframe(df_pred[display_cols], use_container_width=True, height=400)
                            
                            csv_output = df_pred.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Forecast Results (CSV)",
                                data=csv_output,
                                file_name='sales_forecast_results.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
                            print("Prediction complete")
                            
                        except Exception as e:
                            st.error(f"⚠️ Batch forecast error: {e}")
                            st.info("Please check your CSV format and column mappings.")
                            
            except Exception as e:
                st.error(f"⚠️ Error reading file: {e}")
                st.info("Please ensure your CSV is properly formatted.")

# =========================================================
# MODULE 3: ADVANCED PRICE & DEMAND PREDICTION
# =========================================================

elif selected_option == '💎 Price & Demand Prediction':
    st.header("💎 Product Pricing & Demand Analysis")
    
    # Module explanation
    with st.expander("ℹ️ About This Module", expanded=False):
        st.markdown("""
        **What it does:** Predicts the optimal price point and expected monthly sales volume for fashion and retail products based on multiple attributes.
        
        **How it works:** This system analyzes how different product characteristics historically influenced both pricing and sales volume. It uses weighted impact analysis to estimate market value and demand.
        
        **The 6 Major Factors:**
        - **Category:** Product type (Dresses, Shoes, Accessories, etc.)
        - **Subcategory:** Specific style within category
        - **Brand:** Brand reputation and positioning
        - **Season:** Optimal selling season
        - **Material:** Material quality and type
        - **Gender:** Target demographic
        
        **Best for:** New product pricing strategy, market positioning analysis, demand forecasting for product launches.
        
        **Limitations:** Works best for fashion/apparel products. Predictions are based on historical patterns and may not account for sudden market changes or viral trends.
        """)
    
    if advanced_model is None:
        st.error("⚠️ Advanced model not loaded. Please ensure sales_predictor_model.json exists.")
        st.stop()
    
    weights = advanced_model['weights']
    
    # Helper function for prediction (UNCHANGED)
    def predict_advanced(category, subcategory, brand, season, material, gender):
        base_price = weights['basePrice']
        base_units = weights['baseUnits']
        predicted_price = base_price
        predicted_units = base_units
        
        # Track each component for breakdown
        breakdown = {
            'base': {'price': base_price, 'units': base_units},
            'components': []
        }
        
        if category in weights['categoryWeights']:
            cat_price_impact = weights['categoryWeights'][category]['priceDiff']
            cat_units_impact = weights['categoryWeights'][category]['unitsDiff']
            predicted_price += cat_price_impact
            predicted_units += cat_units_impact
            breakdown['components'].append(('Category', category, cat_price_impact, cat_units_impact))
        
        if subcategory in weights['subcategoryWeights']:
            sub_price_impact = weights['subcategoryWeights'][subcategory]['priceDiff']
            sub_units_impact = weights['subcategoryWeights'][subcategory]['unitsDiff']
            predicted_price += sub_price_impact
            predicted_units += sub_units_impact
            breakdown['components'].append(('Subcategory', subcategory, sub_price_impact, sub_units_impact))
        
        if brand in weights['brandWeights']:
            brand_price_impact = weights['brandWeights'][brand]['priceDiff']
            brand_units_impact = weights['brandWeights'][brand]['unitsDiff']
            predicted_price += brand_price_impact
            predicted_units += brand_units_impact
            breakdown['components'].append(('Brand', brand, brand_price_impact, brand_units_impact))
        
        if season in weights['seasonWeights']:
            season_price_impact = weights['seasonWeights'][season]['priceDiff']
            season_units_impact = weights['seasonWeights'][season]['unitsDiff']
            predicted_price += season_price_impact
            predicted_units += season_units_impact
            breakdown['components'].append(('Season', season, season_price_impact, season_units_impact))
        
        if material in weights['materialWeights']:
            mat_price_impact = weights['materialWeights'][material]['priceDiff']
            mat_units_impact = weights['materialWeights'][material]['unitsDiff']
            predicted_price += mat_price_impact
            predicted_units += mat_units_impact
            breakdown['components'].append(('Material', material, mat_price_impact, mat_units_impact))
        
        if gender in weights['genderWeights']:
            gen_price_impact = weights['genderWeights'][gender]['priceDiff']
            gen_units_impact = weights['genderWeights'][gender]['unitsDiff']
            predicted_price += gen_price_impact
            predicted_units += gen_units_impact
            breakdown['components'].append(('Gender', gender, gen_price_impact, gen_units_impact))
        
        return max(0, predicted_price), max(0, predicted_units), breakdown
    
    st.markdown("### Choose Analysis Mode")
    prediction_mode = st.radio(
        "How would you like to analyze?",
        ['🔍 Single Product', '📂 Batch Upload (CSV)'],
        horizontal=True
    )
    st.markdown("---")
    
    if prediction_mode == '🔍 Single Product':
        st.subheader("🔍 Individual Product Analysis")
        st.markdown("Select product attributes to predict pricing and demand.")
        
        with st.form("advanced_form"):
            st.markdown("**📝 Select Product Attributes:**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                category = st.selectbox(
                    "Category",
                    options=sorted(weights['categoryWeights'].keys()),
                    help="Main product category"
                )
                subcategory = st.selectbox(
                    "Subcategory",
                    options=sorted(weights['subcategoryWeights'].keys()),
                    help="Specific product style"
                )
            
            with col2:
                brand = st.selectbox(
                    "Brand",
                    options=sorted(weights['brandWeights'].keys()),
                    help="Product brand"
                )
                season = st.selectbox(
                    "Season",
                    options=sorted(weights['seasonWeights'].keys()),
                    help="Target season"
                )
            
            with col3:
                material = st.selectbox(
                    "Material",
                    options=sorted(weights['materialWeights'].keys()),
                    help="Primary material"
                )
                gender = st.selectbox(
                    "Gender",
                    options=sorted(weights['genderWeights'].keys()),
                    help="Target demographic"
                )
            
            submit_btn = st.form_submit_button("🔮 Analyze Product", type="primary", use_container_width=True)
            
            if submit_btn:
                try:
                    price, units, breakdown = predict_advanced(category, subcategory, brand, season, material, gender)
                    
                    st.success("### ✅ Analysis Complete")
                    
                    # Main results
                    result_col1, result_col2, result_col3 = st.columns(3)
                    result_col1.metric("💰 Predicted Price", f"${price:.2f}")
                    result_col2.metric("📦 Monthly Demand", f"{int(round(units))} units")
                    
                    # Calculate confidence based on extreme values
                    is_luxury = price > 2000
                    is_low_demand = units < 50
                    is_high_demand = units > 100
                    
                    if is_luxury and is_low_demand:
                        confidence = "Medium"
                        confidence_color = "normal"
                    elif is_luxury or is_high_demand:
                        confidence = "High"
                        confidence_color = "normal"
                    else:
                        confidence = "High"
                        confidence_color = "normal"
                    
                    result_col3.metric("🎯 Confidence", confidence)
                    
                    expected_revenue = price * units
                    st.info(f"📊 **Expected Monthly Revenue:** ${expected_revenue:,.2f}")
                    
                    st.markdown("---")
                    
                    # Detailed breakdown
                    st.subheader("📊 Pricing & Demand Breakdown")
                    st.markdown("See how each attribute influences the final prediction:")
                    
                    breakdown_col1, breakdown_col2 = st.columns(2)
                    
                    with breakdown_col1:
                        st.markdown("**💰 Price Components:**")
                        st.write(f"Base Price: ${breakdown['base']['price']:.2f}")
                        for factor, value, price_impact, _ in breakdown['components']:
                            sign = "+" if price_impact >= 0 else ""
                            st.write(f"{factor} ({value}): {sign}${price_impact:.2f}")
                        st.markdown(f"**Final Price: ${price:.2f}**")
                    
                    with breakdown_col2:
                        st.markdown("**📦 Demand Components:**")
                        st.write(f"Base Demand: {breakdown['base']['units']:.1f} units")
                        for factor, value, _, units_impact in breakdown['components']:
                            sign = "+" if units_impact >= 0 else ""
                            st.write(f"{factor} ({value}): {sign}{units_impact:.1f} units")
                        st.markdown(f"**Final Demand: {int(round(units))} units**")
                    
                    st.markdown("---")
                    
                    # Insights and warnings
                    st.subheader("💡 Key Insights")
                    
                    # Find biggest impacts
                    price_impacts = [(f, v, abs(p)) for f, v, p, u in breakdown['components']]
                    price_impacts.sort(key=lambda x: x[2], reverse=True)
                    
                    units_impacts = [(f, v, abs(u)) for f, v, p, u in breakdown['components']]
                    units_impacts.sort(key=lambda x: x[2], reverse=True)
                    
                    insight_col1, insight_col2 = st.columns(2)
                    
                    with insight_col1:
                        st.markdown("**Biggest Price Drivers:**")
                        for i, (factor, value, impact) in enumerate(price_impacts[:3], 1):
                            st.write(f"{i}. {factor}: {value}")
                    
                    with insight_col2:
                        st.markdown("**Biggest Demand Drivers:**")
                        for i, (factor, value, impact) in enumerate(units_impacts[:3], 1):
                            st.write(f"{i}. {factor}: {value}")
                    
                    # Smart warnings
                    warnings_list = []
                    
                    if price > 5000:
                        warnings_list.append("⚠️ **Luxury Pricing:** This is a high-end product. Ensure brand justifies premium.")
                    
                    if units < 30:
                        warnings_list.append("⚠️ **Low Volume:** Expected sales are below 30 units/month. Consider niche marketing.")
                    
                    if price < 50 and 'Luxury' in brand or 'Premium' in brand:
                        warnings_list.append("⚠️ **Brand Mismatch:** Price seems low for a premium brand.")
                    
                    if units > 150:
                        warnings_list.append("📈 **High Demand:** Strong market demand predicted. Ensure adequate inventory.")
                    
                    if warnings_list:
                        st.markdown("---")
                        st.markdown("**⚠️ Recommendations & Warnings:**")
                        for warning in warnings_list:
                            st.markdown(warning)
                    
                    # Explanation section
                    with st.expander("🔍 Understanding This Prediction"):
                        st.markdown(f"""
                        **How We Calculated This:**
                        
                        The model started with baseline values for a typical product:
                        - Base Price: ${breakdown['base']['price']:.2f}
                        - Base Demand: {breakdown['base']['units']:.0f} units/month
                        
                        Then adjusted based on your specific attributes:
                        
                        **Price Adjustments:**
                        Your {brand} brand {'significantly increases' if any(p > 1000 for _, _, p, _ in breakdown['components']) else 'moderately affects'} the price.
                        The {material} material adds {'premium' if any(p > 500 for _, _, p, _ in breakdown['components'] if _ == 'Material') else 'standard'} value.
                        
                        **Demand Adjustments:**
                        The {category} category shows {'strong' if units > 80 else 'moderate'} market demand.
                        {season} season timing {'boosts' if any(u > 5 for _, _, _, u in breakdown['components'] if _ == 'Season') else 'maintains'} expected sales.
                        
                        **Confidence Assessment:**
                        This prediction has {confidence} confidence because:
                        {'- The combination of luxury pricing and niche appeal is well-documented' if is_luxury else '- This product profile matches common market patterns'}
                        {'- Historical data supports these volume expectations' if not is_low_demand else '- Limited historical data for this exact combination'}
                        """)
                    
                    print("Prediction complete")
                    
                except Exception as e:
                    st.error(f"⚠️ Analysis error: {e}")
                    st.info("Please verify all selections are valid.")
    
    elif prediction_mode == '📂 Batch Upload (CSV)':
        st.subheader("📂 Batch Product Analysis")
        st.markdown("Upload a CSV file to analyze multiple products at once.")
        
        uploaded_csv = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            key="advanced_csv",
            help="CSV should contain columns for all 6 attributes"
        )
        
        if uploaded_csv is not None:
            try:
                df_adv = pd.read_csv(uploaded_csv, encoding='utf-8')
                st.success("✅ CSV uploaded successfully")
                print("CSV uploaded")
                
                if len(df_adv) == 0:
                    st.error("The uploaded file is empty.")
                    st.stop()
                
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df_adv.head(10), use_container_width=True)
                    st.caption(f"Showing first 10 of {len(df_adv)} rows")
                
                if len(df_adv) > 10000:
                    st.warning(f"⚠️ Large dataset: {len(df_adv)} rows. Processing may take time.")
                
                st.markdown("---")
                st.subheader("🔗 Map Your Columns")
                st.markdown("Specify which columns contain each product attribute:")
                
                col_map1, col_map2, col_map3 = st.columns(3)
                with col_map1:
                    col_category = st.selectbox(
                        "Column for 'Category'",
                        options=df_adv.columns,
                        help="Product category column"
                    )
                    col_subcategory = st.selectbox(
                        "Column for 'Subcategory'",
                        options=df_adv.columns,
                        help="Product subcategory column"
                    )
                
                with col_map2:
                    col_brand = st.selectbox(
                        "Column for 'Brand'",
                        options=df_adv.columns,
                        help="Brand column"
                    )
                    col_season = st.selectbox(
                        "Column for 'Season'",
                        options=df_adv.columns,
                        help="Season column"
                    )
                
                with col_map3:
                    col_material = st.selectbox(
                        "Column for 'Material'",
                        options=df_adv.columns,
                        help="Material column"
                    )
                    col_gender = st.selectbox(
                        "Column for 'Gender'",
                        options=df_adv.columns,
                        help="Gender/demographic column"
                    )
                
                if st.button("🚀 Analyze Batch", type="primary", use_container_width=True):
                    with st.spinner("Processing batch analysis..."):
                        try:
                            # Validate columns exist
                            required_cols = [col_category, col_subcategory, col_brand, col_season, col_material, col_gender]
                            missing = [c for c in required_cols if c not in df_adv.columns]
                            if missing:
                                st.error(f"Missing columns: {', '.join(missing)}")
                                st.stop()
                            
                            # Convert to string for matching
                            df_adv['cat_str'] = df_adv[col_category].astype(str)
                            df_adv['sub_str'] = df_adv[col_subcategory].astype(str)
                            df_adv['brand_str'] = df_adv[col_brand].astype(str)
                            df_adv['season_str'] = df_adv[col_season].astype(str)
                            df_adv['material_str'] = df_adv[col_material].astype(str)
                            df_adv['gender_str'] = df_adv[col_gender].astype(str)
                            
                            # Create lookup dictionaries for vectorized operations
                            cat_price_map = {k: v['priceDiff'] for k, v in weights['categoryWeights'].items()}
                            cat_units_map = {k: v['unitsDiff'] for k, v in weights['categoryWeights'].items()}
                            sub_price_map = {k: v['priceDiff'] for k, v in weights['subcategoryWeights'].items()}
                            sub_units_map = {k: v['unitsDiff'] for k, v in weights['subcategoryWeights'].items()}
                            brand_price_map = {k: v['priceDiff'] for k, v in weights['brandWeights'].items()}
                            brand_units_map = {k: v['unitsDiff'] for k, v in weights['brandWeights'].items()}
                            season_price_map = {k: v['priceDiff'] for k, v in weights['seasonWeights'].items()}
                            season_units_map = {k: v['unitsDiff'] for k, v in weights['seasonWeights'].items()}
                            material_price_map = {k: v['priceDiff'] for k, v in weights['materialWeights'].items()}
                            material_units_map = {k: v['unitsDiff'] for k, v in weights['materialWeights'].items()}
                            gender_price_map = {k: v['priceDiff'] for k, v in weights['genderWeights'].items()}
                            gender_units_map = {k: v['unitsDiff'] for k, v in weights['genderWeights'].items()}
                            
                            # Vectorized prediction
                            df_adv['predicted_price'] = weights['basePrice']
                            df_adv['predicted_units'] = weights['baseUnits']
                            
                            df_adv['predicted_price'] += df_adv['cat_str'].map(cat_price_map).fillna(0)
                            df_adv['predicted_units'] += df_adv['cat_str'].map(cat_units_map).fillna(0)
                            df_adv['predicted_price'] += df_adv['sub_str'].map(sub_price_map).fillna(0)
                            df_adv['predicted_units'] += df_adv['sub_str'].map(sub_units_map).fillna(0)
                            df_adv['predicted_price'] += df_adv['brand_str'].map(brand_price_map).fillna(0)
                            df_adv['predicted_units'] += df_adv['brand_str'].map(brand_units_map).fillna(0)
                            df_adv['predicted_price'] += df_adv['season_str'].map(season_price_map).fillna(0)
                            df_adv['predicted_units'] += df_adv['season_str'].map(season_units_map).fillna(0)
                            df_adv['predicted_price'] += df_adv['material_str'].map(material_price_map).fillna(0)
                            df_adv['predicted_units'] += df_adv['material_str'].map(material_units_map).fillna(0)
                            df_adv['predicted_price'] += df_adv['gender_str'].map(gender_price_map).fillna(0)
                            df_adv['predicted_units'] += df_adv['gender_str'].map(gender_units_map).fillna(0)
                            
                            # Clean up and finalize
                            df_adv['predicted_price'] = df_adv['predicted_price'].clip(lower=0).round(2)
                            df_adv['predicted_units'] = df_adv['predicted_units'].clip(lower=0).round().astype(int)
                            df_adv['expected_revenue'] = (df_adv['predicted_price'] * df_adv['predicted_units']).round(2)
                            
                            # Remove temporary columns
                            temp_cols = ['cat_str', 'sub_str', 'brand_str', 'season_str', 'material_str', 'gender_str']
                            df_adv = df_adv.drop(columns=temp_cols)
                            
                            st.markdown("---")
                            st.subheader("📊 Batch Analysis Results")
                            
                            valid_prices = df_adv['predicted_price'].dropna()
                            valid_units = df_adv['predicted_units'].dropna()
                            total_revenue = df_adv['expected_revenue'].sum()
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            metric_col1.metric("Total Products", len(df_adv))
                            metric_col2.metric("✅ Successful", len(valid_prices))
                            metric_col3.metric("Avg Price", f"${valid_prices.mean():.2f}")
                            metric_col4.metric("Total Revenue", f"${total_revenue:,.0f}")
                            
                            if len(valid_prices) > 0:
                                st.markdown("**Price Distribution:**")
                                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                                stat_col1.metric("Min Price", f"${valid_prices.min():.2f}")
                                stat_col2.metric("Median Price", f"${valid_prices.median():.2f}")
                                stat_col3.metric("Max Price", f"${valid_prices.max():.2f}")
                                stat_col4.metric("Total Units", f"{valid_units.sum():,.0f}")
                            
                            # Identify notable products
                            st.markdown("---")
                            st.markdown("### 🏆 Notable Products")
                            
                            notable_col1, notable_col2 = st.columns(2)
                            
                            with notable_col1:
                                st.markdown("**💰 Highest Value Products:**")
                                top_price = df_adv.nlargest(5, 'predicted_price')[[col_brand, col_category, 'predicted_price']]
                                st.dataframe(top_price, use_container_width=True, hide_index=True)
                            
                            with notable_col2:
                                st.markdown("**📦 Highest Demand Products:**")
                                top_units = df_adv.nlargest(5, 'predicted_units')[[col_brand, col_category, 'predicted_units']]
                                st.dataframe(top_units, use_container_width=True, hide_index=True)
                            
                            st.markdown("---")
                            st.markdown("### 📄 Complete Results")
                            
                            # Add category-based filters
                            filter_col1, filter_col2 = st.columns(2)
                            with filter_col1:
                                show_luxury = st.checkbox("Show only luxury items (>$1000)", value=False)
                            with filter_col2:
                                show_high_demand = st.checkbox("Show only high demand (>100 units)", value=False)
                            
                            display_df = df_adv.copy()
                            if show_luxury:
                                display_df = display_df[display_df['predicted_price'] > 1000]
                            if show_high_demand:
                                display_df = display_df[display_df['predicted_units'] > 100]
                            
                            st.dataframe(display_df, use_container_width=True, height=400)
                            st.caption(f"Displaying {len(display_df)} of {len(df_adv)} products")
                            
                            csv_out = df_adv.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Complete Analysis (CSV)",
                                data=csv_out,
                                file_name='price_demand_analysis_results.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
                            
                            print("Prediction complete")
                            
                        except Exception as e:
                            st.error(f"⚠️ Analysis error: {e}")
                            st.info("Please check your CSV format and ensure all required columns are present.")
                            
            except Exception as e:
                st.error(f"⚠️ Error reading CSV: {e}")
                st.info("Please ensure your CSV is properly formatted and encoded as UTF-8.")
