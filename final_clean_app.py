import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import json
import warnings
import logging
from datetime import datetime

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
    """Load sentiment analysis models - DO NOT MODIFY"""
    try:
        sent_model = pickle.load(open('trained_model.sav', 'rb'))
        vect = pickle.load(open('vectorizer.sav', 'rb'))
        print("Model loaded")
        return sent_model, vect
    except:
        return None, None

@st.cache_resource
def load_sales_model():
    """Load retail sales forecasting model - DO NOT MODIFY"""
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
def load_advanced_price_model():
    """
    Load ML-based price & sales prediction model (XGBoost)
    FIXED: Added validation to ensure loaded object is PriceSalesModel instance
    Model artifact: final_sales_model.pkl
    """
    try:
        with open('final_sales_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # SAFETY CHECK: Verify it's a PriceSalesModel instance with required methods
        if not hasattr(model, 'predict_single') or not hasattr(model, 'predict_batch'):
            print("ERROR: Loaded model does not have required methods")
            return None
        
        # Verify methods are callable
        if not callable(getattr(model, 'predict_single', None)) or not callable(getattr(model, 'predict_batch', None)):
            print("ERROR: Model methods are not callable")
            return None
            
        print("Model loaded")
        return model
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

# Load all models
sentiment_model, vectorizer = load_sentiment_models()
sales_model, label_encoder = load_sales_model()
advanced_price_model = load_advanced_price_model()

# =========================================================
# SIDEBAR - UNCHANGED
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
# MODULE 2: RETAIL SALES FORECASTING (IMMUTABLE - DO NOT MODIFY)
# =========================================================

elif selected_option == '📦 Sales Forecasting':
    st.header("📦 Daily Sales Forecasting System")
    
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
                            
                            df_pred[col_lag7] = pd.to_numeric(df_pred[col_lag7], errors='coerce')
                            df_pred[col_price] = pd.to_numeric(df_pred[col_price], errors='coerce')
                            df_pred[col_promo] = pd.to_numeric(df_pred[col_promo], errors='coerce')
                            
                            if df_pred[col_lag7].isna().any():
                                st.error(f"⚠️ Invalid values found in '{col_lag7}' column. Must be numeric.")
                                st.stop()
                            if df_pred[col_price].isna().any():
                                st.error(f"⚠️ Invalid values found in '{col_price}' column. Must be numeric.")
                                st.stop()
                            if df_pred[col_promo].isna().any():
                                st.error(f"⚠️ Invalid values found in '{col_promo}' column. Must be numeric.")
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
# MODULE 3: ML-BASED PRICE & DEMAND PREDICTION (FIXED)
# =========================================================

elif selected_option == '💎 Price & Demand Prediction':
    st.header("💎 ML-Powered Price & Demand Analysis")
    
    with st.expander("ℹ️ About This Module", expanded=False):
        st.markdown("""
        **What it does:** Uses XGBoost machine learning to provide data-driven price recommendations and sales volume projections for retail products based on comprehensive product attributes and market conditions.
        
        **How it works:** This system was trained on historical retail transactions. It learns patterns between product characteristics, pricing, market conditions, and actual sales performance to generate recommendations.
        
        **Key Prediction Outputs:**
        - **Recommended Price:** Data-driven price point based on product attributes and market positioning
        - **Predicted Units Sold:** Expected monthly sales volume
        - **Predicted Revenue:** Estimated total revenue (Price × Units)
        
        **Major Factors Analyzed:**
        - Product attributes (category, brand, material, gender, season)
        - Cost structure (original price, competitor pricing)
        - Market performance (rating, reviews, stock levels)
        - Marketing metrics (web views, cart additions, return rate)
        - Temporal factors (month, day of week, days since launch)
        
        **Best for:** New product launches, pricing strategy exploration, revenue forecasting, competitive positioning analysis.
        
        **Model Performance:** Trained on diverse product categories. Results are estimates based on historical patterns.
        
        **Limitations:** Predictions are based on historical patterns and may not capture sudden market disruptions, viral trends, or unprecedented events. Results should be validated against business expertise and current market conditions.
        """)
    
    # FIXED: Enhanced model validation with clear error messaging
    if advanced_price_model is None:
        st.error("⚠️ ML model not loaded. Please ensure final_sales_model.pkl exists in the application directory.")
        st.info("💡 The model file should be generated by running: `python train_model.py`")
        st.stop()
    
    # FIXED: Additional runtime check for method availability
    if not hasattr(advanced_price_model, 'predict_single') or not hasattr(advanced_price_model, 'predict_batch'):
        st.error("⚠️ Model loaded but missing required prediction methods. Please retrain the model.")
        st.stop()
    
    st.markdown("### Choose Analysis Mode")
    prediction_mode = st.radio(
        "How would you like to analyze?",
        ['🔍 Single Product', '📂 Batch Upload (CSV)'],
        horizontal=True
    )
    st.markdown("---")
    
    if prediction_mode == '🔍 Single Product':
        st.subheader("🔍 Individual Product Analysis")
        st.markdown("Enter product details to receive ML-based price and demand predictions.")
        
        with st.form("ml_price_form"):
            st.markdown("**🏷️ Core Product Information:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                brand = st.text_input("Brand", value="Nike", help="Product brand (e.g., Nike, Gucci, Zara)")
                category = st.text_input("Category", value="Shoes", help="Main category (e.g., Shoes, Dresses, Accessories)")
                subcategory = st.text_input("Subcategory", value="Sneakers", help="Specific style (e.g., Sneakers, Running Shoes)")
            
            with col2:
                gender = st.selectbox("Gender", ["Men", "Women", "Unisex", "Kids"], help="Target demographic")
                season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"], help="Optimal selling season")
                material = st.text_input("Material", value="Leather", help="Primary material (e.g., Cotton, Leather, Polyester)")
            
            with col3:
                original_price = st.number_input("Original Cost ($)", min_value=0.0, value=50.0, step=1.0, help="Manufacturing or wholesale cost")
                competitor_price = st.number_input("Competitor Price ($)", min_value=0.0, value=120.0, step=1.0, help="Average market price for similar products")
                discount_pct = st.number_input("Discount %", min_value=0.0, max_value=100.0, value=0.0, step=1.0, help="Current discount percentage")
            
            st.markdown("---")
            st.markdown("**📊 Market Performance Metrics:**")
            st.caption("Use defaults for new products or adjust based on historical data")
            
            col4, col5, col6 = st.columns(3)
            
            with col4:
                rating = st.slider("Product Rating", 1.0, 5.0, 4.0, 0.1, help="Customer rating (1-5 stars)")
                num_reviews = st.number_input("Number of Reviews", min_value=0, value=50, step=10, help="Total customer reviews")
                return_rate = st.slider("Return Rate", 0.0, 1.0, 0.1, 0.01, help="Product return rate (0-1)")
            
            with col5:
                stock_qty = st.number_input("Stock Quantity", min_value=0, value=100, step=10, help="Current inventory level")
                web_views = st.number_input("Website Views", min_value=0, value=500, step=50, help="Monthly product page views")
                cart_adds = st.number_input("Cart Additions", min_value=0, value=20, step=5, help="Monthly add-to-cart actions")
            
            with col6:
                days_launch = st.number_input("Days Since Launch", min_value=0, value=30, step=1, help="Days since product was introduced")
                is_holiday = st.checkbox("Holiday Season", help="Is this a holiday shopping period?")
                is_weekend = st.checkbox("Weekend Pricing", help="Apply weekend pricing strategy?")
            
            submit_btn = st.form_submit_button("🔮 Generate ML Prediction", type="primary", use_container_width=True)
            
            if submit_btn:
                with st.spinner("Running ML analysis..."):
                    try:
                        # FIXED: Schema alignment - building input dict that matches PriceSalesModel expectations
                        # All required and optional fields from model.py are provided with safe defaults
                        product_data = {
                            # Core identifiers (will use defaults if not needed by model)
                            'product_id': 'STREAMLIT_SINGLE',
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            
                            # User-provided categorical fields (match model.py cat_cols)
                            'brand': str(brand).strip(),  # Ensure string type
                            'category': str(category).strip(),
                            'subcategory': str(subcategory).strip(),
                            'gender': str(gender),
                            'season': str(season),
                            'material': str(material).strip(),
                            'color': 'Standard',  # Default for optional field
                            'size': 'M',  # Default for optional field
                            
                            # Numerical fields with proper types
                            'original_price': float(original_price),
                            'current_price': float(original_price),  # Will be predicted by model
                            'competitor_price': float(competitor_price),
                            'discount_percentage': float(discount_pct),
                            'rating': float(rating),
                            'number_of_reviews': int(num_reviews),
                            'stock_quantity': int(stock_qty),
                            'website_views': int(web_views),
                            'cart_additions': int(cart_adds),
                            'return_rate': float(return_rate),
                            'days_since_launch': int(days_launch),
                            
                            # Binary flags (as integers 0/1)
                            'is_holiday_season': 1 if is_holiday else 0,
                            'is_weekend': 1 if is_weekend else 0,
                            
                            # Placeholder for target (will be predicted)
                            'units_sold': 0
                        }
                        
                        # FIXED: Interface enforcement - only calling predict_single method
                        # Defensive check already done at module load, but double-checking
                        if not hasattr(advanced_price_model, 'predict_single'):
                            st.error("⚠️ Model interface error: predict_single method not found")
                            st.stop()
                        
                        result = advanced_price_model.predict_single(product_data)
                        
                        # Extract results safely
                        rec_price = result.get('Recommended_Price', 0)
                        pred_units = result.get('Predicted_Units_Sold', 0)
                        pred_revenue = result.get('Predicted_Revenue', 0)
                        
                        st.success("### ✅ ML Analysis Complete")
                        
                        # Main results
                        result_col1, result_col2, result_col3 = st.columns(3)
                        result_col1.metric("💰 Recommended Price", f"${rec_price:,.2f}")
                        result_col2.metric("📦 Predicted Monthly Sales", f"{pred_units:,} units")
                        result_col3.metric("💵 Expected Revenue", f"${pred_revenue:,.2f}")
                        
                        # Calculate insights
                        profit_margin = ((rec_price - original_price) / rec_price * 100) if rec_price > 0 else 0
                        comp_diff = ((rec_price - competitor_price) / competitor_price * 100) if competitor_price > 0 else 0
                        
                        st.markdown("---")
                        st.subheader("📊 Business Insights")
                        
                        insight_col1, insight_col2 = st.columns(2)
                        
                        with insight_col1:
                            st.markdown("**💡 Pricing Strategy:**")
                            st.write(f"• Profit Margin: {profit_margin:.1f}%")
                            st.write(f"• vs Competitor: {'+' if comp_diff >= 0 else ''}{comp_diff:.1f}%")
                            
                            # FIXED: Academically conservative language
                            if rec_price > competitor_price * 1.2:
                                st.info("📈 Higher than market average")
                            elif rec_price < competitor_price * 0.8:
                                st.info("💵 Lower than market average")
                            else:
                                st.info("⚖️ Aligned with market average")
                        
                        with insight_col2:
                            st.markdown("**📈 Demand Forecast:**")
                            st.write(f"• Expected Monthly Units: {pred_units:,}")
                            st.write(f"• Revenue Estimate: ${pred_revenue:,.2f}")
                            
                            # FIXED: Conservative descriptive language
                            if pred_units < 50:
                                st.warning("⚠️ Low volume projected")
                            elif pred_units > 200:
                                st.success("🎯 High demand projected")
                            else:
                                st.info("✅ Moderate demand projected")
                        
                        # Recommendations
                        st.markdown("---")
                        st.subheader("⚠️ Considerations")
                        
                        recommendations = []
                        
                        if rec_price > original_price * 5:
                            recommendations.append("🔥 **High markup detected** - Strong brand value indicated by model. Verify market positioning supports this.")
                        
                        if pred_units < 30 and rec_price > 1000:
                            recommendations.append("💎 **Low volume, high margin** - Model suggests niche positioning. Consider specialized marketing.")
                        
                        if stock_qty < pred_units:
                            recommendations.append(f"📦 **Inventory note** - Predicted demand ({pred_units}) exceeds current stock ({stock_qty}).")
                        
                        if return_rate > 0.2:
                            recommendations.append("⚠️ **High return rate** - Consider product quality review or improved sizing information.")
                        
                        if cart_adds / web_views < 0.02 and web_views > 100:
                            recommendations.append("🔍 **Low conversion rate** - Product page may benefit from optimization.")
                        
                        if not recommendations:
                            recommendations.append("✅ **Balanced profile** - Product metrics are within typical ranges.")
                        
                        for rec in recommendations:
                            st.markdown(rec)
                        
                        # Model explanation
                        with st.expander("🔬 Understanding The ML Prediction"):
                            st.markdown(f"""
                            **How the Model Generated This Prediction:**
                            
                            The XGBoost model analyzed {len(product_data)} input features and compared your product to patterns learned from historical transactions.
                            
                            **Price Recommendation (${rec_price:,.2f}):**
                            - Base cost: ${original_price:.2f}
                            - Market context: {brand} brand in {category} category
                            - Competitor benchmark: ${competitor_price:.2f}
                            - Quality indicators: {rating}/5 stars, {num_reviews} reviews
                            - Calculated markup: {profit_margin:.1f}%
                            
                            **Sales Forecast ({pred_units} units per month):**
                            - Price point positioning: {rec_price:.2f}
                            - Market engagement: {web_views} views → {cart_adds} cart adds ({cart_adds/web_views*100 if web_views > 0 else 0:.1f}% rate)
                            - Seasonal timing: {season} season
                            - Inventory availability: {stock_qty} units in stock
                            
                            **Important Note:**
                            These predictions are estimates based on historical patterns. Actual results may vary based on market conditions, competitive actions, and other factors not captured in the model. Use these insights alongside business expertise and market knowledge.
                            """)
                        
                        print("Prediction complete")
                        
                    except Exception as e:
                        st.error(f"⚠️ Prediction error: {str(e)}")
                        st.info("Please verify all inputs are valid. If the error persists, the model file may need to be retrained.")
    
    elif prediction_mode == '📂 Batch Upload (CSV)':
        st.subheader("📂 Batch Product Analysis")
        st.markdown("Upload a CSV file to analyze multiple products at once using ML predictions.")
        
        with st.expander("📋 Required CSV Columns", expanded=False):
            st.markdown("""
            Your CSV should contain these columns for best results:
            
            **Essential Columns:**
            - `brand` - Product brand
            - `category` - Product category
            - `original_price` - Base cost
            
            **Recommended Columns:**
            - `subcategory`, `gender`, `season`, `material`
            - `competitor_price`, `discount_percentage`
            - `rating`, `number_of_reviews`, `stock_quantity`
            
            **Optional (will use defaults if missing):**
            - `website_views`, `cart_additions`, `return_rate`
            - `days_since_launch`, `is_holiday_season`, `is_weekend`
            - `color`, `size`, `product_id`, `date`
            
            Missing optional columns will be filled with safe default values.
            """)
        
        uploaded_csv = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            key="ml_price_csv",
            help="CSV with product attributes for ML-based analysis"
        )
        
        if uploaded_csv is not None:
            
                df_ml = pd.read_csv(uploaded_csv, encoding='utf-8')
                st.success("✅ CSV uploaded successfully")
                print("CSV uploaded")
                
                if len(df_ml) == 0:
                    st.error("The uploaded file is empty.")
                    st.stop()
                
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df_ml.head(10), use_container_width=True)
                    st.caption(f"Showing first 10 of {len(df_ml)} rows")
                
                # FIXED: Batch safety - validate required columns
                required_cols = ['brand', 'category', 'original_price']
                missing_cols = [c for c in required_cols if c not in df_ml.columns]
                
                if missing_cols:
                    st.error(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
                    st.info("Please ensure your CSV contains at minimum: brand, category, original_price")
                    st.stop()
                
                # FIXED: Fill missing optional columns with safe defaults matching model.py expectations
                default_values = {
                    'subcategory': 'General',
                    'gender': 'Unisex',
                    'season': 'Spring',
                    'material': 'Standard',
                    'color': 'Standard',
                    'size': 'M',
                    'competitor_price': 0.0,
                    'discount_percentage': 0.0,
                    'rating': 4.0,
                    'number_of_reviews': 50,
                    'stock_quantity': 100,
                    'website_views': 500,
                    'cart_additions': 20,
                    'return_rate': 0.1,
                    'days_since_launch': 30,
                    'is_holiday_season': 0,
                    'is_weekend': 0,
                    'product_id': '',
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'current_price': 0.0,
                    'units_sold': 0
                }
                
                for col, default_val in default_values.items():
                    if col not in df_ml.columns:
                        df_ml[col] = default_val
                
                # Generate product IDs if missing or empty
                if 'product_id' not in df_ml.columns or df_ml['product_id'].isna().all() or (df_ml['product_id'] == '').all():
                    df_ml['product_id'] = [f'BATCH_{i:04d}' for i in range(len(df_ml))]
                
                # FIXED: Ensure categorical fields are strings to prevent encoding errors
                categorical_cols = ['brand', 'category', 'subcategory', 'gender', 'season', 'material', 'color', 'size']
                for col in categorical_cols:
                    if col in df_ml.columns:
                        df_ml[col] = df_ml[col].fillna('Unknown').astype(str).str.strip()
                
                # FIXED: Ensure numerical fields are proper types
                numeric_cols = ['original_price', 'competitor_price', 'discount_percentage', 'rating', 
                               'number_of_reviews', 'stock_quantity', 'website_views', 'cart_additions', 
                               'return_rate', 'days_since_launch']
                for col in numeric_cols:
                    if col in df_ml.columns:
                        df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce').fillna(default_values.get(col, 0))
                
                st.info(f"ℹ️ Processing {len(df_ml)} products with ML model...")
                
                if len(df_ml) > 1000:
                    st.warning(f"⚠️ Large dataset ({len(df_ml)} rows). Processing may take 1-2 minutes.")
                
                if st.button("🚀 Run ML Batch Analysis", type="primary", use_container_width=True):
                    with st.spinner("Running ML predictions on all products..."):
                        try:
                            # FIXED: Interface enforcement - only calling predict_batch method
                            if not hasattr(advanced_price_model, 'predict_batch'):
                                st.error("⚠️ Model interface error: predict_batch method not found")
                                st.stop()
                            
                            # FIXED: Batch safety - try prediction with error handling
                            results_df = advanced_price_model.predict_batch(df_ml)
                            
                            # Validate results
                            if results_df is None or len(results_df) == 0:
                                st.error("⚠️ Batch prediction returned no results")
                                st.stop()
                            
                            st.markdown("---")
                            st.subheader("📊 Batch Analysis Results")
                            
                            # Summary metrics (with safe calculations)
                            total_revenue = results_df['Predicted_Revenue'].sum() if 'Predicted_Revenue' in results_df.columns else 0
                            avg_price = results_df['Recommended_Price'].mean() if 'Recommended_Price' in results_df.columns else 0
                            total_units = results_df['Predicted_Units_Sold'].sum() if 'Predicted_Units_Sold' in results_df.columns else 0
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            metric_col1.metric("Total Products", len(results_df))
                            metric_col2.metric("Avg Price", f"${avg_price:,.2f}")
                            metric_col3.metric("Total Units", f"{total_units:,.0f}")
                            metric_col4.metric("Total Revenue", f"${total_revenue:,.0f}")
                            
                            st.markdown("**Price Distribution:**")
                            price_col1, price_col2, price_col3, price_col4 = st.columns(4)
                            price_col1.metric("Min Price", f"${results_df['Recommended_Price'].min():.2f}")
                            price_col2.metric("Median Price", f"${results_df['Recommended_Price'].median():.2f}")
                            price_col3.metric("Max Price", f"${results_df['Recommended_Price'].max():.2f}")
                            
                            # Safe margin calculation
                            avg_original = df_ml['original_price'].mean()
                            avg_margin = ((avg_price - avg_original) / avg_price * 100) if avg_price > 0 else 0
                            price_col4.metric("Avg Margin", f"{avg_margin:.1f}%")
                            
                            # Top performers
                            st.markdown("---")
                            st.markdown("### 🏆 Top Performers")
                            
                            top_col1, top_col2 = st.columns(2)
                            
                            with top_col1:
                                st.markdown("**💰 Highest Revenue Products:**")
                                if len(results_df) >= 5:
                                    top_revenue = results_df.nlargest(5, 'Predicted_Revenue')[
                                        ['brand', 'category', 'Recommended_Price', 'Predicted_Units_Sold', 'Predicted_Revenue']
                                    ]
                                else:
                                    top_revenue = results_df[
                                        ['brand', 'category', 'Recommended_Price', 'Predicted_Units_Sold', 'Predicted_Revenue']
                                    ]
                                st.dataframe(top_revenue, use_container_width=True, hide_index=True)
                            
                            with top_col2:
                                st.markdown("**📈 Highest Volume Products:**")
                                if len(results_df) >= 5:
                                    top_volume = results_df.nlargest(5, 'Predicted_Units_Sold')[
                                        ['brand', 'category', 'Recommended_Price', 'Predicted_Units_Sold', 'Predicted_Revenue']
                                    ]
                                else:
                                    top_volume = results_df[
                                        ['brand', 'category', 'Recommended_Price', 'Predicted_Units_Sold', 'Predicted_Revenue']
                                    ]
                                st.dataframe(top_volume, use_container_width=True, hide_index=True)
                            
                            # Full results with filters
                            st.markdown("---")
                            st.markdown("### 📄 Complete Results")
                            
                            filter_col1, filter_col2, filter_col3 = st.columns(3)
                            with filter_col1:
                                show_premium = st.checkbox("Premium items only (>$500)", value=False)
                            with filter_col2:
                                show_high_volume = st.checkbox("High volume only (>100 units)", value=False)
                            with filter_col3:
                                show_high_revenue = st.checkbox("High revenue only (>$10k)", value=False)
                            
                            display_df = results_df.copy()
                            if show_premium:
                                display_df = display_df[display_df['Recommended_Price'] > 500]
                            if show_high_volume:
                                display_df = display_df[display_df['Predicted_Units_Sold'] > 100]
                            if show_high_revenue:
                                display_df = display_df[display_df['Predicted_Revenue'] > 10000]
                            
                            st.dataframe(display_df, use_container_width=True, height=400)
                            st.caption(f"Displaying {len(display_df)} of {len(results_df)} products")
                            
                            # Download results
                            csv_out = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Complete Analysis (CSV)",
                                data=csv_out,
                                file_name='ml_price_predictions.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
                            
                            st.info("💡 Note: Predictions are estimates based on historical patterns. Validate results against current market conditions.")
                            
                            print("Prediction complete")
                            
                        except Exception as e:
                            st.error(f"⚠️ Batch analysis error: {str(e)}")
