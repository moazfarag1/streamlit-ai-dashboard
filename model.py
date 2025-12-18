import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings

warnings.filterwarnings('ignore')

class PriceSalesModel:
    def __init__(self):
        self.price_model = None
        self.sales_model = None
        self.encoders = {} 
        self.numerical_defaults = {}
        
        # Explicitly define luxury brands to force-keep them
        self.luxury_brands = [
            'Rolex', 'Gucci', 'Prada', 'Louis Vuitton', 'Hermes', 
            'Burberry', 'Moncler', 'Jimmy Choo', 'Tiffany', 'Cartier'
        ]
        
        self.cat_cols = ['category', 'subcategory', 'brand', 'color', 
                         'size', 'material', 'season', 'gender', 'day_of_week']
        
        # Added 'is_luxury' to features
        self.price_features = [
            'original_price', 'category', 'subcategory', 'brand', 'material', 
            'season', 'gender', 'competitor_price', 'days_since_launch', 
            'is_holiday_season', 'month', 'is_weekend', 'stock_quantity',
            'is_luxury'
        ]
        
        self.sales_features_base = [
            'original_price', 'discount_percentage', 'category', 'brand', 
            'season', 'rating', 'number_of_reviews', 'stock_quantity', 
            'website_views', 'cart_additions', 'return_rate', 
            'price_vs_competitor', 'month', 'day_of_week', 'days_since_launch',
            'is_luxury'
        ]

    def _get_date_features(self, df):
        df_dt = df.copy()
        if 'date' in df_dt.columns:
            df_dt['date'] = pd.to_datetime(df_dt['date'], errors='coerce')
            df_dt['month'] = df_dt['date'].dt.month.fillna(0).astype(int)
            df_dt['day_of_week'] = df_dt['date'].dt.day_name().fillna('Unknown')
        return df_dt

    def _engineer_features(self, df):
        df_eng = df.copy()
        
        # 1. Luxury Flag: Helps the model treat these brands differently
        # Ensure brand column is string to avoid errors
        df_eng['brand'] = df_eng['brand'].astype(str)
        df_eng['is_luxury'] = df_eng['brand'].apply(
            lambda x: 1 if x in self.luxury_brands else 0
        )
        
        # 2. Price context
        ref_price = df_eng.get('current_price', df_eng['original_price'])
        if 'competitor_price' in df_eng.columns:
            df_eng['price_vs_competitor'] = ref_price / (df_eng['competitor_price'] + 0.01)
        else:
            df_eng['price_vs_competitor'] = 1.0
            
        return df_eng

    def _safe_encode(self, df, fit=False):
        df_enc = df.copy()
        for col in self.cat_cols:
            if col not in df_enc.columns: continue
            
            df_enc[col] = df_enc[col].fillna("Unknown").astype(str)
            
            if fit:
                # Keep top 50 categories by frequency
                top_vals = df_enc[col].value_counts().nlargest(50).index.tolist()
                
                # CRITICAL FIX: Always add luxury brands to the 'known' list
                # so they don't get lumped into "Other" even if rare
                if col == 'brand':
                    for lux in self.luxury_brands:
                        if lux not in top_vals:
                            top_vals.append(lux)
                
                unique_vals = list(set(top_vals + ["Unknown", "Other"]))
                self.encoders[col] = {val: i for i, val in enumerate(unique_vals)}
            
            # Map values
            unknown_id = self.encoders[col].get("Unknown", 0)
            other_id = self.encoders[col].get("Other", 0)
            
            df_enc[col] = df_enc[col].map(
                lambda x: self.encoders[col].get(x, other_id)
            ).fillna(unknown_id)
            
        return df_enc

    def prepare_data(self, df, fit=False):
        df_prep = self._get_date_features(df)
        df_prep = self._engineer_features(df_prep)
        df_prep = self._safe_encode(df_prep, fit=fit)
        
        num_cols = [c for c in df_prep.columns if df_prep[c].dtype in ['float64', 'int64']]
        if fit:
            for col in num_cols:
                self.numerical_defaults[col] = df_prep[col].median()
        
        for col in num_cols:
            if col in self.numerical_defaults:
                df_prep[col] = df_prep[col].fillna(self.numerical_defaults[col])
                
        return df_prep

    def train(self, train_df):
        # --- FIX START: REMOVED THE PRICE CAP ---
        # We no longer filter top 1% prices. We keep the luxury items.
        print(">> Preparing Training Data (Including Luxury Items)...")
        df_train = self.prepare_data(train_df, fit=True)
        # --- FIX END ---
        
        # 1. Train Price Model
        print(">> Training Price Model (XGBoost)...")
        X_price = df_train[self.price_features]
        y_price = df_train['current_price']
        
        self.price_model = xgb.XGBRegressor(
            n_estimators=1000, 
            max_depth=8, # Increased depth to capture complex luxury rules
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1, 
            random_state=42
        )
        self.price_model.fit(X_price, y_price)
        
        # 2. Train Sales Model
        print(">> Training Sales Model (XGBoost)...")
        df_train['predicted_price'] = self.price_model.predict(X_price)
        
        sales_features = ['predicted_price'] + self.sales_features_base
        X_sales = df_train[sales_features]
        y_sales = df_train['units_sold']
        
        self.sales_model = xgb.XGBRegressor(
            n_estimators=1000, 
            max_depth=7, 
            learning_rate=0.03,
            subsample=0.8,
            n_jobs=-1, 
            random_state=42
        )
        self.sales_model.fit(X_sales, y_sales)
        print(">> Training Complete.")

    def evaluate(self, test_df):
        print("\n" + "="*50)
        print("EVALUATION ON EXTERNAL TEST SET")
        print("="*50)
        
        df_test = self.prepare_data(test_df, fit=False)
        
        # --- Evaluate Price ---
        X_price_test = df_test[self.price_features]
        y_price_actual = df_test['current_price']
        
        price_preds = self.price_model.predict(X_price_test)
        price_preds = np.maximum(price_preds, 0)
        
        p_rmse = np.sqrt(mean_squared_error(y_price_actual, price_preds))
        p_r2 = r2_score(y_price_actual, price_preds)
        
        print(f"[Price Model]")
        print(f"  RMSE:     ${p_rmse:.2f}")
        print(f"  R2 Score: {p_r2:.4f}")

        # --- Evaluate Sales ---
        df_test['predicted_price'] = price_preds
        sales_features = ['predicted_price'] + self.sales_features_base
        X_sales_test = df_test[sales_features]
        y_sales_actual = df_test['units_sold']
        
        sales_preds = self.sales_model.predict(X_sales_test)
        sales_preds = np.maximum(sales_preds, 0) 
        
        s_rmse = np.sqrt(mean_squared_error(y_sales_actual, sales_preds))
        s_r2 = r2_score(y_sales_actual, sales_preds)
        
        print(f"\n[Sales Model]")
        print(f"  RMSE:     {s_rmse:.1f} units")
        print(f"  R2 Score: {s_r2:.4f}")

    def predict_batch(self, input_df):
        df_prep = self.prepare_data(input_df, fit=False)
        
        X_price = df_prep[self.price_features]
        predicted_prices = self.price_model.predict(X_price)
        predicted_prices = np.maximum(predicted_prices, 0) 
        
        df_prep['predicted_price'] = predicted_prices
        
        sales_cols = ['predicted_price'] + self.sales_features_base
        X_sales = df_prep[sales_cols]
        
        predicted_sales = self.sales_model.predict(X_sales)
        predicted_sales = np.maximum(predicted_sales, 0)
        
        results = input_df.copy()
        results['Recommended_Price'] = np.round(predicted_prices, 2)
        results['Predicted_Units_Sold'] = np.round(predicted_sales, 0).astype(int)
        results['Predicted_Revenue'] = results['Recommended_Price'] * results['Predicted_Units_Sold']
        
        return results

    def predict_single(self, product_dict):
        input_df = pd.DataFrame([product_dict])
        results_df = self.predict_batch(input_df)
        return results_df.iloc[0].to_dict()

    def save(self, path):
        with open(path, 'wb') as f: pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f: return pickle.load(f)