import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle

def fill_tn_with_0(df):
    df['tn'] = df['tn'].fillna(0)
    return df

def create_periodo_dt(df):
    df['periodo_dt'] = pd.to_datetime(df['periodo'].astype(str), format='%Y%m')
    return df

def create_date_features(df):   
    # #### Transformaciones temporales

    # Extraer año, mes, días del mes y quarter
    df['year'] = df['periodo_dt'].dt.year
    df['month'] = df['periodo_dt'].dt.month
    df['days_in_month'] = df['periodo_dt'].dt.days_in_month
    df['quarter'] = df['periodo_dt'].dt.quarter

    # Transformaciones cíclicas de tiempo usando seno y coseno
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter']/4) 
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter']/4)

    return df

def scale_tn(df, scaler_path=None, is_train=True):
    if is_train:
        scalers = {}
        scaled_tn = []

        for product_id, group in df.groupby('product_id'):
            if group.empty or group['tn'].isna().all():
                print(f"⚠️ Skipping empty or NaN-only group for product_id {product_id}")
                continue

            try:
                scaler = StandardScaler()
                tn_scaled = scaler.fit_transform(group[['tn']])
                scalers[product_id] = scaler

                group = group.copy()
                group['tn'] = tn_scaled
                scaled_tn.append(group)
            except Exception as e:
                print(f"❌ Error scaling product_id {product_id}: {e}")

        df_scaled = pd.concat(scaled_tn, axis=0) if scaled_tn else pd.DataFrame()

        if scaler_path:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scalers, f)

    else:
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)

        scaled_tn = []

        for product_id, group in df.groupby('product_id'):
            if group.empty or group['tn'].isna().all():
                print(f"⚠️ Skipping empty or NaN-only group for product_id {product_id}")
                continue

            group = group.copy()
            if product_id in scalers:
                try:
                    group['tn'] = scalers[product_id].transform(group[['tn']])
                except Exception as e:
                    print(f"❌ Error transforming product_id {product_id}: {e}")
                    group['tn'] = 0
            else:
                print(f"⚠️ Warning: No scaler found for product {product_id}, assigning 0s")
                group['tn'] = 0

            scaled_tn.append(group)

        df_scaled = pd.concat(scaled_tn, axis=0) if scaled_tn else pd.DataFrame()

    df_scaled = df_scaled.sort_index()
    return df_scaled

def create_delta_lag_features(df, group_col=['customer_id', 'product_id'], target_col='tn', max_lag=35):
    df = df.sort_values(by=group_col + ['periodo_dt'])

    result_columns = []

    for lag in range(1, max_lag + 1):
        delta_lag = df.groupby(group_col, observed=True)[target_col].shift(lag - 1).diff()
        result_columns.append(delta_lag)

    result_df = pd.concat(result_columns, axis=1)
    result_df.columns = [f'{target_col}_delta_lag_{lag}' for lag in range(1, max_lag + 1)]

    return result_df

def create_lag_features(df, group_col=['customer_id', 'product_id'], target_col='tn', max_lag=35):
    df = df.sort_values(by=group_col + ['periodo_dt'])
    result_columns = []

    for lag in range(1, max_lag + 1):
        lagged_values = df.groupby(group_col, observed=True)[target_col].shift(lag)
        result_columns.append(lagged_values)

    result_df = pd.concat(result_columns, axis=1)
    result_df.columns = [f'{target_col}_lag_{lag}' for lag in range(1, max_lag + 1)]
    return result_df

def create_ma_features(df, group_col=['customer_id', 'product_id'], target_col='tn', max_window=36):
    df = df.sort_values(by=group_col + ['periodo_dt'])
    result_columns = []

    for w in range(1, max_window + 1):
        rolling_means = (
            df.groupby(group_col, observed=False)[target_col]
                .shift(0)
                .rolling(window=w, min_periods=w)
                .mean()
        )
        result_columns.append(rolling_means)

    result_df = pd.concat(result_columns, axis=1)

    result_df.columns = [f'{target_col}_ma_{w}' for w in range(1, max_window + 1)]

    return result_df

def create_delta_ma_features(df, group_col=['customer_id', 'product_id'], target_col='tn', max_lag=36):
    df = df.sort_values(by=group_col + ['periodo_dt'])

    result_columns = []

    for lag in range(2, max_lag + 1):
        delta_ma = df[target_col] - df['tn_ma_' + str(lag)]         
        result_columns.append(delta_ma)

    result_df = pd.concat(result_columns, axis=1)
    result_df.columns = [f'delta_{target_col}_ma_lag_{lag}' for lag in range(2, max_lag + 1)]

    return result_df

def create_std_features(df, group_col=['customer_id', 'product_id'], target_col='tn', max_window=36):
    df = df.sort_values(by=group_col + ['periodo_dt'])
    result_columns = []

    for w in range(1, max_window + 1):
        rolling_std = (
            df.groupby(group_col, observed=False)[target_col]
                .shift(0)
                .rolling(window=w, min_periods=w)
                .std()
        )
        result_columns.append(rolling_std)
    result_df = pd.concat(result_columns, axis=1)
    result_df.columns = [f'{target_col}_std_{w}' for w in range(1, max_window + 1)]

    return result_df
        
def create_min_features(df: pd.DataFrame, group_col=['customer_id', 'product_id'], target_col='tn', max_window=36):
    df = df.sort_values(by=group_col + ['periodo_dt'])
    result_columns = []
    
    for w in range(1, max_window + 1):
        rolling_min = df.groupby(group_col, observed=False)[target_col].shift(0).rolling(window=w, min_periods=1).min()
        tn_lagged = df.groupby(group_col, observed=False)[target_col].shift(w-1)
        is_min_col = tn_lagged == rolling_min
        result_columns.append(is_min_col)

    df_result = pd.concat(result_columns, axis=1)
    df_result.columns = [f'{target_col}_is_min_{w}' for w in range(1, max_window + 1)]

    return df_result

def create_max_features(df: pd.DataFrame, group_col=['customer_id', 'product_id'], target_col='tn', max_window=36):
    df = df.sort_values(by=group_col + ['periodo_dt'])
    result_columns = []
    
    for w in range(1, max_window + 1):
        rolling_max = df.groupby(group_col, observed=False)[target_col].shift(0).rolling(window=w, min_periods=1).max()
        tn_lagged = df.groupby(group_col, observed=False)[target_col].shift(w-1)
        is_max_col = tn_lagged == rolling_max
        result_columns.append(is_max_col)

    df_result = pd.concat(result_columns, axis=1)
    df_result.columns = [f'{target_col}_is_max_{w}' for w in range(1, max_window + 1)]

    return df_result
    
def set_categorical_features(df):
    df['product_id'] = df['product_id'].astype('category')
    df['customer_id'] = df['customer_id'].astype('category')
    df['cat1'] = df['cat1'].astype('category')
    df['cat2'] = df['cat2'].astype('category')
    df['cat3'] = df['cat3'].astype('category')
    df['brand'] = df['brand'].astype('category')
    df['sku_size'] = df['sku_size'].astype('category')
    return df

def limit_categorical_values(df, columns, max_values=250, other_prefix='other_'):
    for col in columns:
        top_values = df.groupby(col)['tn'].sum().sort_values(ascending=False).head(max_values).index
        
        other_value = -1
        
        df[f"{col}_limited"] = np.where(df[col].isin(top_values), df[col], other_value)
        
        df[f"{col}_limited"] = df[f"{col}_limited"].astype('category')
    
    return df

def set_ordinal_features(
        df: pd.DataFrame,
        categorical_cols,
        is_train: bool = True,
        encoders_path: str = "./encoders/encoders.pkl",
        suffix: str = "_encoded",
        drop_original: bool = False,
        verbose: bool = True,
    ):
    if is_train:
        encoders = {}
        os.makedirs(os.path.dirname(encoders_path), exist_ok=True)

        for i, col in enumerate(categorical_cols):
            if verbose:
                print(f"[Train] ({i+1}/{len(categorical_cols)}) Encoding '{col}'")
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

            df[f"{col}{suffix}"] = oe.fit_transform(df[[col]]).astype("int32")
            encoders[col] = oe

        with open(encoders_path, "wb") as f:
            pickle.dump(encoders, f)

    else:
        with open(encoders_path, "rb") as f:
            encoders = pickle.load(f)

        for i, col in enumerate(categorical_cols):
            if verbose:
                print(f"[Infer] ({i+1}/{len(categorical_cols)}) Encoding '{col}'")
            oe = encoders[col]
            df[f"{col}{suffix}"] = oe.transform(df[[col]]).astype("int32")

    if drop_original:
        df.drop(columns=categorical_cols, inplace=True)

    return df

def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype

        if col_type in ['int64', 'int32']:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype('uint8')
                elif c_max < 65535:
                    df[col] = df[col].astype('uint16')
                else:
                    df[col] = df[col].astype('uint32')
            else:
                if np.iinfo(np.int8).min < c_min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype('int8')
                elif np.iinfo(np.int16).min < c_min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype('int16')
                else:
                    df[col] = df[col].astype('int32')

        elif col_type in ['float64', 'float32']:
            df[col] = df[col].astype('float32')

        elif col_type == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

    return df

def meses_sin_ventas(serie_sin_ventas: pd.Series) -> pd.Series:
    sin_ventas = []
    count = 0
    for valor in serie_sin_ventas:
        if valor == 1:
            count += 1
        else:
            count = 0
        sin_ventas.append(count)
    return pd.Series(sin_ventas, index=serie_sin_ventas.index)


def get_linear_regression_params(group):
    y = group.values
    X = np.arange(len(y)).reshape(-1, 1)

    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return pd.Series({'slope': 0.0, 'intercept': 0.0})

    reg = LinearRegression()
    reg.fit(X[mask], y[mask])

    return pd.Series({'slope': reg.coef_[0], 'intercept': reg.intercept_, 'r_squared': reg.score(X[mask], y[mask])})

def apply_loess_smoothing(group):
    if len(group) < 2 or group.isna().all():
        return pd.Series({
            'loess_smoothed': 0,
            'detrended_loess': 0,
            'future_trend': 0
        })
    
    time = np.arange(len(group))
    values = group.values
    
    mask = ~np.isnan(values)
    if sum(mask) < 2:
        return pd.Series({
            'loess_smoothed': 0, 
            'detrended_loess': 0,
            'future_trend': 0
        })
        
    time_clean = time[mask]
    values_clean = values[mask]
    
    try:
        smoothed = lowess(values_clean, time_clean, frac=0.05)[:, 1]
        
        detrended = values_clean - smoothed
        
        model = ExponentialSmoothing(smoothed, trend="additive", seasonal=None, 
                                   initialization_method="estimated")
        fitted = model.fit()
        future = fitted.forecast(1)[0]
        
        smoothed_full = np.zeros(len(group)) 
        detrended_full = np.zeros(len(group))
        smoothed_full[mask] = smoothed
        detrended_full[mask] = detrended
        
        return pd.Series({
            'loess_smoothed': smoothed_full[-1],
            'detrended_loess': detrended_full[-1], 
            'future_trend': future
        })
        
    except:
        return pd.Series({
            'loess_smoothed': 0,
            'detrended_loess': 0, 
            'future_trend': 0
        })
