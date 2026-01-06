import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class InsightModule:
    """
    Advanced Analytics Module for Nexus AI.
    NOW WITH AUTO-PRINTING to ensure explanations appear with charts.
    """

    def __init__(self):
        pass

    def check_anomalies(self, df, column_name, contamination=0.05):
        # Prep Data
        data = df[[column_name]].dropna()

        # Train Model
        model = IsolationForest(contamination=contamination, random_state=42)
        data['anomaly'] = model.fit_predict(data[[column_name]])
        anomalies = data[data['anomaly'] == -1]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[column_name], color='blue', label='Normal', alpha=0.6)
        plt.scatter(anomalies.index, anomalies[column_name], color='red', label='Anomaly', s=50)
        plt.title(f"Anomaly Detection: {column_name}")
        plt.legend()

        # PRINT THE INSIGHT (Don't just return it)
        print(f"### 🔍 Anomaly Report: {column_name}")
        print(f"- **Total Anomalies Found:** {len(anomalies)}")
        if len(anomalies) > 0:
            avg_anom = anomalies[column_name].mean()
            print(
                f"- **Context:** The anomalies have an average value of {avg_anom:.2f}, which is significantly different from the norm.")
            print("- **Visual:** Look for the RED dots in the chart above.")
        else:
            print("- No significant anomalies detected.")

    def forecast_series(self, df, date_col, value_col, periods=30):
        # Prep Data
        temp_df = df.copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col])
        temp_df = temp_df.sort_values(by=date_col).set_index(date_col)
        series = temp_df[value_col].dropna()

        if len(series) < 10:
            print("❌ Not enough data points to forecast (Need at least 10).")
            return

        # Train Model
        try:
            model = ExponentialSmoothing(series, seasonal_periods=None, trend='add', seasonal=None).fit()
            forecast = model.forecast(periods)

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(series.index, series, label='Historical')
            plt.plot(forecast.index, forecast, label='Forecast', color='green', linestyle='--')
            plt.title(f"Forecast: {value_col} ({periods} steps)")
            plt.legend()

            # PRINT THE INSIGHT
            print(f"### 📈 Forecast Report: {value_col}")
            print(f"- **Prediction Window:** Next {periods} periods.")
            print(f"- **Trend:** The model predicts the value will end around {forecast.iloc[-1]:.2f}.")
            print("- **Visual:** The GREEN dotted line in the chart shows the future trend.")

        except Exception as e:
            print(f"❌ Forecasting Error: {str(e)}")

    def get_correlation_drivers(self, df, target_col):
        numeric_df = df.select_dtypes(include=['number'])
        if target_col not in numeric_df.columns:
            print(f"❌ Target column '{target_col}' must be numeric.")
            return

        corr = numeric_df.corr()[target_col].sort_values(ascending=False).drop(target_col)
        top_driver = corr.index[0]
        top_score = corr.iloc[0]

        # Plot
        plt.figure(figsize=(8, 5))
        sns.barplot(x=corr.values, y=corr.index, palette="coolwarm")
        plt.title(f"Correlation Drivers for '{target_col}'")
        plt.axvline(0, color='black', linewidth=1)

        # PRINT THE INSIGHT
        print(f"### 🎯 Key Drivers for '{target_col}'")
        print(f"- **Top Driver:** {top_driver} (Correlation: {top_score:.2f})")
        print("- **Interpretation:**")
        if top_score > 0.5:
            print(f"  - Strong positive relationship. As '{top_driver}' goes up, '{target_col}' likely goes up.")
        elif top_score < -0.5:
            print(f"  - Strong negative relationship. As '{top_driver}' goes up, '{target_col}' goes down.")
        else:
            print("  - Relationships are moderate. Other factors may be involved.")