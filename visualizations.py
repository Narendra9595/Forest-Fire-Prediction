import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd

def create_visualizations(df, model, X_test, shap_values):
    # 1. Fire Intensity Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='fire_intensity')
    plt.title('Distribution of Fire Intensity')
    plt.savefig('outputs/fire_intensity_dist.png')
    plt.close()
    
    # 2. Correlation Heatmap - using only numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()
    
    # 3. Feature Importance Plot
    importances = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importances, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.savefig('outputs/feature_importance.png')
    plt.close()
    
    # 4. SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP Summary Plot')
    plt.savefig('outputs/shap_summary.png')
    plt.close()
    
    # 5. Temperature vs Humidity scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='temperature', y='humidity', hue='fire_intensity')
    plt.title('Temperature vs Humidity by Fire Intensity')
    plt.savefig('outputs/temp_humidity_scatter.png')
    plt.close()