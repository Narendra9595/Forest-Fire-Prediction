from data_preprocessing import preprocess_forestfire_data
from model_training import train_explainable_model
from visualizations import create_visualizations
import pickle

def main():
    # Paths
    forest_path = 'data/forestfires.csv'
    ambee_path = 'data/ambee_data.csv'
    
    # Preprocess data
    X, y, df = preprocess_forestfire_data(forest_path, ambee_path)
    
    # Train model and get explanations
    model, X_test, y_test, shap_values, lime_explainer,featuer_names = train_explainable_model(X, y)
    
    # Create visualizations
    create_visualizations(df, model, X_test, shap_values)
    
    # Save model
    with open('models/forest_fire_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Example prediction with explanation
    example_instance = X_test.iloc[0]
    
    # Get LIME explanation
    exp = lime_explainer.explain_instance(
        example_instance.values, 
        model.predict_proba,
        num_features=6
    )
    
    # Print prediction and explanation
    pred_prob = model.predict_proba([example_instance])[0]
    print("\nPrediction Probabilities:")
    print(f"No Fire: {pred_prob[0]:.2f}")
    print(f"Fire: {pred_prob[1]:.2f}")
    
    print("\nFeature Contributions:")
    for feature, importance in exp.as_list():
        print(f"{feature}: {importance:.3f}")

if __name__ == "__main__":
    main()
