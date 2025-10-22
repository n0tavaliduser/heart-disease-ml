"""
Main script to run the heart disease prediction project.
"""
from src.utils import setup_directories, save_results
from src.data_preprocessing import load_data, preprocess_data, get_feature_importance
from src.model_training import create_models, train_all_models
from src.evaluating import (
    evaluate_all_models, plot_roc_curves,
    plot_confusion_matrices, plot_feature_importance
)

def main():
    # Setup directories
    setup_directories()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Create and train models
    print("Creating and training models...")
    models = create_models()
    trained_models = train_all_models(models, X_train, y_train)
    
    # Evaluate models
    print("Evaluating models...")
    metrics = evaluate_all_models(trained_models, X_test, y_test)
    
    # Save metrics
    save_results(metrics)
    
    # Generate plots
    print("Generating plots...")
    plot_roc_curves(trained_models, X_test, y_test)
    plot_confusion_matrices(trained_models, X_test, y_test)
    
    # Plot feature importance for supported models
    for name, model in trained_models.items():
        importance = get_feature_importance(model, X_train.columns)
        if importance is not None:
            plot_feature_importance(importance, name)
    
    print("Done! Results have been saved to the 'results' directory.")

if __name__ == "__main__":
    main()