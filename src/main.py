"""
Main script to run the heart disease prediction project.
"""
from sklearn.model_selection import train_test_split
from src.utils import setup_directories, save_results, RANDOM_STATE, TEST_SIZE
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
    X_train_temp, X_test, y_train_temp, y_test, scaler = preprocess_data(df)
    
    # Create validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_temp
    )
    
    # Create and train models
    print("\nCreating and training models...")
    models = create_models()
    trained_models = train_all_models(models, X_train, y_train, X_val, y_val)
    
    # Evaluate models
    print("\nEvaluating models...")
    metrics = evaluate_all_models(trained_models, X_test, y_test)
    
    # Save metrics
    save_results(metrics)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_roc_curves(trained_models, X_test, y_test)
    plot_confusion_matrices(trained_models, X_test, y_test)
    
    # Plot feature importance for supported models
    for name, model in trained_models.items():
        importance = get_feature_importance(model, X_train.columns)
        if importance is not None:
            plot_feature_importance(importance, name)
    
    print("\nDone! Results have been saved to the 'results' directory.")

if __name__ == "__main__":
    main()