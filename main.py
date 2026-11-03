import joblib
import pandas as pd

from src import config
from src.config import load_config, ensure_dirs
from src.data_generator import generate_flood_data
from src.data_loader import save_csv, load_csv
from src.preprocessing import clean_data, split_data, build_preprocessor
from src.features import add_features
from src.models import get_models
from src.benchmark import cross_validate_models, evaluate_on_test, confusion
from src.visualize import (
    plot_class_distribution, plot_correlation_heatmap,
    plot_roc_curves, plot_confusion_matrix,
    plot_feature_importance, plot_model_comparison,
)


def main() -> None:
    config = load_config()
    ensure_dirs(config)

    # 1. Load or generate data
    raw_path = config["data"]["raw_path"]
    try:
        df = load_csv(raw_path)
        print(f"Loaded existing data: {raw_path}")
    except FileNotFoundError:
        df = generate_flood_data(config)   # tự chọn nguồn theo config
        save_csv(df, raw_path)
        print(f"Generated new data from source='{config['data']['source']}': {raw_path}")

    # 2. Clean + feature engineering
    df = clean_data(df)
    df = add_features(df)
    save_csv(df, config["data"]["processed_path"])

    target = config["features"]["target"]
    plot_class_distribution(df[target], f"{config['output']['figures_dir']}/class_distribution.png")
    plot_correlation_heatmap(df, f"{config['output']['figures_dir']}/correlation_heatmap.png")

    # 3. Split
    X_train, X_test, y_train, y_test = split_data(
        df, target,
        test_size=config["features"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # 4. Preprocess (fit trên train, transform cả train/test)
    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # 5. Benchmark
    models = get_models(config["models"]["random_state"])
    cv_results = cross_validate_models(
        models, X_train_scaled, y_train, cv=config["features"]["cv_folds"]
    )
    print("\n=== Cross-Validation Results ===")
    print(cv_results.to_string(index=False))

    test_results, fitted_models = evaluate_on_test(
        models, X_train_scaled, y_train, X_test_scaled, y_test
    )
    print("\n=== Test Set Results ===")
    print(test_results.to_string(index=False))

    cv_results.to_csv(config["output"]["metrics_path"], index=False)

    # 6. Visualize benchmark
    plot_model_comparison(test_results, f"{config['output']['figures_dir']}/model_comparison.png")
    plot_roc_curves(fitted_models, X_test_scaled, y_test, f"{config['output']['figures_dir']}/roc_curves.png")

    best_model_name = test_results.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    plot_confusion_matrix(best_model, X_test_scaled, y_test, f"{config['output']['figures_dir']}/confusion_matrix.png")
    plot_feature_importance(best_model, X_train.columns, f"{config['output']['figures_dir']}/feature_importance.png")

    # 7. Save best model
    joblib.dump(best_model, config["output"]["model_path"])
    print(f"Saved best model: {config['output']['model_path']}")


if __name__ == "__main__":
    main()