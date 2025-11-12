"""
工具模組

包含資料載入、視覺化和模型管理功能。
"""

from .data_loader import load_csv_file, validate_dataframe, get_data_info
from .data_preprocessor import detect_categorical_features, preprocess_features
from .data_augmenter import augment_dataframe, get_augmentation_stats, add_gaussian_noise, add_uniform_noise
from .evaluator import evaluate_model, repeated_random_split_evaluate, format_metrics_with_std
from .visualizer import (
    create_histogram,
    create_histograms_for_numeric_columns,
    create_scatter_matrix,
    create_statistics_summary,
    plot_training_loss,
    display_model_parameters,
    display_evaluation_metrics
)
from .model_manager import (
    save_model,
    load_model,
    list_models,
    delete_model,
    get_model_info
)

__all__ = [
    'load_csv_file',
    'validate_dataframe',
    'get_data_info',
    'detect_categorical_features',
    'preprocess_features',
    'augment_dataframe',
    'get_augmentation_stats',
    'add_gaussian_noise',
    'add_uniform_noise',
    'evaluate_model',
    'repeated_random_split_evaluate',
    'format_metrics_with_std',
    'create_histogram',
    'create_histograms_for_numeric_columns',
    'create_scatter_matrix',
    'create_statistics_summary',
    'plot_training_loss',
    'display_model_parameters',
    'display_evaluation_metrics',
    'save_model',
    'load_model',
    'list_models',
    'delete_model',
    'get_model_info'
]

