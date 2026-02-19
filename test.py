from src.data_utils import load_parkinsons, get_X_y_groups, FEATURE_COLS
from src.eval_utils import compute_metrics, plot_confusion_matrix
from src.pipeline_utils import build_pipeline, run_grouped_cv, get_classifiers

df = load_parkinsons()
X, y, groups = get_X_y_groups(df)
print(f'Loaded: {X.shape[0]} samples, {X.shape[1]} features, {groups.nunique()} subjects')
print(f'Features: {len(FEATURE_COLS)}')
print(f'Classifiers available: {list(get_classifiers().keys())}')
print('All imports OK!')