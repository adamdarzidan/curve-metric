from components.metric import Metric

if __name__ == '__main__':
    model = Metric()
    model.load_model('data/saved-models/best_metric_20260413_190916.json')
    print('Loaded model:', model.model_filepath)
    metrics = model.validate('data/train/corupus.csv')
    print('\nValidation metrics:')
    for key, value in metrics.items():
        print(f'{key}: {value}')
    if model.model is not None:
        print('\nTop 30 feature importances:')
        for idx, score in sorted(enumerate(model.model.feature_importances_), key=lambda x: x[1], reverse=True)[:30]:
            print(f'{idx:03d}: {score:.6f}')
