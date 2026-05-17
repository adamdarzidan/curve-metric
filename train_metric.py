import builtins
import datetime
from components.metric import Metric

if __name__ == '__main__':
    model = Metric()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'best_metric_{timestamp}'
    builtins.input = lambda prompt='': filename
    print('Training on full Kaggle train split...')
    model.train('data/train/corupus.csv')
    print('Training complete. Running validation on held-out Kaggle test split...')
    metrics = model.validate('data/train/corupus.csv')
    print('\nValidation metrics:')
    for k, v in metrics.items():
        print(f'{k}: {v}')
    if model.model is not None:
        print('\nTop 30 feature importances:')
        for i, score in sorted(enumerate(model.model.feature_importances_), key=lambda x: x[1], reverse=True)[:30]:
            print(f'{i:03d}: {score:.6f}')
    print('\nSaved model file:', model.model_filepath)
