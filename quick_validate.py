from components.metric import Metric
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __name__ == '__main__':
    m = Metric()
    m.load_model('data/saved-models/best_metric_20260413_190916.json')
    rows = m._Metric__format_csv_data('data/train/corupus.csv', False)
    print('Rows to validate:', len(rows))
    expected = []
    preds = []
    for row in rows:
        y = float(row['bt_easiness'])
        preds.append(m.score(row['excerpt'], row))
        expected.append(y)
    mae = mean_absolute_error(expected, preds)
    rmse = mean_squared_error(expected, preds, squared=False)
    r2 = r2_score(expected, preds)
    print('mae:', mae)
    print('rmse:', rmse)
    print('r2:', r2)
    print('top features:')
    for idx, score in sorted(enumerate(m.model.feature_importances_), key=lambda x: -x[1])[:30]:
        print(f'{idx:03d}: {score:.6f}')
