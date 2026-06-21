from clearlane.features import model_feature_columns


def test_feature_list_contains_weekly_lag() -> None:
    config = {
        "temporal": {
            "lag_windows": [1, 2, 84],
            "rolling_windows": [12, 84, 336],
        }
    }
    features = model_feature_columns(config)
    assert "count_lag_84" in features
    assert "impact_roll_mean_336" in features
    assert "trend_ratio" in features
