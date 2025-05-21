import importlib

cal_std_mean_module = importlib.import_module(".cal_std_mean", __name__)
cal_std_mean = cal_std_mean_module.cal_std_mean
cal_std_mean_simd = cal_std_mean_module.cal_std_mean_simd

cal_cpr = importlib.import_module(".cal_cpr", __name__).cal_cpr
cal_all_largest_indicators = importlib.import_module(".cal_all_largest_indicators", __name__).cal_all_largest_indicators
cal_all_longest_indicators = importlib.import_module(".cal_all_longest_indicators", __name__).cal_all_longest_indicators
cal_longest_dd_recover = importlib.import_module(".cal_longest_dd_recover", __name__).cal_longest_dd_recover
cal_max_dd = importlib.import_module(".cal_max_dd", __name__).cal_max_dd
cal_rolling_gain_loss = importlib.import_module(".cal_rolling_gain_loss", __name__).cal_rolling_gain_loss
