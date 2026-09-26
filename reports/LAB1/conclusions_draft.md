# Выводы ЛР1 (сгенерированы src/eda.py, числа из eda_stats.csv)

1. Объём датасета - 20000 строк и 21 столбцов (eda_stats.csv: n_rows, n_cols; src/eda.py).
2. Все 20 признаков имеют тип float64, целевое поле target - int64 (eda_stats.csv: dtype_features).
3. Пропусков в данных 0 (eda_stats.csv: n_missing).
4. Полностью дублирующихся строк 0 (eda_stats.csv: n_full_duplicates).
5. Доля строк с аномальными значениями (|z| > 3.0) равна 0.04385 (eda_stats.csv: outlier_row_share).
6. Доля класса 1 равна 0.4995, дисбаланса классов нет (eda_stats.csv: class_1_share; class_share.png).
7. Шумовых признаков (|AUC-0.5| <= 0.02) - 3 из 20 (eda_stats.csv: n_noise_features; problem.png).
8. Минимальный однопризнаковый ROC-AUC равен 0.506664 у признака feature_17, что соответствует случайному угадыванию (eda_stats.csv: min_feature_auc; problem.png).
9. Максимальный однопризнаковый ROC-AUC равен 0.681384 у признака feature_07 (eda_stats.csv: max_feature_auc).
10. Пар признаков с |r| >= 0.99 - 7, максимальная корреляция 1.0 между feature_00 и feature_06 (eda_stats.csv: n_dup_pairs, max_abs_corr; corr_heatmap.png).
11. Заявленная проблема варианта 10 в данных видна: 3 признаков не связаны с целью и 7 пар дублируют друг друга, поэтому в ЛР2 применяется L1-регуляризация, обнуляющая такие признаки (problem.png).
