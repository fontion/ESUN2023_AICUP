db_name="db-v3-UsedBeforeFraud"
mdlname="catboost"
param1="14_3_0.66_31_0.03_8"
# param1="14_3_0.66_31_0.03_8"
# param2="1_3_0.66_31_0.03_8"
# param3="7_3_0.66_31_0.03_8"
eval_metric="F1"
# python GridSearch.py $db_name $mdlname \
# --eval_metric $eval_metric --custom_metric $eval_metric \
# --early_stopping_rounds=350 --min_data_in_leaf=1
python train_main.py $db_name $mdlname $param1 --combineTrainTest --no-early-stopping \
--no-calc_shap --eval_metric $eval_metric --custom_metric $eval_metric --n_estimators=893