db_name="db-v3-FirstUse"
mdlname="catboost"
param1="1_3_0.66_31_0.04_8"
param2="1_3_0.75_31_0.03_7"
param3="1_2.5_0.8_31_0.04_6"
eval_metric="F1"
python train_main.py $db_name $mdlname $param1 $param2 $param3 \
--eval_metric $eval_metric --custom_metric $eval_metric \
--no-calc_shap --combineTrainTest --no-early-stopping --n_estimators 138 234 309