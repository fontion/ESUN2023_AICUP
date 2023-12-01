db_name="db-v4-FirstUse"
mdlname="catboost"
# param1="12_3_0.8_31_0.8_6"
# param2="10_3_0.8_31_0.8_6"
# param3="7_3_0.8_31_0.8_6"
# param4="4_3_0.8_31_0.8_6"
# param5="1_3_0.8_31_0.8_6"
eval_metric="F1"
# python train_main.py $db_name $mdlname $param1 $param2 $param3 $param4 $param5 \
# --eval_metric $eval_metric --custom_metric $eval_metric \
# --early_stopping_rounds=350 --no-save_model --no-calc_shap

python GridSearch.py $db_name $mdlname \
--eval_metric $eval_metric --custom_metric $eval_metric \
--task_type=CPU --bootstrap_type=MVS --early_stopping_rounds=350