db_name="db-v3-AfterFraud"
mdlname="catboost"
# param1="14_0.02_0.66_31_0.8_2"
# param1="14_3_0.66_31_0.03_8"
# param2="1_3_0.66_31_0.03_8"
# param3="7_3_0.66_31_0.03_8"
eval_metric="F1"
python GridSearch.py $db_name $mdlname \
--eval_metric $eval_metric --custom_metric $eval_metric \
--early_stopping_rounds=350 --save_model_thres=1.803
# python train_main.py $db_name $mdlname $param1 --no-save_model --no-calc_shap \
# --eval_metric $eval_metric --custom_metric $eval_metric
