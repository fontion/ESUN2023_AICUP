echo "1. 執行運算"
db_name="db-v4-AfterFraud"
mdlname="catboost"
# param1="1_3_0.66_31_0.8_6"
# param1="1_3.5_0.66_31_0.8_8"
param1="1_3_0.8_31_0.8_8"
eval_metric="F1"
python train_main.py $db_name $mdlname $param1 --combineTrainTest --no-early-stopping \
--eval_metric $eval_metric --custom_metric $eval_metric --n_estimators=482
# python -m Model.GridSearch $db_name $mdlname --save_model_thres=2 \
# --eval_metric $eval_metric --custom_metric $eval_metric --early_stopping_rounds=350

echo "2. 刪除開發型容器"
machine_name=$(hostname)
ccs_id=$(eval "twccli ls ccs -json | jq '.[]| select(.name == \"${machine_name:6:-6}\").id'")
TWCC_CLI_CMD="/home/$(whoami)/.local/bin/twccli"
$TWCC_CLI_CMD rm ccs -f -s $ccs_id