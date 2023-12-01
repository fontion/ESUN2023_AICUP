echo "1. 執行運算"
db_name="db-v4-UsedBeforeFraud"
mdlname="catboost"
param1="1_3_0.66_31_0.8_6"
eval_metric="F1:use_weights=False"
python train_main.py $db_name $mdlname $param1 --scale_pos_weight=10 \
--eval_metric $eval_metric --custom_metric $eval_metric

echo "2. 刪除開發型容器"
machine_name=$(hostname)
ccs_id=$(eval "twccli ls ccs -json | jq '.[]| select(.name == \"${machine_name:6:-6}\").id'")
TWCC_CLI_CMD="/home/$(whoami)/.local/bin/twccli"
$TWCC_CLI_CMD rm ccs -f -s $ccs_id