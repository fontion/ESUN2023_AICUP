echo "1. 執行運算"
db_name="db-v4-UsedBeforeFraud"
mdlname="catboost"
eval_metric="F1"
python -m Model.GridSearch $db_name $mdlname 2 1 \
--eval_metric $eval_metric --custom_metric $eval_metric --save_model_thres=2

echo "2. 刪除開發型容器"
machine_name=$(hostname)
ccs_id=$(eval "twccli ls ccs -json | jq '.[]| select(.name == \"${machine_name:6:-6}\").id'")
TWCC_CLI_CMD="/home/$(whoami)/.local/bin/twccli"
$TWCC_CLI_CMD rm ccs -f -s $ccs_id