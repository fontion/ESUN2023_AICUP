echo "1. 執行運算"
python train_main.py db-v2-UsedBefore catboost 3_0.66_31_0.03_8 3_0.66_31_0.02_8 2_0.7_31_0.03_8 --n_estimators=550

echo "2. 刪除開發型容器"
machine_name=$(hostname)
ccs_id=$(eval "twccli ls ccs -json | jq '.[]| select(.name == \"${machine_name:6:-6}\").id'")
TWCC_CLI_CMD="/home/$(whoami)/.local/bin/twccli"
$TWCC_CLI_CMD rm ccs -f -s $ccs_id