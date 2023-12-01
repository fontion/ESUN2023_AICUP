echo "1. 執行運算"
a="3"
b="None"
c="None"
d="None"
e="None"
f="None"
gpu_id="0"
python GridSearch_twcc.py xgboost $a $b $c $d $e $f $gpu_id
python GridSearch_twcc2.py xgboost

# echo "2. 刪除開發型容器"
machine_name=$(hostname)
ccs_id=$(eval "twccli ls ccs -json | jq '.[]| select(.name == \"${machine_name:6:-6}\").id'")
TWCC_CLI_CMD="/home/$(whoami)/.local/bin/twccli"
$TWCC_CLI_CMD rm ccs -f -s $ccs_id