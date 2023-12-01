echo "1. 執行運算"
a="None"
b="1"
c="None"
d="None"
e="None"
python GridSearch.py catboost $a $b $c $d $e

echo "2. 刪除開發型容器"
machine_name=$(hostname)
ccs_id=$(eval "twccli ls ccs -json | jq '.[]| select(.name == \"${machine_name:6:-6}\").id'")
TWCC_CLI_CMD="/home/$(whoami)/.local/bin/twccli"
$TWCC_CLI_CMD rm ccs -f -s $ccs_id