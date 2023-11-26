# 請先按readme說明將檔案放置於dataset_1st, dataset_2nd與根目錄，再執行本程式

# 建立環境
pip install -r requirements.txt

# 資料前處理
cd Preprocess
python3 preprocess.py dataset_1st
python3 preprocess.py dataset_2nd

# 建立訓練模型的database並做feature engineering (建議CPU邏輯核心數量24以上，系統記憶體大於70GB)
source create_dataset.sh

# 將訓練database分成無過去刷卡記錄(FirstUse)與有過去刷卡記錄(UsedBefore)兩種類型分別訓練模型
python3 create_dataset_split.py db-v4 FirstUse UsedBefore

# 取出最佳的參數做最後的訓練 (合併training與validation set)
cd ../
#  > 無過去刷卡記錄(FirstUse)，訓練3個模型做embedding
db_name="db-v4-FirstUse"
mdlname="catboost"
param1="1_2_0.7_31_0.02_7"
param2="1_2_0.75_31_0.04_7"
param3="1_2_0.75_31_0.8_7"
eval_metric="F1"
python3 train_main.py $db_name $mdlname $param1 $param2 $param3 \
--eval_metric $eval_metric --custom_metric $eval_metric --no-calc_shap \
--no-early-stopping --combineTrainTest --n_estimators 722 381 54

#  > 有過去刷卡記錄(UsedBefore)，訓練1個模型
db_name="db-v4-UsedBefore"
mdlname="catboost"
param1="14_3_0.66_31_0.03_8"
eval_metric="F1"
python3 train_main.py $db_name $mdlname $param1 --combineTrainTest --no-early-stopping \
--no-calc_shap --eval_metric $eval_metric --custom_metric $eval_metric --n_estimators=893

# 計算最佳threshold (供FirstUse embedding模型使用)
db_name="db-v4-FirstUse"
mdlname="catboost"
param1="1_2_0.7_31_0.02_7"
param2="1_2_0.75_31_0.04_7"
param3="1_2_0.75_31_0.8_7"
python3 -m Model.opt_thres_final $db_name $mdlname $param1 $param2 $param3 --subfolder=""

# 計算預測時所需的額外features
cd Preprocess/
python3 create_testset.py

# 模型正式預測 (先手動將db-v4-UsedBefore-14_3_0.66_31_0.03_8(allin)_opt_thres.json檔中的opt_thres改回0.5)
cd ../
folder="dataset_2nd"
json_path1="train_test/catboost_search/embedding_3_models.json"
json_path2="train_test/catboost_search/db-v4-UsedBefore-14_3_0.66_31_0.03_8(allin)_opt_thres.json"
python3 test_main.py --folder=$folder --model_FirstUse=$json_path1 --model_UsedBefore=$json_path2

# 最後的預測結果會輸出放在submit資料夾，檔名 31_{日期}upload.csv