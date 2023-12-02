# 檔案用途
## 根目錄/
 - codes/: 本來置放程式的位置，配合大會公告的資料夾結構，已將多數程式碼按要求改放置於Preprocess與Model資料夾，並為此修改程式碼呼叫方式
 - Preprocess/: 存放前處理的code
 - Model/: 存放模型相關code
 - requirements.txt: 需要的套件
 - features.txt: 關於產生新features的說明
 - main.sh: 完整流程shell script
 - train_main.py: 模型訓練主程式
 - test_main.py: 模型預測主程式
 - utils.py: 分析時共用的函式庫
## /codes/
 - shap_importance.py: 計算feature importance
## /Model/
 - GridSearch.py: 供搜尋模型訓練時的最佳參數
 - opt_thres_final.py: 計算模型判斷positive或negative最佳的threshold
## /Preprocess/
 - create_dataset.py: 回顧過去的刷卡記錄，產生訓練模型所需的所有features
 - create_dataset_parts.py: 將create_dataset.py工作拆分，以方便使用shell多線程處理，縮短處理時間
 - create_dataset.sh: 配合create_dataset_parts.py使用的shell script
 - create_dataset_split.py: 將產生的db-v4.joblib檔再進行分割
 - create_testset.py: 產生正式預測時public或private set時所需的所有features
 - preprocess.py: 初步的資料前處理，妥善轉換資料格式，如category, boolean, integer, ...


# 資料結構
 - 資料夾"dataset_1st"為初賽第一階段釋出的資料，裡面應包含"training.csv"與"public_processed.csv"兩個檔案
 - 資料夾"dataset_2nd"為初賽第二階段釋出的資料，裡面應包含"public.csv"與"private_1_processed.csv"兩個檔案
 - 根目錄下需放置檔案"31_範例繳交檔案.csv"
```
.
├ dataset_1st
│ ├ public_processed.csv
│ └ training.csv
│
├ dataset_2nd
│ ├ private_1_processed.csv
│ └ public.csv
│
└ 31_範例繳交檔案.csv
```

# 執行流程
## 建立環境
```
pip install -r requirements.txt
```

## 資料前處理
```
cd Preprocess
python3 preprocess.py dataset_1st
python3 preprocess.py dataset_2nd
```

## 建立訓練模型的database並做feature engineering
```
python3 create_dataset.py
```
> 若處理時間較久，可使用 create_dataset_parts.py 搭配 create_dataset.sh 做平行處理，建議CPU邏輯核心數量24以上，系統記憶體大於70GB
```
source create_dataset.sh
```
> 最後會在 dataset_2nd 資料夾產生 db-v4.joblib，內含一個dictionary包含train與pred兩個欄位，分別儲存訓練與測試所需的pandas.DataFrame
> 經過此處理，共計有212個features，前面日期的資料做為模型訓練用(存在train欄位)，倒數4或5日的資料做為validation(存在pred欄位)，調參使用
相關features說明請見 features.txt

## 將訓練database分成無過去刷卡記錄(FirstUse)與有過去刷卡記錄(UsedBefore)兩種類型分別訓練模型
```
python3 create_dataset_split.py db-v4 FirstUse UsedBefore
```

## 使用Catboost模型搭配Grid Search找出表現最佳的模型與參數 (下一步已列出參數，可跳過)
(以搜尋FirstUse為例)
```
cd ../
db_name="db-v4-FirstUse"
mdlname="catboost"
eval_metric="F1"
python3 -m Model.GridSearch $db_name $mdlname --eval_metric $eval_metric --custom_metric $eval_metric --early_stopping_rounds=350 --min_data_in_leaf=1
```
> GridSearch.py可設定的參數請見Model/GridSearch.py與utils.py中的parse_catboost函式
> 搜尋結果會儲存在train_test/catboost_search/資料夾，evaluation_catboost.csv與catboost_search.csv兩個檔案

## 取出最佳的參數做最後的訓練 (合併training與validation set)
> 若跳過上一步，先回到根目錄
```
cd ../
```
> 引數param依序為min_data_in_leaf, reg_lambda, subsample, max_leaves, learning_rate，max_depth, 以下底線_分隔
> 使用n_estimators引數指定最佳iteration次數
* 無過去刷卡記錄(FirstUse)，訓練3個模型做embedding
```
db_name="db-v4-FirstUse"
mdlname="catboost"
param1="1_2_0.7_31_0.02_7"
param2="1_2_0.75_31_0.04_7"
param3="1_2_0.75_31_0.8_7"
eval_metric="F1"
python3 train_main.py $db_name $mdlname $param1 $param2 $param3 \
--eval_metric $eval_metric --custom_metric $eval_metric --no-calc_shap \
--no-early-stopping --combineTrainTest --n_estimators 722 381 54
```
* 有過去刷卡記錄(UsedBefore)，訓練1個模型
```
db_name="db-v4-UsedBefore"
mdlname="catboost"
param1="14_3_0.66_31_0.03_8"
eval_metric="F1"
python3 train_main.py $db_name $mdlname $param1 --combineTrainTest --no-early-stopping \
--no-calc_shap --eval_metric $eval_metric --custom_metric $eval_metric --n_estimators=893
```
> 最後的模型儲存在train_test/catboost_search資料夾.cbm檔

## 計算最佳threshold (供FirstUse embedding模型使用)
```
db_name="db-v4-FirstUse"
mdlname="catboost"
param1="1_2_0.7_31_0.02_7"
param2="1_2_0.75_31_0.04_7"
param3="1_2_0.75_31_0.8_7"
python3 -m Model.opt_thres_final $db_name $mdlname $param1 $param2 $param3 --subfolder=""
```
> 合併三個模型最佳的threshold輸出在train_test/catboost_search/embedding_3_models.json
> UsedBefore因為模型太大，僅使用單一模型，最佳threshold會在上一步執行train_main.py時輸出在train_test/catboost_search/db-v4-UsedBefore-14_3_0.66_31_0.03_8(allin)_opt_thres.json

## 計算預測時所需的額外features
```
cd Preprocess/
python3 create_testset.py
```
> 這一步會回溯目前已知的training與public資料，新增features供模型預測private set，最後預測所需的所有features輸出在 dataset_2nd/private_add_features.joblib

## 模型正式預測
> 因為最佳threshold不見得有最好的預測表現，在UsedBefore的情況，先手動將db-v4-UsedBefore-14_3_0.66_31_0.03_8(allin)_opt_thres.json檔中的opt_thres改回0.5，再執行以下程式碼
```
cd ../
folder="dataset_2nd"
json_path1="train_test/catboost_search/embedding_3_models.json"
json_path2="train_test/catboost_search/db-v4-UsedBefore-14_3_0.66_31_0.03_8(allin)_opt_thres.json"
python3 test_main.py --folder=$folder --model_FirstUse=$json_path1 --model_UsedBefore=$json_path2
```
> 最後的預測結果會輸出放在submit資料夾，檔名 31_{日期}upload.csv


## 複賽
```
db_name="db-v5-FirstUse"
mdlname="catboost"
param1="1_2_0.7_31_0.02_7"
param2="1_2_0.75_31_0.04_7"
param3="1_2_0.75_31_0.8_7"
python3 -m Model.opt_thres_final $db_name $mdlname $param1 $param2 $param3
```


```
json_path1="pretrain_models/embedding_3_models.json"
json_path2="pretrain_models/UsedBeforeFraud/"
```