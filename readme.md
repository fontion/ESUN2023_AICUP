# 建立環境
1. 請先安裝miniconda
2. 使用conda建立名為"ESUN"的environment
```
cd build_environment
source create_env.sh
```

# 資料前處理
資料夾"dataset_1st"為初賽第一階段釋出的資料，裡面應包含"training.csv"與"public_processed.csv"兩個檔案
資料夾"dataset_2nd"為初賽第二階段釋出的資料，裡面應包含"public.csv"與"private_1_processed.csv"兩個檔案
根目錄下需放置檔案"31_範例繳交檔案.csv"
```
cd ../codes
python preprocess.py dataset_1st
python preprocess.py dataset_2nd
```

# 建立訓練模型的database並做feature engineering
```
python create_dataset.py split_from_training
python create_dataset.py combine_training_and_public
```
> 實際上因為處理時間較久，會使用 create_dataset_parts.py 搭配 scripts/create_dataset.sh 做平行處理
最後會在 dataset_1st 與 dataset_2nd 資料夾分別產生 db.joblib，根據修改版本，最後命名為db-v4.joblib
經過此處理，共計有212個features，前面日期的資料做為模型訓練用(存在train欄位)，倒數4或5日的資料做為validation，調參使用(存在pred欄位)
相關features說明請見 features.txt

# 將訓練database分成無過去刷卡記錄(FirstUse)與有過去刷卡記錄(UsedBefore)兩種類型分別訓練模型
 > 需手動修改path_db的路徑
```
python create_dataset_FirstUse.py
python create_dataset_UsedBefore.py
```

# 使用Catboost模型搭配Grid Search找出表現最佳的模型與參數
(以搜尋FirstUse為例)
```
db_name="db-v4-FirstUse"
mdlname="catboost"
eval_metric="F1"
python GridSearch.py $db_name $mdlname --eval_metric $eval_metric --custom_metric $eval_metric --early_stopping_rounds=350 --min_data_in_leaf=1
```
> GridSearch.py可設定的參數請見GridSearch.py與utils.py中的parse_catboost函式
> 搜尋結果會儲存在train_test/catboost_search/資料夾，evaluation_catboost.csv與catboost_search.csv兩個檔案

# 取出最佳的參數做最後的訓練 (合併training與validation set)
> 參數依序為min_data_in_leaf, reg_lambda, subsample, max_leaves, learning_rate，max_depth, 以下底線_分隔
* 無過去刷卡記錄(FirstUse)
```
db_name="db-v4-FistUse"
mdlname="catboost"
param1="1_2_0.7_31_0.02_7"
param2="1_2_0.75_31_0.04_7"
param3="1_2_0.75_31_0.8_7"
eval_metric="F1"
python train_main.py $db_name $mdlname $param1 $param2 $param3 \
--eval_metric $eval_metric --custom_metric $eval_metric \
--no-early-stopping --combineTrainTest --n_estimators 722 381 54
```
* 有過去刷卡記錄(UsedBefore)
```
db_name="db-v4-UsedBefore"
mdlname="catboost"
param1="14_3_0.66_31_0.03_8"
eval_metric="F1"
python train_main.py $db_name $mdlname $param1 --combineTrainTest --no-early-stopping \
--eval_metric $eval_metric --custom_metric $eval_metric --n_estimators=893
```
> 最後的模型儲存在train_test/catboost_search

# 計算最佳threshold
```
db_name="db-FirstUse"
mdlname="catboost"
param1="1_2_0.7_31_0.02_7"
param2="1_2_0.75_31_0.04_7"
param3="1_2_0.75_31_0.8_7"
python opt_thres_final.py $db_name $mdlname $param1 $param2 $param3 --subfolder=""
```
> FirstUse使用三個模型做embedding，預期可以提升預測準確性
> 合併三個模型最佳的threshold輸出在train_test/catboost_search/embedding_3_models.json
> UsedBefore因為模型太大，僅使用單一模型，最佳threshold會在上一步執行train_main.py時輸出在train_test/catboost_search/db-v4-UsedBefore-14_3_0.66_31_0.03_8(allin)_opt_thres.json

# 計算預測時所需的額外features
```
python create_testset.py dataset_2nd
```
> 這一步會回溯目前已知的training與public資料，新增features供模型預測

# 模型正式預測
> 因為最佳threshold不見得有最好的預測表現，在UsedBefore的情況，手動將json檔中的opt_thres改回0.5
```
folder="dataset_2nd"
json_path1="../train_test/catboost_search/embedding_3_models.json"
json_path2="../train_test/catboost_search/db-v4-UsedBefore-14_3_0.66_31_0.03_8(allin)_opt_thres.json"
python test_main.py --folder=$folder --model_FirstUse=$json_path1 --model_UsedBefore=$json_path2
```
> 最後的預測結果會輸出放在submit資料夾，檔名 31_{日期}upload.csv
