"""
初步的資料前處理, 妥善轉換資料格式, 如category, boolean, integer, ...
"""
import os
import pandas as pd
import numpy as np
import joblib
from argparse import ArgumentParser

def loctm_mapper():
    """
    建立轉換dictionary, 將本來loctm儲存的hhmmss格式映射成秒數
    """
    to_sec = dict()
    s = 0
    for hour in range(24):
        for min in range(60):
            for sec in range(60):
                v = hour*10000 + min*100 + sec
                to_sec[v] = s
                s += 1
    return to_sec

def format_dtypeI(df_train, df_pred):
    """
    low level data preprocessing
    """
    # deal with df_train
    df = df_train
    df.locdt = df.locdt.astype(np.uint8) # 授權日期 train: 0-55; pub: 56-59
    to_sec = loctm_mapper()
    assert df.loctm.isin(set(to_sec)).all(), 'Unexpect time'
    df.loctm = df.loctm.map(to_sec).astype(np.uint32) # convert to second number (0-86399)
    df.chid = df.chid.astype('category') # 顧客ID train: 482667位客戶； pub: 219005位客戶(其中5163不在train)
    df.cano = df.cano.astype('category') # 交易卡號 train: 618898; pub: 239866(其中8887不在train)
    df.contp = df.contp.astype(np.uint8).astype('category') # 交易類別: 數值0-6
    Dtype = pd.CategoricalDtype(categories=np.int8([0,1,2,3,4,5,7,8,9,10]), ordered=False)
    df.etymd = df.etymd.astype(Dtype) # 交易型態: 數值0,1,2,3,4,5,7,8,9,10,nan(沒有6), trian空值203455(2.34%); pub空值13826(2.30%)
    df.mchno = df.mchno.astype('category') # 特店代號 train: 163797; pub: 52303(其中4627不在train)
    df.acqic = df.acqic.astype('category') # 收單行代碼 train: 8334; pub: 2651(其中169不在train)
    notna = df.mcc.notna()
    Dtype = pd.CategoricalDtype(categories=np.unique(df.loc[notna,'mcc'].astype(np.uint16).to_numpy()), ordered=False)
    df.mcc = df.mcc.astype(Dtype) # mcc_code train: 459; pub: 335(其中6種不在train)
    df.conam = df.conam.astype(float) # 交易金額-台幣 維持原本float格式
    df.ecfg = df.ecfg.astype(bool) # 網路交易註記
    df.insfg = df.insfg.astype(bool) # 是否分期交易
    Dtype = pd.CategoricalDtype(categories=np.unique(df.iterm.astype(np.uint8)), ordered=True) # (應該不會分期超過21.25年)
    df.iterm = df.iterm.astype(Dtype) # 分期期數
    df.bnsfg = df.bnsfg.astype(bool) # 是否紅利交易
    df.flam1 = df.flam1.astype(np.uint64) # 實付金額 單位應該是台幣，建議取消這個feature，改成是與交易金額之間的差值
    notna = df.stocn.notna()
    Dtype = pd.CategoricalDtype(categories=np.unique(df.loc[notna,'stocn'].astype(np.uint8).to_numpy()), ordered=False)
    df.stocn = df.stocn.astype(Dtype) # 消費地國別 train: 122; pub: 84(其中3不在train)
    notna = df.scity.notna()
    Dtype = pd.CategoricalDtype(categories=np.unique(df.loc[notna,'scity'].astype(np.uint16).to_numpy()), ordered=False)
    df.scity = df.scity.astype(Dtype) # 消費城市 train: 12003; pub: 2672(其中518不在train)
    notna = df.stscd.notna()
    Dtype = pd.CategoricalDtype(categories=np.r_[-1, np.unique(df.loc[notna,'stscd'].astype(np.int8).to_numpy())], ordered=False)
    df.stscd = df.stscd.fillna(-1).astype(Dtype) # 狀態碼 0,1,2,3,4 絕大多數皆為空值(99.73%)，空值應該代表交易正常，視為其中一種類別(設為-1)
    df.ovrlt = df.ovrlt.astype(bool) # 超額註記碼
    df.flbmk = df.flbmk.astype(bool) # Fallback註記
    notna = df.hcefg.notna()
    Dtype = pd.CategoricalDtype(categories=np.unique(df.loc[notna,'hcefg'].astype(np.uint8).to_numpy()), ordered=False)
    df.hcefg = df.hcefg.astype(Dtype) # 支付型態 0-10
    notna = df.csmcu.notna()
    Dtype = pd.CategoricalDtype(categories=np.unique(df.loc[notna,'csmcu'].astype(np.uint8).to_numpy()), ordered=False)
    df.csmcu = df.csmcu.astype(Dtype) # 消費地幣別 train: 79; pub: 57(其中3不在train)
    df.csmam = df.csmam.astype(np.uint64) # 消費地金額 在train中，有8096169(93.18%)與flam1實付金額相同，2.37%消費金額較高，4.45%消費金額較低
    df.flg_3dsmk = df.flg_3dsmk.astype(bool)
    df.label = df.label.astype(bool)
    # deal with df_pred
    columns = df.columns.tolist()
    if 'label' not in df_pred.columns:
        columns.remove('label') # remove label
    columns.remove('chid') # 顧客ID
    columns.remove('cano') # 交易卡號
    df_pred.loctm = df_pred.loctm.map(to_sec)
    df_pred.stscd.fillna(-1, inplace=True)
    for col in columns:
        dtype = df[col].dtype
        if dtype.name=='category':
            notna = df_pred[col].notna()
            lg_in = df_pred.loc[notna,col].isin(df[col].cat.categories) # 排除nan造成的新類別
            if not lg_in.all(): # 當出現train沒有的類別，加入成為新的類別
                categories = np.r_[dtype.categories, df_pred.loc[notna,col].loc[~lg_in].unique().astype(dtype.categories.dtype)]
                if issubclass(dtype.categories.dtype.type, np.integer):
                    categories = np.sort(categories)
                dtype = pd.CategoricalDtype(categories=categories, ordered=dtype.ordered)
        df_pred[col] = df_pred[col].astype(dtype)
    df_pred.chid = df_pred.chid.astype('category')
    df_pred.cano = df_pred.cano.astype('category')

def sanity_card2customer(df):
    """
    sanity check: 一張卡被幾個客戶持有
    """
    ix = np.argsort(df.cano.cat.codes)
    n_card = df.cano.cat.categories.size
    codes_sorted = df.cano[ix].cat.codes.to_numpy()
    lg_adj = codes_sorted[1:]!=codes_sorted[:-1]
    unqix = np.nonzero(np.r_[True, lg_adj])[0] # first appearance
    unqix = np.r_[unqix, df.shape[0]]
    assert len(unqix)-1==n_card, 'Unexpect error'
    chid_sorted = df.chid[ix].to_numpy()
    for i in range(n_card):
        n_customer = pd.unique(chid_sorted[unqix[i]:unqix[i+1]]).size
        if n_customer!=1:
            print(f'card belong to {n_customer} customers', df.cano.cat.categories[i])

def sanity_always_last(df):
    """
    sanity check: 盜刷是否總發生在刷卡記錄的最後一筆
    """
    df.sort_values(['cano','locdt','loctm'], inplace=True)
    kwargs = {'observed':True, 'sort':False}
    gb = df.groupby('cano', **kwargs)
    is_fraud = gb.label.any()
    last_fraud = gb.label.last()
    n_fraud_card = is_fraud.sum()
    fraud_and_last = (is_fraud & last_fraud).sum()
    print('盜刷為最後一筆的卡片共有{}，佔全部 {:.2f}%'.format(fraud_and_last, fraud_and_last/n_fraud_card*100))
    # n_card = df.cano.cat.categories.size
    # codes = df.cano.cat.codes.to_numpy()
    # lg_adj = codes[1:]!=codes[:-1]
    # unqix = np.nonzero(np.r_[True, lg_adj])[0] # first appearance
    # unqix = np.r_[unqix, df.shape[0]]
    # assert len(unqix)-1==n_card, 'Unexpect error'
    # label_loc = df.columns.get_loc('label')
    # txkey_loc = df.columns.get_loc('txkey')
    # count = 0
    # fraud_card = 0
    # for i in range(n_card):
    #     if df.iloc[unqix[i]:unqix[i+1], label_loc].any():
    #         fraud_card += 1
    #         if df.iat[unqix[i+1]-1, label_loc]:
    #             count += 1
    #             # print('盜刷為最後一筆，txkey:', df.iat[unqix[i+1]-1, txkey_loc])
    # print('盜刷為最後一筆的卡片共有{}，佔全部 {:.2f}%'.format(count, count/fraud_card*100))

def ana_usage(df):
    """
    sanity check: 剖析客戶刷卡記錄
    """
    ix = np.argsort(df.chid.cat.codes)
    n_customer = df.chid.cat.categories.size
    codes_sorted = df.chid.iloc[ix].cat.codes.to_numpy()
    lg_adj = codes_sorted[1:]!=codes_sorted[:-1]
    unqix = np.nonzero(np.r_[True, lg_adj])[0] # first appearance
    unqix = np.r_[unqix, df.shape[0]]
    assert len(unqix)-1==n_customer, 'Unexpect error'
    cano_sorted = df.cano.iloc[ix].to_numpy()
    label_sorted = df.label.iloc[ix].to_numpy()
    n_txkey = unqix[1:]-unqix[:-1] # 刷卡次數
    n_card = np.zeros(n_customer, dtype=np.uint8) # 持有信用卡張數
    n_fraud = np.zeros(n_customer, dtype=np.uint16) # 被盜刷次數
    for i in range(n_customer):
        n_card[i] = pd.unique(cano_sorted[unqix[i]:unqix[i+1]]).size # 每位客戶擁有的卡片數量
        n_fraud[i] = label_sorted[unqix[i]:unqix[i+1]].sum() # 每位客戶被盜刷的總次數
    df_ch = pd.DataFrame(data={'n_txkey': n_txkey, 'n_card': n_card, 'n_fraud': n_fraud}, index=df.chid.cat.categories)
    print('總刷卡次數統計(人)')
    print(df_ch.n_txkey.value_counts().sort_index())
    print('持有信用卡數量統計(人)')
    print(df_ch.n_card.value_counts().sort_index())
    print('被盜刷的總次數統計(人)')
    print(df_ch.n_fraud.value_counts().sort_index())
    nt = np.unique(n_txkey)
    nc = np.unique(n_card)
    y1 = np.zeros(nt.size)
    y2 = np.zeros(nc.size)
    # 分析刷卡次數與被盜刷次數兩者間的關係
    for i,n in enumerate(nt):
        lg = n_txkey==n
        y1[i] = n_fraud[lg].sum()/lg.sum() # 在該刷卡次數的客戶中，平均每人被盜刷的次數
    # 分析持有信用卡張數與被盜刷次數兩者間的關係
    for i,n in enumerate(nc):
        lg = n_card==n
        y2[i] = n_fraud[lg].sum()/lg.sum()
    print('刷卡次數對應平均每人被盜刷次數')
    print(pd.Series(y1, index=nt))
    print('持有信用卡張數對應平均每人被盜刷次數')
    print(pd.Series(y2, index=nc))

def chk_rule(df):
    """
    若前一筆是盜刷, 就預測下一筆為盜刷, 這樣正確率有多少
    """
    df = df.sort_values(['cano','locdt','loctm'])
    n_card = df.cano.cat.categories.size
    codes = df.cano.cat.codes.to_numpy()
    lg_adj = codes[1:]!=codes[:-1]
    unqix = np.nonzero(np.r_[True, lg_adj])[0] # first appearance
    unqix = np.r_[unqix, df.shape[0]]
    assert len(unqix)-1==n_card, 'Unexpect error'
    label_loc = df.columns.get_loc('label')
    result = []
    for i in range(n_card):
        sli = slice(unqix[i], unqix[i+1])
        if df.iloc[sli,label_loc].any():
            six = df.iloc[sli, label_loc].tolist().index(True)
            result += df.iloc[unqix[i]+six+1:unqix[i+1], label_loc].tolist()
    print('正確率:', sum(result)/len(result)*100, '%')


if __name__=='__main__':
    # __file__ = '/root/ESUN/codes/load_db.py'
    parser = ArgumentParser()
    parser.add_argument('folder', type=str, choices=['dataset_1st','dataset_2nd','dataset_3rd'], default='dataset_3rd')
    args = parser.parse_args()
    print('Dataset folder:', args.folder)
    db_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.folder)
    if args.folder=='dataset_1st':
        path_train = os.path.join(db_folder, 'training.joblib')
        path_pub = os.path.join(db_folder, 'public_processed.joblib')
    elif args.folder=='dataset_2nd':
        path_train = os.path.join(db_folder, 'public.joblib')
        path_pub = os.path.join(db_folder, 'private_1_processed.joblib')
    elif args.folder=='dataset_3rd':
        path_train = os.path.join(db_folder, 'private_1.joblib')
        path_pub = os.path.join(db_folder, 'private_2_processed.joblib')
    if os.path.isfile(path_train) and os.path.isfile(path_pub):
        df_train = joblib.load(path_train)
        df_pub = joblib.load(path_pub)
    else:
        os.path.basename(path_train)
        train_file_name = os.path.splitext(os.path.basename(path_train))[0]
        pub_file_name = os.path.splitext(os.path.basename(path_pub))[0]
        df_train = pd.read_csv(os.path.join(db_folder, train_file_name+'.csv'))
        df_pub = pd.read_csv(os.path.join(db_folder, pub_file_name+'.csv'))
        # save raw data
        joblib.dump(df_train, os.path.splitext(path_train)[0]+'_raw.joblib', compress=3, protocol=4)
        joblib.dump(df_pub, os.path.splitext(path_pub)[0]+'_raw.joblib', compress=3, protocol=4)
        # preprocessing
        format_dtypeI(df_train, df_pub)
        # output to compressed joblib file
        joblib.dump(df_train, path_train, compress=3, protocol=4)
        joblib.dump(df_pub, path_pub, compress=3, protocol=4)
    # sanity check
    print('檢查一張卡是否只屬於一位客戶 - training')
    sanity_card2customer(df_train)
    print('檢查一張卡是否只屬於一位客戶 - public')
    sanity_card2customer(df_pub)
    print('')
    # 卡片被盜刷後仍可以使用
    sanity_always_last(df_train.copy())
    # 信用卡使用分析
    ana_usage(df_train)
    # 檢測每張卡片，只要前一次是被盜刷，下次就一定是盜刷的正確性
    chk_rule(df_train)
