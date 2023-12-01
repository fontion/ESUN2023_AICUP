"""
回顧過去的刷卡記錄, 產生訓練模型所需的所有features
"""

# Syntax
#       python create_dataset.py [split_from_training/combine_training_and_public]
import os
import pandas as pd
import numpy as np
import joblib
import sys
import scipy.stats as stats
import concurrent.futures
from argparse import ArgumentParser
from datetime import datetime
from preprocess import format_dtypeI

def unique_vec(v:np.ndarray):
    """
    抓出已排序向量中unique element第一次出現的index
    """
    lg_adj = v[1:]!=v[:-1]
    unqix = np.nonzero(np.r_[True, lg_adj])[0] # first appearance
    unqix = np.r_[unqix, v.size]
    return unqix

def add_features_group(df_train, df_pred, gp, parallel=False):
    """
    回溯此人或此卡過去刷卡記錄, 產生新的features
    """
    if gp=='P':
        print('Processing chid records...')
        ID = 'chid'
    elif gp=='C':
        print('Processing cano records...')
        ID = 'cano'
    codes = df_train[ID].cat.codes.to_numpy()
    lg_adj = codes[1:]!=codes[:-1]
    lg_unq = np.r_[True, lg_adj]
    lg_in = df_pred[ID].isin(df_train.loc[lg_unq,ID])
    ix_in = np.nonzero(lg_in.to_numpy())[0]
    # 顧客無之前的刷卡紀錄
    lg_notin = ~lg_in
    df_add = add_features_default(df_pred.loc[lg_notin], pd.DataFrame(), gp)
    df_add.index = np.nonzero(lg_notin.to_numpy())[0]
    # 顧客有之前的刷卡紀錄
    codes = df_pred.loc[lg_in, ID].cat.codes.to_numpy()
    lg_adj = codes[1:]!=codes[:-1]
    reps = np.cumsum(np.r_[False, lg_adj])
    lg_unq = np.r_[True, lg_adj] # first appearance
    unq_ids = df_pred[ID].iloc[ix_in[lg_unq]]
    in_pred = df_train[ID].isin(unq_ids)
    # pick intersection contents (df_pre與df_inf出現chid或cano的順序會完全相同)
    df_pre = df_train.loc[in_pred]
    df_inf = df_pred.loc[lg_in]
    # start adding features
    kwargs = {'observed':True, 'sort':False} # 僅列出有出現的chid或cano，且按出現順序排序
    gb = df_pre.groupby(ID, **kwargs)
    if gp=='P':
        data = {'n_card': gb.cano.nunique().iloc[reps].to_numpy(dtype=np.uint8)}
    elif gp=='C':
        data = {'n_person': gb.chid.nunique().iloc[reps].to_numpy(dtype=np.uint8)}
    
    group_size = gb.size()
    data[f'{gp}usage'] = group_size.iloc[reps].to_numpy(dtype=np.uint32)

    last_time = gb.days_from_start.last()
    data[f'{gp}duration'] = df_inf.days_from_start.to_numpy() - last_time.iloc[reps].to_numpy()

    diff_mean = df_pre[[ID,'days_from_start']].set_index(ID).groupby(level=0,**kwargs).diff().groupby(level=0,**kwargs).mean()
    data[f'{gp}interval'] = diff_mean.iloc[reps,0].to_numpy()
    data[f'{gp}interval'][data[f'{gp}usage']==1] = 100

    data[f'{gp}fraud_accu'] = gb.label.sum().iloc[reps].to_numpy(dtype=np.uint16)

    data[f'{gp}fraud_last'] = gb.label.last().iloc[reps].to_numpy()

    stocn_eq_0 = (df_pre[[ID,'stocn']].set_index(ID)==0).groupby(level=0,**kwargs).sum()
    data[f'{gp}stocn_ratio_tw'] = stocn_eq_0.iloc[reps,0].to_numpy()/data[f'{gp}usage']

    stscd_eq_n1 = (df_pre[[ID,'stscd']].set_index(ID)!=-1).groupby(level=0,**kwargs).sum()
    data[f'{gp}stscd_abnormal'] = stscd_eq_n1.iloc[reps,0].to_numpy()/data[f'{gp}usage']

    # get distribution properties of flam1
    group_value = gb.flam1.agg(['mean','sem'])
    meanval = group_value['mean'].to_numpy()
    sem = group_value['sem'].to_numpy()
    dof = (group_size-1).to_numpy()
    upper99ci = meanval + stats.t.ppf(1-(0.01/2), dof)*sem # upper bound of 99% confidence interval
    popmean = df_inf.flam1.to_numpy()
    data[f'{gp}flam1_pvalue'] = stats.t.sf(np.abs((meanval[reps]-popmean)/sem[reps]), dof[reps])*2 # https://docs.scipy.org/doc/scipy/tutorial/stats.html#t-test-and-ks-test
    
    # numeric feature (flam1)
    col = 'flam1'
    data[f'{gp}{col}_maxval'] = gb[col].max().iloc[reps].to_numpy()
    data[f'{gp}{col}_minval'] = gb[col].min().iloc[reps].to_numpy()
    data[f'{gp}{col}_avgval'] = meanval[reps]
    data[f'{gp}{col}_stdval'] = gb[col].std().iloc[reps].to_numpy()
    data[f'{gp}{col}_newrcd'] = df_inf[col].to_numpy() > data[f'{gp}{col}_maxval']
    data[f'{gp}{col}_gt99ci'] = df_inf[col].to_numpy() > upper99ci[reps]

    # boolean features
    boolcols = df_inf.columns[df_inf.dtypes==bool].tolist()
    if 'label' not in boolcols:
        boolcols.append('label')
    for col in boolcols:
        data[f'{gp}{col}_incidence'] = (gb[col].sum()/group_size).iloc[reps].to_numpy()
    
    # categorical features
    catcols = df_inf.columns[df_inf.dtypes=='category'].tolist()
    catcols.remove('chid')
    catcols.remove('cano')
    if parallel:
        n_catcols = len(catcols)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            data_catcols = executor.map(
                add_features_category,
                [df_pre]*n_catcols,
                [df_inf]*n_catcols,
                gp*n_catcols,
                [ID]*n_catcols,
                catcols,
                [reps]*n_catcols,
                [gb]*n_catcols
            )
        for d in data_catcols:
            data.update(d)
    else:
        for col in catcols:
            print('  - categorical feature:',col)
            data.update(add_features_category(df_pre, df_inf, gp, ID, col, reps, gb))
    df_add = pd.concat([df_add, pd.DataFrame(data, index=ix_in)],axis=0)
    df_add.sort_index(axis=0, inplace=True)
    df_add.index = df_pred.index
    return df_add, lg_in

def add_features_category(df_pre,df_inf,gp,ID,col,reps,gb):
    """
    回溯此人或此卡過去刷卡記錄, 產生新的features(針對categorical feature), 被add_features_group所呼叫
    """
    kwargs = {'observed':True, 'sort':False}
    data = dict()
    dtype = sel_int_type(df_inf[col].cat.categories.size)
    hasNA = df_pre[col].isna().any()
    # mode
    s_code = pd.Series(df_pre[col].cat.codes.to_numpy(), index=df_pre[ID], name='code')
    if hasNA:
        s_code.drop(index=s_code.index[s_code==-1], inplace=True)
        mode_cat = s_code.groupby(level=0,**kwargs).agg(lambda c: c.mode()[0])
        lg_ID_in = df_inf[ID].isin(mode_cat.index).to_numpy()
        if lg_ID_in.all():
            codes = mode_cat.iloc[reps].to_numpy()
        else:
            codes = -np.ones(df_inf.shape[0], dtype=int)
            codes[lg_ID_in] = mode_cat[df_inf.loc[lg_ID_in,ID]].to_numpy()
    else:
        mode_cat = s_code.groupby(level=0,**kwargs).agg(lambda c: c.mode()[0])
        codes = mode_cat.iloc[reps].to_numpy()
    data[f'{gp}{col}_mode'] = pd.Categorical.from_codes(codes, dtype=df_pre[col].dtype)
    # scope
    data[f'{gp}{col}_scope'] = gb[col].nunique(dropna=False).iloc[reps].to_numpy(dtype=dtype)
    # focus
    df_flam1 = pd.DataFrame({ID:df_pre[ID], 'code':df_pre[col].cat.codes, 'flam1':df_pre['flam1']})
    if hasNA:
        df_flam1.drop(index=df_flam1.index[df_flam1.code==-1], inplace=True)
        s_flam1 = df_flam1.groupby([ID,'code'],**kwargs).flam1.sum()
        max_cat = s_flam1.groupby(level=0,**kwargs).idxmax().transform(lambda t:t[1])
        lg_ID_in = df_inf[ID].isin(max_cat.index).to_numpy()
        if lg_ID_in.all():
            codes = max_cat.iloc[reps].to_numpy()
        else:
            codes = -np.ones(df_inf.shape[0], dtype=int)
            codes[lg_ID_in] = max_cat[df_inf.loc[lg_ID_in,ID]].to_numpy()
    else:
        s_flam1 = df_flam1.groupby([ID,'code'],**kwargs).flam1.sum()
        max_cat = s_flam1.groupby(level=0,**kwargs).idxmax().transform(lambda t:t[1])
        codes = max_cat.iloc[reps].to_numpy()
    data[f'{gp}{col}_focus'] = pd.Categorical.from_codes(codes, dtype=df_pre[col].dtype)
    # newcat
    categories_set = gb[col].agg(lambda x: set(x))
    oldcat = np.vectorize(a_isin_b)(df_inf[col], categories_set.iloc[reps])
    data[f'{gp}{col}_newcat'] = np.logical_not(oldcat)
    # ratio
    data[f'{gp}{col}_ratio'] = np.zeros(df_inf.shape[0])
    data[f'{gp}{col}_ratio'][df_inf[col].isna().to_numpy()] = np.nan
    value_count = df_pre.groupby([ID,col],**kwargs).size() # 不包含col為空值的記錄，本來使用value_counts指令也不包含
    group_size_woNA = value_count.groupby(level=0,**kwargs).sum()
    value_ratio = value_count/group_size_woNA.loc[value_count.index.get_level_values(0)].to_numpy()
    query_index = pd.MultiIndex.from_frame(df_inf[[ID,col]])
    lg_ratio_in = query_index.isin(value_ratio.index)
    data[f'{gp}{col}_ratio'][lg_ratio_in] = value_ratio.loc[query_index[lg_ratio_in]].to_numpy()
    # TopN
    value_ratio_descend = value_ratio.to_frame().reset_index().sort_values([ID,0], ascending=False).set_index([ID,col])[0]
    lg_lt75 = value_ratio_descend.groupby(level=0,**kwargs).cumsum() <= 0.75
    ix_1st = unique_vec(lg_lt75.index.get_level_values(0).codes)[:-1]
    lg_lt75.iloc[ix_1st] = True
    categories_set = lg_lt75.loc[lg_lt75].reset_index(level=1)[col].groupby(level=0,**kwargs).agg(lambda x:set(x))
    lg_ID_in = df_inf[ID].isin(categories_set.index).to_numpy() # 之前的記錄若皆為空值，該ID就會在groupby時被排除
    if lg_ID_in.all():
        data[f'{gp}{col}_TopN'] = np.vectorize(a_isin_b)(df_inf[col], categories_set[df_inf[ID]])
    else:
        data[f'{gp}{col}_TopN'] = np.zeros(df_inf.shape[0], dtype=bool)
        data[f'{gp}{col}_TopN'][lg_ID_in] = np.vectorize(a_isin_b)(df_inf.loc[lg_ID_in,col], categories_set[df_inf.loc[lg_ID_in,ID]])
    return data

def a_isin_b(a,b):
    """
    check if a is in set b
    """
    return a in b

def add_features(df_train, df_pred, df_add, parallel=False):
    """
    回溯此人及此卡過去刷卡記錄, 產生新的features
    """
    pred_idx = df_pred.index # 保留df_pred原來的index
    # 按客戶排序
    df_train.sort_values(['chid','days_from_start'], inplace=True)
    df_pred.sort_values(['chid','days_from_start'], inplace=True)
    df_addP, lg_inP = add_features_group(df_train, df_pred, 'P', parallel)
    # 按卡片排序
    df_train.sort_values(['cano','days_from_start'], inplace=True)
    df_pred.sort_values(['cano','days_from_start'], inplace=True)
    df_addC, lg_inC = add_features_group(df_train, df_pred, 'C', parallel)
    # Combine two DataFrames
    df_addPC = pd.concat([df_addP.loc[pred_idx], df_addC.loc[pred_idx]], axis=1)
    if df_add.empty:
        df_add = df_addPC
    else:
        df_add = pd.concat([df_addPC, df_add], axis=0)
    lg_in = lg_inP.loc[pred_idx] | lg_inC.loc[pred_idx]
    return df_add, lg_in

def sel_int_type(n):
    """
    依input數量選擇合適的integer格式
    """
    p = np.log2(n)
    lg = [p < k for k in (8, 16, 32, 64)]
    dtype = [np.uint8, np.uint16, np.uint32, np.uint64][lg.index(True)]
    return dtype

def add_features_default(df_pred, df_add, gp:str): # gp should be "P"(此人), "C"(此卡) or "PC"
    """
    根據testing set維度建立一個新features矩陣, 並賦予預設值
    """
    data = dict()
    n_record = df_pred.shape[0]
    # boolean features
    boolcols = df_pred.columns[df_pred.dtypes==bool].to_list()
    if 'label' not in boolcols:
        boolcols.append('label')
    # categorical features
    catcols = df_pred.columns[df_pred.dtypes=='category'].tolist()
    catcols.remove('chid')
    catcols.remove('cano')
    for s in gp:
        if s=='P':
            data['n_card'] = np.ones(n_record, dtype=np.uint8)
        elif s=='C':
            data['n_person'] = np.ones(n_record, dtype=np.uint8)
        else:
            raise AssertionError('Unexpect related source')
        data[f'{s}usage'] = np.zeros(n_record, dtype=np.uint32)
        data[f'{s}duration'] = 100*np.ones(n_record)
        data[f'{s}interval'] = 100*np.ones(n_record)
        data[f'{s}fraud_accu'] = np.zeros(n_record, dtype=np.uint16)
        data[f'{s}fraud_last'] = np.zeros(n_record, dtype=bool)
        data[f'{s}stocn_ratio_tw'] = 100*np.ones(n_record)
        data[f'{s}stscd_abnormal'] = np.zeros(n_record)
        data[f'{s}flam1_pvalue'] = np.zeros(n_record)
        # numeric feature (flam1)
        col = 'flam1'
        data[f'{s}{col}_maxval'] = np.zeros(n_record)
        data[f'{s}{col}_minval'] = np.zeros(n_record)
        data[f'{s}{col}_avgval'] = np.zeros(n_record)
        data[f'{s}{col}_stdval'] = np.zeros(n_record)
        data[f'{s}{col}_newrcd'] = np.ones(n_record, dtype=bool)
        data[f'{s}{col}_gt99ci'] = np.ones(n_record, dtype=bool)
        # boolean features
        for col in boolcols:
            data[f'{s}{col}_incidence'] = np.zeros(n_record)
        # categorical features
        for col in catcols:
            data[f'{s}{col}_mode'] = pd.Categorical.from_codes([-1]*n_record, dtype=df_pred[col].dtype)
            dtype = sel_int_type(df_pred[col].cat.categories.size)
            data[f'{s}{col}_scope'] = np.zeros(n_record, dtype=dtype)
            data[f'{s}{col}_focus'] = pd.Categorical.from_codes([-1]*n_record, dtype=df_pred[col].dtype)
            data[f'{s}{col}_newcat'] = np.ones(n_record, dtype=bool)
            data[f'{s}{col}_ratio'] = np.zeros(n_record)
            if df_pred[col].isna().any():
                data[f'{s}{col}_ratio'][df_pred[col].isna().to_numpy()] = np.nan
            data[f'{s}{col}_TopN'] = np.zeros(n_record, dtype=bool)
    df_new = pd.DataFrame(data, index=df_pred.index)
    if df_add.empty:
        return df_new
    else:
        return pd.concat([df_new, df_add], axis=0)

def renew_chid_cano(df:pd.DataFrame):
    """
    調整chid與cano兩個feature的值, 若無過去盜刷記錄設為'clear'
    """
    df.loc[df.Plabel_incidence==0,'chid'] = 'clear'
    df.loc[df.Clabel_incidence==0,'cano'] = 'clear'
    df.chid = df.chid.cat.remove_unused_categories()
    df.cano = df.cano.cat.remove_unused_categories()

def assign_chid_cano_categories(df_train_new):
    """
    chid與cano兩個feature適當轉換成包含所有類別的格式, 供後續pd.concat使用
    """
    for col in ('chid','cano'):
        categories = set()
        for df in df_train_new:
            categories.update(df[col].cat.categories.tolist())
        categories = list(categories)
        categories.sort()
        Dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        for df in df_train_new:
            df[col] = df[col].astype(Dtype)

def union_chid_cano_categories(df_train_new, df_pred_new):
    """
    testing set的類別必須包含所有training set的類別
    """
    for col in ('chid','cano'):
        dtype = df_train_new[col].dtype
        lg_in = df_pred_new[col].isin(df_train_new[col].cat.categories)
        if not lg_in.all():
            categories = np.r_[dtype.categories, df_pred_new.loc[~lg_in,col].unique()]
            dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        df_pred_new[col] = df_pred_new[col].astype(dtype)

def format_dtypeII(df_train, df_pred):
    df_train = df_train.set_index('txkey').drop_duplicates()
    df_pred = df_pred.set_index('txkey').drop_duplicates()
    assert df_train.columns.equals(df_pred.columns), 'features mismatch'
    assert df_train.index.is_unique, 'txkey is not unique'
    assert df_pred.index.is_unique, 'txkey is not unique'
    # add days_from_start
    df_train.insert(0,'days_from_start',df_train.locdt + df_train.loctm/86400) # 取代授權日期locdt和授權時間loctm
    df_pred.insert(0,'days_from_start',df_pred.locdt + df_pred.loctm/86400) # 取代授權日期locdt和授權時間loctm
    df_train.sort_values(['locdt','chid','loctm'], inplace=True)
    df_pred.sort_values(['locdt','chid','loctm'], inplace=True)

    # replace loctm as category data type
    categories = ['00-03','03-06','06-09','09-12','12-15','15-18','18-21','21-24'] # 每三小時為一個類別
    df_train.loctm = pd.Categorical.from_codes(df_train.loctm//(3*60*60), categories=categories, ordered=True)
    df_pred.loctm = pd.Categorical.from_codes(df_pred.loctm//(3*60*60), categories=categories, ordered=True)
    # add discount
    df_train.insert(df_train.shape[1],'discount', df_train.conam-df_train.flam1)
    df_pred.insert(df_pred.shape[1],'discount', df_pred.conam-df_pred.flam1)
    # add new category 'clear' in chid and cano which represent no fraud before
    df_train.chid = df_train.chid.cat.add_categories('clear')
    df_train.cano = df_train.cano.cat.add_categories('clear')
    df_pred.chid = df_pred.chid.cat.add_categories('clear')
    df_pred.cano = df_pred.cano.cat.add_categories('clear')
    return df_train, df_pred

if __name__=='__main__':
    # __file__ = '/root/ESUN/codes/create_dataset.py'
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['split_from_training','combine_training_and_public'], default='combine_training_and_public')
    args = parser.parse_args()
    tStart = datetime.now()
    db_folder1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_1st')
    db_folder2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_2nd')
    print('Create mode:', args.mode)

    path_train = os.path.join(db_folder1,'training_raw.joblib')
    path_public = os.path.join(db_folder2,'public_raw.joblib')
    if args.mode=='split_from_training':
        df_raw = joblib.load(path_train)
        db_output = db_folder1
        # 取出52, 53, 54, 55四天的記錄做為testing set，佔原來training data的7.19%
        period = 4 # 一次預測的週期是幾天(public set是4天)
        
    elif args.mode=='combine_training_and_public':
        df_raw = pd.concat([joblib.load(path_train), joblib.load(path_public)], axis=0, ignore_index=True)
        db_output = db_folder2
        # 取出55, 56, 57, 58, 59五天的記錄做為testing set，佔原來training data的8.24%
        period = 5 # 一次預測的週期是幾天(private set是5天) 複賽時視情況調整

    lg = df_raw.locdt > (df_raw.locdt.max()-period)
    df_train = df_raw.loc[~lg].copy()
    df_pred = df_raw.loc[lg].copy() # for validation
    df_raw = '' # release memory occupied by df_raw
    format_dtypeI(df_train, df_pred)
    df_train, df_pred = format_dtypeII(df_train, df_pred)

    drop_cols = ['locdt','flam1']
    # deal with df_pred
    print('[Deal with df_pred]')
    df_pred_add, lg_in = add_features(df_train.copy(),df_pred.copy(),pd.DataFrame(),parallel=True)
    df_pred.drop(columns=drop_cols, inplace=True)
    df_pred_new = pd.concat([df_pred.loc[df_pred_add.index], df_pred_add], axis=1)
    renew_chid_cano(df_pred_new)

    # deal with df_train
    locdt = df_train.locdt.to_numpy()
    unqix = unique_vec(locdt)
    assert unqix.size-1==locdt[-1]+1, 'Not everyday has records'
    df_train_new = []
    for i in range(locdt[-1],0,-1): # 到第1天
        print('[Deal with df_train]',f'day-{i}')
        sli_pred = slice(unqix[i], unqix[i+1])
        sli_train = slice(0, unqix[i])
        df_add, lg_in = add_features(df_train.iloc[sli_train].copy(), df_train.iloc[sli_pred].copy(), pd.DataFrame(), parallel=True)
        sli_pred = np.r_[sli_pred][lg_in.to_numpy()]
        # data augmentation
        for j in range(1,period):
            if i > j:
                sli_train = slice(0, unqix[i-j])
                df_add, lg_in = add_features(df_train.iloc[sli_train].copy(), df_train.iloc[sli_pred].copy(), df_add, parallel=True)
                sli_pred = sli_pred[lg_in.to_numpy()]
                if sli_pred.size==0: break
        df_train_daily = pd.concat([df_train.loc[df_add.index].drop(columns=drop_cols), df_add], axis=1)
        renew_chid_cano(df_train_daily)
        df_train_new.append(df_train_daily.drop_duplicates())

    sli_pred = slice(0, unqix[1])
    df_add = add_features_default(df_train.iloc[sli_pred], pd.DataFrame(), 'PC')
    df_train_daily = pd.concat([df_train.loc[df_add.index].drop(columns=drop_cols), df_add], axis=1)
    renew_chid_cano(df_train_daily)
    df_train_new.append(df_train_daily.drop_duplicates())

    assign_chid_cano_categories(df_train_new)
    df_train_new = pd.concat(df_train_new, axis=0)

    # df_pred_new需包含所有df_train_new的類別，所以從df_train_new的categories再去新增
    union_chid_cano_categories(df_train_new, df_pred_new)

    # output database
    path_db = os.path.join(db_output,'db-v4.joblib')
    joblib.dump({'train':df_train_new, 'pred':df_pred_new}, path_db, compress=3, protocol=4)

    assert df_pred_new.shape[0]==df_pred_new.drop_duplicates().shape[0], 'Found duplicate records'
    assert df_train_new.shape[0]==df_train_new.drop_duplicates().shape[0], 'Found duplicate records'

    tElapsed = datetime.now() - tStart
    print('Elapsed time:', tElapsed)