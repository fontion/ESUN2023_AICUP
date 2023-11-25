# Syntax
#       python create_dataset.py [split_from_training/combine_training_and_public]
import os
import pandas as pd
import numpy as np
import joblib
import sys
import scipy.stats as stats
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
from preprocess import format_dtypeI

def unique_vec(v:np.ndarray):
    lg_adj = v[1:]!=v[:-1]
    unqix = np.nonzero(np.r_[True, lg_adj])[0] # first appearance
    unqix = np.r_[unqix, v.size]
    return unqix

# def get99CI(x:pd.Series, popmean):
#     n = x.notna().sum()
#     m = x.mean()
#     s = x.std()
#     sem = s / np.sqrt(n)
#     dof = n - 1
#     alpha = 0.01
#     t = stats.t.ppf(1 - (alpha / 2), dof)
#     d = t*sem
#     upper_ci = m + d
#     lower_ci = m - d
#     t = (m - popmean)/sem
#     pvalue = stats.t.sf(np.abs(t), dof)*2
#     return upper_ci, lower_ci, pvalue

def add_features_P(df_train, df_pred):
    df_addP = add_features_default(df_pred, pd.DataFrame(), 'P')
    unqix_train = unique_vec(df_train.chid.cat.codes.to_numpy())
    lg_inP = df_pred.chid.isin(df_train.chid.iloc[unqix_train[:-1]])
    # 顧客有之前的刷卡紀錄
    catidx2ix = {catidx:ix for catidx,ix in zip(df_train.chid.cat.codes.iloc[unqix_train[:-1]], range(unqix_train.size-1))} # categories的index 對應 unqix_train的index
    ix_in = np.nonzero(lg_inP.to_numpy())[0]
    unqix = unique_vec(df_pred.loc[lg_inP,'chid'].to_numpy())
    train_unq_chid_idx = [catidx2ix[df_train.chid.cat.categories.get_loc(id)] for id in df_pred.chid.iloc[ix_in[unqix[:-1]]]]
    get_loc = df_addP.columns.get_loc
    print('Processing chid records...')
    for p in tqdm(range(unqix.size-1)):
        # 挑出同一位顧客的資料
        sli_train = slice(unqix_train[train_unq_chid_idx[p]], unqix_train[train_unq_chid_idx[p]+1])
        idx_pred = ix_in[unqix[p]:unqix[p+1]]
        df_pre = df_train.iloc[sli_train]
        df_inf = df_pred.iloc[idx_pred]
        # chid = df_train.chid.iat[unqix_train[train_unq_chid_idx[p]]]
        # assert (df_pre.chid==chid).all(), 'Unexpect error'
        # assert (df_inf.chid==chid).all(), 'Unexpect error'
        # assert (df_train.chid==chid).sum()==df_pre.shape[0], 'Unexpect error'
        # assert (df_pred.chid==chid).sum()==df_inf.shape[0], 'Unexpect error'
        # assert df_addP.index[idx_pred].equals(df_inf.index), 'Unexpect error'
        df_addP.iloc[idx_pred, get_loc('n_card')] = df_pre.cano.unique().size
        add_features_group(df_pre, df_inf, df_addP, 'P', idx_pred)
    return df_addP, lg_inP

def add_features_C(df_train, df_pred):
    df_addC = add_features_default(df_pred, pd.DataFrame(), 'C')
    unqix_train = unique_vec(df_train.cano.cat.codes.to_numpy())
    lg_inC = df_pred.cano.isin(df_train.cano.iloc[unqix_train[:-1]])
    # 顧客有之前的刷卡紀錄
    catidx2ix = {catidx:ix for catidx,ix in zip(df_train.cano.cat.codes.iloc[unqix_train[:-1]], range(unqix_train.size-1))} # categories的index 對應 unqix_train的index
    ix_in = np.nonzero(lg_inC.to_numpy())[0]
    unqix = unique_vec(df_pred.loc[lg_inC,'cano'].to_numpy())
    train_unq_cano_idx = [catidx2ix[df_train.cano.cat.categories.get_loc(id)] for id in df_pred.cano.iloc[ix_in[unqix[:-1]]]]
    get_loc = df_addC.columns.get_loc
    print('Processing cano records...')
    for p in tqdm(range(unqix.size-1)):
        # 挑出同一位顧客的資料
        sli_train = slice(unqix_train[train_unq_cano_idx[p]], unqix_train[train_unq_cano_idx[p]+1])
        idx_pred = ix_in[unqix[p]:unqix[p+1]]
        df_pre = df_train.iloc[sli_train]
        df_inf = df_pred.iloc[idx_pred]
        # cano = df_train.cano.iat[unqix_train[train_unq_cano_idx[p]]]
        # assert (df_pre.cano==cano).all(), 'Unexpect error'
        # assert (df_inf.cano==cano).all(), 'Unexpect error'
        # assert (df_train.cano==cano).sum()==df_pre.shape[0], 'Unexpect error'
        # assert (df_pred.cano==cano).sum()==df_inf.shape[0], 'Unexpect error'
        # assert df_addC.index[idx_pred].equals(df_inf.index), 'Unexpect error'
        df_addC.iloc[idx_pred, get_loc('n_person')] = df_pre.chid.unique().size
        add_features_group(df_pre, df_inf, df_addC, 'C', idx_pred)
    return df_addC, lg_inC

def add_features_group(df_pre, df_inf, df_add, gp, idx_pred): # gp should be "P" or "C"
    get_loc = df_add.columns.get_loc
    df_add.iloc[idx_pred, get_loc(f'{gp}usage')] = df_pre.shape[0]
    df_add.iloc[idx_pred, get_loc(f'{gp}duration')] = df_inf.days_from_start - df_pre.days_from_start.iloc[-1]
    if df_pre.shape[0] > 1:
        df_add.iloc[idx_pred, get_loc(f'{gp}interval')] = df_pre.days_from_start.diff().mean()
    df_add.iloc[idx_pred, get_loc(f'{gp}fraud_accu')] = df_pre.label.sum()
    df_add.iloc[idx_pred, get_loc(f'{gp}fraud_last')] = df_pre.label.iloc[-1]
    df_add.iloc[idx_pred, get_loc(f'{gp}stocn_ratio_tw')] = (df_pre.stocn==0).sum()/df_pre.shape[0]
    df_add.iloc[idx_pred, get_loc(f'{gp}stscd_abnormal')] = (df_pre.stscd!=-1).sum()/df_pre.shape[0] # -1代表正常
    # get distribution properties of flam1
    meanval = df_pre.flam1.mean()
    sem = df_pre.flam1.std()/np.sqrt(df_pre.shape[0])
    dof = df_pre.shape[0]-1
    ttest_1samp = lambda popmean: stats.t.sf(np.abs((meanval-popmean)/sem), dof)*2 # https://docs.scipy.org/doc/scipy/tutorial/stats.html#t-test-and-ks-test
    df_add.iloc[idx_pred, get_loc(f'{gp}flam1_pvalue')] = ttest_1samp(df_inf.flam1)
    upper99ci = meanval + stats.t.ppf(1-(0.01/2), dof)*sem # upper bound of 99% confidence interval
    # flam1_pvalue_loc = get_loc(f'{gp}flam1_pvalue')
        # for i,popmean in enumerate(df_inf.flam1):
        #     df_add.iat[idx_pred[i], flam1_pvalue_loc] = stats.ttest_1samp(a=df_pre.flam1, popmean=popmean)[1]
    # numeric feature (flam1)
    col = 'flam1'
    df_add.iloc[idx_pred, get_loc(f'{gp}{col}_maxval')] = df_pre[col].max()
    df_add.iloc[idx_pred, get_loc(f'{gp}{col}_minval')] = df_pre[col].min()
    df_add.iloc[idx_pred, get_loc(f'{gp}{col}_avgval')] = df_pre[col].mean()
    df_add.iloc[idx_pred, get_loc(f'{gp}{col}_stdval')] = df_pre[col].std()
    df_add.iloc[idx_pred, get_loc(f'{gp}{col}_newrcd')] = df_inf[col] > df_pre[col].max()
    df_add.iloc[idx_pred, get_loc(f'{gp}{col}_gt99ci')] = df_inf[col] > upper99ci
    # boolean features
    boolcols = df_inf.columns[df_inf.dtypes==bool].tolist()
    if 'label' not in boolcols:
        boolcols.append('label')
    for col in boolcols:
        df_add.iloc[idx_pred, get_loc(f'{gp}{col}_incidence')] = df_pre[col].sum()/df_pre.shape[0]
    # categorical features
    catcols = df_inf.columns[df_inf.dtypes=='category'].tolist()
    catcols.remove('chid')
    catcols.remove('cano')
    for col in catcols:
        if df_pre[col].notna().any():
            df_add.iloc[idx_pred, get_loc(f'{gp}{col}_mode')] = df_pre[col].mode().iloc[0]
            df_add.iloc[idx_pred, get_loc(f'{gp}{col}_scope')] = df_pre[col].unique().size # nan也算一類
            groupby = df_pre[[col,'flam1']].groupby(col, observed=True).sum().flam1 # A series indexed by categories and values are summary value correspond to the category
            df_add.iloc[idx_pred, get_loc(f'{gp}{col}_focus')] = groupby.index[groupby.argmax()]
        newcat_loc = get_loc(f'{gp}{col}_newcat')
        ratio_loc = get_loc(f'{gp}{col}_ratio')
        TopN_loc = get_loc(f'{gp}{col}_TopN')
        for i in np.nonzero(df_inf[col].notna().to_numpy())[0]:
            tf = df_inf[col].iat[i] not in set(df_pre[col])
            df_add.iat[idx_pred[i], newcat_loc] = tf
            if not tf:
                value_count = df_pre[col].value_counts()
                df_add.iat[idx_pred[i], ratio_loc] = value_count[df_inf[col].iat[i]]/value_count.sum()
                cum_percent = value_count.cumsum()/value_count.sum()
                ix = max(1, np.nonzero(cum_percent.to_numpy() > 0.75)[0][0])
                df_add.iat[idx_pred[i], TopN_loc] = df_inf[col].iat[i] in set(value_count.index[:ix])

def add_features(df_train, df_pred, df_add):
    pred_idx = df_pred.index # 保留df_pred原來的index
    # 按客戶排序
    df_train.sort_values(['chid','days_from_start'], inplace=True)
    df_pred.sort_values(['chid','days_from_start'], inplace=True)
    df_addP, lg_inP = add_features_P(df_train, df_pred)
    # 按卡片排序
    df_train.sort_values(['cano','days_from_start'], inplace=True)
    df_pred.sort_values(['cano','days_from_start'], inplace=True)
    df_addC, lg_inC = add_features_C(df_train, df_pred)
    assert df_addP.shape[0]==df_pred.shape[0], 'Unexpect error'
    assert df_addC.shape[0]==df_pred.shape[0], 'Unexpect error'
    df_addPC = pd.concat([df_addP.loc[pred_idx], df_addC.loc[pred_idx]], axis=1)
    df_add = pd.concat([df_addPC, df_add], axis=0)
    lg_in = lg_inP.loc[pred_idx] | lg_inC.loc[pred_idx]
    return df_add, lg_in

def sel_int_type(n):
    p = np.log2(n)
    lg = [p < k for k in (8, 16, 32, 64)]
    dtype = [np.uint8, np.uint16, np.uint32, np.uint64][lg.index(True)]
    return dtype

def add_features_default(df_pred, df_add, gp:str): # gp should be "P"(此人), "C"(此卡) or "PC"
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
    df.loc[df.Plabel_incidence==0,'chid'] = 'clear'
    df.loc[df.Clabel_incidence==0,'cano'] = 'clear'
    df.chid = df.chid.cat.remove_unused_categories()
    df.cano = df.cano.cat.remove_unused_categories()

def union_chid_cano_categories(df_train_new, df_pred_new):
    for col in ('chid','cano'):
        dtype = df_train_new[col].dtype
        lg_in = df_pred_new[col].isin(df_train_new[col].cat.categories)
        if not lg_in.all():
            categories = np.r_[dtype.categories, df_pred_new.loc[~lg_in,col].unique()]
            dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        df_pred_new[col] = df_pred_new[col].astype(dtype)

if __name__=='__main__':
    # __file__ = '/root/ESUN/ceodes/create_dataset.py'
    tStart = datetime.now()
    db_folder1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_1st')
    db_folder2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_2nd') # TODO: need modify
    mode = 'split_from_training' # [split_from_training/combine_training_and_public]
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    print('Create mode:', mode)

    if mode=='split_from_training':
        path_train = os.path.join(db_folder1, 'training_raw.joblib')
        if os.path.isfile(path_train):
            df_raw = joblib.load(path_train)
        else:
            df_raw = pd.read_csv(os.path.join(db_folder1, 'training.csv'))
            joblib.dump(df_raw, path_train, compress=3, protocol=4)
        lg = df_raw.locdt > 51 # 取出52, 53, 54, 55四天的記錄做為testing set，佔原來training data的7.19%
        df_train = df_raw.loc[~lg].copy()
        df_pred = df_raw.loc[lg].copy() # for validation
        df_raw = '' # release memory occupied by df_raw
        format_dtypeI(df_train, df_pred)
        db_output = db_folder1
        period = 4 # 一次預測的週期是幾天(public set是4天)
    elif mode=='combine_training_and_public':
        path_train = os.path.join(db_folder1,'training.joblib')
        path_pub = os.path.join(db_folder1,'public_processed.joblib')
        path_pub_gt = os.path.join(db_folder2,'public_processed_label.joblib') # TODO: need modifty
        df_train = joblib.load(path_train)
        df_pred = joblib.load(path_pub)
        df_pred.insert(df_pred.shape[0],'label', joblib.load(path_pub_gt))
        db_output = db_folder2
        period = 4 # TODO: 一次預測的週期是幾天(public set是4天) 複賽時視情況調整

    else:
        raise AssertionError(f'Unexpect mode: {mode}')

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

    drop_cols = ['locdt','flam1']
    # deal with df_pred
    print('[Deal with df_pred]')
    df_pred_add, lg_in = add_features(df_train.copy(),df_pred.copy(),pd.DataFrame())
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
        df_add, lg_in = add_features(df_train.iloc[sli_train].copy(), df_train.iloc[sli_pred].copy(), pd.DataFrame())
        sli_pred = np.r_[sli_pred][lg_in.to_numpy()]
        # data augmentation
        for j in range(1,period):
            if i > j:
                sli_train = slice(0, unqix[i-j])
                df_add, lg_in = add_features(df_train.iloc[sli_train].copy(), df_train.iloc[sli_pred].copy(), df_add)
                sli_pred = sli_pred[lg_in.to_numpy()]
                if sli_pred.size==0: break
        df_train_daily = pd.concat([df_train.loc[df_add.index].drop(columns=drop_cols), df_add], axis=1)
        df_train_new.append(df_train_daily.drop_duplicates())

    sli_pred = slice(0, unqix[1])
    df_add = add_features_default(df_train.iloc[sli_pred], pd.DataFrame(), 'PC')
    df_train_daily = pd.concat([df_train.loc[df_add.index].drop(columns=drop_cols), df_add], axis=1)
    df_train_new.append(df_train_daily.drop_duplicates())
    df_train_new = pd.concat(df_train_new, axis=0)
    renew_chid_cano(df_train_new)

    # df_pred_new需包含所有df_train_new的類別，所以從df_train_new的categories再去新增
    union_chid_cano_categories(df_train_new, df_pred_new)

    # output database
    path_db = os.path.join(db_output,'db-v3-new.joblib')
    joblib.dump({'train':df_train_new, 'pred':df_pred_new}, path_db, compress=3, protocol=4)

    assert df_pred_new.shape[0]==df_pred_new.drop_duplicates().shape[0], 'Found duplicate records'
    assert df_train_new.shape[0]==df_train_new.drop_duplicates().shape[0], 'Found duplicate records'

    tElapsed = datetime.now() - tStart
    print('Elapsed time:', tElapsed)