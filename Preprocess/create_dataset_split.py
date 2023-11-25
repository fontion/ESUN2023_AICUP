import os
import joblib
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('db_name', type=str, default='db-v4')
parser.add_argument('mode',nargs='+',help='Allow modes are: FirstUse, UsedBefore, NoFraudBefore, AfterFraud')
args = parser.parse_args()


# __file__ = '/root/ESUN/codes/create_dataset_split.py'
prj_folder = os.path.dirname(os.path.dirname(__file__))
ver = int(args.db_name[4])
if ver < 4:
    db_folder = 'dataset_1st'
else:
    db_folder = 'dataset_2nd'

D = joblib.load(os.path.join(prj_folder, db_folder, f'{args.db_name}.joblib'))
newdb = dict.fromkeys(['train','pred'])
for m in args.mode:
    if m=='FirstUse':
        lg_train = (D['train'].Pusage==0) | (D['train'].Cusage==0)
        lg_pred = (D['pred'].Pusage==0) | (D['pred'].Cusage==0)
    elif m=='UsedBefore':
        lg_train = (D['train'].Pusage > 0) & (D['train'].Cusage > 0)
        lg_pred = (D['pred'].Pusage > 0) & (D['pred'].Cusage > 0)
    elif m=='NoFraudBefore':
        lg_train = (D['train'].Plabel_incidence==0) & (D['train'].Clabel_incidence==0)
        lg_pred = (D['pred'].Plabel_incidence==0) & (D['pred'].Clabel_incidence==0)
    elif m=='AfterFraud':
        lg_train = (D['train'].Plabel_incidence > 0) | (D['train'].Clabel_incidence > 0)
        lg_pred = (D['pred'].Plabel_incidence > 0) | (D['pred'].Clabel_incidence > 0)
    
    newdb['train'] = D['train'].loc[lg_train]
    newdb['pred'] = D['pred'].loc[lg_pred]
    file_name = f'{args.db_name}-{m}.joblib'
    file_path = os.path.join(prj_folder, db_folder, file_name)
    joblib.dump(newdb, file_path, compress=3, protocol=4)
    print('Processed:', file_name)
    print('取出 {:.2f}% 的 training samples'.format(lg_train.sum()/lg_train.size*100))
    print('取出 {:.2f}% 的 testing samples'.format(lg_pred.sum()/lg_pred.size*100))