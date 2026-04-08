import argparse
import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='iTransformer')
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')
# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--drop_last', type=int, default=False,
                    help='drop_last_batch')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')

#Common Parameters
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)

#Other Model Parameters 
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=24, help='decomposition-kernel')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                    help='whether to use future_temporal_feature; True 1 False 0')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--patch_len', type=int, default=8, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')


# imputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
#Optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--comment', type=str, default='none', help='com')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, timeF, learned]')  
parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                    help='experiemnt name, options:[MTSF, partial_train]')
parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                        'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')
#Standarization
parser.add_argument('--std', type=int, default=0, help='std')
#Cyclenet
parser.add_argument('--cycle', type=int, default=24, help='cycle length')
parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
#PatchTST
parser.add_argument('--dropout', type=float, default=0.15, help='dropout')
parser.add_argument('--patch_num', type=int, default=12, help='patch number, this must be divisible by input sequence length')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--fc_dropout', type=float, default=0.2, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout') 
#iTransformer
parser.add_argument('--d_model2', type=int, default=512, help='dimension of iTransformer backbone')
parser.add_argument('--d_ff2', type=int, default=512, help='dimension of fcn')
parser.add_argument('--dropout2', type=float, default=0.15, help='dropout')
#iTransformer and PatchTST
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
#Loss
parser.add_argument('--criterion', type=str, default='MAE',
                    help='l1 or l2 loss')  
#MLOW
parser.add_argument('--rank', type=int, default=10, help='components')
parser.add_argument('--frequency_level', type=int, default=48, help='frequency_level')
parser.add_argument('--lamb', type=int, default=20, help='regularization_factor')
parser.add_argument('--optimze_H_from_scratch', type=bool, default=True, help='whether to optimze H from scratch, otherwise we provide our pretrained H')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

if args.task_name == 'long_term_forecast':
    Exp = Exp_Long_Term_Forecast
elif args.task_name == 'short_term_forecast':
    Exp = Exp_Short_Term_Forecast
elif args.task_name == 'imputation':
    Exp = Exp_Imputation
elif args.task_name == 'anomaly_detection':
    Exp = Exp_Anomaly_Detection
elif args.task_name == 'classification':
    Exp = Exp_Classification
else:
    Exp = Exp_Long_Term_Forecast

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments

        if args.model=="iTransformer":
            setting = 'new_{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dr{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.comment,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.d_model2,
                args.n_heads,
                args.e_layers,
                args.dropout2,
                args.d_ff2,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
        else:
            setting = 'new_{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dr{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.comment,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.dropout,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    if args.model=="iTransformer":
        setting = 'new_{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dr{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.comment,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_model2,
            args.n_heads,
            args.e_layers,
            args.dropout2,
            args.d_ff2,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
    else:
        setting = 'new_{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dr{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.comment,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.dropout,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
