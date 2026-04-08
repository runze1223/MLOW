from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider 
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv, visual_weights
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import math

warnings.filterwarnings('ignore')


class MLOW_Decomposition(object):
    def __init__(self, args):
        self.args=args
        self.device = self._acquire_device()
        self.context_window=args.seq_len
        self.seq_len=args.seq_len
        
        ts = 1.0/self.context_window
        t = np.arange(0,1,ts)
        t=torch.tensor(t).cuda()
        for i in range(self.context_window//2+1):
            if i==0:
                cos=0.5*torch.cos(2*math.pi*i*t).unsqueeze(0)
                sin=-0.5*torch.sin(2*math.pi*i*t).unsqueeze(0)
            else:
                if i==(self.context_window//2+1):
                    cos=torch.vstack([cos,0.5*torch.cos(2*math.pi*i*t).unsqueeze(0)])
                    sin=torch.vstack([sin,-0.5*torch.sin(2*math.pi*i*t).unsqueeze(0)])
                else:
                    cos=torch.vstack([cos,torch.cos(2*math.pi*i*t).unsqueeze(0)])
                    sin=torch.vstack([sin,-torch.sin(2*math.pi*i*t).unsqueeze(0)])
        self.cos=cos.float().to(self.device) 
        self.sin=sin.float().to(self.device) 


        self.data_path=args.data_path

        folder_path = './NMF/' + self.data_path + '/'
        H=np.load(folder_path +"H.npy")
        self.weight=torch.from_numpy(H.T).float().to(self.device) 


    def decomp(self, data_x):

        data_x=data_x.permute(0,2,1)
            
        norm=data_x.size()[-1]
        frequency=torch.fft.rfft(data_x,axis=-1)
        X_oneside=frequency/(norm)*2

        basis_cos=torch.einsum('bkp,pt->bkpt', X_oneside.real, self.cos[:,-self.seq_len:])
        basis_sin=torch.einsum('bkp,pt->bkpt', X_oneside.imag, self.sin[:,-self.seq_len:])
        x=basis_cos+basis_sin
        average=x[:,:,0,:]
        x=x[:,:,1:-1,:]
        X_oneside=X_oneside[:,:,1:-1]
        amplitude=torch.sqrt(X_oneside.real**2+ X_oneside.imag**2)
        basis=torch.einsum('bkpt,bkp->bkpt', x, (amplitude+1e-10)**(-1))
        X_new_scores=torch.einsum('bkp,ps->bks', amplitude, self.weight)

        basis_p=torch.einsum('bkpt,ps->bkst', basis, self.weight)
        output=2*torch.einsum('bks,bkst->bkst', X_new_scores, basis_p)
        residual=data_x[:,:,-self.seq_len:]-torch.sum(output,axis=-2)-average
        x=torch.cat([average.unsqueeze(-2),output,residual.unsqueeze(-2)],axis=-2)
        x=x.permute(0,2,3,1).detach()

        return x


    def _acquire_device(self):
        if self.args.use_gpu:
            import platform
            if platform.system() == 'Darwin':
                device = torch.device('mps')
                print('Use MPS')
                return device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            if self.args.use_multi_gpu:
                print('Use GPU: cuda{}'.format(self.args.device_ids))
            else:
                print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device




class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        print(model)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


    def _get_MLOW(self):
        MLOW=MLOW_Decomposition(self.args)
        return MLOW


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.criterion == 'MAE':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)



                if self.args.embed == 'timestamp':
                    batch_x_mark = batch_x_mark.long().to(self.device)
                    # batch_y_mark = batch_y_mark.long().to(self.device)
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    # batch_y_mark = batch_y_mark.float().to(self.device)

                batch_y_mark = batch_y_mark.int().to(self.device)

                # if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                #     batch_x_mark = None
                #     batch_y_mark = None

                batch_x=self.MLOW.decomp(batch_x)
                batch_x_mark=self.MLOW.decomp(batch_x_mark)

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        self.MLOW=self._get_MLOW()



        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        



        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)




                if self.args.embed == 'timestamp':
                    batch_x_mark = batch_x_mark.long().to(self.device)
                    # batch_y_mark = batch_y_mark.long().to(self.device)
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    # batch_y_mark = batch_y_mark.float().to(self.device)

                batch_y_mark = batch_y_mark.int().to(self.device)
                
                batch_x=self.MLOW.decomp(batch_x)
                batch_x_mark=self.MLOW.decomp(batch_x_mark)

                # if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                #     batch_x_mark = None
                #     batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None


                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                    loss = criterion(outputs, batch_y)#+0.01*torch.mean(torch.nn.functional.cosine_similarity(self.model.linear.weight[:,:96], self.model.linear.weight[:,96:], dim=-1)) #+1000*add_loss

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST' or self.args.lradj == 'type3':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

                    
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj == 'TST':
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        self.MLOW=self._get_MLOW()
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        checkpoints_path = './checkpoints/' + setting + '/'
        preds = []
        trues = []
        inputx = []


        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)



        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)



                if self.args.embed == 'timestamp':
                    batch_x_mark = batch_x_mark.long().to(self.device)
                    # batch_y_mark = batch_y_mark.long().to(self.device)
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    # batch_y_mark = batch_y_mark.float().to(self.device)

                batch_y_mark = batch_y_mark.int().to(self.device)
                # if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                #     batch_x_mark = None
                #     batch_y_mark = None

                batch_x=self.MLOW.decomp(batch_x)
                batch_x_mark=self.MLOW.decomp(batch_x_mark)

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:

                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)



                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # outputs = outputs.detach()
                # batch_y = batch_y.detach()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))


        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)


        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        # if self.args.data == 'PEMS':
        #     f.write('mae:{}, mape:{}, rmse:{}'.format(mae, mape, rmse))
        # else:
        #     f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'inputx.npy', inputx)
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)


        return
