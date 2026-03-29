import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from sktime.datasets import load_from_tsfile_to_dataframe
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import Normalizer, interpolate_missing
import warnings
import math

from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time

warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',levels=168,rank=10,lam=20, optimze_H_from_scratch=True, cycle=24):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.optimze_H_from_scratch=optimze_H_from_scratch

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.rank=rank
        self.lamb=lam

        self.freq_level=levels
        self.context_window=2*levels
    
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

        self.cos=cos
        self.sin=sin
        self.cycle = cycle


        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.context_window, 12 * 30 * 24 + 4 * 30 * 24 - self.context_window]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)


        data_x = data[border1:border2]
        data_y = data[border1:border2]

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


        if self.set_type==0:
            if self.optimze_H_from_scratch:
                L= data_x.shape[0]
                window = self.context_window
                stride = 1

                num_segments = (L - window) // stride + 1

                segments = np.array([
                    data_x[i : i + window]
                    for i in range(0, L - window + 1, stride)
                ])

                fft_segments = np.fft.rfft(segments, axis=1)
                fft_segments=fft_segments/(window)*2
                
                print("fft_segments.shape =", fft_segments.shape) 
                fft_segments=np.sqrt(fft_segments.real**2+ fft_segments.imag**2)
                log_amp = fft_segments 
                x = np.transpose(log_amp[:, 1:-1, :], (0, 2, 1))  
                new_x = x.reshape(-1, x.shape[-1]) 
                mean=np.mean(new_x,axis=0, keepdims=True)
                std=np.std(new_x,axis=0, keepdims=True)
                folder_path = './NMF/' + self.data_path + '/'

                k = self.rank
                pca = PCA(n_components=k)
                pca.fit(new_x)
                H = (pca.components_.T+np.abs(pca.components_.T))/2
                mask = ( H== 0)
                rand_array = np.random.uniform(low=0, high=0.1, size=H.shape)
                noise=mask*rand_array*np.transpose(std, (1, 0))
                H=H+noise
                W,H=hyperplane_nmf(new_x,H.T,self.lamb)
                folder_path = './NMF/' + self.data_path + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                np.save(folder_path +"H.npy", H)
                self.weight=torch.from_numpy(H.T).cuda()

            else:
                folder_path = './NMF/' + self.data_path + '/'
                H=np.load(folder_path +"H.npy")
                self.weight=torch.from_numpy(H.T).cuda()                

        if self.set_type!=0:
            folder_path = './NMF/' + self.data_path + '/'
            H=np.load(folder_path +"H.npy")
            self.weight=torch.from_numpy(H.T).cuda()

        L= data_x.shape[0]
        window = self.context_window+self.pred_len
        stride = 1

        num_segments = (L - window) // stride + 1

        segments = np.array([
            data_x[i : i + window]
            for i in range(0, L - window + 1, stride)
        ])

        data_x_total=segments[:,:-self.pred_len,:]
        self.data_y=segments[:,-self.pred_len:,:]

        timestamp = np.array([
            data_stamp[i : i + window]
            for i in range(0, L - window + 1, stride)
        ])

        self.data_stamp_x=  timestamp[:,:-self.pred_len,:] 
        self.data_stamp_y=  timestamp[:,-self.pred_len:,:]


        a,_,d= data_x_total.shape

        self.data_x=torch.zeros(a,self.rank+2,self.seq_len,d)
        data_x_total=torch.from_numpy(data_x_total).cuda()

        self.data_stamp_x=torch.zeros(a,self.rank+2,self.seq_len,4)
        data_stamp_total=torch.from_numpy(timestamp[:,:-self.pred_len,:]).cuda()


        if a % 100==0:
            k=a//100
        else:
            k=a//100+1

        for i in range(k):
            if i ==k-1:
                data_x=data_x_total[100*i:,:,:].permute(0,2,1)
            else:
                data_x=data_x_total[100*i:100*(i+1),:,:].permute(0,2,1)

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
            
            # x=w-mean-torch.sum(output,axis=-2)-average
            residual=data_x[:,:,-self.seq_len:]-torch.sum(output,axis=-2)-average

            x=torch.cat([average.unsqueeze(-2),output,residual.unsqueeze(-2)],axis=-2)
            x=x.permute(0,2,3,1)

            if i ==k-1:
                self.data_x[100*i:,:,:,:]=x
            else:
                self.data_x[100*i:100*(i+1),:,:,:]=x

            if i ==k-1:
                data_x=data_stamp_total[100*i:,:,:].permute(0,2,1)
            else:
                data_x=data_stamp_total[100*i:100*(i+1),:,:].permute(0,2,1)

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
            x=x.permute(0,2,3,1)

            if i ==k-1:
                self.data_stamp_x[100*i:,:,:,:]=x
            else:
                self.data_stamp_x[100*i:100*(i+1),:,:,:]=x

    def __getitem__(self, index):
        seq_x=self.data_x[index]
        seq_y=self.data_y[index]
        
        seq_x_mark = self.data_stamp_x[index]
        seq_y_mark = self.data_stamp_y[index]

        cycle_index = torch.tensor(self.cycle_index[index+ self.context_window])

        return seq_x, seq_y, seq_x_mark, cycle_index

    def __len__(self):
        return len(self.data_x) 

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', levels=168 ,rank=10,lam=20, optimze_H_from_scratch=True, cycle=24):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.rank=rank
        self.lamb=lam
        self.freq_levels=levels
        self.context_window=2*levels

        self.optimze_H_from_scratch=optimze_H_from_scratch



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

        self.cos=cos
        self.sin=sin
        self.cycle = cycle


        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.context_window, len(df_raw) - num_test - self.context_window]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        data_x = data[border1:border2]

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

        data_stamp = data_stamp

        if self.set_type==0:
            if self.optimze_H_from_scratch:
                L= data_x.shape[0]
                window = self.context_window
                stride = 1
                num_segments = (L - window) // stride + 1

                segments = np.array([
                    data_x[i : i + window]
                    for i in range(0, L - window + 1, stride)
                ])

                fft_segments = np.fft.rfft(segments, axis=1)
                fft_segments=fft_segments/(window)*2
                print("fft_segments.shape =", fft_segments.shape)  
                fft_segments=np.sqrt(fft_segments.real**2+ fft_segments.imag**2)

                log_amp = fft_segments 

                x = np.transpose(log_amp[:, 1:-1, :], (0, 2, 1))  
                new_x = x.reshape(-1, x.shape[-1])  

                mean=np.mean(new_x,axis=0, keepdims=True)
                std=np.std(new_x,axis=0, keepdims=True)
                folder_path = './NMF/' + self.data_path + '/'

                k = self.rank
                pca = PCA(n_components=k)
                pca.fit(new_x)

                H = (pca.components_.T+np.abs(pca.components_.T))/2
                mask = ( H== 0)
                rand_array = np.random.uniform(low=0, high=0.1, size=H.shape)
                noise=mask*rand_array*np.transpose(std, (1, 0))
                H=H+noise
                W,H=hyperplane_nmf(new_x,H.T,self.lamb)
                
                folder_path = './NMF/' + self.data_path + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                np.save(folder_path +"H.npy", H)
                self.weight=torch.from_numpy(H.T).cuda()

            else:
                folder_path = './NMF/' + self.data_path + '/'
                H=np.load(folder_path +"H.npy")
                self.weight=torch.from_numpy(H.T).cuda()                



        if self.set_type!=0:
            folder_path = './NMF/' + self.data_path + '/'
            H=np.load(folder_path +"H.npy")
            self.weight=torch.from_numpy(H.T).cuda()

        L= data_x.shape[0]
        window = self.context_window+self.pred_len
        stride = 1

        num_segments = (L - window) // stride + 1

        segments = np.array([
            data_x[i : i + window]
            for i in range(0, L - window + 1, stride)
        ])

        data_x_total=segments[:,:-self.pred_len,:]
        self.data_y=segments[:,-self.pred_len:,:]

        timestamp = np.array([
            data_stamp[i : i + window]
            for i in range(0, L - window + 1, stride)
        ])

        self.data_stamp_x=  timestamp[:,:-self.pred_len,:] 
        self.data_stamp_y=  timestamp[:,-self.pred_len:,:]


        a,_,d= data_x_total.shape

        self.data_x=torch.zeros(a,self.rank+2,self.seq_len,d)
        data_x_total=torch.from_numpy(data_x_total).cuda()

        self.data_stamp_x=torch.zeros(a,self.rank+2,self.seq_len,4)
        data_stamp_total=torch.from_numpy(timestamp[:,:-self.pred_len,:]).cuda()

        k=a//30+1

        if a % 30==0:
            k=a//30
        else:
            k=a//30+1

        for i in range(k):
            if i ==k-1:
                data_x=data_x_total[30*i:,:,:].permute(0,2,1)
            else:
                data_x=data_x_total[30*i:30*(i+1),:,:].permute(0,2,1)

            
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
            x=x.permute(0,2,3,1)

            if i ==k-1:
                self.data_x[30*i:,:,:,:]=x
            else:
                self.data_x[30*i:30*(i+1),:,:,:]=x


            if i ==k-1:
                data_x=data_stamp_total[30*i:,:,:].permute(0,2,1)
            else:
                data_x=data_stamp_total[30*i:30*(i+1),:,:].permute(0,2,1)

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
            x=x.permute(0,2,3,1)

            if i ==k-1:
                self.data_stamp_x[30*i:,:,:,:]=x
            else:
                self.data_stamp_x[30*i:30*(i+1),:,:,:]=x
    
    def __getitem__(self, index):
        seq_x=self.data_x[index]
        seq_y=self.data_y[index]
        seq_x_mark = self.data_stamp_x[index]
        seq_y_mark = self.data_stamp_y[index]

        cycle_index = torch.tensor(self.cycle_index[index+ self.context_window])

        return seq_x, seq_y, seq_x_mark, cycle_index

    def __len__(self):
        return len(self.data_x) 

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', levels=168, rank=10,lam=20, optimze_H_from_scratch=True, cycle=24):
        # size [seq_len, label_len, pred_len]
        # info

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.rank=rank
        self.lamb=lam
        self.contex_window=2*levels

        self.optimze_H_from_scratch=optimze_H_from_scratch

        ts = 1.0/self.contex_window
        t = np.arange(0,1,ts)
        t=torch.tensor(t).cuda()
        for i in range(self.contex_window//2+1):
            if i==0:
                cos=0.5*torch.cos(2*math.pi*i*t).unsqueeze(0)
                sin=-0.5*torch.sin(2*math.pi*i*t).unsqueeze(0)
            else:
                if i==(self.contex_window//2+1):
                    cos=torch.vstack([cos,0.5*torch.cos(2*math.pi*i*t).unsqueeze(0)])
                    sin=torch.vstack([sin,-0.5*torch.sin(2*math.pi*i*t).unsqueeze(0)])
                else:
                    cos=torch.vstack([cos,torch.cos(2*math.pi*i*t).unsqueeze(0)])
                    sin=torch.vstack([sin,-torch.sin(2*math.pi*i*t).unsqueeze(0)])

        self.cos=cos
        self.sin=sin
        self.cycle = cycle


        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        print('data file:', data_file)
        data = np.load(data_file, allow_pickle=True)

        data = data['data'][:, :, 0]
        index=np.arange(len(data))
        index1=index%288
        index=(index-1)/288-0.5

        cycle_index = (np.arange(len(data)) % self.cycle)
 
        train_ratio = 0.7
        valid_ratio = 0.1
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data))-self.contex_window:int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data))-self.contex_window:]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        train_index = index[:int(train_ratio * len(index))]
        valid_index = index[int(train_ratio * len(index))-self.contex_window:int((train_ratio + valid_ratio) * len(index))]
        test_index = index[int((train_ratio + valid_ratio) * len(index))-self.contex_window:]
        total_index = [train_index, valid_index, test_index]
        index = total_index[self.set_type]

        train_cycle = cycle_index[:int(train_ratio * len(cycle_index))]
        valid_cycle = cycle_index[int(train_ratio * len(cycle_index))-self.contex_window:int((train_ratio + valid_ratio) * len(cycle_index))]
        test_cycle = cycle_index[int((train_ratio + valid_ratio) * len(cycle_index))-self.contex_window:]
        total_cycle = [train_cycle, valid_cycle, test_cycle]
        self.cycle_index = total_cycle[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)


        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        data_x = df
        # data_y = df_y

        df2 = pd.DataFrame(index)
        data_stamp = df2.fillna(method='ffill', limit=len(df2)).fillna(method='bfill', limit=len(df2)).values


        if self.set_type==0:

            if self.optimze_H_from_scratch:
                L= data_x.shape[0]
                window = self.contex_window
                stride = 1

                num_segments = (L - window) // stride + 1

                segments = np.array([
                    data_x[i : i + window]
                    for i in range(0, L - window + 1, stride)
                ])

                fft_segments = np.fft.rfft(segments, axis=1)
                fft_segments=fft_segments/(window)*2
                
                print("fft_segments.shape =", fft_segments.shape)  
                fft_segments=np.sqrt(fft_segments.real**2+ fft_segments.imag**2)

                log_amp = fft_segments 

                pca_models = []

                # np.save("frequency.npy", fft_segments[:1000,:,:])

                x = np.transpose(log_amp[:, 1:-1, :], (0, 2, 1))  # 变为 (A, C, B)
                new_x = x.reshape(-1, x.shape[-1])  # 合并前两维，保持最后一维B不变

                mean=np.mean(new_x,axis=0, keepdims=True)
                std=np.std(new_x,axis=0, keepdims=True)

                folder_path = './NMF/' + self.data_path + '/'

                k = self.rank
                pca = PCA(n_components=k)
                pca.fit(new_x)

                H = (pca.components_.T+np.abs(pca.components_.T))/2
                mask = ( H== 0)
                rand_array = np.random.uniform(low=0, high=0.1, size=H.shape)
                noise=mask*rand_array*np.transpose(std, (1, 0))
                H=H+noise

                if self.contex_window<200:
                    epoch=600
                else:
                    epoch=1000

                W,H=hyperplane_nmf(new_x,H.T,self.lamb,epoch)

                folder_path = './NMF/' + self.data_path + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                np.save(folder_path +"H.npy", H)

                self.weight=torch.from_numpy(H.T).cuda()

            else:
                folder_path = './NMF/' + self.data_path + '/'
                H=np.load(folder_path +"H.npy")
                self.weight=torch.from_numpy(H.T).cuda()


        if self.set_type!=0:
            folder_path = './NMF/' + self.data_path + '/'
            H=np.load(folder_path +"H.npy")
            self.weight=torch.from_numpy(H.T).cuda()


        L= data_x.shape[0]
        window = self.contex_window+self.pred_len
        stride = 1

        num_segments = (L - window) // stride + 1

        segments = np.array([
            data_x[i : i + window]
            for i in range(0, L - window + 1, stride)
        ])

        data_x_total=segments[:,:-self.pred_len,:]
        self.data_y=segments[:,-self.pred_len:,:]

        timestamp = np.array([
            data_stamp[i : i + window]
            for i in range(0, L - window + 1, stride)
        ])

        # self.data_stamp_x=  timestamp[:,:-self.pred_len,:] 
        self.data_stamp_y=  timestamp[:,-self.pred_len:,:]


        a,_,d= data_x_total.shape

        self.data_x=torch.zeros(a,self.rank+2,self.seq_len,d)
        data_x_total=torch.from_numpy(data_x_total).cuda()

        self.data_stamp_x=torch.zeros(a,self.rank+2,self.seq_len,1)
        data_stamp_total=torch.from_numpy(timestamp[:,:-self.pred_len,:]).cuda()


        if a % 30==0:
            k=a//30
        else:
            k=a//30+1
        
        for i in range(k):
            if i ==k-1:
                data_x=data_x_total[30*i:,:,:].permute(0,2,1)
            else:
                data_x=data_x_total[30*i:30*(i+1),:,:].permute(0,2,1)

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
            x=x.permute(0,2,3,1)

            if i ==k-1:
                self.data_x[30*i:,:,:,:]=x
            else:
                self.data_x[30*i:30*(i+1),:,:,:]=x

            if i ==k-1:
                data_x=data_stamp_total[30*i:,:,:].permute(0,2,1)
            else:
                data_x=data_stamp_total[30*i:30*(i+1),:,:].permute(0,2,1)

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
            x=x.permute(0,2,3,1)

            if i ==k-1:
                self.data_stamp_x[30*i:,:,:,:]=x
            else:
                self.data_stamp_x[30*i:30*(i+1),:,:,:]=x

    def __getitem__(self, index):
        if self.set_type == 2: 
            index = index * 12
        else:
            index = index

        seq_x=self.data_x[index]
        seq_y=self.data_y[index]
        
        seq_x_mark = self.data_stamp_x[index]
        seq_y_mark = self.data_stamp_y[index]

        cycle_index = torch.tensor(self.cycle_index[index+ self.contex_window])

        return seq_x, seq_y, seq_x_mark, cycle_index

    def __len__(self):
        if self.set_type == 2:  
            return (len(self.data_x) ) // 12
        else:
            return len(self.data_x) 

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def hyperplane_nmf(X, H, lam=20 ,max_iter=1000):
    X=torch.from_numpy(X).cuda()

    H=torch.from_numpy(H).cuda()

    W = torch.einsum('bk,kp->bp', X, H.T)

    k=0

    for it in range(max_iter):

        postive,negative=cosine_similarity_regularizer(H)


        alpha=1

        numerator=torch.einsum('bk,kp->bp', W.T, X)

        WW=torch.einsum('bk,kp->bp', W.T , W)

        denominator=torch.einsum('bk,kp->bp', WW , H)


        update = (  numerator +lam*negative+ 1e-9) / (denominator + lam*postive+ 1e-9)


        H=  (1 - alpha) * H + alpha * (H * update)

        W = torch.einsum('bk,kp->bp', X, H.T)

        loss = torch.sum(torch.abs( (X -  torch.einsum('bk,kp->bp', W , H)))) / torch.sum(torch.abs(X)) *100
        # More memory for using the this.
        # loss= torch.linalg.norm(X - torch.einsum('bk,kp->bp', W , H), 'fro')**2 / torch.linalg.norm(X, 'fro')**2


        if it % 20 == 0:
            print(f"Iter {it}: loss={loss:.6f}")

        print("loss1",loss)

        if it > 0 and loss- prev_loss < 0:
            k=k+1
            if k>2 :
                break       
        prev_loss = loss
        k=0


    H=H.detach().cpu().numpy()
    W=W.detach().cpu().numpy()

    return W, H


# def cosine_similarity_regularizer(H, gamma=1.0, eps=1e-12):
#     """
#     Row-wise cosine similarity regularizer to minimize similarity (or flip sign for minimizing similarity).
    
#     H: (n_rows, dim) matrix, rows are vectors
#     gamma: scaling factor
#     eps: small value to prevent division by zero

#     Returns:
#         grad_pos: (n_rows, dim) numerator contribution (∇+)
#         grad_neg: (n_rows, dim) denominator contribution (∇-)
#     """
#     n_rows, dim = H.shape

#     # ---------------------
#     # row norms
#     row_norms = torch.norm(H, dim=1) + eps  # (n_rows,)
#     norm_matrix = row_norms[:, None]        # (n_rows, 1) for broadcasting

#     # ---------------------
#     # normalized rows
#     A = H / norm_matrix                     # (n_rows, dim)

#     # ---------------------
#     # grad positive term ∇+
#     sum_all = A.sum(dim=0, keepdim=True).repeat(n_rows, 1)  # sum over all rows
#     grad_pos = (sum_all - A) / norm_matrix                   # (n_rows, dim)
#     grad_pos *= gamma

#     # ---------------------
#     # grad negative term ∇-
#     M = H @ A.T                              # (n_rows, n_rows) row-wise dot products
#     sum_inner_excl = torch.sum(M, dim=1) - torch.diag(M)  # sum over all rows except self
#     grad_neg = H * (sum_inner_excl / (row_norms**3 + eps))[:, None]  # (n_rows, dim)
#     grad_neg *= gamma

#     return grad_pos, grad_neg



def cosine_similarity_regularizer(H, gamma=1.0, eps=1e-12):
    """
    Compute gamma * grad_pos and gamma * grad_neg for column-wise
    cosine similarity regularizer (minimize similarity) for WH update.

    H: (r, n) matrix, columns are vectors
    gamma: scaling factor
    eps: small value to prevent division by zero

    Returns:
        grad_pos: (r, n) numerator contribution for WH
        grad_neg: (r, n) denominator contribution for WH
    """
    r, n = H.shape

    # column norms
    # np.linalg.norm()
    col_norms = torch.norm(H, dim=0) + eps  # shape (n,)
    norm_matrix = col_norms[None, :]            # shape (1, n)

    # normalized columns
    A = H / norm_matrix  # shape (r, n)

    # grad positive term (sum_{q != p} h_q / ||h_q||)
    sum_all = A.sum(axis=1, keepdims=True)      # (r,1)
    grad_pos = (sum_all - A) / norm_matrix      # (r,n)
    grad_pos *= gamma

    # grad negative term (sum_{q != p} (h_p^T h_q)/||h_p||^3 / ||h_q|| * h_p)

    M = H.T@A 
    M = torch.einsum('bk,kp->bp', H.T , A)                             # (n,n)
    sum_inner_excl = M.sum(axis=1) - torch.diag(M) # (n,)
    grad_neg = H * (sum_inner_excl / (col_norms**3 + eps))[None, :]
    grad_neg *= gamma

    return grad_pos, grad_neg



class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)


        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w        


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', levels=168,rank=10,lam=20, optimze_H_from_scratch=True,cycle=24):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.optimze_H_from_scratch=optimze_H_from_scratch
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.rank=rank
        self.lamb=lam


        self.freq_level=levels
        self.context_window=2*levels

        


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

        self.cos=cos
        self.sin=sin

        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.context_window, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.context_window]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)


        data_x = data[border1:border2]
        data_y = data[border1:border2]

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


        if self.set_type==0:
            if self.optimze_H_from_scratch:
                L= data_x.shape[0]
                window = self.context_window
                stride = 1

                num_segments = (L - window) // stride + 1

                segments = np.array([
                    data_x[i : i + window]
                    for i in range(0, L - window + 1, stride)
                ])

                fft_segments = np.fft.rfft(segments, axis=1)
                fft_segments=fft_segments/(window)*2
                
                print("fft_segments.shape =", fft_segments.shape) 
                fft_segments=np.sqrt(fft_segments.real**2+ fft_segments.imag**2)
                log_amp = fft_segments 
                x = np.transpose(log_amp[:, 1:-1, :], (0, 2, 1))  
                new_x = x.reshape(-1, x.shape[-1]) 
                mean=np.mean(new_x,axis=0, keepdims=True)
                std=np.std(new_x,axis=0, keepdims=True)
                folder_path = './NMF/' + self.data_path + '/'

                k = self.rank
                pca = PCA(n_components=k)
                pca.fit(new_x)
                H = (pca.components_.T+np.abs(pca.components_.T))/2
                mask = ( H== 0)
                rand_array = np.random.uniform(low=0, high=0.1, size=H.shape)
                noise=mask*rand_array*np.transpose(std, (1, 0))
                H=H+noise
                W,H=hyperplane_nmf(new_x,H.T,self.lamb)
                folder_path = './NMF/' + self.data_path + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                np.save(folder_path +"H.npy", H)
                self.weight=torch.from_numpy(H.T).cuda()

            else:
                folder_path = './NMF/' + self.data_path + '/'
                H=np.load(folder_path +"H.npy")
                self.weight=torch.from_numpy(H.T).cuda()                


        if self.set_type!=0:
            folder_path = './NMF/' + self.data_path + '/'
            H=np.load(folder_path +"H.npy")
            self.weight=torch.from_numpy(H.T).cuda()

        L= data_x.shape[0]
        window = self.context_window+self.pred_len
        stride = 1

        num_segments = (L - window) // stride + 1

        segments = np.array([
            data_x[i : i + window]
            for i in range(0, L - window + 1, stride)
        ])

        data_x_total=segments[:,:-self.pred_len,:]
        self.data_y=segments[:,-self.pred_len:,:]

        timestamp = np.array([
            data_stamp[i : i + window]
            for i in range(0, L - window + 1, stride)
        ])

        self.data_stamp_x=  timestamp[:,:-self.pred_len,:] 
        self.data_stamp_y=  timestamp[:,-self.pred_len:,:]


        a,_,d= data_x_total.shape

        self.data_x=torch.zeros(a,self.rank+2,self.seq_len,d)
        data_x_total=torch.from_numpy(data_x_total).cuda()

        self.data_stamp_x=torch.zeros(a,self.rank+2,self.seq_len,4)
        data_stamp_total=torch.from_numpy(timestamp[:,:-self.pred_len,:]).cuda()


        if a % 100==0:
            k=a//100
        else:
            k=a//100+1

        for i in range(k):
            if i ==k-1:
                data_x=data_x_total[100*i:,:,:].permute(0,2,1)
            else:
                data_x=data_x_total[100*i:100*(i+1),:,:].permute(0,2,1)

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
            
            # x=w-mean-torch.sum(output,axis=-2)-average
            residual=data_x[:,:,-self.seq_len:]-torch.sum(output,axis=-2)-average

            x=torch.cat([average.unsqueeze(-2),output,residual.unsqueeze(-2)],axis=-2)
            x=x.permute(0,2,3,1)

            if i ==k-1:
                self.data_x[100*i:,:,:,:]=x
            else:
                self.data_x[100*i:100*(i+1),:,:,:]=x

            if i ==k-1:
                data_x=data_stamp_total[100*i:,:,:].permute(0,2,1)
            else:
                data_x=data_stamp_total[100*i:100*(i+1),:,:].permute(0,2,1)

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
            x=x.permute(0,2,3,1)

            if i ==k-1:
                self.data_stamp_x[100*i:,:,:,:]=x
            else:
                self.data_stamp_x[100*i:100*(i+1),:,:,:]=x

    def __getitem__(self, index):

        seq_x=self.data_x[index]
        seq_y=self.data_y[index]
        
        seq_x_mark = self.data_stamp_x[index]
        seq_y_mark = self.data_stamp_y[index]

        cycle_index = torch.tensor(self.cycle_index[index+ self.context_window])

        return seq_x, seq_y, seq_x_mark, cycle_index

    def __len__(self):
        return len(self.data_x) 

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        self.val = test_data
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for dataset included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads dataset from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                   replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
               torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)






class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # initn
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')  # 去除文本中的换行符
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_x_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




