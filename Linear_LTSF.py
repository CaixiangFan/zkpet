from typing import Tuple, Dict

import torch
import torch.nn as nn
import numpy as np
from data_provider.data_factory import data_provider
from models import DLinear, Linear, NLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from torch import optim

import os
import time, datetime
import warnings
warnings.filterwarnings('ignore')


class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x # [Batch, Output length, Channel]

ROOT_PATH = "./dataset"
DATA_PATH = "electricity.csv"
device = torch.device('cpu')

class Args():
    def __init__(self, model: str, target: str, batch_size: int, 
                 seq_len: int, pred_len: int) -> None:
        self.model = model
        self.data = 'custom'
        self.embed = 'timeF'
        self.train_only = False
        self.freq = 'h'
        self.features = 'S'
        self.num_workers = 0
        self.patience = 3 # early stopping patience
        self.learning_rate = 0.001
        self.train_epochs = 100 # default train epochs
        self.output_attention = False
        self.checkpoints = './checkpoints/'
        self.enc_in = 321
        self.individual = False
        self.lradj = 'type1' # adjust learning rate
        self.root_path = ROOT_PATH
        self.data_path = DATA_PATH
        self.target = target
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len

        self.setting = 'Electricity_{}_{}_{}_{}'.format(
            self.seq_len, self.pred_len, self.model, self.data
        )

def load_data(args) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """Load (training and test set)."""
    trainset, trainloader = data_provider(args, flag='train')
    testset, testloader = data_provider(args, flag='test')
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples

def vali(model, args, vali_data, vali_loader, criterion):

    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            # encoder - decoder

            if 'Linear' in args.model:
                outputs = model(batch_x)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
    total_loss = np.average(total_loss)
    model.train()
    return total_loss

def train(model, args):

    time_now = time.time()
    train_data, train_loader = data_provider(args, flag='train')
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, dummy_input=next(iter(train_loader))[0].float().to(device))
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    
    path = os.path.join(args.checkpoints, args.setting)
    if not os.path.exists(path):
        os.makedirs(path)

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)


            if 'Linear' in args.model:
                    outputs = model(batch_x)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
            # print(outputs.shape,batch_y.shape)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()


            loss.backward()
            model_optim.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        if not args.train_only:
            vali_data, vali_loader = data_provider(args, flag='val')
            test_data, test_loader = data_provider(args, flag='test')
            
            vali_loss = vali(model, args, vali_data, vali_loader, criterion)
            test_loss = vali(model, args, test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, model, path)
        else:
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))
            early_stopping(train_loss, model, path)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(model_optim, epoch + 1, args)

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))

def test(model, args, test=0):
    test_data, test_loader = data_provider(args, flag='test')
    
    if test:
        print('loading model')
        model.load_state_dict(torch.load(os.path.join('./checkpoints/' + args.setting, 'checkpoint.pth')))

    preds = []
    trues = []
    inputx = []
    folder_path = './test_results/' + args.setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            if 'Linear' in args.model:
                    outputs = model(batch_x)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            # print(outputs.shape,batch_y.shape)
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)
            trues.append(true)
            inputx.append(batch_x.detach().cpu().numpy())
            if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputx = np.concatenate(inputx, axis=0)

    # result save
    folder_path = './results/' + args.setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))
    time_now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    f = open("result.txt", 'a')
    f.write(args.setting + "_" + args.target + "_" + time_now + "  \n")
    f.write('mse:{}, mae:{}, mape: {}, rse:{}, corr:{}'.format(mse, mae, mape, rse, corr))
    f.write('\n')
    f.write('\n')
    f.close()

    # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
    np.save(folder_path + 'pred.npy', preds)
    # np.save(folder_path + 'true.npy', trues)
    # np.save(folder_path + 'x.npy', inputx)
    return mse, float(1.0 - mape)
    
def main(args=None):
    
    args = Args(
        model='Linear', target='0', batch_size=16, seq_len=96, pred_len=24
    )

    model_dict = {
        'DLinear': DLinear,
        'NLinear': NLinear,
        'Linear': Linear,
    }
    model = model_dict[args.model].Model(args).float()
    
    print("Training the model...")
    train(model=model, args=args)
    print("Testing the model...")
    test(model=model, args=args)

if __name__ == "__main__":
  main()    
