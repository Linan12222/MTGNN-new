import argparse
import math
import time

import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib

from util import *
from trainer import Optim


def evaluate(data, X, Y, model, batch_size):
    model.eval()
    total_mse = 0
    total_mae = 0
    total_samples = 0
    total_mape = 0
    predict = None
    test = None

    mse_loss = nn.MSELoss(reduction='sum')
    mae_loss = nn.L1Loss(reduction='sum')

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        Y = Y.view(-1, args.seq_out_len, Y.size(-1))  # 修改Y的形状为[batch_size, seq_out_len, num_nodes]
        with torch.no_grad():
            output = model(X)

        output = output.view(-1, args.seq_out_len, Y.size(-1))  # 调整输出的形状

        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output), dim=0)
            test = torch.cat((test, Y), dim=0)

        # 计算整个序列的损失
        total_mse += mse_loss(output, Y).item()
        total_mae += mae_loss(output, Y).item()
        total_samples += Y.numel()

        # 计算MAPE
        mape_loss = torch.mean(torch.abs((Y - output) / Y)) if Y.sum() != 0 else 0
        total_mape += mape_loss.item() * Y.size(0)

    mse = total_mse / total_samples
    rmse = math.sqrt(mse)
    mae = total_mae / total_samples
    mape = total_mape / total_samples

    # 计算R方
    ss_total = ((test - test.mean()) ** 2).sum()
    ss_res = ((test - predict) ** 2).sum()
    r_squared = 1 - ss_res / ss_total

    return rmse, mae, mse, r_squared, mape



def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device)
            tx = X[:, :, id, :]
            ty = Y[:, :, id]  # 修改Y的形状为[batch_size, seq_out_len, num_nodes]
            output = model(tx, id)
            output = output.view(-1, args.seq_out_len, num_sub)

            loss = criterion(output, ty)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
            grad_norm = optim.step()

        if iter % 100 == 0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item() / (output.size(0) * data.m)))
        iter += 1

    return total_loss / n_samples



parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='data/17+18_已处理.txt',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=False)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda',help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=14,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=13,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=24*7,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--layers',type=int,default=5,help='number of layers')

parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.001,help='weight decay rate')

parser.add_argument('--clip',type=int,default=5,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

parser.add_argument('--epochs',type=int,default=1,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')


args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

def main():

    Data = DataLoaderS(args.data, 0.8, 0.1, device, args.seq_out_len, args.seq_in_len)

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device,None,None,dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)
    model = model.to(device)

    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)


    best_val = 20
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)

            # 在验证集上评估模型
            val_rmse, val_mae, val_mse, val_r_squared, val_mape = evaluate(Data, Data.valid[0], Data.valid[1], model,
                                                                           args.batch_size)

            # 在测试集上评估模型
            test_rmse, test_mae, test_mse, test_r_squared, test_mape = evaluate(Data, Data.test[0], Data.test[1], model,
                                                                                args.batch_size)

            print(
                "| Validation Metrics | RMSE: {:5.4f} | MAE: {:5.4f} | MSE: {:5.4f} | R^2: {:5.4f} | MAPE: {:5.4f}".format(
                    val_rmse, val_mae, val_mse, val_r_squared, val_mape))
            print(
                "| Test Metrics       | RMSE: {:5.4f} | MAE: {:5.4f} | MSE: {:5.4f} | R^2: {:5.4f} | MAPE: {:5.4f}".format(
                    test_rmse, test_mae, test_mse, test_r_squared, test_mape))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')



    return val_rmse, val_mae, val_mse, val_r_squared, val_mape, test_rmse, test_mae, test_mse, test_r_squared, test_mape

if __name__ == "__main__":
    for i in range(1):
        val_rmse, val_mae, val_mse, val_r_squared, val_mape, test_rmse, test_mae, test_mse, test_r_squared, test_mape = main()


