import numpy as np
import pandas as pd

import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, BCELoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

class FraudDataset(Dataset):
    def __init__(self, X_df, y_df):
        self.X = X_df.to_numpy().astype(np.float32)
        self.y = y_df.to_numpy().astype(np.float32)
      
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


get_model = lambda : Sequential(
    Linear(4,8),
    ReLU(),
    Linear(8,1),
    Sigmoid()
)


def train_model(model, dataloader, epochs):
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    cum_loss = 0.0

    for epoch in range(epochs):
        cum_loss = 0.0
        for i, (X,y) in enumerate(dataloader):
            pred_y = model(X)
            loss = criterion(pred_y, y)
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
        cum_loss /= len(dataloader)

    return model, cum_loss


def test_model(model, dataloader):
    correct_preds = 0.0
    total_preds = 0.0

    preds = []
    pred_conf = []

    for i, (X,y) in enumerate(dataloader):
        res = model(X)

        preds.extend(torch.flatten(res > 0.5).tolist())
        pred_conf.extend(torch.flatten(res).tolist())

        y_pred = (res > 0.5).type(torch.FloatTensor)
        correct_preds += torch.sum(y_pred == y).item()
        total_preds += len(y)

    return correct_preds/total_preds, preds, pred_conf


def process_df(csv_path):
    df = pd.read_csv(csv_path, index_col=0)

    df['is_int'] = df['location'].map(lambda x: (False if x=='US' else True))
    df['is_withdrawal'] = df['transaction_type'].map(lambda x: (True if x=='withdrawal' else False))
    df['is_payment'] = df['transaction_type'].map(lambda x: (True if x=='payment' else False))
    df['transaction_amount'] /= 1000
    df = df.drop(['balance', 'transaction_id', 'account_id', 'transaction_time', 'transaction_type', 'location'], axis=1)
    
    X = df[['transaction_amount', 'is_int', 'is_withdrawal', 'is_payment']]
    y = df[['is_fraud']]

    ds = FraudDataset(X,y)
    dataloader = DataLoader(ds, batch_size=128, shuffle=False)

    return dataloader




from stadle import AdminAgent, BaseModelConvFormat, BasicClient
from stadle.lib.entity.model import BaseModel

from time import time

fd_bm = BaseModel("Fraud-Detection-Model", get_model(), BaseModelConvFormat.pytorch_format)

def on_start_fl_click(agg_addr):
    global dataloader

    # Upload base model
    admin_agent = AdminAgent(aggregator_ip_address=agg_addr, base_model=fd_bm)
    admin_agent.preload()
    admin_agent.initialize()

    # Start FL
    stadle_client = BasicClient(agent_name=f'agent_{time()}')

    model = get_model()
    stadle_client.set_bm_obj(model)

    fl_progress.style={'bar_color': 'blue', 'description_width': 'initial'}

    for rnd in range(round_lim.value):
        fl_progress.description='Training'
        fl_progress.value=rnd+1
        model, loss = train_model(model, dataloader, 2)
        fl_progress.description='Aggregating'
        stadle_client.send_trained_model(model, perf_values={'loss_training':loss})
        fl_sd = stadle_client.wait_for_sg_model().state_dict()
        model.load_state_dict(fl_sd)

    fl_progress.description='FL complete'
    fl_progress.style={'bar_color': 'green', 'description_width': 'initial'}

    trained_models.append(('FL_model', model))
    inf_model_select.options = [t for t in trained_models]