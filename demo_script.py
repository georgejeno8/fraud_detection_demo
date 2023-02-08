import streamlit as st

import os
from time import sleep, time
from datetime import datetime

import pandas as pd
import numpy as np

import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, BCELoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# if (not os.path.exists('/app/fraud_detection_demo/logs')):
#     os.makedirs('/app/fraud_detection_demo/logs')

from stadle import AdminAgent, BaseModelConvFormat, BasicClient
from stadle.lib.entity.model import BaseModel

from model import *

# import nest_asyncio
# nest_asyncio.apply()

st.title('XYZ Bank Fraud Detection App')

rows = st.tabs(['Load Data', 'Local Training', 'STADLE Training', 'Run Fraud Detection'])

def get_data_files():
    return [fn for fn in os.listdir() if ((len(fn) > 4) and (fn[-4:] == '.csv'))]

def sd2str(sd):
    return '\n'.join([f'[{n}]\n{str(p)}\n' for n,p in sd.items()])

def get_timestamp():
    t = time()
    dt = datetime.fromtimestamp(t)
    return dt.strftime('%y-%m-%d_[%H:%M:%S]')

if ('clear_files' not in st.session_state):
    for file in get_data_files():
        if (file != 'XYZ_international_transactions.csv' and file != 'XYZ_US_transactions.csv'):
            os.remove(file)
    st.session_state.clear_files = True

upload_counter = 0

data_files = get_data_files()

data_file = ''

def on_file_upload():
    global data_files
    global data_file
    global rows
    global upload_counter

    data_files = get_data_files()
    print(data_files)

    upload_counter += 1

    with rows[0]:
        with file_uploader_container:
            with st.container():
                data_file = st.selectbox(
                    'Select data:',
                    data_files,
                    key=upload_counter
                )

                with open(data_file, 'rb') as f:
                    dl_btn = st.download_button(
                        'Download data .csv file',
                        f,
                        file_name=data_file,
                        key=f'dbk_{upload_counter}'
                    )

file_uploader_container = st.empty()

with rows[0]:
    with file_uploader_container:
        with st.container():
            data_file = st.selectbox(
                'Select data:',
                data_files,
                key=upload_counter
            )
            
            with open(data_file, 'rb') as f:
                dl_btn = st.download_button(
                    'Download data .csv file',
                    f,
                    file_name=data_file,
                    key=f'dbk_{upload_counter}'
                )

    df1 = pd.read_csv(data_file, index_col=0).style.format(subset=['balance', 'transaction_amount'], formatter='{:.2f}')
    df1 = df1.format(subset=['transaction_time'], formatter='{:.0f}')
    st.write(df1)

    uploaded_file = st.file_uploader(
        'Upload data file',
        help='Upload dataframe .csv file to use for training and validation',
        type=['csv'],
        on_change=on_file_upload
    )

    if uploaded_file is not None:
        with open(uploaded_file.name, 'wb') as f:
            f.write(uploaded_file.getbuffer())

lt_container = None
ft_container = None

num_epoch_input = None

if ('lt_text' not in st.session_state):
    st.session_state.lt_text = 'Waiting for training to start...'

if ('ft_text' not in st.session_state):
    st.session_state.ft_text = ''

if ('models' not in st.session_state):
    st.session_state.models = {}

def on_train_click():
    global lt_container
    global num_epoch_input
    global data_file
    global rows
    global models

    dataloader = process_df(data_file)

    model = get_model()

    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epoch_input):
        model, loss, optimizer = train_model(model, dataloader, optimizer, 1)
        lt_container.text(f'Epoch {epoch+1}/{num_epoch_input}' + '\n' + sd2str(model.state_dict()))

    st.session_state.lt_text = 'Training completed.\n' + sd2str(model.state_dict())
    lt_container.text(st.session_state.lt_text)
    
    st.session_state.models['local_model_' + get_timestamp()] = model

with rows[1]:
    print(st.session_state.lt_text)
    num_epoch_input = st.number_input('Number of epochs:', 1, 256)

    train_btn = st.button(
        'Train Model',
        on_click=on_train_click
    )

    lt_container = st.empty()
    lt_container.text(st.session_state.lt_text)

def on_up_click(agg_addr):
    with rows[2]:
        with st.spinner('Uploading model metadata to STADLE...'):
            admin_agent = AdminAgent(aggregator_ip_address=agg_addr, base_model=fd_bm)
            admin_agent.preload()
            admin_agent.initialize()

        st.success('Uploaded metadata successfully!')

def on_fl_click(agg_addr, rounds):
    global data_file

    dataloader = process_df(data_file)

    with rows[2]:
        ft_container.text('Starting agent...')
        stadle_client = BasicClient(aggregator_ip_address=agg_addr, agent_name=f'agent_{get_timestamp()}', simulation_flag=True)
        model = get_model()
        stadle_client.set_bm_obj(model)
        sleep(15)

        optimizer = Adam(model.parameters(), lr=1e-3)

        start_sd = stadle_client.wait_for_sg_model().state_dict()
        model.load_state_dict(start_sd)

        for rnd in range(rounds):
            model, loss, optimizer = train_model(model, dataloader, optimizer, 3)
            ft_container.text(f'Round {rnd+1}/{rounds} - Waiting for aggregation...\nModel after local training:\n' + sd2str(model.state_dict()))
            ret = stadle_client.send_trained_model(model, perf_values={'performance':loss})

            print(rnd, ret)

            fl_sd = stadle_client.wait_for_sg_model().state_dict()
            ft_container.text(f'Round {rnd+1}/{rounds}\nAggregate model:\n' + sd2str(fl_sd))
            model.load_state_dict(fl_sd)
            sleep(10)

        stadle_client.disconnect()
        
        st.session_state.models['fl_model_' + get_timestamp()] = model

with rows[2]:
    agg_addr = st.text_input('STADLE Aggregator Address:')

    upload_btn = st.button(
        'Upload Model Metadata',
        on_click=on_up_click,
        args=[agg_addr]
    )

    num_rounds_input = st.number_input('Number of FL rounds:', 1, 20)

    train_btn = st.button(
        'Start FL Agent',
        on_click=on_fl_click,
        args=[agg_addr, num_rounds_input]
    )

    ft_container = st.empty()
    ft_container.text(st.session_state.ft_text)

def on_validate_click(mkey):
    global data_file

    dataloader = process_df(data_file)

    stats, preds, real, pred_conf = test_model(st.session_state.models[mkey], dataloader)

    st.session_state.pred_data = (real, preds)
    st.session_state.stats = stats

    vdf = pd.read_csv(data_file, index_col=0)

    print(preds)

    vdf[f'flagged_by_{mkey}'] = pd.Series(preds)
    # vdf[f'{mkey}_probs'] = pd.Series(pred_conf)

    vdf = vdf.style.format(subset=['balance', 'transaction_amount'], formatter='{:.2f}').format(subset=['transaction_time'], formatter='{:.0f}')

    st.session_state.vdf = vdf

with rows[3]:
    mkey = st.selectbox(
        'Select model to use',
        st.session_state.models.keys(),
        help='Models created through local training or with STADLE are stored.  Select the model to use for running fraud detection.'
    )

    if (mkey is not None):
        minfo = st.text(sd2str(st.session_state.models[mkey].state_dict()))

        validate_btn = st.button(
            'Run Fraud Detection',
            on_click=on_validate_click,
            args=[mkey]
        )

        if ('stats' in st.session_state):
            tp,tn,fp,fn = st.session_state.stats
            st.text(f'Out of {tp+tn+fp+fn} transactions:\n\t{tp+tn} transactions were classified correctly\n\t\t{tp} fradulent transactions were caught\n\t\t{tn} valid transactions were allowed to continue\n\t{fn} fraudlent transactions were missed\n\t{fp} valid transactions were incorrectly flagged as fraud')
            st.text(f'Accuracy: {(tp+tn)/(tp+tn+fp+fn):.2f} | Precision: {tp/(tp+fp+1e-5):.2f} | Recall: {tp/(tp+fn+1e-5):.2f} | F1-Score: {tp/(tp + (fp+fn)/2 + 1e-5):.2f}')

            fig, ax = plt.subplots()

            ConfusionMatrixDisplay.from_predictions(
                *st.session_state.pred_data,
                display_labels=['Not Fraud', 'Fraud'],
                ax=ax,
                cmap='GnBu'
            )
            st.pyplot(fig)

            with st.expander("Show model predictions"):
                st.dataframe(st.session_state.vdf)