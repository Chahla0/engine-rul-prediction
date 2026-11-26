import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

WINDOW = 30
COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]


class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


@st.cache_resource
def load_model():
    features = json.load(open('features.json'))
    scaler = joblib.load('scaler.pkl')
    model = LSTMModel(len(features))
    model.load_state_dict(torch.load('lstm_rul.pt', map_location='cpu'))
    model.eval()
    return features, scaler, model


@st.cache_data
def load_data():
    test = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=COLS)
    rul = pd.read_csv('RUL_FD001.txt', header=None, names=['RUL'])
    return test, rul


def make_windows(scaled, last_cycle):
    """One window (last 30 cycles) for every cycle from WINDOW to last_cycle."""
    wins = [scaled[i - WINDOW:i] for i in range(WINDOW, last_cycle + 1)]
    return np.array(wins, dtype=np.float32)


st.set_page_config(page_title='Engine RUL Predictor', layout='wide')
st.title('Turbofan Engine: Remaining Useful Life')
st.write('LSTM model trained on NASA C-MAPSS (FD001). '
         'Pick an engine and move the slider to see how the prediction changes as the engine ages.')

features, scaler, model = load_model()
test, rul = load_data()

# ---- Sidebar controls ----
unit = st.sidebar.selectbox('Test engine', sorted(test['unit'].unique()))
engine = test[test['unit'] == unit].reset_index(drop=True)
n_cycles = len(engine)
cycle = st.sidebar.slider('Current cycle', WINDOW, n_cycles, n_cycles)
shown = st.sidebar.multiselect('Sensors to plot', features,
                               default=[s for s in ['s11', 's9', 's4'] if s in features])

# ---- Predict ----
scaled = scaler.transform(engine[features])
windows = make_windows(scaled, cycle)
with torch.no_grad():
    preds = model(torch.tensor(windows)).numpy() * 125
current_pred = float(max(preds[-1], 0))

# ---- Headline numbers ----
c1, c2, c3 = st.columns(3)
c1.metric('Predicted remaining life (cycles)', f'{current_pred:.0f}')
if cycle == n_cycles:
    actual = int(rul.iloc[unit - 1]['RUL'])
    c2.metric('Actual remaining life (cycles)', actual)
    c3.metric('Error (predicted - actual)', f'{current_pred - min(actual, 125):+.0f}')
else:
    c2.metric('Cycle', f'{cycle} of {n_cycles}')
    c3.metric('Actual remaining life', 'only known at the last cycle')

# Illustrative thresholds only
if current_pred < 30:
    st.error('Status: critical. Failure is close.')
elif current_pred < 60:
    st.warning('Status: plan maintenance soon.')
else:
    st.success('Status: healthy.')

# ---- Charts ----
left, right = st.columns(2)
with left:
    st.subheader('Predicted remaining life over time')
    st.line_chart(pd.DataFrame({'Predicted RUL': np.clip(preds, 0, None)},
                               index=range(WINDOW, cycle + 1)))
with right:
    st.subheader('Sensor readings (scaled 0-1)')
    if shown:
        sensor_df = pd.DataFrame(scaled[:cycle], columns=features,
                                 index=engine['cycle'][:cycle])[shown]
        st.line_chart(sensor_df)
    else:
        st.info('Choose at least one sensor in the sidebar.')

st.caption('The model was trained with RUL capped at 125, so it never predicts above 125. '
           'Status thresholds (30 and 60 cycles) are illustrative, not from NASA.')
st.write('Scaler max values:', scaler.data_max_[:4])
