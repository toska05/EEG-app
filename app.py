import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import joblib
import sys
import os
import mne
from scipy.signal import welch
import streamlit.components.v1 as components
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
labram_dir = os.path.join(BASE_DIR, "LaBraM")

if labram_dir not in sys.path:
    sys.path.append(labram_dir)

from modeling_finetune import labram_base_patch200_200

class LaBraM_Finetune_Wrapper(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = labram_base_patch200_200(
            num_classes=num_classes, 
            in_chans=1, 
            init_values=0.0,
            EEG_size=1000
        )

    def forward(self, x):
        B, N, L = x.shape
        A = L // 200
        x = x.view(B, N, A, 200)
        input_chans = torch.arange(N + 1, device=x.device)
        return self.model(x, input_chans=input_chans)

SFREQ = 250
CH_NAMES = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
CH_TYPES = ['eeg'] * 8
BANDS = {
    'Delta': (1.0, 4.0), 'Theta': (4.0, 8.0), 'Alpha': (8.0, 13.0),
    'Beta': (13.0, 30.0), 'Gamma': (30.0, 40.0)
}
mne.set_log_level('WARNING')

def z_score_normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1e-6
    return (data - mean) / std

def process_flight_data_unified(df_raw):
    eeg_cols = ['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8']
    eeg_data = df_raw[eeg_cols].values.T * 1e-6
    
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types=CH_TYPES)
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    raw = mne.io.RawArray(eeg_data, info)
    
    raw_filtered = raw.copy().filter(l_freq=4.0, h_freq=40.0)
    
    ica = mne.preprocessing.ICA(n_components=8, random_state=42, method='fastica')
    ica.fit(raw_filtered)
    ica.exclude = [0, 1]
    raw_clean = raw_filtered.copy()
    ica.apply(raw_clean)
    
    events = mne.make_fixed_length_events(raw_clean, duration=5.0, overlap=3.75)
    epochs = mne.Epochs(
        raw_clean, events, tmin=0.0, tmax=5.0, 
        baseline=None, reject=dict(eeg=160e-6), preload=True
    )
    
    df_epochs = epochs.to_data_frame()
    original_epoch_numbers = epochs.selection
    
    features = []
    grouped = df_epochs.groupby('epoch')
    for (epoch_num, group), orig_idx in zip(grouped, original_epoch_numbers):
        time_sec = (orig_idx * 1.25) + 2.5
        row = {'epoch': orig_idx, 'Time_sec': time_sec}
        
        for ch_name in CH_NAMES:
            freqs, psd = welch(group[ch_name].to_numpy(), fs=SFREQ, nperseg=SFREQ)
            for band_name, (fmin, fmax) in BANDS.items():
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                row[f"{ch_name}_{band_name}"] = np.mean(psd[freq_mask])
        features.append(row)
        
    df_feat = pd.DataFrame(features)
    
    if not df_feat.empty:
        df_feat['Arousal_Index'] = (df_feat['Fz_Beta'] + df_feat['Cz_Beta'] + df_feat['Pz_Beta']) / \
                                   (df_feat['Fz_Alpha'] + df_feat['Cz_Alpha'] + df_feat['Pz_Alpha'])
        df_feat['Relaxation_Index'] = (df_feat['Oz_Alpha'] + df_feat['Pz_Alpha']) / \
                                      (df_feat['Oz_Theta'] + df_feat['Pz_Theta'])
        df_feat['Valence_Index'] = df_feat['C4_Alpha'] - df_feat['C3_Alpha']
        df_feat['Readiness_Index'] = (df_feat['C3_Beta'] + df_feat['C4_Beta'] + df_feat['Cz_Beta']) / \
                                     (df_feat['C3_Alpha'] + df_feat['C4_Alpha'] + df_feat['C3_Theta'] + df_feat['C4_Theta'])
        df_feat['TBR_Fz'] = df_feat['Fz_Theta'] / df_feat['Fz_Beta'].replace(0, 0.0001)
        
        cols_for_trend = ['Arousal_Index', 'Relaxation_Index', 'Fz_Beta', 'Valence_Index', 'Readiness_Index', 'TBR_Fz']
        for col in cols_for_trend:
            df_feat[f'{col}_trend'] = df_feat[col].diff().fillna(0)

    raw_resampled = raw_clean.copy().resample(200)
    data_resampled = raw_resampled.get_data()
    mapped_signals = data_resampled[[0, 2, 1, 3, 5, 7, 4, 6], :].T 
    
    X_windows = []
    labram_times = []
    
    WINDOW_SIZE = int(5.0 * 200)
    STEP_NORMAL = int(1.25 * 200) 
    
    for orig_idx in original_epoch_numbers:
        start = orig_idx * STEP_NORMAL
        end = start + WINDOW_SIZE
        if end <= len(mapped_signals):
            window_signal_raw = mapped_signals[start:end, :]
            X_windows.append(z_score_normalize(window_signal_raw))
            labram_times.append((orig_idx * 1.25) + 2.5)

    labram_tensors = None
    if X_windows:
        X_final = np.transpose(np.array(X_windows), (0, 2, 1))
        labram_tensors = torch.tensor(X_final, dtype=torch.float32)

    return df_feat, labram_tensors, labram_times

@st.cache_resource
def load_xgb_model(path):
    print("Loading XGBoost model...")
    if os.path.exists(path):
        obj = joblib.load(path)
        if isinstance(obj, dict) and 'model' in obj: return obj['model']
        if isinstance(obj, dict) and 'clf' in obj: return obj['clf']
        return obj
    return None

@st.cache_resource
def load_labram_models(base_path):
    print("Loading LaBram models...")
    models = []
    for i in range(1, 6):
        path = os.path.join(base_path, f"labram_augmented_best_fold_{i}.pth")
        if os.path.exists(path):
            model = LaBraM_Finetune_Wrapper(num_classes=2)
            state_dict = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
    return models


st.set_page_config(layout="wide")

st.sidebar.header("1. Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload EEG CSV", type=['csv'])

st.sidebar.markdown("---")

st.sidebar.header("2. Analysis Settings")

st.sidebar.markdown("**Models to Display**")
show_trad = st.sidebar.checkbox("Traditional Analysis", value=True)
show_xgb = st.sidebar.checkbox("XGBoost Model", value=True)
show_labram = st.sidebar.checkbox("LaBram Model", value=True)

st.sidebar.markdown("**Smoothing & Trend**")
show_smooth = st.sidebar.checkbox("Enable Smoothing (Fast)", value=True)
show_trend = st.sidebar.checkbox("Show Macro Trend (Slow)", value=True)


xgb_path = os.path.join(BASE_DIR, "data", "model_optimized_stress_class_BATCH_AUC_0654.pkl")
xgb_model = load_xgb_model(xgb_path)
labram_models = load_labram_models(labram_dir)

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file, skipinitialspace=True)
    df_raw.columns = df_raw.columns.str.strip()
    
    features_df, labram_tensors, labram_times = process_flight_data_unified(df_raw.copy())
    
    time_axis = []
    trad_stress = []
    xgb_stress = []
    labram_stress = []
    
    if not features_df.empty:
        time_axis = features_df['Time_sec'].tolist()
        trad_stress = features_df['Arousal_Index'].tolist()
        
        if xgb_model is not None:
            cols_to_drop = ['epoch', 'Time_sec']
            X = features_df.drop(columns=[c for c in cols_to_drop if c in features_df.columns], errors='ignore')
            try:
                features_to_use = xgb_model.feature_names_in_
                xgb_stress = xgb_model.predict_proba(X[features_to_use])[:, 1].tolist()
            except AttributeError:
                xgb_stress = xgb_model.predict_proba(X)[:, 1].tolist()

    if labram_tensors is not None and len(labram_models) > 0:
        predictions = []
        with torch.no_grad():
            for model in labram_models:
                outputs = model(labram_tensors)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                predictions.append(probs)
        labram_stress = np.mean(predictions, axis=0).tolist()

    chart_data = {
        "trad_time": time_axis, "trad_stress": trad_stress,
        "xgb_time": time_axis, "xgb_stress": xgb_stress,
        "labram_time": labram_times, "labram_stress": labram_stress
    }
    json_data = json.dumps(chart_data)

    str_show_trad = str(show_trad).lower()
    str_show_xgb = str(show_xgb).lower()
    str_show_labram = str(show_labram).lower()
    str_show_smooth = str(show_smooth).lower()
    str_show_trend = str(show_trend).lower()

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{ font-family: sans-serif; margin: 0; padding: 0; color: #333; }}
            .main-content {{ width: 100%; box-sizing: border-box; padding: 10px; }}
            .plot-container {{ width: 100%; height: 350px; }}
            
            .layout-wrapper {{
                display: flex;
                flex-direction: column;
                gap: 20px;
                width: 100%;
            }}
            
            .plots-column {{
                flex: 1;
                min-width: 0;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }}
            
            .video-column {{
                width: 100%;
                display: flex;
                flex-direction: column;
            }}
            
            .video-section {{
                display: flex;
                flex-direction: row;
                gap: 15px;
                align-items: flex-start;
                background: #f8f9fa;
                padding: 15px;
                border-radius: 12px;
                border: 1px solid #e9ecef;
            }}
            
            .video-controls-panel {{ 
                display: flex;
                flex-direction: column;
                gap: 10px;
                width: 200px;
                flex-shrink: 0; 
            }}
            
            .control-block {{
                background: #ffffff; 
                padding: 10px; 
                border-radius: 8px; 
                border: 1px dashed #ced4da;
            }}
            
            .video-wrapper {{
                flex: 1;
                display: flex;
                justify-content: center;
                align-items: center;
                min-width: 0;
                width: 100%;
            }}
            
            video {{ 
                width: 100%; 
                max-height: 350px;
                min-height: 250px;
                object-fit: contain; 
                background: #000; 
                border-radius: 8px; 
            }}
            
            h3 {{ margin-top: 0; font-size: 13px; margin-bottom: 5px; color: #495057; }}
            input[type="number"] {{ width: 100%; padding: 4px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }}
            input[type="file"] {{ width: 100%; font-size: 12px; }}

            @media (min-width: 1100px) {{
                .layout-wrapper {{
                    flex-direction: row-reverse;
                    align-items: flex-start;
                }}
                .video-column {{
                    width: 480px;
                    flex-shrink: 0;
                    position: sticky;
                    top: 15px; 
                    z-index: 10;
                }}
                .video-section {{
                    flex-direction: column;
                    align-items: stretch;
                }}
                .video-controls-panel {{
                    width: 100%;
                    flex-direction: row;
                }}
                .control-block {{
                    flex: 1;
                }}
                .video-wrapper {{
                    width: 100%;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="main-content">
            <div class="layout-wrapper">
                
                <div class="video-column">
                    <div class="video-section">
                        <div class="video-controls-panel">
                            <div class="control-block">
                                <h3>Local Video Upload</h3>
                                <input type="file" id="video-upload" accept="video/*">
                            </div>
                            <div class="control-block">
                                <h3>Sync Offset (sec)</h3>
                                <input type="number" id="time-offset" value="0" step="0.1">
                            </div>
                        </div>
                        <div class="video-wrapper">
                            <video id="video-player" controls></video>
                        </div>
                    </div>
                </div>

                <div class="plots-column">
                    <div id="plot-trad" class="plot-container" style="display: {'block' if show_trad else 'none'}"></div>
                    <div id="plot-xgb" class="plot-container" style="display: {'block' if show_xgb else 'none'}"></div>
                    <div id="plot-labram" class="plot-container" style="display: {'block' if show_labram else 'none'}"></div>
                </div>

            </div>
        </div>

        <script>
            const rawData = {json_data};
            
            const vidUpload = document.getElementById('video-upload');
            const video = document.getElementById('video-player');
            const offsetInput = document.getElementById('time-offset');

            vidUpload.addEventListener('change', function(e) {{
                const file = e.target.files[0];
                if (file) {{
                    video.src = URL.createObjectURL(file);
                    console.log("Video loaded successfully");
                }}
            }});

            const layoutBase = {{
                margin: {{ l: 40, r: 20, t: 30, b: 30 }},
                xaxis: {{ title: 'Time (s)' }},
                shapes: [],
                autosize: true,
                showlegend: true,
                legend: {{ orientation: "h", y: 1.1, x: 0 }}
            }};

            function minMaxScale(arr) {{
                if (!arr || arr.length === 0) return [];
                const min = Math.min(...arr);
                const max = Math.max(...arr);
                if (max === min) return arr.map(() => 0);
                return arr.map(val => (val - min) / (max - min));
            }}

            function movingAverage(data, windowSize) {{
                if (!data || data.length < windowSize) return data;
                let result = [];
                for (let i = 0; i < data.length; i++) {{
                    let start = Math.max(0, i - windowSize + 1);
                    let end = i + 1;
                    let slice = data.slice(start, end);
                    let sum = slice.reduce((a, b) => a + b, 0);
                    result.push(sum / slice.length);
                }}
                return result;
            }}

            function centeredMovingAverage(data, windowSize) {{
                if (!data || data.length < windowSize) return data;
                let result = [];
                let half = Math.floor(windowSize / 2);
                for (let i = 0; i < data.length; i++) {{
                    let start = Math.max(0, i - half);
                    let end = Math.min(data.length, i + half + 1);
                    let slice = data.slice(start, end);
                    let sum = slice.reduce((a, b) => a + b, 0);
                    result.push(sum / slice.length);
                }}
                return result;
            }}

            function getTraces(xData, yData, name, color, normalize, smooth, showTrend) {{
                let traces = [];
                let processedY = normalize ? minMaxScale(yData) : yData;
                
                let baseY = smooth ? movingAverage(processedY, 5) : processedY;
                traces.push({{
                    x: xData,
                    y: baseY,
                    name: name,
                    type: 'scatter',
                    mode: smooth ? 'lines' : 'lines+markers',
                    line: {{ color: color, shape: 'spline', width: smooth ? 2 : 1 }},
                    fill: 'tozeroy',
                    fillcolor: color + '22',
                    connectgaps: true
                }});

                if (showTrend) {{
                    let trendWindow = Math.max(5, Math.floor(processedY.length / 10)); 
                    let trendY = centeredMovingAverage(processedY, trendWindow);
                    traces.push({{
                        x: xData,
                        y: trendY,
                        name: 'Macro Trend',
                        type: 'scatter',
                        mode: 'lines',
                        line: {{ color: 'black', width: 3, dash: 'dot', shape: 'spline' }},
                        hoverinfo: 'skip'
                    }});
                }}
                
                return traces;
            }}

            function renderPlots() {{
                const useSmooth = {str_show_smooth};
                const useTrend = {str_show_trend};
                const config = {{responsive: true}};
            
                if ({str_show_trad}) {{
                    const tradData = getTraces(rawData.trad_time, rawData.trad_stress, 'Arousal Index', '#d62728', true, useSmooth, useTrend);
                    Plotly.newPlot('plot-trad', tradData, {{...layoutBase, title: 'Traditional Analysis (Normalized)'}}, config);
                }}

                if ({str_show_xgb}) {{
                    const xgbData = getTraces(rawData.xgb_time, rawData.xgb_stress, 'Stress Prediction', '#ff7f0e', true, useSmooth, useTrend);
                    Plotly.newPlot('plot-xgb', xgbData, {{...layoutBase, title: 'XGBoost (Normalized)' }}, config);
                }}

                if ({str_show_labram}) {{
                    const labramData = getTraces(rawData.labram_time, rawData.labram_stress, 'Stress Prediction', '#9467bd', true, useSmooth, useTrend);
                    Plotly.newPlot('plot-labram', labramData, {{...layoutBase, title: 'LaBram (Normalized)' }}, config);
                }}
            }}

            video.addEventListener('timeupdate', function() {{
                const currentOffset = parseFloat(offsetInput.value) || 0;
                const adjustedTime = Math.max(0, video.currentTime - currentOffset); 
                
                const shape = {{
                    type: 'line',
                    x0: adjustedTime,
                    x1: adjustedTime,
                    y0: 0,
                    y1: 1,
                    yref: 'paper',
                    line: {{ color: 'blue', width: 2, dash: 'solid' }}
                }};

                if ({str_show_trad}) Plotly.relayout('plot-trad', {{ shapes: [shape] }});
                if ({str_show_xgb}) Plotly.relayout('plot-xgb', {{ shapes: [shape] }});
                if ({str_show_labram}) Plotly.relayout('plot-labram', {{ shapes: [shape] }});
            }});

            renderPlots();
            
            window.addEventListener('resize', function() {{
                if ({str_show_trad}) Plotly.Plots.resize('plot-trad');
                if ({str_show_xgb}) Plotly.Plots.resize('plot-xgb');
                if ({str_show_labram}) Plotly.Plots.resize('plot-labram');
            }});
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=1800, scrolling=True)
else:
    st.info("Upload CSV file in the sidebar to start.")