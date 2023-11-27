from dash import Dash, dash_table, dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import lightgbm as lgb
import base64
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import moduls

import warnings
warnings.filterwarnings("ignore")

app = Dash(__name__)
server = app.server

app.layout = html.Div(children=[
    html.H2("Análisis de Datos", style={
                        "margin": "8px",
                        "display": "flex",
                        "justifyContent": "center",
                        "fontSize": "30px",
                        "fontWeight": "700",
                        "letterSpacing": "0",
                        "lineHeight": "1.5em",
                        "paddingBottom": "0px",
                        "marginBottom": "0px",
                        "position": "relative",
                        "color": "#000000",
                        "fontFamily": "'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif"
                    }),
    html.H2("Comparación Entrenamiento - Validación", style={
                        "fontSize": "24px",
                        "fontWeight": "700",
                        "color": "#49494d",
                        "fontFamily": "'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif"
                    }),
    html.H2("Entrenamiento", style={
                        "fontSize": "24px",
                        "fontWeight": "700",
                        "color": "#3f48cc",
                        "fontFamily": "'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif"
                    }),

    dcc.Upload(
        id='upload-data-1',
        children=html.Div([
            'Seleccione',
            html.A(' datos EEG y TRs')
        ]),
        style={
            'width': '80%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
  
    html.Div(id='metrics'),
    dcc.Graph(id='vtc-entrenamiento', className="plot",style={"height": "60vh", "width": "85%"}),
    dcc.Graph(id='boxplot-caracteristicas', className="plot",style={"height": "80vh", "width": "85%"}),

    html.H2("Validación", style={
                        "fontSize": "24px",
                        "fontWeight": "700",
                        "color": "#3f48cc",
                        "fontFamily": "'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif"
                    }),

    dcc.Upload(
        id='upload-data-2',
        children=html.Div([
            'Seleccione',
            html.A(' datos EEG y TRs')
        ]),
        style={
            'width': '80%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
  
    html.Div(id='output-data-upload-2'),
    dcc.Graph(id='vtc-validacion', className="plot",style={"height": "60vh", "width": "85%"}),
    dcc.Graph(id='boxplot-validacion', className="plot",style={"height": "80vh", "width": "85%"}),



    html.H2("Comparación de Estímulos Auditivos", style={
                        "fontSize": "24px",
                        "fontWeight": "700",
                        "color": "#49494d",
                        "fontFamily": "'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif"
                    }),
    dcc.Upload(
        id='upload-data-3',
        children=html.Div([
            'Seleccione',
            html.A(' datos TRs')
        ]),
        style={
            'width': '80%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    dcc.Upload(
        id='upload-data-4',
        children=html.Div([
            'Seleccione',
            html.A(' datos TRs')
        ]),
        style={
            'width': '80%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload-3'),
    html.Div(id='output-data-upload-4'),

    html.Button('Comparar', id='comparar', n_clicks=0, 
                    style={
                        "backgroundColor": "#3f48cc",
                        "borderRadius": "14px",
                        "borderStyle": "none",
                        "color": "white",
                        "fontFamily": "Roboto,Arial,sans-serif",
                        "fontSize": "14px",
                        "fontWeight": "bold",
                        "height": "48px",
                        "width": "220px",
                    }),
    
    dcc.Graph(id='boxplot-auditivos', className="plot",style={"height": "60vh", "width": "85%"}),   
                    
                    ])


def way(df_eeg, df_rt, task):
    global clf_class
    global clf_regre
    global gain_importance_df

    df_eeg = df_eeg.copy()
    df_rt = df_rt.copy()

    # Transform RT
    df_rt['time'] =  pd.to_datetime(df_rt['time'])
    df_rt['tag'] = df_rt['tag'].fillna('click')
    df_rt['tr'] = df_rt['tr'].map(lambda x: float(x[-8:-1]) if not isinstance(x, float) else x)

    # Transform EEG
    #df_eeg['time'] = df_eeg['time'].map(lambda x: datetime.fromtimestamp(x))
    df_eeg['time'] =  pd.to_datetime(df_eeg['time'])
    df_eeg = df_eeg.iloc[:,:-1]
    df_eeg = df_eeg[df_eeg['time']>df_rt['time'].iloc[0]]
    df_eeg = df_eeg[df_eeg['time']<df_rt['time'].iloc[-1]]
    df_eeg.reset_index(inplace=True, drop=True)

    # generate date range with RT mean
    df_rt_date = moduls.generate_df_rt_date_no_mean(df_rt)

    # interpolate Values for trials without responses (omission errors and correct trials)
    df_rt_date = moduls.interp_rt(df_rt_date)

    # compute VTC
    df_rt_date = moduls.compute_VTC(df_rt_date)
    ori_med = df_rt_date['vtc'].median()

    # Set label classification, In the zone and Out of the zone
    df_rt_date['class'] = np.where(df_rt_date['vtc'] >= ori_med, 0, 1)  # 0:out   1:in
    
    # preprocessing eeg data
    df_eeg = moduls.preprocessimg_data(df_eeg)

    # Extract features from eeg
    df_features = moduls.wavelet_packet_decomposition(df_eeg, df_rt_date)
    
    # Normalize values
    df_features = moduls.normalization(df_features)
    
    # Number of experiment 
    df_features['n_experiment'] = [100 for l in range(df_features.shape[0])]

    #df_features.to_excel("prueba.xlsx", index=False)
    
    if task == 'train':
        # Split to train LGBM
        X = df_features[df_features.columns.difference(['rt', 'vtc', 'class', 'time','flag','n_experiment'])]  # rt, vtc, class, n_experiment
        y = df_features['class']
        X = X.filter(regex='^(?!.*p_min).', axis=1) # filter cols with p_min
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 3)

        # Train LGBM Classification
        clf_class = lgb.LGBMClassifier(importance_type='gain', max_depth=5, num_leaves=15) #, min_data_in_leaf=100,min_gain_to_split=50)#,  n_estimators=10, learning_rate=0.1)
        clf_class.fit(X_train, y_train)
        y_pred = clf_class.predict(X_test)
        accuracy = round(accuracy_score(y_test, y_pred)*100,2)
        gain_importance = clf_class.feature_importances_

        feature_names = list(X.columns)
        gain_importance_df = pd.DataFrame({'Feature': feature_names, 'Gain': gain_importance})
        gain_importance_df = gain_importance_df.sort_values(by=['Gain'], ascending=False)

        # Train LGBM Regressor
        y = df_features['rt']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 3)

        params = {
            'task': 'train', 
            'boosting': 'gbdt',
            'objective': 'regression',
            'learnnig_rage': 0.05,
            'feature_fraction': 0.9,
            'metric': 'rmse'
        }

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        clf_regre  = lgb.train(params,
                        train_set=lgb_train,
                        valid_sets=lgb_eval)

        y_pred = clf_regre.predict(X_test)
        y_pred = [0.45 if y<0.45 else y for y in y_pred]
        y_pred = [1.12 if y>1.12 else y for y in y_pred]

        # accuracy check
        mse = mean_squared_error(y_test, y_pred)
        rmse = round((mse**(0.5))*1000,2)
    else:
        X = df_features[df_features.columns.difference(['rt', 'vtc', 'class', 'time','flag','n_experiment'])]  # rt, vtc, class, n_experiment
        X = X.filter(regex='^(?!.*p_min).', axis=1) # filter cols with p_min

        # Classification prediction
        df_features['class'] = clf_class.predict(X)
        
        # Regression prediction
        y_pred = clf_regre.predict(X)
        y_pred = [0.45 if y<0.45 else y for y in y_pred]
        y_pred = [1.12 if y>1.12 else y for y in y_pred]
        df_features['rt'] = clf_regre.predict(X)
        df_features = moduls.compute_VTC_validity(df_features)
        ori_med = df_features['vtc'].median()


    # Etiquet experiment parts
    lst_part = []

    delta_2 = df_features['time'].iloc[0] + timedelta(minutes = 2)
    delta_3 = df_features['time'].iloc[0] + timedelta(minutes = 3)
    
    for i in range(df_features.shape[0]):
        if df_features['time'].iloc[i] < delta_2:
            lst_part.append('no notify 1')
        elif (df_features['time'].iloc[i] >= delta_2) and (df_features['time'].iloc[i] < delta_3):
            lst_part.append('notify')
        else:
            lst_part.append('no notify 2')
        
    df_features['part'] =  lst_part

    # Most important features
    imp = gain_importance_df['Feature'].to_list()[:3]

    f1 = df_features[[imp[0],'part']].copy()
    f1.columns = ["feature", "part"]
    f1['name'] = [imp[0] for i in range(f1.shape[0])]

    f2 = df_features[[imp[1],'part']].copy()
    f2.columns = ["feature", "part"]
    f2['name'] = [imp[1] for i in range(f2.shape[0])]

    f3 = df_features[[imp[2],'part']].copy()
    f3.columns = ["feature", "part"]
    f3['name'] = [imp[2] for i in range(f3.shape[0])]

    frames = [f1, f2, f3]
    df_box = pd.concat(frames)

    # Fig VTC
    print(df_features['part'].value_counts())
    lst_step = list(np.linspace(0, 2, num=df_features[df_features['part']=='no notify 1'].shape[0]))+list(np.linspace(2.05, 3, num=df_features[df_features['part']=='notify'].shape[0]))+list(np.linspace(3.05, 5, num=df_features[df_features['part']=='no notify 2'].shape[0]))
    df_features['step'] = lst_step
    fig_vtc = go.Figure()   
    fig_vtc.add_trace(go.Scatter(x=df_features['step'], y=df_features['class'],
                        mode='lines',
                        name='class')) 
    fig_vtc.add_trace(go.Scatter(x=df_features['step'], y=df_features['vtc'],
                        mode='lines',
                        name='VTC'))
    fig_vtc.add_trace(go.Scatter(x=[0.25], y=[ori_med+0.05],text=["Mediana"],mode="text", showlegend=False))
    fig_vtc.add_shape(legendrank=1, showlegend=False, type="line", xref="paper", line=dict(dash="5px"), x0=0, x1=4,y0=ori_med, y1=ori_med)
    fig_vtc.update_xaxes(title_text = "Time(min.)")
    fig_vtc.update_yaxes(title_text = "VTC")
    fig_vtc.update_layout(title='VTC', legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1), font=dict(size=16))

    # Box Plot Features
    fig_box = px.box(df_box, x="part", y="feature", color='name')
    fig_box.update_traces(quartilemethod="linear") # or "inclusive", or "linear" by default
    fig_box.update_layout(title="Parte de experimento vs características", legend=dict(
                                                                            yanchor="top",
                                                                            y=0.99,
                                                                            xanchor="left",
                                                                            x=0.01
                                                                        ),
                                                                    font=dict(size=16))
    fig_box.update_traces(boxmean=True)

    if task == 'train':
        metrics = 'Accuraccy:'+str(accuracy)+'%'+'             '+'RMSE: '+str(rmse)+'ms'
        return fig_vtc, fig_box, metrics
    else:
        return fig_vtc, fig_box

def way_auditivos(df_rt_au1, df_rt_au2):
    df_rt_au1 = df_rt_au1.copy()
    df_rt_au2 = df_rt_au2.copy()

    # Transform RT
    df_rt_au1['time'] =  pd.to_datetime(df_rt_au1['time'])
    df_rt_au1['tag'] = df_rt_au1['tag'].fillna('click')
    df_rt_au1['tr'] = df_rt_au1['tr'].map(lambda x: float(x[-8:-1]) if not isinstance(x, float) else x)

    df_rt_au2['time'] =  pd.to_datetime(df_rt_au2['time'])
    df_rt_au2['tag'] = df_rt_au2['tag'].fillna('click')
    df_rt_au2['tr'] = df_rt_au2['tr'].map(lambda x: float(x[-8:-1]) if not isinstance(x, float) else x)

    # generate date range with RT mean
    df_rt_date_1 = moduls.generate_df_rt_date_no_mean(df_rt_au1)
    df_rt_date_2 = moduls.generate_df_rt_date_no_mean(df_rt_au2)

    # interpolate Values for trials without responses (omission errors and correct trials)
    df_rt_date_1 = moduls.interp_rt(df_rt_date_1)
    df_rt_date_2 = moduls.interp_rt(df_rt_date_2)

    # compute VTC
    df_rt_date_1 = moduls.compute_VTC(df_rt_date_1)
    ori_med_1 = df_rt_date_1['vtc'].median()
    df_rt_date_2 = moduls.compute_VTC(df_rt_date_2)
    ori_med_2 = df_rt_date_2['vtc'].median()

    # Set label classification, In the zone and Out of the zone
    df_rt_date_1['class'] = np.where(df_rt_date_1['vtc'] >= ori_med_1, "Desatento", "Atento")  # 0:out   1:in
    df_rt_date_2['class'] = np.where(df_rt_date_2['vtc'] >= ori_med_2, "Desatento", "Atento")  # 0:out   1:in

    # Etiquet experiment parts
    lst_part = []

    delta_2 = df_rt_date_1['start'].iloc[0] + timedelta(minutes = 2)
    delta_3 = df_rt_date_1['start'].iloc[0] + timedelta(minutes = 3)
    
    for i in range(df_rt_date_1.shape[0]):
        if df_rt_date_1['start'].iloc[i] < delta_2:
            lst_part.append('no notify 1')
        elif (df_rt_date_1['start'].iloc[i] >= delta_2) and (df_rt_date_1['start'].iloc[i] < delta_3):
            lst_part.append('notify')
        else:
            lst_part.append('no notify 2')
        
    df_rt_date_1['part'] =  lst_part
    print(df_rt_date_1['part'].value_counts())

    # Etiquet experiment parts
    lst_part = []

    delta_2 = df_rt_date_2['start'].iloc[0] + timedelta(minutes = 2)
    delta_3 = df_rt_date_2['start'].iloc[0] + timedelta(minutes = 3)
    
    for i in range(df_rt_date_2.shape[0]):
        if df_rt_date_2['start'].iloc[i] < delta_2:
            lst_part.append('no notify 1')
        elif (df_rt_date_2['start'].iloc[i] >= delta_2) and (df_rt_date_2['start'].iloc[i] < delta_3):
            lst_part.append('notify')
        else:
            lst_part.append('no notify 2')
        
    df_rt_date_2['part'] =  lst_part
    print(df_rt_date_2['part'].value_counts())

    df_rt_date_1['Estímulo'] =  ['estímulo 1' for i in range(df_rt_date_1.shape[0])]
    df_rt_date_2['Estímulo'] =  ['estímulo 2' for i in range(df_rt_date_2.shape[0])]

    frames = [df_rt_date_1, df_rt_date_2]
    df_all_rt = pd.concat(frames)

    # Box Plot Features
    fig_box = px.box(df_all_rt, x="part", y="rt", color='Estímulo')
    fig_box.update_traces(quartilemethod="linear") # or "inclusive", or "linear" by default
    fig_box.update_layout(title="Parte de experimento vs TR vs Estímulo", legend=dict(
                                                                            yanchor="top",
                                                                            y=0.99,
                                                                            xanchor="left",
                                                                            x=0.01
                                                                        ),
                                                                    font=dict(size=16))
    fig_box.update_traces(boxmean=True)

    return fig_box



@callback(
            Output('vtc-entrenamiento', 'figure'),
            Output('boxplot-caracteristicas', 'figure'),
            Output('metrics', 'children'),
            Input('upload-data-1', 'contents'),
            State('upload-data-1', 'filename'),
            State('upload-data-1', 'last_modified'),
              prevent_initial_call=True)
def update_output_1(list_of_contents, list_of_names, list_of_dates):
    print(list_of_names)
    
    # import EEG
    content_type, content_string = list_of_contents[1].split(',')
    decoded = base64.b64decode(content_string)
    df_eeg = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
    
    # import RTs
    content_type, content_string = list_of_contents[0].split(',')
    decoded = base64.b64decode(content_string)
    df_rt = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
   
    fig_vtc, fig_box, metrics = way(df_eeg, df_rt, task='train')
    

    return fig_vtc, fig_box, metrics


@callback(
            Output('vtc-validacion', 'figure'),
            Output('boxplot-validacion', 'figure'),
            Input('upload-data-2', 'contents'),
            State('upload-data-2', 'filename'),
            State('upload-data-2', 'last_modified'),
              prevent_initial_call=True)
def update_output_2(list_of_contents, list_of_names, list_of_dates):
    print(list_of_names)
    
    content_type, content_string = list_of_contents[1].split(',')
    decoded = base64.b64decode(content_string)
    df_eeg = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

    content_type, content_string = list_of_contents[0].split(',')
    decoded = base64.b64decode(content_string)
    df_rt = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

    fig_vtc, fig_box= way(df_eeg, df_rt, task='validity')

    return fig_vtc, fig_box


@callback(
            Output('output-data-upload-3', 'children'),
            Input('upload-data-3', 'contents'),
            State('upload-data-3', 'filename'),
            State('upload-data-3', 'last_modified'),
              prevent_initial_call=True)
def update_output_3(list_of_contents, list_of_names, list_of_dates):
    print(list_of_names)

    content_type, content_string = list_of_contents[0].split(',')
    decoded = base64.b64decode(content_string)
    df_rt_au1 = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
   
    df_rt_au1.to_excel("rts/rt1.xlsx", index=False)

    return "✔️"


@callback(
            Output('output-data-upload-4', 'children'),
            Input('upload-data-4', 'contents'),
            State('upload-data-4', 'filename'),
            State('upload-data-4', 'last_modified'),
              prevent_initial_call=True)
def update_output_4(list_of_contents, list_of_names, list_of_dates):
    print(list_of_names)

    content_type, content_string = list_of_contents[0].split(',')
    decoded = base64.b64decode(content_string)
    df_rt_au2 = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
    df_rt_au2.to_excel("rts/rt2.xlsx", index=False)

    return "✔️"


@callback(
            Output('boxplot-auditivos', 'figure'),
            Input('comparar', 'n_clicks'),
            prevent_initial_call=True)
def update_output_5(n_clicks):
    df_rt_au1 = pd.read_excel('rts/rt1.xlsx')
    df_rt_au2 = pd.read_excel('rts/rt2.xlsx')

    fig = way_auditivos(df_rt_au1, df_rt_au2)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)