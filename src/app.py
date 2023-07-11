import requests
import json
import pandas as pd
import numpy as np
from time import sleep
import os
from pulp import *
import matplotlib.pyplot as plt
import plotly.express as px
from dash import Dash, dcc, html, Input, Output,callback
from dash.dependencies import Input, Output
import dash_table
import numpy as np
import dash_core_components as dcc
import plotly.graph_objects as go
import dash_html_components as html
from plotly.subplots import make_subplots

df = pd.read_csv('data/game_logs_sorare.csv')
df = df.sort_values(by='date')
players_slug_list = list(df['player_name'].unique())
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__,external_stylesheets=external_stylesheets)
# Declare server for Heroku deployment. Needed for Procfile.
server = app.server
app.layout = html.Div(children=[
    html.H1(
        children='Player Rankings',
        style={
            'textAlign': 'center',

        }
    ),#end of H1
    html.Div([dcc.Dropdown(options=players_slug_list,
        id='player_dropdown',
        value=players_slug_list[0]

    ),
    dcc.Graph(id='player_progression'),
             ])])
@callback(
    Output('player_progression', 'figure'),
    Input('player_dropdown', 'value'))
def update_figure(selected_player):
    # temp = x[x['player']==selected_player]
    # temp = temp.set_index(pd.to_datetime(temp['date']))
    #
    # temp = temp[temp.price!=0].resample('1H').mean()
    # temp['date'] =temp.index
    temp_df = df[df['player_name']==selected_player]
    temp_df = temp_df.set_index(pd.to_datetime(temp_df['date']))

    temp_df = temp_df.groupby('gw').max()
    
    # print(temp_df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.add_scatter(x=temp["date"], y=temp["price"],secondary_y=False,name='Price',mode='markers')
#     fig.add_trace(go.scatter(x=temp['date'],y=temp['price'],mode='markers',name='Price'))
    fig.add_scatter(x=temp_df['date'],y=temp_df['tenGameAverage'].shift(-1),mode='lines+markers',name='Cap',secondary_y=True)
    fig.add_scatter(x=temp_df['date'],y=temp_df['score'],mode='lines+markers',name='Score',secondary_y=True)
#     fig = px.scatter(temp_df, x="date", y="tenGameAverage")
#     fig = px.scatter(temp_df, x="date", y="score")
#     for gw,gw_starts in gw_start.items():
#         fig.add_vline(x=gw_starts,line_width=3, line_dash='dash',name=gw)
    fig.update_layout(transition_duration=500)

    return fig

if __name__ == "__main__":
    app.run_server(debug=True,port=8051)