import requests
import json
import pandas as pd
import numpy as np
from time import sleep
import os
from pulp import *
import dash
import matplotlib.pyplot as plt
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback, dash_table
from dash.dependencies import Input, Output
import dash_table
import numpy as np
import dash_core_components as dcc
import plotly.graph_objects as go
import dash_html_components as html
from plotly.subplots import make_subplots

df = pd.read_csv('data/game_logs_sorare.csv')
prices_df = pd.read_csv('data/historical_prices_sorare.csv')
df = df.sort_values(by='score', ascending=False)
gw_start = pd.read_pickle('data/gw_starts.pkl')
players_slug_list = list(df['player_name'].unique())
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

dash.register_page(__name__)
layout = html.Div(children=[
    html.H1(
        children='Player Rankings',
        style={
            'textAlign': 'center',

        }
    ),  # end of H1
    html.Div([dcc.Dropdown(options=players_slug_list,
                           id='player_dropdown',
                           value=players_slug_list[0]

                           ),
              dcc.Graph(id='player_progression', style={'height': '80vh'}),

              ]),
    html.P("Game Week Start"),
    dcc.Slider(
        id='slider-position',
        min=1, max=len(gw_start), value=1, step=1,
        marks=gw_start
    ),
    html.P("Optimal Contender Lineup"),
    html.Div(id='optimal_contender'),
    html.P("Optimal Champion Lineup"),
    html.Div(id='optimal_champion'),
])


@callback(
    Output('player_progression', 'figure'),
    Output('optimal_contender', 'children'),
    Output('optimal_champion', 'children'),
    Input('player_dropdown', 'value'),
    Input('slider-position', 'value'))
def update_figure(selected_player, gw_number):
    temp = prices_df[prices_df['player'] == selected_player]
    temp = temp.set_index(pd.to_datetime(temp['date']))

    temp = temp[temp.price != 0].resample('24H').mean()
    temp['date'] = temp.index
    temp_df = df[df['player_name'] == selected_player]
    temp_df = temp_df.set_index(pd.to_datetime(temp_df['date']))

    temp_df = temp_df.groupby('gw').max()

    # print(temp_df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=temp["date"], y=temp["price"], secondary_y=False, name='Price', mode='markers')
    # fig.add_trace(go.scatter(x=temp['date'],y=temp['price'],mode='markers',name='Price'))
    fig.add_scatter(x=temp_df['date'], y=temp_df['tenGameAverage'].shift(-1), mode='lines+markers', name='Cap',
                    secondary_y=True)
    fig.add_scatter(x=temp_df['date'], y=temp_df['score'], mode='markers', name='Score', secondary_y=True)
    # fig = px.scatter(temp_df, x="date", y="tenGameAverage")
    # fig = px.scatter(temp_df, x="date", y="score")
    # for gw,gw_starts in gw_start.items():
    fig.add_vline(x=gw_start[f'nba-gameweek-{gw_number}'], line_width=1, line_dash='dash')
    fig.add_vline(x=gw_start[f'nba-gameweek-{gw_number + 1}'], line_width=1, line_dash='dash')
    fig.update_yaxes(title_text="<b>Price in Euro</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Score and Cap</b>", secondary_y=True)
    # fig.update_layout(transition_duration=500)
    optimal_dict = []

    def optimize_lineups(df, gw_number, contender):
        df_week = df[df['gw'] == gw_number].sort_values(by='score', ascending=False)
        df_week = df_week.drop_duplicates(subset=['player_name'], keep='first')
        optimal_lineup = []
        if not contender:
            optimal_lineup.append(df_week['player_name'].iloc[0])
            df_week = df_week.iloc[1:]
        inv_item = list(df_week['player_name'])
        cap = dict(zip(inv_item, df_week['tenGameAverage']))
        score = dict(zip(inv_item, df_week['score']))
        prob = LpProblem('Sorare', LpMaximize)
        inv_vars = LpVariable.dicts('Variable', inv_item, lowBound=0, cat='Integer')
        prob += lpSum([inv_vars[i] * score[i] for i in inv_item])
        if contender:
            prob += lpSum([inv_vars[i] * cap[i] for i in inv_item]) <= 110, 'Contender Cap'
            # prob += lpSum([inv_vars[i] * z[i] for i in inv_item]) <= 800, 'Price Cap'
            prob += lpSum([inv_vars[i] for i in inv_item]) == 5, 'Number of Players'
        else:
            prob += lpSum([inv_vars[i] * cap[i] for i in inv_item]) <= 120, 'Champion Cap'
            # prob += lpSum([inv_vars[i] * z[i] for i in inv_item]) <= 800, 'Price Cap'
            prob += lpSum([inv_vars[i] for i in inv_item]) == 4, 'Number of Players'
        for name in inv_vars:
            prob += inv_vars[name] <= 1

        prob.solve()
        print('The optimal answer\n' + '-' * 70)

        for v in prob.variables():

            if v.varValue > 0:
                print(v.name, '=', v.varValue)
                name_temp = v.name.split('Variable_')[1]
                tmp = df_week[df_week['player_name'] == name_temp.replace('_', '-')].iloc[0][
                    ['score', 'tenGameAverage']].values

                optimal_lineup.append(v.name.split('Variable_')[1].replace('_', '-'))
        return optimal_lineup

    contender_lineup = optimize_lineups(df, gw_number, contender=True)
    champion_lineup = optimize_lineups(df, gw_number, contender=False)
    contender_df = \
    df[np.logical_and(df['gw'] == gw_number, df['player_name'].isin(contender_lineup))].groupby('player_name').max()[
        ['team_name', 'against', 'tenGameAverage', 'score']].sort_values(by='score', ascending=False)
    contender_df.insert(0, 'player_name', contender_df.index)
    datatable = dash_table.DataTable(
        id='table_team,',
        columns=[{"name": i, "id": i} for i in contender_df.columns],
        style_cell={'textAlign': 'left'},
        editable=True,
        # filter_action="native",
        sort_action="native",
        sort_mode="single",
        column_selectable="single",
        #     fixed_columns={'headers':True,'data':2},
        #     style_table ={'max-width':'100% !important'},
        page_action="native",
        page_current=0,
        page_size=50,
        # row_selectable="multi",
        # row_deletable=True,
        # column_deletable = True,
        # selected_columns=[],
        # selected_rows=[],

        data=contender_df.round(3).to_dict('records'),
    )
    champion_df = \
        df[np.logical_and(df['gw'] == gw_number, df['player_name'].isin(champion_lineup))].groupby('player_name').max()[
            ['team_name', 'against', 'tenGameAverage', 'score']].sort_values(by='score', ascending=False)
    champion_df.insert(0, 'player_name', champion_df.index)
    datatable_1 = dash_table.DataTable(
        id='table_team,',
        columns=[{"name": i, "id": i} for i in champion_df.columns],
        style_cell={'textAlign': 'left'},
        editable=True,
        # filter_action="native",
        sort_action="native",
        sort_mode="single",
        column_selectable="single",
        #     fixed_columns={'headers':True,'data':2},
        #     style_table ={'max-width':'100% !important'},
        page_action="native",
        page_current=0,
        page_size=50,
        # row_selectable="multi",
        # row_deletable=True,
        # column_deletable = True,
        # selected_columns=[],
        # selected_rows=[],

        data=champion_df.round(3).to_dict('records'),
    )
    return fig, [html.Div([datatable])], [html.Div([datatable_1])]


