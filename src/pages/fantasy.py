import dash
from basketball_reference_scraper.constants import TEAM_TO_TEAM_ABBR as teams
from bs4 import BeautifulSoup
from bs4 import Comment
import requests
import pandas as pd
import json
import plotly.express as px
from dash import callback
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import os
from dash_table.Format import Format, Scheme, Sign, Symbol
from datetime import datetime as dt
import time
import plotly.graph_objects as go
from sklearn.cluster import KMeans
dash.register_page(__name__)
# Load Data
years = [2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011]
total_df = pd.read_csv('data/players_current_data.csv')


def calculate_z_scores(df, punt_cats, cats=['FG%', '3P', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']):
    #     year = int(year)
    #     df = df[df['year']==year]
    # df = df[df['MP']>3]
    if 'Age' in cats:

        stats = ['games', 'MP', 'FGA', 'FTA']
    else:
        stats = ['games', 'Age', 'MP', 'FGA', 'FTA']
    stats.extend(cats)

    df = df[stats]
    cats_std = {}

    z_cats = []
    for cat in cats:
        df_cat = df.copy()
        #     if cat in ['FG%','FT%']:
        #         df_cat = df_cat[np.logical_and(f'{cat[:-1]}A']>0,df_cat[cat]>0)]
        #         df_cat[cat] = df_cat[cat]*df_cat[f'{cat[:-1]}A']
        df_cat = df_cat[df_cat[cat] > 0]
        df_cat = df_cat.sort_values(cat, ascending=False)
        cat_std = df_cat[cat].iloc[:460].std()
        cat_mean = df_cat[cat][:460].mean()

        cats_std[f'{cat}_mean'] = cat_mean
        cats_std[f'{cat}_std'] = cat_std
        if cat in ['FG%', 'FT%']:

            #         df_cat = df_cat[df_cat[f'{cat[:-1]}A']>0]
            #             cat_std= df_cat[cat].iloc[:450].std()
            #             cat_mean = df_cat[cat][:450].mean()
            #             print(f'{cat_mean}_{cat}')
            temp_z = ((df_cat[cat]) - cat_mean) / cat_std

            temp_z = temp_z * df_cat[f'{cat[:-1]}A']
            cat_mean = temp_z.mean()
            cat_std = temp_z.std()
            cats_std[f'{cat}_weighted_mean'] = cat_mean
            cats_std[f'{cat}_weighted_std'] = cat_std
            #             print(temp_z)
            df[f'z_{cat}'] = ((temp_z) - cat_mean) / cat_std
        else:
            if cat in ['TOV', 'Age']:
                df[f'z_{cat}'] = -(df[cat] - cat_mean) / cat_std
            else:
                df[f'z_{cat}'] = (df[cat] - cat_mean) / cat_std
        z_cats.append(f'z_{cat}')
    for cat in punt_cats:
        z_cats.remove(f'z_{cat}')
    df['z_score'] = df[z_cats].mean(axis=1)
    #     df = df.dropna(axis=1)

    df = df.fillna(df.mean())
    df = df.sort_values('z_score', ascending=False)
    #     column_names = df.columns
    #     df = np.ceil(df.values,3)
    #     df = pd.DataFrame(df,columns = column_names)
    df.insert(loc=0, column='rank', value=np.arange(1, len(df) + 1))
    return df.round(3), cats_std
def turn_value_into_cat(value):
    if value>2.5:
        return "++++"
    elif (value>1.5)& (value<=2.5):
        return "+++"
    elif (value>0.5)& (value<=1.5):
        return "++"
    elif (value>0)& (value<=0.5):
        return "+"
    elif (value>-0.5)& (value<=0):
        return "-"
    elif (value>-1.5)& (value<=-0.5):
        return "--"
    elif (value>-2.5)& (value<=-1.5):
        return "--"
    elif value<-2.5:
        return "----"
data_columns  = [
       'MP','Age', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GmSc', '+/-',
       ]
age = total_df['Age'].values
for i in range(0,len(age),1):
    age[i] = str(age[i].split('-')[0])
total_df['age'] = age
data = total_df[total_df['G'].notnull()]

for element in data_columns[1:-1]:
    data[element] = data[element].astype('float32')
data.dtypes
# Build App
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# df = pd.DataFrame(columns = [
#        'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB',
#        'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GmSc', '+/-','name'
#        ])
PAGE_SIZE = 600
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
data_columns = [
    'MP', 'Age', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB',
    'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'name', 'position', 'team', 'games'
]
players_list = []
for year in years:
    z_cats = ['z_PTS', 'z_FG%', 'z_3P', 'z_FT%', 'z_TRB', 'z_AST', 'z_STL', 'z_BLK', 'z_TOV']
    z = data[data['year'] == year]
    z = z.drop_duplicates()

    #     games= z.groupby('name').size()
    #     z = z[data_columns]
    z = z.groupby('name').agg({'MP': 'mean', 'Age': 'mean', 'FG': 'mean', 'FGA': 'mean', 'FG%': 'mean', '3P': 'mean',
                               '3PA': 'mean', '3P%': 'mean', 'FT': 'mean', 'FTA': 'mean', 'FT%': 'mean', 'ORB': 'mean',
                               'DRB': 'mean', 'TRB': 'mean', 'AST': 'mean', 'STL': 'mean', 'BLK': 'mean', 'TOV': 'mean',
                               'PF': 'mean', 'PTS': 'mean', 'name': 'last', 'position': 'last',
                               'Tm': 'last', 'G': 'count'})
    z.columns = data_columns

    z['FT%'] = z['FT'] / z['FTA']
    z['FG%'] = z['FG'] / z['FGA']

    z, _ = calculate_z_scores(z, [])
    players_list.append(z.iloc[:140][z_cats].values)
players_list = np.concatenate(players_list)
kmeans = KMeans(n_clusters=7, random_state=0)

cluster_dict = {}

kmeans.fit(players_list)
for i in range(1,len(kmeans.cluster_centers_)+1):
    cluster_dict[f'Cluster {i}'] = []
    for j in range(0,len(kmeans.cluster_centers_[i-1])):
        cluster_dict[f'Cluster {i}'].append(turn_value_into_cat(kmeans.cluster_centers_[i-1][j]))
cluster_df = pd.DataFrame(cluster_dict).T
cluster_df.columns = ['PTS', 'FG%', '3P', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV']
cluster_df.insert(0,'Cluster', cluster_df.index)
layout = html.Div(style={'backgroundColor': colors['background'], 'display': 'inline-block'}, children=[
    html.H1(
        children='Player Rankings',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),  # end of H1
    html.Div(id='intermediate_value', style={'display': 'none'}),
    html.Div(id='container', style={'display': 'flex', 'paddingRight': '15px', 'color': colors['text']}, children=[
        html.Div(style={'color': colors['text'], },
                 children=[html.Label('Ranking Year'),
                           dcc.RadioItems(
                               options=[
                                   {'label': i, 'value': i} for i in years
                               ], id='year_rank', style={'display': 'flex', 'paddingRight': '15px'},
                               value=2023
                           )])
        ,
        html.Div([html.Label('Select date'),
                  dcc.DatePickerRange(
                      id='date_picker',
                      min_date_allowed=dt(2018, 1, 1),
                      max_date_allowed=dt(2024, 1, 1),
                      #       initial_visible_month=dt(current_year,df['Date'].max().to_pydatetime().month, 1),
                      #       start_date=(df['Date'].max() - timedelta(6)).to_pydatetime(),
                      #       end_date=df['Date'].max().to_pydatetime(),
                  ),

                  ], ),
        html.Div([html.Label('Punt Categories'),
                  dcc.Checklist(
                      options=[{'label': cat, 'value': cat} for cat in
                               ['FG%', '3P', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']

                               ],
                      value=[], id='punt_cats', style={'display': 'flex', 'paddingRight': '15px'}
                  )
                  ]),
        html.Div(id='radar_chart_container'),
    ]),  ####end of Div after first

    html.Div([ dash_table.DataTable(
            fixed_rows={'headers': True},
            id='table_z',
            # columns=[{"name": i, "id": i} for i in df.columns],
            style_cell={'textAlign': 'left'},
            # editable=True,
            # filter_action="native",
            sort_action="native",
            sort_mode="single",
            column_selectable="single",
            #     fixed_columns={'headers':True,'data':2},
            #     style_table ={'max-width':'100% !important'},
            page_action="native",
            page_current=0,
            page_size=500,
            # row_selectable="multi",
            # row_deletable=True,
            # column_deletable = True,
            # selected_columns=[],
            # selected_rows=[],

            # data=df.round(3).to_dict('records'),
        )]),
html.P("Clusters Explained"),
html.Div([
        dash_table.DataTable(data=cluster_df.round(2).to_dict("records"),columns=[{"name":i,"id":i} for i in cluster_df.columns])

    ]),
    html.Div(id='z_scores_container'),

]  # end of children
                      ,
                      )  # end of first div


@callback (
    Output('date_picker', 'max_date_allowed'), Output('date_picker', 'min_date_allowed'),
     Output('date_picker', 'end_date'), Output('date_picker', 'start_date'),
    Input('year_rank', 'value')
)
def calculate_dates(year):
    year = int(year)
    z = data[data['year'] == year]
    start_date = z.sort_values('Date').iloc[0].Date
    end_date = z.sort_values('Date').iloc[-1].Date
    return end_date, start_date, end_date, start_date


@callback(
    [Output('table_z', 'data'), Output('table_z', 'columns'), Output('intermediate_value', 'children')],
    [Input('year_rank', 'value'), Input('date_picker', 'end_date'), Input('date_picker', 'start_date'),
     Input('punt_cats', 'value')]
)
def update_table(year, end_date, start_date, punt_cats):
    year = int(year)
    years = [year]
    #     years.append(year-1)
    #     years.append(year-2)
    print(years)
    data_columns = [
        'MP', 'Age', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB',
        'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'name', 'position', 'team', 'games'
    ]
    z_cats = ['z_PTS', 'z_FG%', 'z_3P', 'z_FT%', 'z_TRB', 'z_AST', 'z_STL', 'z_BLK', 'z_TOV']
    z = data[data['year'].isin(years)]
    z = z.drop_duplicates()
    #     z = z[np.logical_and(z['Date']>=start_date,z['Date']<=end_date)]
    #     games= z.groupby('name').size()
    #     z = z[data_columns]
    z = z.groupby('name').agg({'MP': 'mean', 'Age': 'mean', 'FG': 'mean', 'FGA': 'mean', 'FG%': 'mean', '3P': 'mean',
                               '3PA': 'mean', '3P%': 'mean', 'FT': 'mean', 'FTA': 'mean', 'FT%': 'mean', 'ORB': 'mean',
                               'DRB': 'mean', 'TRB': 'mean', 'AST': 'mean', 'STL': 'mean', 'BLK': 'mean', 'TOV': 'mean',
                               'PF': 'mean', 'PTS': 'mean', 'name': 'last', 'position': 'last',
                               'Tm': 'last', 'G': 'count'})
    z.columns = data_columns

    z['FT%'] = z['FT'] / z['FTA']
    z['FG%'] = z['FG'] / z['FGA']

    z, _ = calculate_z_scores(z, punt_cats)
    # print(z)
    #     position = z.index.get_level_values('position')
    #     z.index = z.index.droplevel(1)
    # z['games'] = games
    # z.insert(loc=1,column = 'position' ,value = position)

    y = kmeans.predict(z[z_cats])

    z['cluster'] = y
    for cluster in list(z['cluster'].unique()):

        for cat in z_cats:
            temp_z = z.iloc[:200]
            temp_z = temp_z[temp_z['cluster'] == cluster]
            # print(f'Cluster {cluster} has {cat} mean:{temp_z[cat].mean()}')
        # print(temp_z.index)
    z.insert(loc=1, column='name', value=z.index.get_level_values('name'))
    z['MP'] = pd.to_datetime(z["MP"], unit='s').dt.strftime("%M:%S")
    #     z['rank'] = z['rank'].astype('str')
    #     cols_to_be_formated = list(z.keys())
    #     cols_to_be_formated.remove('rank')
    #     cols_to_be_formated.remove('games')

    # print(np.round(z,3).to_dict('records')[0]
    return z.round(3).to_dict('records'), [{"name": i, "id": i, 'type': 'numeric', 'format': Format(
        precision=3,
        scheme=Scheme.fixed,

    ), } for i in z.keys()], z.to_json(date_format='iso', orient='split')


@callback(
    Output('z_scores_container', 'children'),
    [Input('year_rank', 'value'), Input('date_picker', 'end_date'), Input('date_picker', 'start_date')]
)
def update_z_score_graphs(year, end_date, start_date):
    year = int(year)
    data_columns = [
        'MP', 'Age', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB',
        'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'name', 'position', 'team', 'games'
    ]
    z_cats = ['z_PTS', 'z_FG%', 'z_3P', 'z_FT%', 'z_TRB', 'z_AST', 'z_STL', 'z_BLK']
    z = data[data['year'] == year]
    z = z[np.logical_and(z['Date'] >= start_date, z['Date'] <= end_date)]
    #     games= z.groupby('name').size()
    #     z = z[data_columns]
    z = z.groupby('name').agg({'MP': 'mean', 'Age': 'mean', 'FG': 'mean', 'FGA': 'mean', 'FG%': 'mean', '3P': 'mean',
                               '3PA': 'mean', '3P%': 'mean', 'FT': 'mean', 'FTA': 'mean', 'FT%': 'mean', 'ORB': 'mean',
                               'DRB': 'mean', 'TRB': 'mean', 'AST': 'mean', 'STL': 'mean', 'BLK': 'mean', 'TOV': 'mean',
                               'PF': 'mean', 'PTS': 'mean', 'name': 'last', 'position': 'last',
                               'Tm': 'last', 'G': 'count'})
    z.columns = data_columns
    #     position = z.index.get_level_values('position')
    #     z.index = z.index.droplevel(1)
    #     z['games'] = games
    z['FT%'] = z['FT'] / z['FTA']
    z['FG%'] = z['FG'] / z['FGA']

    z, _ = calculate_z_scores(z, [])

    #     z.insert(loc=1,column = 'position' ,value = position)
    z.insert(loc=1, column='name', value=z.index.get_level_values('name'))

    children = []
    for z_cat in z_cats:
        fig = px.scatter(z.iloc[:300], x='rank', y=z_cat,
                         hover_data={"name": True, f'{z_cat[2:]}': ':.3f'}, trendline="lowess")
        #         fig.update_xaxes(hoverformat= '.2f')
        children.append(html.Div(dcc.Graph(id=z_cat, figure=fig)))
    return children




# @callback(
#     [Output('radar_chart_container', 'children'), Output('z_team_container', 'children')],
#     [Input('table_z', 'selected_rows'), Input('intermediate_value', 'children')]
# )
# def show_radar_chart(player, df):
#     z_cats = ['z_FG%', 'z_3P', 'z_FT%', 'z_TRB', 'z_AST', 'z_STL', 'z_BLK', 'z_PTS']
#     z_tov = ['z_FG%', 'z_3P', 'z_FT%', 'z_TRB', 'z_AST', 'z_STL', 'z_BLK', 'z_PTS', 'z_TOV']
#     children = []
#     if len(player) == 0:
#         return children, []
#     else:
#         df = pd.read_json(df, orient='split')
#         fig = go.Figure()
#
#         df_team = df.iloc[player][z_tov]
#
#         for i in player:
#             name = df.iloc[i].name
#             df_team.append(list(df.iloc[i][z_tov].values))
#
#             r = list(df.iloc[i][z_cats].values)
#
#             # print(r)
#             #             df = pd.DataFrame(dict(r=r,theta = z_cats))
#             fig.add_trace(go.Scatterpolar(
#                 r=r,
#                 theta=z_cats,
#                 #               fill='toself',
#                 name=name))
#         #             fig = px.line_polar(df, r='r',theta='theta', line_close=True)
#
#         fig.update_layout(
#             polar=dict(
#                 radialaxis=dict(
#                     visible=True,
#                     range=[-3, 5]
#                 )),
#             showlegend=False
#         )
#
#         names = list(df_team.index)
#         df_team = df_team.append(df_team.sum(numeric_only=True), ignore_index=True)
#         names.append('Total')
#         df_team.insert(0, 'name', names)
#
#         datatable = dash_table.DataTable(
#             id='table_team,',
#             columns=[{"name": i, "id": i} for i in df_team.columns],
#             style_cell={'textAlign': 'left'},
#             editable=True,
#             # filter_action="native",
#             sort_action="native",
#             sort_mode="single",
#             column_selectable="single",
#             #     fixed_columns={'headers':True,'data':2},
#             #     style_table ={'max-width':'100% !important'},
#             page_action="native",
#             page_current=0,
#             page_size=50,
#             row_selectable="multi",
#             # row_deletable=True,
#             # column_deletable = True,
#             selected_columns=[],
#             selected_rows=[],
#
#             data=df_team.round(3).to_dict('records'),
#         )
#
#         return [html.Div(dcc.Graph(id=name, figure=fig))], [html.Div([datatable])]
