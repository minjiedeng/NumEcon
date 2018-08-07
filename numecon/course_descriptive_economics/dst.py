from io import StringIO
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets

def load_meta_dst(table):
    
    # a. build
    json_dict = dict()
    json_dict['table'] = table
    json_dict['format'] = 'json'
    
    # b. request
    r = requests.post('https://api.statbank.dk/v1/tableinfo', json=json_dict)
    
    # c. return
    return json.loads(r.text)

def print_meta_data(table):

    out = load_meta_dst('NAN1') 

    # variables
    print('variable:')
    var_dict = out['variables'][0]
    for key in var_dict['values']:
        print('{:15s} {}'.format(key['id'],key['text']))
    print('')

    # units
    print('units:')
    unit_dict = out['variables'][1]
    for key in unit_dict['values']:
        print('{:10s} {}'.format(key['id'],key['text']))
    print('')
    # time
    print('time:')
    time_dict = out['variables'][2]
    for key in time_dict['values']:
        print('{:15s}'.format(key['id']))

def load_data_dst(table,varlist,unitlist,timelist): 
    
    # a. variables
    var_dict = dict()
    var_dict = {'code': 'TRANSAKT', 'values': varlist}
    
    # b. units
    unit_dict = dict()
    unit_dict = {'code': 'PRISENHED', 'values': unitlist}

    # c. time
    time_dict = dict()
    time_dict = {'code': 'TID', 'values': timelist}
    
    # d. full
    data_dict = dict()
    data_dict['table'] = table
    data_dict['format'] = 'BULK'
    data_dict['valuePresentation'] = 'Value'
    data_dict['delimiter'] = 'Semicolon'
    data_dict['allowVariablesInHead'] = 'false'
    data_dict['allowCodeOverrideInColumnNames'] = 'false'
    data_dict['variables'] = [var_dict,unit_dict,time_dict]

    # e. json string
    _json_str = json.dumps(data_dict, sort_keys=True, indent=4)
    
    # f. requires
    r = requests.post('https://api.statbank.dk/v1/data', json=data_dict)
    
    # g. load into pandas
    output_pd = r.text.replace(',','.')
    output_pd = StringIO(output_pd)
    df = pd.read_csv(output_pd, sep=';')
    df.sort_values('TID')

    return df

def plot():

    widgets.interact(plot_, 
        var=widgets.Dropdown(description='variable',options=['GDP','Investment']),  
        unit=widgets.RadioButtons(description='units',options=['chained','current']),
        years=widgets.IntRangeSlider(description='years',min=1966, max=2017, value=[1980,2017], continuous_update=False)
    ) 

def plot_(var,unit,years):
    
    # a. load data
    if unit == 'chained':
        unit = 'Lan_M'
    elif unit == 'current':
        unit = 'V_M'
    else:
        raise('unknown unit')

    if var == 'GDP':
        var = 'B1GQK'
    elif var == 'Investment':
        var = 'P5GD'
    else:
        raise('unknown variable')

    df = load_data_dst('NAN1',[var],[unit],['*'])
    I = (df['TID'] >= years[0]) & (df['TID'] <= years[1])

    # b. figure
    fig = plt.figure(figsize=(8,6),dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(df['TID'][I], df['INDHOLD'][I])
    
    ax.set_xlabel('year')
    ax.set_xticks(list(range(years[0],years[1]+1,5)))
    ax.set_ylabel('billion DKK')

    ax.grid(ls='--', lw=1)

    plt.show()
