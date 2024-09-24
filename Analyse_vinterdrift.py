import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Driftsdata", page_icon="🔥")

with open("styles/main.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

def akk_to_power(df_col):
    #new_col = np.zeros(len(df_col))
    #for i in range(1,len(df_col)):
    #    new_col[i]=df_col.iloc[i]-df_col.iloc[i-1]
    #new_col = pd.Series(new_col)
    df_col = df_col.diff()
    new_col = df_col
    return new_col

def temperature_plot(df, series, min_value = 0, max_value = 10): #self,
    fig = px.line(df, x=df['Tidsverdier'], y=series, labels={'Value': series, 'Timestamp': 'Tid'}, color_discrete_sequence=[f"rgba(29, 60, 52, 0.75)"])
    fig.update_xaxes(type='category')
    fig.update_xaxes(
        title='',
        type='category',
        gridwidth=0.3,
        tickmode='auto',
        nticks=4,  
        tickangle=30)
    fig.update_yaxes(
        title=f"Temperatur (ºC)",
        tickformat=",",
        ticks="outside",
        gridcolor="lightgrey",
        gridwidth=0.3,
    )
    fig.update_layout(
        #xaxis=dict(showticklabels=False),
        showlegend=False,
        yaxis=dict(range=[min_value, max_value]),
        margin=dict(l=20,r=20,b=20,t=20,pad=0),
        #separators="* .*",
        #yaxis_title=f"Temperatur {series_name.lower()} (ºC)",
        xaxis_title="",
        height = 300,
        )
    st.plotly_chart(fig, use_container_width=True, config = {'displayModeBar': False, 'staticPlot': False})

def energy_effect_plot(df, series, series_label, average = False, separator = False, min_value = None, max_value = None, chart_type = "Line"):
    if chart_type == "Line":
        fig = px.line(df, x=df['Tidsverdier'], y=series, labels={'Value': series, 'Timestamp': 'Tid'}, color_discrete_sequence=["rgba(29, 60, 52, 0.75)"])
    elif chart_type == "Bar":
        fig = px.bar(df, x=df['Tidsverdier'], y=series, labels={'Value': series, 'Timestamp': 'Tid'}, color_discrete_sequence=["rgba(29, 60, 52, 0.75)"])
    fig.update_xaxes(
        title='',
        type='category',
        gridwidth=0.3,
        tickmode='auto',
        nticks=4,  
        tickangle=30)
    fig.update_yaxes(
        title=f"Temperatur (ºC)",
        tickformat=",",
        ticks="outside",
        gridcolor="lightgrey",
        gridwidth=0.3,
    )
    if average == True:
        average = df[series].mean()
        delta_average = average * 0.98
        fig.update_layout(yaxis=dict(range=[average - delta_average, average + delta_average]))
    if separator == True:
        fig.update_layout(separators="* .*")
        
    fig.update_layout(
            #xaxis=dict(showticklabels=False),
            showlegend=False,
            margin=dict(l=20,r=20,b=20,t=20,pad=0),
            yaxis_title=series_label,
            yaxis=dict(range=[min_value, max_value]),
            xaxis_title="",
            height = 300
            )
    st.plotly_chart(fig, use_container_width=True, config = {'displayModeBar': False, 'staticPlot': False})

    avg_nonzero = df[df[series] > 0][series].mean()
    sum_series = np.sum(df[series])
    st.write(f'Gjennomsnitt av verdier som er >0: {avg_nonzero} kW')
    st.write(f'Total energimengde i perioden: {sum_series} kWh')
    st.write('---')

###############################################################################################################################################################

st.title('Driftsdata for sesongvarmelager 🥵🥶')

data = pd.read_csv('Kolbotn_driftsdata_240924.csv')

st.write(data.columns)

data['Tid'] = pd.to_datetime(data['Tid'])
df = data.loc[(data['Tid'] >= '2023-10-01 00:00:00') & (data['Tid'] < '2024-04-01 00:00:00')]
df = df.reset_index(drop=True)

df['Tilført energi - Bane 1_ABS'] = akk_to_power(df['Tilført energi - Bane 1'])
df['Tilført energi - Bane 2_ABS'] = akk_to_power(df['Tilført energi - Bane 2'])
df['Energi levert fra varmepumpe_ABS'] = akk_to_power(df['Energi levert fra varmepumpe'])

df['Forskjell_Bane1'] = df['Tilført energi - Bane 1_ABS']-df['Tilført effekt - Bane 1']

st.subheader('Temperaturer til baner')
temperature_plot(df=df, series='Til bane 1', min_value = -5, max_value = 10)
temperature_plot(df=df, series='Fra bane 1', min_value = -5, max_value = 10)

st.subheader('Effekt tilført bane 1')
df_3deg = df.loc[(df['Tid'] >= '2023-12-07 00:00:00') & (df['Tid'] < '2023-12-19 00:00:00')]
df_3deg = df_3deg.reset_index(drop=True)
df_23deg = df.loc[(df['Tid'] >= '2023-12-19 00:00:00') & (df['Tid'] < '2023-12-24 00:00:00')]
df_23deg = df_23deg.reset_index(drop=True)
df_2deg = df.loc[(df['Tid'] >= '2023-12-24 00:00:00') & (df['Tid'] < '2024-01-02 00:00:00')]
df_2deg = df_2deg.reset_index(drop=True)

st.write('Effekt fra akkumulert energi til bane 1 ved 3 C returtemperatur:')
energy_effect_plot(df = df_3deg, series = "Tilført energi - Bane 1_ABS", series_label = "Effekt (kW)", separator = True, chart_type = "Bar", min_value=0, max_value = 400)
st.write('Effekt tilført bane 1 ved 3 C returtemperatur::')
energy_effect_plot(df = df_3deg, series = "Tilført effekt - Bane 1", series_label = "Effekt (kW)", separator = True, chart_type = "Bar", min_value=0, max_value = 400)

st.write('Forskjell bane 1 ved 3 C returtemperatur:')
energy_effect_plot(df = df_3deg, series = "Forskjell_Bane1", series_label = "Effekt (kW)", separator = True, chart_type = "Bar", min_value=-200, max_value = 200)

#st.write('Effekt fra akkumulert energi til bane 1 ved 2 C returtemperatur:')
#energy_effect_plot(df = df_2deg, series = "Tilført energi - Bane 1_ABS", series_label = "Effekt (kW)", separator = True, chart_type = "Bar", min_value=0, max_value = 400)
#st.write('Effekt tilført bane 1 ved 2 C returtemperatur::')
#energy_effect_plot(df = df_2deg, series = "Tilført effekt - Bane 1", series_label = "Effekt (kW)", separator = True, chart_type = "Bar", min_value=0, max_value = 400)

st.subheader('Videre')
st.write('Effekt tilført bane 1 ved 3 C returtemperatur:')
energy_effect_plot(df = df_3deg, series = "Tilført effekt - Bane 1", series_label = "Effekt (kW)", separator = True, chart_type = "Bar", min_value=0, max_value = 400)
st.write('Effekt varmepumpe bane 1 ved 3 C returtemperatur:')
energy_effect_plot(df = df_3deg, series = "Tilført effekt - Varmepumpe", series_label = "Effekt (kW)", separator = True, chart_type = "Bar", min_value=0, max_value = 400)
st.write('Effekt tilført bane 1 ved 2 C returtemperatur::')
energy_effect_plot(df = df_2deg, series = "Tilført effekt - Bane 1", series_label = "Effekt (kW)", separator = True, chart_type = "Bar", min_value=0, max_value = 400)
st.write('Effekt varmepumpe bane 1 ved 2 C returtemperatur:')
energy_effect_plot(df = df_2deg, series = "Tilført effekt - Varmepumpe", series_label = "Effekt (kW)", separator = True, chart_type = "Bar", min_value=0, max_value = 400)
