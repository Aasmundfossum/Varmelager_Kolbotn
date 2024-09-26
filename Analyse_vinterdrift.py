import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Driftsdata Kolbotn", page_icon="🔥", layout='wide')

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

def energy_effect_plot(df, x_series, y_series, series_label, average = False, separator = False, min_value = None, max_value = None, chart_type = "Line"):
    if chart_type == "Line":
        fig = px.line(df, x=x_series, y=y_series, labels={'Value': y_series, 'Timestamp': 'Tid'}, color_discrete_sequence=["rgba(29, 60, 52, 0.75)"])
    elif chart_type == "Bar":
        fig = px.bar(df, x=x_series, y=y_series, labels={'Value': y_series, 'Timestamp': 'Tid'}, color_discrete_sequence=["rgba(29, 60, 52, 0.75)"])
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
        average = df[y_series].mean()
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

    avg_nonzero = df[df[y_series] > 0][y_series].mean()
    sum_series = np.sum(df[y_series])
    st.write(f'Gjennomsnitt av verdier som er >0: {avg_nonzero} kW')
    st.write(f'Total energimengde i perioden: {sum_series} kWh')
    st.write('---')

def other_plot(df, xseries, yseries, min_value = 0, max_value = 10): #self,
    df = df.sort_values(by=xseries)
    fig = px.scatter(df, x=xseries, y=yseries, labels={'Value': yseries, 'Timestamp': 'Tid'}, color_discrete_sequence=[f"rgba(29, 60, 52, 0.75)"])
    fig.update_xaxes(type='category')
    fig.update_xaxes(
        title='Returtemperatur (ºC)',
        type='category',
        gridwidth=0.3,
        tickmode='auto',
        nticks=4,  
        tickangle=30)
    fig.update_yaxes(
        title=f"Effekt (kW)",
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

def other_plot_2(df, xseries, yseries, bin_series, step=2, min_value = 0, max_value = 10):
    min_bin = np.min(df[bin_series])
    max_bin = np.max(df[bin_series])
    
    # Create the figure outside the loop
    fig = go.Figure()

    # Loop through the bins and add each scatter plot as a trace
    #for i in range(math.floor(min_bin), math.ceil(max_bin), step):
    for i in np.arange(1, 4, step):
        relevant_df = df.loc[(df[bin_series] >= i) & (df[bin_series] <= i + step)]
        
        # Add scatter trace to the figure for each iteration
        fig.add_trace(go.Scatter(
            x=relevant_df[xseries], 
            y=relevant_df[yseries], 
            mode='markers', 
            name=f'Returtemp ({i}) - ({i+step}) ºC',
            showlegend=True
        ))
    
    # Update the axes and layout after all traces have been added
    fig.update_xaxes(
        title='Utetemp. (ºC)',
        gridwidth=0.3,
        tickmode='auto',
        nticks=4,  
        tickangle=30,
        range=[df[xseries].min(), df[xseries].max()]  # Set fixed range for x-axis
    )
    fig.update_yaxes(
        title="Effekt (kW)",
        tickformat=",",
        ticks="outside",
        gridcolor="lightgrey",
        gridwidth=0.3,
        range=[min_value, max_value]  # Set fixed range for y-axis
    )
    fig.update_layout(
        showlegend=True,  # Ensure the legend is enabled
        margin=dict(l=20, r=20, b=20, t=20, pad=0),
        height=300,
    )
    
    # Plot the single figure with all traces
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'staticPlot': False})


###############################################################################################################################################################

st.title('Driftsdata for sesongvarmelager 🥵🥶')

data = pd.read_csv('Kolbotn_driftsdata_240924.csv')

st.write(data.columns)

data['Tid'] = pd.to_datetime(data['Tid'])
df = data.loc[(data['Tid'] >= '2023-10-01 00:00:00') & (data['Tid'] < '2024-04-01 00:00:00')]
df = df.reset_index(drop=True)
df['Utetemp_seklima'] = pd.read_excel('Værdata_Kolbotn_Vinter_2324.xlsx', usecols='D')

#df['Tilført energi - Bane 1_ABS'] = akk_to_power(df['Tilført energi - Bane 1'])
#df['Tilført energi - Bane 2_ABS'] = akk_to_power(df['Tilført energi - Bane 2'])
#df['Energi levert fra varmepumpe_ABS'] = akk_to_power(df['Energi levert fra varmepumpe'])

#st.write(df)

st.subheader('Temperaturer til/fra baner')
c1,c2 = st.columns(2)
with c1:
    temperature_plot(df=df, series='Til bane 1', min_value = -5, max_value = 10)
with c2:
    temperature_plot(df=df, series='Fra bane 1', min_value = -5, max_value = 10)

st.subheader('Utetemperatur')
c1,c2 = st.columns(2)
with c1:
    temperature_plot(df=df, series='Utetemperatur', min_value = -20, max_value = 25)
with c2:
    temperature_plot(df=df, series='Utetemp_seklima', min_value = -20, max_value = 25)

st.subheader('Effekt tilført bane 1 og effekt fra varmepumpe')
c1,c2 = st.columns(2)
with c1:
    st.write('Effekt tilført bane 1')
    energy_effect_plot(df = df, x_series = 'Tid', y_series = "Tilført effekt - Bane 1", series_label = "Effekt (kW)", separator = True, chart_type = "Bar", min_value=0, max_value = 400)
with c2:
    st.write('Effekt varmepumpe bane 1')
    energy_effect_plot(df = df, x_series = 'Tid', y_series = "Tilført effekt - Varmepumpe", series_label = "Effekt (kW)", separator = True, chart_type = "Bar", min_value=0, max_value = 400)

st.subheader('Effekt som funksjon av returtemp')
c1,c2 = st.columns(2)
with c1:
    st.write('Tilført bane 1')
    other_plot(df=df, xseries='Fra bane 1', yseries='Tilført effekt - Bane 1', min_value = 0, max_value = 400)
with c2:
    st.write('Fra varmepumpe')
    other_plot(df=df, xseries='Fra bane 1', yseries='Tilført effekt - Varmepumpe', min_value = 0, max_value = 400)

st.subheader('Effekt som funksjon av utetemp')
c1,c2 = st.columns(2)
with c1:
    st.write('Effekt tilført bane 1')
    other_plot_2(df, 'Utetemp_seklima', 'Tilført effekt - Bane 1', 'Fra bane 1', step=0.5, min_value = 0, max_value = 400)
with c2:
    st.write('Effekt til varmepumpe')
    other_plot_2(df, 'Utetemp_seklima', 'Tilført effekt - Varmepumpe', 'Fra bane 1', step=0.5, min_value = 0, max_value = 400)
c1,c2 = st.columns(2)
with c1:
    st.write('Tilført bane 1')
    other_plot(df=df, xseries='Utetemp_seklima', yseries='Tilført effekt - Bane 1', min_value = 0, max_value = 400)
with c2:
    st.write('Fra varmepumpe')
    other_plot(df=df, xseries='Utetemp_seklima', yseries='Tilført effekt - Varmepumpe', min_value = 0, max_value = 400)

st.subheader('COP')
temperature_plot(df=df, series='COP', min_value = -5, max_value = 10)

st.subheader('Summert på dager')
df_daily = df.resample('D', on='Tid').sum()
df_daily = df_daily.reset_index()
c1,c2 = st.columns(2)
with c1:
    st.write('Tilført effekt til bane:')
    energy_effect_plot(df = df_daily, x_series = 'Tid', y_series = "Tilført effekt - Bane 1", series_label = "Energi (kWh)", separator = True, chart_type = "Bar", min_value=0, max_value = 10000)
with c2:
    st.write('Tilført effekt varmepumpe:')    
    energy_effect_plot(df = df_daily, x_series = 'Tid', y_series = "Tilført effekt - Varmepumpe", series_label = "Energi (kWh)", separator = True, chart_type = "Bar", min_value=0, max_value = 10000)
