import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime as datetime
from datetime import datetime as dt

st.set_page_config(page_title="TRT-beregning", page_icon="🔥")

with open("styles/main.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

st.title('Driftsdata for sesongvarmelager 🥵🥶')

filnavn = 'Trd_2023-06-27.csv'

def custom_to_datetime(timestamp):
    return pd.to_datetime(timestamp, format="%d.%m.%Y %H:%M:%S")


def funk_les_datafil(filnavn):

    df = pd.read_csv(filnavn, 
            sep=";", 
            skiprows=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], 
            usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12],
            header=0,
            names = ['utropstegn','tid','klokkeslett','til_bane','fra_bane','RT401','RT501','RT402','RT502','RT403','RT501_2','RT404','RT502_2'])
    
    
    df['tid'] = df['tid']+' '+df['klokkeslett']
    df['tid'] = df['tid'].apply(custom_to_datetime)
    df = df.drop('klokkeslett',axis=1)
    df['til_bane'] = df['til_bane'].astype(float)
    df['fra_bane'] = df['fra_bane'].astype(float)

    df['RT401'] = df['RT401'].astype(float)
    df['RT501'] = df['RT501'].astype(float)
    df['RT402'] = df['RT402'].astype(float)
    df['RT502'] = df['RT502'].astype(float)
    df['RT403'] = df['RT403'].astype(float)
    df['RT501_2'] = df['RT501_2'].astype(float)
    df['RT404'] = df['RT404'].astype(float)
    df['RT502_2'] = df['RT502_2'].astype(float)

    return df

df = funk_les_datafil(filnavn)
print(df)

def plottefunksjon(x_data,x_navn,y_data1,y_navn1,y_data2,y_navn2,yakse_navn,tittel):

    til_plot = pd.DataFrame({x_navn : x_data, y_navn1 : y_data1, y_navn2 : y_data2})
    fig = px.line(til_plot, x=x_navn, y=[y_navn1,y_navn2], title=tittel, color_discrete_sequence=['#367A2F', '#FFC358'])
    fig.update_layout(xaxis_title=x_navn, yaxis_title=yakse_navn,legend_title=None)
    st.plotly_chart(fig)

plottefunksjon(x_data=df['tid'],x_navn='Tid',y_data1=df['til_bane'],y_navn1='Til bane',y_data2=df['fra_bane'],y_navn2='Fra bane',yakse_navn='Til og fra',tittel='Til og fra')