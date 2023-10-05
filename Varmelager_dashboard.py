import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime as datetime, timedelta
from datetime import datetime as dt

import plotly.graph_objs as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Driftsdata", page_icon="🔥")

with open("styles/main.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

st.title('Driftsdata for sesongvarmelager 🥵🥶')

tidshorisont = st.selectbox('Tidshorisont',options=['Siste 24 timer', 'Siste 7 dager', 'Siste 30 dager', 'Siste 3 måneder', 'Siste 12 måneder','All data'])

filnavn = 'Trd_2023-06-27.csv'
filnavn2 = 'Trd_2023-06-27 (1).csv'

def custom_to_datetime(timestamp):
    return pd.to_datetime(timestamp, format="%d.%m.%Y %H:%M:%S")

def temp_to_datetime(timestamp):
    return pd.to_datetime(timestamp, format="%d.%m.%Y %H:%M")


# DEL I AV DATASETT
df = pd.read_csv(filnavn, 
        sep=";", 
        skiprows=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], 
        usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12],
        header=0,
        names = ['utropstegn','tid','klokkeslett','temp_tilbane1','temp_frabane1','temp_tilbane2','temp_frabane2','RT402','RT502','temp_tilbronn40','temp_frabronn40','temp_tilbronn20','temp_frabronn20'])

df['tid'] = df['tid']+' '+df['klokkeslett']
df['tid'] = df['tid'].apply(custom_to_datetime)
df = df.drop('klokkeslett',axis=1)
df['temp_tilbane1'] = df['temp_tilbane1'].astype(float)
df['temp_frabane1'] = df['temp_frabane1'].astype(float)

df['temp_tilbane2'] = df['temp_tilbane2'].astype(float)
df['temp_frabane2'] = df['temp_frabane2'].astype(float)
#df['RT402'] = df['RT402'].astype(float)
#df['RT502'] = df['RT502'].astype(float)
df['temp_tilbronn40'] = df['temp_tilbronn40'].astype(float)
df['temp_frabronn40'] = df['temp_frabronn40'].astype(float)
df['temp_tilbronn20'] = df['temp_tilbronn20'].astype(float)
df['temp_frabronn20'] = df['temp_frabronn20'].astype(float)

snittemp_tilbaner = (df['temp_tilbane1']+df['temp_tilbane2'])/2
snittemp_frabaner = (df['temp_frabane2']+df['temp_frabane2'])/2
vaesketemp_baner = (snittemp_tilbaner+snittemp_frabaner)/2

snittemp_tilbronn = (df['temp_tilbronn40']+df['temp_tilbronn20'])/2
snittemp_frabronn = df['temp_frabronn40']*(40/60)+df['temp_frabronn20']*(20/60)
vaesketemp_bronn = (snittemp_tilbronn+snittemp_frabronn)/2

tempforskjell_bronn = snittemp_tilbronn-snittemp_frabronn
effekt_tilbronn = tempforskjell_bronn*6*4.2

# DEL II AV DATASETT
df2 = pd.read_csv(filnavn2, 
        sep=";", 
        skiprows=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], 
        usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12],
        header=0,
        names = ['utropstegn','tid','klokkeslett','RT503','temp_VP_varm_ut','temp_VP_varm_inn','temp_VP_kald_ut','temp_VP_kald_inn','bronntemp40','bronntemp20','RP001','RP002','RN001'])
df2['tid'] = df2['tid']+' '+df2['klokkeslett']
df2['tid'] = df2['tid'].apply(custom_to_datetime)
df2 = df2.drop('klokkeslett',axis=1)

temp_forskjell_kald = df2['temp_VP_kald_inn']-df2['temp_VP_kald_ut']
temp_forskjell_varm = df2['temp_VP_varm_ut']-df2['temp_VP_varm_inn']

varmekap = 4200 #J/kgK
massestrom = 6 #kg/s

effekt_VP = varmekap*massestrom*temp_forskjell_varm/1000  # kW


# Leser inn lufttemperaturer
# Last ned CSV-fil for dette fra   https://seklima.met.no/ 
df_lufttemp = pd.read_csv('Temperaturdata_TRD.csv', 
        sep=";", 
        usecols=[2,3],
        header=0,
        names = ['tid','lufttemp'])
df_lufttemp['tid'] = df_lufttemp['tid'].apply(temp_to_datetime)

df_lufttemp = df_lufttemp.replace(',','.',regex=True)
df_lufttemp['lufttemp'] = df_lufttemp['lufttemp'].astype(float)


# Henter ut lufttemperaturer for relevant tidspunkt
starttid = df['tid'].iloc[0].replace(minute=0, second=0)
if df['tid'].iloc[0].minute >= 30:
    starttid += timedelta(hours=1)

sluttid = df['tid'].iloc[-1].replace(minute=0, second=0)
if df['tid'].iloc[-1].minute >= 30:
    starttid += timedelta(hours=1)

start_indeks = np.where(df_lufttemp['tid'] == starttid)
start_indeks = int(start_indeks[0])
slutt_indeks = np.where(df_lufttemp['tid'] == sluttid)
slutt_indeks = int(slutt_indeks[0])
relevante_lufttemp = df_lufttemp.iloc[start_indeks:slutt_indeks+1,:]
relevante_lufttemp = relevante_lufttemp.reset_index()

# Plottefunksjoner
if tidshorisont == 'Siste 24 timer':
    startindeks = -24
elif tidshorisont == 'Siste 7 dager':
    startindeks = -168
elif tidshorisont == 'Siste 30 dager':
    startindeks = -720
elif tidshorisont == 'Siste 3 måneder':
    startindeks = -2208
elif tidshorisont == 'Siste 12 måneder':
    startindeks = -8760
elif tidshorisont == 'All data':
    startindeks = 0


def plottefunksjon2stk(x_data,x_navn,y_data1,y_navn1,y_data2,y_navn2,yakse_navn,tittel):
    til_plot = pd.DataFrame({x_navn : x_data[startindeks:], y_navn1 : y_data1[startindeks:], y_navn2 : y_data2[startindeks:]})
    fig = px.line(til_plot, x=x_navn, y=[y_navn1,y_navn2], title=tittel, color_discrete_sequence=['#367A2F', '#FFC358'])
    fig.update_layout(xaxis_title=x_navn, yaxis_title=yakse_navn,legend_title=None)
    #st.plotly_chart(fig)
    return fig

def plottefunksjon_bar(x_data, x_navn, y_data1, y_navn1, y_data2, y_navn2, yakse_navn, tittel):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(x=x_data[startindeks:], y=y_data1[startindeks:], name=y_navn1, marker=dict(color='#367A2F')))
    fig.add_trace(go.Scatter(x=x_data[startindeks:], y=y_data2[startindeks:], name=y_navn2, marker=dict(color='#FFC358')), secondary_y=True)
    
    fig.update_layout(title=tittel, xaxis_title=x_navn, yaxis_title=yakse_navn, legend_title=None)
    fig.update_yaxes(title_text=y_navn1, secondary_y=False)
    fig.update_yaxes(title_text=y_navn2, secondary_y=True)
    
    return fig

def plottefunksjon_2akser(x_data, x_navn, y_data1, y_navn1, y_data2, y_navn2, yakse_navn, tittel):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x_data[startindeks:], y=y_data1[startindeks:], mode="lines", name=y_navn1, line=dict(color='#367A2F')), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_data[startindeks:], y=y_data2[startindeks:], mode="lines", name=y_navn2, line=dict(color='#FFC358')), secondary_y=True)
    fig.update_layout(title=tittel, xaxis_title=x_navn, yaxis_title=yakse_navn, legend_title=None)
    fig.update_yaxes(title_text=y_navn1, secondary_y=False)
    fig.update_yaxes(title_text=y_navn2, secondary_y=True)
    #st.plotly_chart(fig)
    return fig

def plottefunksjon3stk(x_data,x_navn,y_data1,y_navn1,y_data2,y_navn2,y_data3,y_navn3,yakse_navn,tittel):
    til_plot = pd.DataFrame({x_navn : x_data[startindeks:], y_navn1 : y_data1[startindeks:], y_navn2 : y_data2[startindeks:],  y_navn3 : y_data3[startindeks:]})
    fig = px.line(til_plot, x=x_navn, y=[y_navn1,y_navn2,y_navn3], title=tittel, color_discrete_sequence=['#367A2F', '#C2CF9F', '#FFC358', '#FFE7BC'])
    fig.update_layout(xaxis_title=x_navn, yaxis_title=yakse_navn,legend_title=None)
    #st.plotly_chart(fig)
    return fig

def plottefunksjon4stk(x_data,x_navn,y_data1,y_navn1,y_data2,y_navn2,y_data3,y_navn3,y_data4,y_navn4,yakse_navn,tittel):
    til_plot = pd.DataFrame({x_navn : x_data[startindeks:], y_navn1 : y_data1[startindeks:], y_navn2 : y_data2[startindeks:],  y_navn3 : y_data3[startindeks:],  y_navn4 : y_data4[startindeks:]})
    fig = px.line(til_plot, x=x_navn, y=[y_navn1,y_navn2,y_navn3,y_navn4], title=tittel, color_discrete_sequence=['#367A2F', '#C2CF9F', '#FFC358', '#FFE7BC'])
    fig.update_layout(xaxis_title=x_navn, yaxis_title=yakse_navn,legend_title=None)
    #st.plotly_chart(fig)
    return fig

## Figurer:
# Baner
fig1 = plottefunksjon4stk(x_data=df['tid'],
                   x_navn='Tid',
                   y_data1=df['temp_tilbane1'],
                   y_navn1='Temperatur til bane 1',
                   y_data2=df['temp_tilbane2'],
                   y_navn2='Temperatur til bane 2',
                   y_data3=df['temp_frabane1'],
                   y_navn3='Temperatur fra bane 1',
                   y_data4=df['temp_frabane2'],
                   y_navn4='Temperatur fra bane 2',
                   yakse_navn='Temperatur (\u2103)',
                   tittel='Tur- og returtemperaturer for baner')
st.plotly_chart(fig1)

#fig2 = plottefunksjon2stk(x_data=df['tid'],
#                   x_navn='Tid',
#                   y_data1=df['temp_tilbane2'],
#                   y_navn1='Temperatur til banen',
#                   y_data2=df['temp_frabane2'],
#                   y_navn2='Temperatur fra banen',
#                   yakse_navn='Temperatur (\u2103)',
#                   tittel='Tur- og returtemperaturer for bane 2')
#st.plotly_chart(fig2)

# Ukjent
fig3 = plottefunksjon2stk(x_data=df['tid'],
                   x_navn='Tid',
                   y_data1=df['RT402'],
                   y_navn1='Ukjent temp. 1',
                   y_data2=df['RT502'],
                   y_navn2='Ukjent temp. 2',
                   yakse_navn='Temperatur (\u2103)',
                   tittel='Ukjente temperaturer')
st.plotly_chart(fig3)

# Brønner
fig4 = plottefunksjon4stk(x_data=df['tid'],
                   x_navn='Tid',
                   y_data1=df['temp_tilbronn40'],
                   y_navn1='Temperatur til 40 brønner',
                   y_data2=df['temp_tilbronn20'],
                   y_navn2='Temperatur til 20 brønner',
                    y_data3=df['temp_frabronn40'],
                   y_navn3='Temperatur fra 40 brønner',
                   y_data4=df['temp_frabronn20'],
                   y_navn4='Temperatur fra 20 brønner',
                   yakse_navn='Temperatur (\u2103)',
                   tittel='Tur- og returtemperaturer for brønner')
st.plotly_chart(fig4)

#fig5 = plottefunksjon2stk(x_data=df['tid'],
#                   x_navn='Tid',
#                   y_data1=df['temp_tilbronn20'],
#                   y_navn1='Temperatur til brønn',
#                   y_data2=df['temp_frabronn20'],
#                   y_navn2='Temperatur fra brønn',
#                   yakse_navn='Temperatur (\u2103)',
#                   tittel='Tur- og returtemperaturer for brønn 2')
#st.plotly_chart(fig5)

fig6 = plottefunksjon4stk(x_data=df['tid'],
                   x_navn='Tid',
                   y_data1=snittemp_tilbronn,
                   y_navn1='Temperatur til brønner',
                   y_data2=snittemp_frabronn,
                   y_navn2='Temperatur fra brønn',
                   y_data3=vaesketemp_bronn,
                   y_navn3='Gj. snittlig væsketemp.',
                   y_data4=relevante_lufttemp['lufttemp'],
                   y_navn4='Lufttemperatur.',
                   yakse_navn='Temperatur (\u2103)',
                   tittel='Gjennomsnittlige tur- og returtemperaturer for brønner')
st.plotly_chart(fig6)

fig7 = plottefunksjon2stk(x_data=df2['tid'],
                   x_navn='Tid',
                   y_data1=df2['bronntemp40'],
                   y_navn1='Temperatur i brønn 40',
                   y_data2=df2['bronntemp20'],
                   y_navn2='Temperatur i brønn 20',
                   yakse_navn='Temperatur (\u2103)',
                   tittel='Temperatur i brønner')
st.plotly_chart(fig7)

fig8 = plottefunksjon_2akser(x_data=df2['tid'],
                      x_navn='Tid',
                      y_data1=tempforskjell_bronn,
                      y_navn1='Temperaturendring i brønn',
                      y_data2=effekt_tilbronn,
                      y_navn2='Effekt levert til brønnen',
                      yakse_navn='Temperatur (\u2103)',
                      tittel='Temperatur- og effektendring i brønner')
st.plotly_chart(fig8)

# Varmepumpe
#fig9 = plottefunksjon2stk(x_data=df2['tid'],
#                   x_navn='Tid',
#                   y_data1=df2['temp_VP_varm_ut'],
#                   y_navn1='Temperatur ut varm side',
#                   y_data2=df2['temp_VP_varm_inn'],
#                   y_navn2='Temperatur inn varm side',
#                   yakse_navn='Temperatur (\u2103)',
#                   tittel='Temperaturer varm side av VP')
#st.plotly_chart(fig9)

#fig10 = plottefunksjon2stk(x_data=df2['tid'],
#                   x_navn='Tid',
#                   y_data1=df2['temp_VP_kald_ut'],
#                   y_navn1='Temperatur ut kald side',
#                   y_data2=df2['temp_VP_kald_inn'],
#                   y_navn2='Temperatur inn kald side',
#                   yakse_navn='Temperatur (\u2103)',
#                   tittel='Temperaturer kald side av VP')
#st.plotly_chart(fig10)

fig11 = plottefunksjon4stk(x_data=df2['tid'],
                   x_navn='Tid',
                   y_data1=df2['temp_VP_kald_ut'],
                   y_navn1='Temperatur ut kald side',
                   y_data2=df2['temp_VP_kald_inn'],
                   y_navn2='Temperatur inn kald side',
                   y_data3=df2['temp_VP_varm_ut'],
                   y_navn3='Temperatur ut varm side',
                   y_data4=df2['temp_VP_varm_inn'],
                   y_navn4='Temperatur inn varm side',
                   yakse_navn='Temperatur (\u2103)',
                   tittel='Temperaturer inn og ut av VP')
st.plotly_chart(fig11)

fig12 = plottefunksjon_bar(x_data=df2['tid'],
                   x_navn='Tid',
                   y_data1=effekt_VP,
                   y_navn1='Effekt (kW)',
                   y_data2=relevante_lufttemp['lufttemp'],
                   y_navn2='Lufttemperatur (\u2103)',
                   yakse_navn='Effekt (kW)',
                   tittel='Effekt fra varmepumpen til brønner')
st.plotly_chart(fig12)