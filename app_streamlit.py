import streamlit as st
import pickle
import numpy as np


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://drive.google.com/u/0/uc?id=1umkcLfyjipUYSaV9Hjx9qX_IxCmN-Esx&export=download");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# st.markdown(f'<h1 style="color:#33ff33;font-size:40px;">{"Laptop Price Predictor"}</h1>', unsafe_allow_html=True)
st.title('Laptop Price Predictor')

st.warning('All fields are mandatory')

# brand
company = st.selectbox('Brand*', df['Company'].unique())

# type
type = st.selectbox('Type*', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)*', [2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the laptop*')
st.warning('Note - Most common laptop weights are between 2 to 3 (kgs)')

# Touchscreen
touchscreen = st.selectbox('Touchscreen*', ['No','Yes'])

# IPS
ips = st.selectbox('IPS*', ['No','Yes'])

# screen size
screen_size = st.number_input('Screen_Size*')

# resolution
resolution = st.selectbox('Screen Resolution*',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# cpu
cpu = st.selectbox('CPU*', df['Cpu Brand'].unique())

hdd = st.selectbox('HDD(in GB)*',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)*',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU*',df['Gpu brand'].unique())

os = st.selectbox('OS*',df['os'].unique())

st.caption('© Dhananjay Raut')

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = np.sqrt((X_res ** 2) + (Y_res ** 2)) / screen_size

    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)

    price = np.around(int(np.exp(pipe.predict(query)[0])),2)
    st.title('Approximate laptop price: ' + str(price) + ' Rs.')





    










