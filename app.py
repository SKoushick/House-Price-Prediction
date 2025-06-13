import pandas as pd 
import numpy as np
import streamlit as st 
import matplotlib.pyplot as plt 
import seaborn as sns 
from xgboost import plot_importance
import joblib as jl 


st.set_page_config(page_title='Melbourne House Price Prediction App',page_icon='🏚️',layout='wide')

st.title('🏚️ Melbourne House Price Prediction App')
st.caption('built with XGBoost and Streamlit -- Predict house prices interactively')


@st.cache_resource

def load_data():
    df=pd.read_csv("Melbourne_df.csv")
    return df

df=load_data()

@st.cache_resource
def load_model():
    model = jl.load("xgboost_model.pkl")
    return model

model = load_model()


st.sidebar.header('🔍Input Features')
st.sidebar.text('Adjiust the values to predict house price')
rooms = st.sidebar.slider('🛏️Rooms',0,10)
property_type = st.sidebar.selectbox('Property Type',["h - House", "u - Unit", "t - Townhouse"])
postcode = st.sidebar.number_input('📍Postcode', min_value=3000,max_value=3999,value = 3000)
distance = st.sidebar.slider('🚗Distance to city',min_value = 0.0,max_value = 50.0,value = 10.2,step=0.2)
prop_count = st.sidebar.number_input('Property Count in Suburb',min_value=100,max_value=10000,value = 1000)



type_map={'h - House':0,'u - Unit':1,'t - Townhouse':2}

type_encoded = type_map[property_type]

input_data = np.array([[rooms, type_encoded, postcode, distance, prop_count]])
prediction = model.predict(input_data)[0]
formatted_prediction = "${:,.2f}".format(prediction)

col1,col2 = st.columns([2,3])

with col1:
    st.header('🏠Your Input Features')
    input_df = pd.DataFrame({
        'Feature': ['Rooms', 'Property Type', 'Postcode', 'Distance to City (km)', 'Property Count'],
        'Value': [rooms, property_type.split(" - ")[1], postcode, distance, prop_count]
    })
    st.table(input_df)

with col2:
    st.header('🔮Predicted House Price')
    st.markdown(f"<h2 style='color:#4CAF50;'>{formatted_prediction}</h2>", unsafe_allow_html=True)
    st.info("This is an estimate based on the selected features and our trained XGBoost model.")


st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Distribution of House Prices")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(df["Price"], kde=True, ax=ax1)
    ax1.set_xlabel("Price")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

with col2:
    st.markdown("### 📈 Feature Importance")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    plot_importance(model, ax=ax2, max_num_features=5, color="#4285F4")
    st.pyplot(fig2)


st.markdown("---") 
st.subheader("🧾 Dataset Sample")
st.write(df.head())

st.markdown("---")
st.markdown("© 2025 House Price Predictor | Built By Koushick Saravanan")