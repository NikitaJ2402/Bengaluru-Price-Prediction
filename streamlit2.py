# #streamlit run streamlitminip.py 
# #bengaluru house project

# import streamlit as st 
# import pandas as pd
# import numpy as np 
# import matplotlib.pyplot as plt 
# import seaborn as sns

# st.title('WEB APP FOR HOUSE PRICE PREDICTION FOR BENGELURU')
# st.title('Case study on House Price Dataset')

# data=pd.read_csv("D:\\Programmes\\Python\\DataSets\\Bengaluru_House_Data.csv")

# st.write('Shape of Dataset : ',data.shape)
# menu=st.sidebar.radio('Menu',['home','Prediction Price'])

# if menu=='home':
#     st.header('Tabular Data of a Home Prices')
#     if st.checkbox('Tabular Data'):
#         st.table(data.head(100))

#     st.header('Statistical Summary of a DataFrame')
#     if st.checkbox('Statistics'):
#         st.table(data.describe())

#     st.header('Top Rated Housing in Bengaluru : ')
#     if st.checkbox('Top-Ranked Housing Details'):
#         st.table(data.describe(include='object'))


#     st.title('Graphs')
#     graph=st.selectbox('Different Types of Graph',['Scatter Plot','Bar Plot','Histogram'])
#     if graph=='Scatter Plot':
#          value=st.slider('Distribution For Number of Balconies',0,6)
#          data=data.loc[data['balcony']>=value]
#          fig,ax=plt.subplots(figsize=(10,5))
#          sns.scatterplot(data=data,x='balcony',y='price',hue='area_type')    
#          st.pyplot(fig)

#     if graph=='Bar Plot':
#         fig,ax=plt.subplots(figsize=(5,3))
#         sns.barplot(x='area_type',y=data.area_type.index,data=data)
#         st.pyplot(fig)

#     if graph=='Histogram':
#         fig,ax=plt.subplots(figsize=(5,3))
#         sns.distplot(data.price,kde=True)
#         st.pyplot(fig)
   
# if menu=='Prediction Price':
#     st.title('Prediction Price of Houses in Bengaluru')
#     data1=data.fillna('0')
#     from sklearn.model_selection import train_test_split
#     X_train,X_test,y_train,y_test=train_test_split(data1[['bath','balcony']],data.price,test_size=0.2)
#     from sklearn.linear_model import LinearRegression

#     model=LinearRegression()
#     model.fit(X_train,y_train)
    
    
#     value1=st.text_input('Enter the Bathroom numbers : ')
#     value2=st.text_input('Enter the Balcony Number : ')
#     prediction=model.predict(value1,value2)

#     if st.button('Price Prediction($)'):
#         st.write(f'{prediction}')



# streamlit run streamlitminip.py 
# bengaluru house project

import streamlit as st 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title('WEB APP FOR HOUSE PRICE PREDICTION FOR BENGALURU')
st.title('Case study on House Price Dataset')

data = pd.read_csv("D:\\Programmes\\Python\\DataSets\\Bengaluru_House_Data.csv")

st.write('Shape of Dataset : ', data.shape)
menu = st.sidebar.radio('Menu', ['home', 'Prediction Price'])

if menu == 'home':
    st.header('Tabular Data of Home Prices')
    if st.checkbox('Tabular Data'):
        st.table(data.head(100))

    st.header('Statistical Summary of the DataFrame')
    if st.checkbox('Statistics'):
        st.table(data.describe())

    st.header('Top Rated Housing in Bengaluru:')
    if st.checkbox('Top-Ranked Housing Details'):
        st.table(data.describe(include='object'))

    st.title('Graphs')
    graph = st.selectbox('Different Types of Graphs', ['Scatter Plot', 'Bar Plot', 'Histogram'])
    
    if graph == 'Scatter Plot':
        value = st.slider('Distribution For Number of Balconies', 0, 6)
        data_filtered = data.loc[data['balcony'] >= value]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=data_filtered, x='balcony', y='price', hue='area_type', ax=ax)
        st.pyplot(fig)

    if graph == 'Bar Plot':
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x='area_type', y=data['price'], data=data, ax=ax)
        st.pyplot(fig)

    if graph == 'Histogram':
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(data['price'], kde=True, ax=ax)
        st.pyplot(fig)

if menu == 'Prediction Price':
    st.title('Prediction Price of Houses in Bengaluru')
    data1 = data.fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(data1[['bath', 'balcony']], data['price'], test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    value1 = st.text_input('Enter the number of bathrooms:')
    value2 = st.text_input('Enter the number of balconies:')
    
    if st.button('Price Prediction ($)'):
        try:
            value1 = float(value1)
            value2 = float(value2)
            prediction_input = np.array([[value1, value2]])
            prediction = model.predict(prediction_input)
            st.write(f'Predicted Price: ${prediction[0]:,.2f}')
        except ValueError:
            st.write('Please enter valid numerical values for bathroom and balcony numbers.')
