import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Split the dataset into features (X) and target variable (y)
data = pd.read_csv('happydata.csv')
X = data.drop('happy', axis=1)
y = data['happy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = model.predict(X_test)
filename = 'model_happiness.sav'
pickle.dump(model,open(filename,'wb'))

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

@st.cache(suppress_st_warning=True)
def get_value(val, my_dict):    
    for key, value in my_dict.items():        
        if val == key:            
            return value



# Load model
model = pickle.load(open('model_happiness.sav', 'rb'))
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction'])  # two pages
if app_mode == 'Home':    
    # Title
    st.markdown("<h1 style='text-align: center;'>Happiness Classification Analysis</h1><br>", unsafe_allow_html=True)
    
    # Image
    st.image('hpy_pict.jpg')
    
    # Header for Dataset
    st.header("Dataset")

    # Read CSV  
    df1 = pd.read_csv('happydata.csv')

    # Display DataFrame
    st.dataframe(df1)
    
    # Chart for Infoavail
    st.write("Grafik Infoavail")
    chart_infoavail = pd.DataFrame(df1, columns=["infoavail"])
    st.line_chart(chart_infoavail)
    
    # Chart for House Cost
    st.write("Grafik Housecost")
    chart_housecost = pd.DataFrame(df1, columns=["housecost"])
    st.line_chart(chart_housecost)
    
    # Chart for School Quality
    st.write("Grafik Schoolquality")
    chart_schoolquality = pd.DataFrame(df1, columns=["schoolquality"])
    st.line_chart(chart_schoolquality)
    
    # Chart for Police Trust
    st.write("Grafik Policetrust")
    chart_policetrust = pd.DataFrame(df1, columns=["policetrust"])
    st.line_chart(chart_policetrust)

    # Chart for Events
    st.write("Grafik Events")
    chart_policetrust = pd.DataFrame(df1, columns=["events"])
    st.line_chart(chart_policetrust)
    
    # Chart for Street Quality
    st.write("Grafik Schoolquality")
    chart_streetquality = pd.DataFrame(df1, columns=["streetquality"])
    st.line_chart(chart_streetquality)
    
elif app_mode == 'Prediction':    
    st.image('hpy_pict.jpg')    

    # Input features
    infoavail = st.number_input('Infoavail', 0, 10000000)
    housecost = st.number_input('Housecost', 0, 10000000)
    schoolquality = st.number_input('Schoolquality', 0, 10000000)
    policetrust = st.number_input('Policetrust', 0, 10000000)
    streetquality = st.number_input('Streetquality', 0, 10000000)
    event = st.number_input('events', 0, 10000000)

    # Prediction button
    if st.button('Prediksi'):
        happiness_classification = model.predict([[infoavail, housecost, schoolquality, policetrust, streetquality, event]])
        # Format and display the prediction
        happiness_classification_str = np.array(happiness_classification)
        happiness_classification_float = float(happiness_classification_str[0])
        happiness_classification_formatted = "{:,.2f}".format(happiness_classification_float)
        st.markdown(f'Tingkat Bahagia:  {happiness_classification_formatted}')
        if happiness_classification_float == 1.0 :
            st.markdown('Bahagia')
        else :
            st.markdown('Tidak bahagia')


    
    
