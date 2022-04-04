import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report



st.title('Machine Learning (Classification)')

st.sidebar.header("""
This is an amateur Web App using python libraries such as Streamlit, Sklearn etc.   
""")

st.sidebar.write ("Feel free to flex your codes to me. For more info, we could get in touch through:")

image = Image.open('passpo.png')
st.sidebar.image(image)

st.sidebar.write("<a href='https://www.linkedin.com/in/saamuhymin/'>Br. Abdul Muhaimin </a>", unsafe_allow_html=True)   
    
data = pd.read_csv(r'https://raw.githubusercontent.com/saamuhymin/interviewdata/main/Interview.csv')

data = data.drop(['Have you obtained the necessary permission to start at the required time','Hope there will beNounScheduledmeetings','Can I Call you three hours before the interview and follow up on your attendance for the interview','Can I have an alternative number/ desk number. I assure you that I will Not trouble you too much','Have you taken a printout of your updated resume. Have you read the JD and understood the same','Are you clear with the venue details and the landmark.','Has the call letter been shared'], axis=1)
data = data.drop(['Client name', 'Name(Cand ID)', 'Location','Date of Interview','Candidate Current Location','Candidate Job Location'], axis=1)

data = data.dropna()

labelencoder1 = LabelEncoder()
labelencoder2 = LabelEncoder()
labelencoder3 = LabelEncoder()
labelencoder4 = LabelEncoder()
labelencoder5 = LabelEncoder()
labelencoder6 = LabelEncoder()
labelencoder7 = LabelEncoder()
labelencoder8 = LabelEncoder()
labelencoder9 = LabelEncoder()
labelencoder10 = LabelEncoder()

data['Industry'] = labelencoder1.fit_transform(data['Industry'])
data['Position to be closed'] = labelencoder2.fit_transform(data['Position to be closed'])
data['Nature of Skillset'] = labelencoder3.fit_transform(data['Nature of Skillset'])
data['Interview Type'] = labelencoder4.fit_transform(data['Interview Type'])
data['Gender'] = labelencoder5.fit_transform(data['Gender'])
data['Interview Venue'] = labelencoder6.fit_transform(data['Interview Venue'])
data['Candidate Native location'] = labelencoder7.fit_transform(data['Candidate Native location'])
data['Expected Attendance'] = labelencoder8.fit_transform(data['Expected Attendance'])
data['Observed Attendance'] = labelencoder9.fit_transform(data['Observed Attendance'])
data['Marital Status'] = labelencoder10.fit_transform(data['Marital Status'])


X = data.drop(['Observed Attendance'], axis=1)
y = data['Observed Attendance']

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest') )

test_data_ratio = st.sidebar.slider('Select testing size or ratio', 
                                    min_value= 0.10, 
                                    max_value = 0.50,
                                    value=0.2)
random_state = st.sidebar.slider('Select random state', 1, 9999,value=1234)

st.write("## 1: Summary (X variables)")


if len(X)==0:
   st.write("<font color='Aquamarine'>Note: Predictors @ X variables have not been selected.</font>", unsafe_allow_html=True)
else:
   st.write('Shape of predictors @ X variables :', X.shape)
   st.write('Summary of predictors @ X variables:', pd.DataFrame(X).describe())

st.write("## 2: Summary (y variable)")

if len(y)==0:
   st.write("<font color='Aquamarine'>Note: Label @ y variable has not been selected.</font>", unsafe_allow_html=True)
elif len(np.unique(y)) <5:
     st.write('Observed Attendance:', len(np.unique(y)))

else: 
   st.write("<font color='red'>Warning: System detects an unusual number of unique classes. Please make sure that the label @ y is a categorical variable. Ignore this warning message if you are sure that the y is a categorical variable.</font>", unsafe_allow_html=True)
   st.write('Observed Attendance:', len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0,value=1.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15,value=5)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15,value=5)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100,value=10)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=random_state)
    return clf

clf = get_classifier(classifier_name, params)


st.write("## 3: Classification Report")

if len(X)!=0 and len(y)!=0: 


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_ratio, random_state=random_state)

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)    

  clf.fit(X_train_scaled, y_train)
  y_pred = clf.predict(X_test_scaled)


  st.write('Classifier:',classifier_name)
  st.write('Classification report:')
  report = classification_report(y_test, y_pred,output_dict=True)
  df = pd.DataFrame(report).transpose()
  st.write(df)

else: 
   st.write("<font color='Aquamarine'>Note: No classification report generated.</font>", unsafe_allow_html=True)


st.write("## 4: Principal Component Analysis Plot")
suitable = 1
if len(X) <2:
  st.write("<font color='Aquamarine'>Note: No PCA plot as it requires at least two predictors.</font>", unsafe_allow_html=True)
  suitable = 0
else:
    for names in X:
        if names:
           st.write("<font color='Aquamarine'>Note: No PCA plot as it only supports numerical predictors.</font>", unsafe_allow_html=True)
           suitable = 0
           break

if suitable == 1:
   pca = PCA(2)
   X_projected = pca.fit_transform(X)

   x1 = X_projected[:, 0]
   x2 = X_projected[:, 1]

   fig = plt.figure()
   plt.scatter(x1, x2,
               c=y, alpha=0.8,
               cmap='viridis')

   plt.xlabel('Principal Component 1')
   plt.ylabel('Principal Component 2')
   plt.colorbar()
   st.pyplot(fig)
            
            
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
