import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image


diabetes_df = pd.read_csv('diabetes.csv')


X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)


model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)


train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_y_pred)
test_acc = accuracy_score(y_test, test_y_pred)


def app():

    
    img = Image.open("img.jpeg")
    img = img.resize((200, 200))
    st.image(img, caption="Diabetes Image", width=200)

    st.title("Diabetes Disease Prediction")

    
    st.write(f"**Training Accuracy:** {train_acc * 100:.2f}%")
    st.write(f"**Testing Accuracy:** {test_acc * 100:.2f}%")

    
    st.sidebar.title("Input Features")

    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider(
        'Diabetes Pedigree Function', 0.078, 2.42, 0.3725, step=0.001
    )
    age = st.sidebar.slider('Age', 21, 81, 29)

    
    input_data = np.array([
        preg, glucose, bp, skinthickness,
        insulin, bmi, dpf, age
    ]).reshape(1, -1)

    
    input_data_scaled = scaler.transform(input_data)

    
    prediction = model.predict(input_data_scaled)

    
    if prediction[0] == 1:
        st.warning("This person **has Diabetes**")
    else:
        st.success("This person **does NOT have Diabetes**")


if __name__ == "__main__":
    app()
