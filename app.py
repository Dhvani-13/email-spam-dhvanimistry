# for run -> streamlit run app.py

import streamlit as st
import pickle

model = pickle.load(open('spam.pkl','rb'))
cv = pickle.load(open('vectorization.pkl','rb'))

st.title("Email spam classification Application")
st.write("This is a Machine Learning Application to Classify email as spam or ham.")
user_input = st.text_area("Enter an email to classify",height=150)

if st.button("Classify"):
    if user_input:  
        data = [user_input]
        vect = cv.transform(data).toarray()
        pred = model.predict(vect)
        if pred[0] == 0:
            st.write("This email is not spam..")
            st.success("This email is not spam..")
        else:
            st.error("This is spam email..")
    else:
        print("Please enter email..")
