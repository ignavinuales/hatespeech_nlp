import streamlit as st
import requests
from update_db import UpdateDB


def main():
    if 'updateDB' not in st.session_state:
        st.session_state.updateDB = None

    st.title("Hate speech detection")
    st.markdown("""
    This web app detects hate speech from a text. Enter any suitable text to try it out.
    """)
    st.caption('It only supports English as of now.')
    input_text = st.text_input("Enter text:"
    )

    if st.button("Predict"):
        with st.spinner("Wait for it..."):

            predict_api_endpoint = "http://144.22.214.61:8000/predict"  # HTTP endpoint to make prediction
            # predict_api_endpoint = "http://127.0.0.1:8000/predict"  # HTTP endpoint to make prediction
            response = requests.post(predict_api_endpoint, json={'text': input_text})  # Send request to make prediction
            prediction = response.text
            st.success(prediction, icon="🤖") # Display prediciton to user

        entry_db = UpdateDB(input=input_text, output=prediction) # Instantiate UpdateDB object
        entry_db.new_entry_db() # Insert input and output values to the database
        st.session_state.updateDB = entry_db  # Save object in the current streamlit session

    with st.expander('**Please give me a feeback whether the result was satisfactory**'):
        st.markdown("Was the prediction correct?")

        yes_check = st.checkbox("Yes")
        no_check = st.checkbox("No")
        if yes_check and no_check:
            st.warning('You cannot check both boxes')

        # Send feedback. Add the user's feedback to the database:
        if st.button("Send feedback"):
            try:
                if yes_check and not no_check:
                    st.session_state.updateDB.add_feedback_db('Yes')
                    st.success("Feedback sent, thank you!")
                    st.balloons()
                    
                if no_check and not yes_check:
                    st.session_state.updateDB.add_feedback_db('No')
                    st.success("Feedback sent, thank you!")
                if yes_check and no_check :
                    st.warning('You cannot check both boxes')
                if not yes_check and not no_check:
                    st.warning("Please select an option")
            except:
                pass
                # st.warning("First make a prediction")

                

    
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("**Github:** https://github.com/ignavinuales/nlp_hatespeech")
    st.markdown("""**Objective of project:** 
     The goal of hate speech detection is to determine if communication contains hatred or promotes
     violence against an individual or a group of individuals. Prejudice against "protected traits" such their ethnicity, 
     gender, sexual orientation, religion, age, etc., is typically the basis for this. """)

if __name__ == '__main__':
    main()
