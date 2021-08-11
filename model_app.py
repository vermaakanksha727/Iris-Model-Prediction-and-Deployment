import streamlit as st 
import pickle


log=pickle.load(open('log_model.pkl','rb'))
dtree=pickle.load(open('dtree_model.pkl','rb'))

def main():
    st.title('IRIS MODEL')
    html_temp="""
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Classification</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    models=['Logistic Regression','Decision Tree']
    option=st.sidebar.selectbox('Which model would you like to use?',models)
    st.subheader(option)
    sl=st.slider('Select Sepal Length',0.0,10.0)
    sw=st.slider('Select Sepal Width',0.0,10.0)
    pl=st.slider('Select Petal Length',0.0,10.0)
    pw=st.slider('Select Petal Width',0.0,10.0)
    inputs=[[sl,sw,pl,pw]]
    if st.button('Prediction'):
        if option=='Logistic Regression':
            st.success(log.predict(inputs))
        else:
            st.success(dtree.predict(inputs))
        

if __name__=='__main__':
    main()

