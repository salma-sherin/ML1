import streamlit as st 
import pickle
from PIL import Image

def main():
    st.title(":rainbow[TITANIC]")
    image=Image.open('titanicship.jpg')

    st.image(image,width=600)


    PassengerId=st.text_input(':red[PassengerId]','')
    Pclass=st.text_input(':red[Pclass]','')
    Name=st.text_input(':red[Name]','')
    Age=st.text_input(':red[Age]','')
    SibSp=st.text_input(':red[SibSp]','')
    Parch=st.text_input(':red[Parch]','')
    Ticket=st.text_input(':red[Ticket]','')
    Fare=st.text_input(':red[Fare]','')
    Cabin=st.text_input(':red[Cabin]','')
    female=st.text_input(':red[female]','')
    male=st.text_input(':red[male]','')
    
    
    # opt1=['0','1']
    # female = st.radio(':green[female]', opt1)
    # male = st.radio(':green[male]', opt1)

    opt2=['0','1','2']
    Embarked_C = st.radio(':green[Embarked_C]', opt2)
    Embarked_Q = st.radio(':green[Embarked_Q]', opt2)   
    Embarked_S = st.radio(':green[Embarked_S]', opt2)


    

    features=[PassengerId,Pclass,Name,Age,SibSp,Parch,Ticket,Fare,Cabin,female,male,Embarked_C,Embarked_Q,Embarked_S]
    
    model=pickle.load(open('knn.sav','rb'))
    standard=pickle.load(open('sd1.sav','rb'))

    pred=st.button('PREDICT')

    if pred:

        prediction=model.predict(standard.transform([features]))

        if prediction==0:
            st.write('NOT SURVIVED')

        else:
            st.write('SURVIVED')
    

  

    
main()
  
