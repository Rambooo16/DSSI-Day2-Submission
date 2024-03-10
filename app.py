import streamlit as st
from src.Evaluating import evaluate



def main():

    # Set page config
    apptitle = 'DSSI Workshop App'

    st.set_page_config(page_title=apptitle, layout='wide')

    st.title('Workshop Streamlit Application for Prediction')
    st.balloons()

    evaluation = evaluate()

    st.dataframe(evaluation,use_container_width=True)
    return None

if __name__ == "__main__":
    main()

