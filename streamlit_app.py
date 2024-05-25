import pickle
import streamlit as st
# loading in the model to predict on the data
pickle_in = open('rf2.pkl', 'rb')
rf2 = pickle.load(pickle_in)

def main():
    st.set_page_config(layout="wide", page_title="Insurance Fraud Prediction App")
    st.title("Insurance Fraud Prediction")

    html_temp = """
    <div style ="background-color:lightblue;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Application for Predicting Insurance Fraud </h1>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html = True)

    csv_path = "df2fraud.csv"
    df2 = pd.read_csv(csv_path)
    st.write("This app employs a Random Forest Model trained on an insurance claims dataset to predict fraud.")

    st.write("## Preview of Dataset")
    st.write(df2.head())

    st.write("## Predictions from the Random Forest Model")
    user_input = {}

    X = df2.drop(columns=['fraud_reported'])

    categorical_columns = ['policy_csl', 'insured_education_level', 'insured_hobbies', 'insured_relationship',
                           'incident_severity', 'authorities_contacted', 'incident_state', 'incident_city']

    descriptions = {
    'incident_severity': '0-Major_Damage, 1-Minor_Damage, 2-Total_Loss, 3-Trivial_Damage',
    'insured_hobbies': '0-basejumping, 1-basketball, 2-boardgames, 3-bungiejumping, 4-camping, 5-chess, 6-crossfit, 7-dancing, 8-exercise, 9-golf, 10-hiking, 11-kayaking, 12-movies, 13-paintball, 14-polo, 15-reading, 16-skydiving, 17-sleeping, 18-videogames, 19-yachting',
    'insured_education_level': '0-Associate, 1-College, 2- HighSchool, 3-JD, 4-Masters, 5-MD, 6-PhD',
    'incident_city': '0-Arlington, 1-Columbus, 2-Hillsdale, 3-Northbend, 4-Northbrook, 5-Riverwood, 6-Springfield',
    'incident_state': '0-NC, 1-NY, 2-OH, 3-PA, 4-SC, 5-VA, 6-WV',
    'insured_relationship': '0-husband, 1-not_in_family, 2-other_relative, 3-own_child, 4-unmarried, 5-wife',
    'authorities_contacted': '0-Ambulance, 1-Fire, 2-Other, 3-Police',
    'policy_csl': '0-100/300, 1-250/500, 2-500/1000',}

    for column in X.columns:
        if column in categorical_columns:
            description = descriptions.get(column, '')
            sorted_unique_values = sorted(df2[column].unique())
            user_input[column] = st.selectbox(f"{column}: {description}", sorted_unique_values)
        else:
            description = descriptions.get(column, '')
            user_input[column] = st.number_input(column, min_value=float(df2[column].min()), max_value=float(df2[column].max()))

    # Convert user input to dataframe
    user_input_df2 = pd.DataFrame([user_input])

    if st.button('Predict'):
       prediction = rf2.predict(user_input_df2)
       if prediction[0] == 1:
           st.write('The claim is predicted to be fraudulent.')
       else:
           st.write('The claim is predicted to be non-fraudulent.')

if __name__=='__main__':
	main()
