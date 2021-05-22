import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Growify ML Models", page_icon="🌳", layout='centered', initial_sidebar_state="collapsed")

def main(): 

    def load_model(file):
        loaded_model = load(file)
        return loaded_model

    col1, col2 = st.beta_columns([1,1])

    with col1:
        # st.write("""
        # ## **Environment Rating** 
        # (get a metric to score your growing conditions)\n
        # [Link to ML notebook](https://discuss.streamlit.io/t/changing-width-of-columns/6816/2) & [Crop Classification](https://discuss.streamlit.io/t/changing-width-of-columns/6816/2)
        # """)

        st.markdown(f'<h2 style="font-weight: bold; padding-bottom: 0px;">Environment Rating</h2>', unsafe_allow_html=True)
        st.write("""
        (get the optimal crop based on measurments)
        [Link to ML notebook](https://colab.research.google.com/drive/1H39jp1yB-RBOhcuYaA8FdUhtt-a2xOFG#scrollTo=inHzThqlauJ4) & [Crop Classification](https://colab.research.google.com/drive/1b1s7uUVeev2qgadHmpHian4ZEage4Uti#scrollTo=A9aBeA-wcok-)
        """)


        rf_env = load_model('models/crop_outlook_rfg1.joblib')

        def env_rating(temp, soil_mois, hum):
            result = rf_env.predict([[temp, soil_mois, hum]])
            # print('Environment Rating: ' + str(round(result[0], 2)) + '%')
            return str(round(result[0], 2))

        temp2 = st.number_input('Temperature (between 15°C  and 30°C due to arduino sensor) ', 15, 30)
        soil_mois2 = st.number_input('Soil Moisture (%)', 0, 100)
        hum2 = st.number_input('Humidity (%) ', 0, 100)

        if st.button('Run Environment Rating Model'):
            rating = env_rating(temp2, soil_mois2, hum2)
            print('Rating:', rating)
            st.write("AI's Environment Rating (%): " + rating)
            if (float(rating) < 70):
                st.markdown(f'<p style="font-weight: bold; color:red;">Not Viable</p>', unsafe_allow_html=True)
            elif (float(rating) >= 85):
                st.markdown(f'<p style="font-weight: bold; color:#00ff59;">Optimal</p>', unsafe_allow_html=True)
            elif (float(rating) >= 70):
                st.markdown(f'<p style="font-weight: bold; color:#3dbb3d;">Viable</p>', unsafe_allow_html=True)

    with col2:
        # st.write("""
        # ## **Crop Recommendation**
        # (get the optimal crop based on measurments)\n
        # [Link to ML notebook](https://discuss.streamlit.io/t/changing-width-of-columns/6816/2) & [Link to analysis](https://discuss.streamlit.io/t/changing-width-of-columns/6816/2)
        # """)

        st.markdown(f'<h2 style="font-weight: bold; padding-bottom: 0px;">Crop Recommendation</h2>', unsafe_allow_html=True)
        st.write("""
        (get the optimal crop based on measurments)
        [Link to ML notebook](https://colab.research.google.com/drive/1KRE3yEfb3US_50EM-YE1c4n8Ov9-m8xl#scrollTo=xhAr_Fv4bZqn) & [Link to analysis](https://colab.research.google.com/drive/1veCUGr0bnJeSwiExd3ilQN-aFP5bmmHx#scrollTo=OTNVqh6Yij9k)
        """)

        rf_crop = load_model('models/rf_crop_recommendation.joblib')

        def recommend_crop(N, P, K, temp, hum, pH, rain):
            result = rf_crop.predict([[N, P, K, temp, hum, pH, rain]])[0]
            # print('Recommended Crop: ' + result)
            return result

        N = st.number_input("Nitrogen NPK Ratio", 0, 10000)
        P = st.number_input("Phosporus NPK Ratio", 0, 10000)
        K = st.number_input("Potassium NPK Ratio", 0, 10000)
        temp = st.number_input("Temperature (°C)", 0, 100)
        humidity = st.number_input("Humidity (%)", 0, 100)
        pH = st.number_input("pH level", 0, 14)
        rainfall = st.number_input("Rainfall (mm)", 0, 500)

        if st.button('Run Recommendation Model'):
            best_crop = recommend_crop(N, P, K, temp, humidity, pH, rainfall)
            print('Best Crop:', best_crop)
            st.write("AI's Recommended Crop:", best_crop)
    
    st.write("""
    ----
    ### Optimal Growth Factors per Crop  
    """)
    df = pd.read_csv('data/optimal_growth_factors.csv')
    df.rename(columns={'Unnamed: 0':'Crop'}, inplace=True)
    st.dataframe(df)
    st.write('The dataframe above shows the ideal metrics for each crop type. \
        Meaning if you’re looking for the best growth conditions for ‘watermelon’, \
        you would go to row 13 and read the data in each column. ')

    col3, col4 = st.beta_columns([1,1])

if __name__ == '__main__':
	main()
# if st.button('Run Environment Rating Model'):
#     best_crop = recommend_crop(21, 40, 55)
#     print('Best Crop:', best_crop)

# st.text



