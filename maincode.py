from pandas.io.parsers import read_csv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

@st.cache_data(hash_funcs={pd.DataFrame: lambda x: None})
def load_data():
    df = pd.read_csv('googleplaystore.csv')
    df['Installs'] = df['Installs'].str.replace(',', '')  # Remove commas
    df['Installs'] = df['Installs'].str.replace('+', '')  # Remove plus sign
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')  # Convert to numeric, handle errors
    df = df.dropna(subset=['Installs', 'Reviews', 'Rating'])
    return df
    


custom_css = """
<style>
body {
    background-color: yellow;  /* Replace with your desired background color */
    color: black;  /* Replace with your desired text color */
}
</style>
"""



# Apply custom CSS
st.markdown(custom_css, unsafe_allow_html=True)



def plot_top_categories():
    st.header("Top Categories on Playstore")
    st.image("category.png")
    st.info('''The Google Play Store apps data analysis provides enough 
    potential to drive apps making businesses to succeed. Actionable stats
    can be drawn for developers to work on and capture the Android market.
    The dataset is web scrapped data of 10 thousand Playstore applications to analyze the android competition.''')

    st.info('''There are 34 categories in this dataset. The Family category has around 2000 apps, followed 
    by the Game category with 1200 apps. The '1.9' Category only has 1 app and is not visible on the graph.''')

def plot_ratings_histogram():
    fig, ax = plt.subplots()
    df['Rating'].replace(to_replace=[19.0], value=[1.9], inplace=True)
    ax.hist(df['Rating'], bins=20)
    plt.title("RATINGS OF APPS")
    st.pyplot(fig)

def plot_ratings_bar_chart():
    fig, ax = plt.subplots()
    df.head(15).plot.bar(x='App', y='Rating', figsize=(25, 20), ax=ax)
    plt.title("RATINGS OF APPS")
    plt.xlabel("APPS")
    plt.ylabel("RATING")
    st.pyplot(fig)

# ... (existing code)

def recommend_apps():
    st.header("App Recommendation System")

    # Scale the 'Installs' and 'Rating' columns for better comparison
    scaler = MinMaxScaler()
    df['Installs_scaled'] = scaler.fit_transform(df[['Installs']])
    df['Rating_scaled'] = scaler.fit_transform(df[['Rating']])

    # Define a simple recommendation metric (you can customize this based on your business logic)
    df['Recommendation_Metric'] = df['Installs_scaled'] + df['Rating_scaled']

    # Recommend top N apps
    top_n = 20
    recommended_apps = df.nlargest(top_n, 'Recommendation_Metric')[['App', 'Installs', 'Rating']]

    # Display recommended apps
    st.table(recommended_apps)

# ... (rest of the code)


    



def evaluate_random_forest_regression():
    st.header("Random Forest Regression Evaluation")

    
    # Defining features and target variable
    features = df[['Installs', 'Reviews']]
    target = df['Rating']

    # Splitting the dataset into train and test set
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2,
                                                                              random_state=42)

    # Applying Random Forest Regression
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    target_train = target_train.dropna()

    model.fit(features_train, target_train)

    # Predicting the results
    predictions = model.predict(features_test)

    # Evaluating the algorithm
    st.info(f"Mean Absolute Error: {metrics.mean_absolute_error(target_test, predictions)}")
    st.info(f"Mean Squared Error: {metrics.mean_squared_error(target_test, predictions)}")
    st.info(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(target_test, predictions))}")

    # Prediction System
    st.header("Random Forest Regression Prediction System")

    # User inputs
    user_installs = st.slider("Number of Installs", min_value=0, max_value=int(df['Installs'].max()), step=1000)
    user_reviews = st.slider("Number of Reviews", min_value=0, max_value=int(df['Reviews'].max()), step=100)

    if st.button("Predict Rating"):
    # Predicting the rating for user inputs
        user_prediction = model.predict([[user_installs, user_reviews]])
        st.success(f"Predicted Rating: {user_prediction[0]:.2f}")
    
    st.info(f"Hence we see that we are increasing the number of install and number of reviews we are getting increased ratings . ")

# Main Streamlit app
st.image("head.png")
st.sidebar.header("Project Options")

options = ['Top Categories', 'Rating', 'Total Reviews on each App',
           'Installs', 'Distributed value of installs on each Category', 'Scatter plot on Total reviews on each app',
           'App types', 'Android Versions', 'Categories', 'App Types ', 'Random Forest Regression','Recommendation System']

choice = st.sidebar.selectbox("SELECT AN OPTION", options)

df = load_data()

if choice == options[0]:
    plot_top_categories()

elif choice == options[1]:
    plot_ratings_histogram()

elif choice == options[2]:
    plot_ratings_bar_chart()

elif choice == options[3]:
    fig, ax = plt.subplots(figsize=(4, 2), dpi=10)  # Adjust the DPI as needed
    sum_inst = df.groupby(['Category'])['Installs'].sum().sort_values(ascending=False)
    ax.barh(sum_inst.index, sum_inst)
    st.pyplot(fig)

elif choice == options[4]:
    fig, ax = plt.subplots()
    df['Installs'].replace(to_replace=['0', 'Free'], value=['0+', '0+'], inplace=True)
    Installs = []

    for x in df['Installs']:
        x = x.replace(',', '')
        Installs.append(float(x[:-1]))
    df['installs'] = Installs

    # Use plt.hist for a distribution plot
    ax.hist(Installs, bins=20)  # Adjust the number of bins as needed
    plt.title("Distribution of Installs")
    plt.xlabel("Number of Installs")
    plt.ylabel("Frequency")

    st.pyplot(fig)

elif choice == options[5]:
    fig, ax = plt.subplots()
    df.head(20).plot.scatter(x='Reviews', y='App', s=30, figsize=(18, 8), color='red', ax=ax)
    plt.title("Scatter plot on Total reviews on each app")
    st.pyplot(fig)

elif choice == options[6]:
    fig, ax = plt.subplots()
    df.Type.unique()
    df['Type'].replace(to_replace=['0'], value=['Free'], inplace=True)
    df['Type'].fillna('Free', inplace=True)
    print(df.groupby('Category')['Type'].value_counts())
    Type_cat = df.groupby('Category')['Type'].value_counts().unstack().plot.barh(figsize=(10, 20), width=0.7, ax=ax)
    st.pyplot(fig)

elif choice == options[7]:
    fig, ax = plt.subplots()
    Type_cat = df.groupby('Category')['Android Ver'].value_counts().unstack().plot.barh(figsize=(10, 18), width=1, ax=ax)
    st.pyplot(fig)

elif choice == options[8]:
    fig, ax = plt.subplots()
    df['Category'].value_counts().head(10).plot.pie(
        figsize=(8, 7),
        startangle=90,
        wedgeprops={'width': .5},
        radius=1,
        autopct='%.1f%%',

        pctdistance=.9,

        textprops={'color': 'black'}, ax=ax
    )
    plt.title("CATEGORIES")
    st.pyplot(fig)

elif choice == options[9]:
    fig, ax = plt.subplots()
    df['Type'].value_counts().head(10).plot.pie(
        figsize=(8, 7),
        startangle=90,
        wedgeprops={'width': .5},
        radius=1,
        autopct='%.1f%%',

        pctdistance=.9,

        textprops={'color': 'red'}, ax=ax
    )
    plt.title(" APP TYPES")
    st.pyplot(fig)

# Continue for other plotting functions...

elif choice == options[10]:
    evaluate_random_forest_regression()
elif choice == options[11]:
    recommend_apps()

