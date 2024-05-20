import pandas
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def Process_sentences(text):
    temp_sent = []
    # Tokenize words
    words = word_tokenize(text)
    for word in words:
        # Remove stop words and non alphabet tokens
        if word not in stop_words and word.isalpha():
            temp_sent.append(word)
    # Some other clean-up
    full_sentence = " ".join(temp_sent)
    return full_sentence


def Import_Zomata_data():
    df = pandas.read_csv("ProcessedData.csv")
    return df


def Convert_List_to_string(List):
    str = ""
    for item in List:
        str = str + " " + item
    return str


def RecommendRestaurant(input_cuisine, input_Features, input_PriceRange, input_Area):

    input_cuisine = Convert_List_to_string(input_cuisine)
    input_Features = Convert_List_to_string(input_Features)
    input_PriceRange = Convert_List_to_string(input_PriceRange)
    input_Area = input_Area
    # Convert user input to lowercase

    description = input_cuisine + " " + input_Features + " " + input_PriceRange
    description = description.lower()

    data = Import_Zomata_data()

    data = data[data["Area"] == input_Area]

    # Process user description text input
    # description = Process_sentences(description)
    description = description.strip()

    # Fit data on processed reviews
    vec = tfidf.fit(data["bag_of_words"])
    Zomato_Data_features = vec.transform(data["bag_of_words"])

    # Transform user input data based on fitted model
    User_description_vector = vec.transform([description])

    # Calculate the similarity for the differnt models
    similarity = model(User_description_vector, Zomato_Data_features)

    # Add similarities to data frame
    data["similarity"] = similarity[0]

    # Sort data frame by similarities
    data.sort_values(by="similarity", ascending=False, inplace=True)

    return data[
        [
            "Name",
            "Area",
            "Full_Address",
            "AverageCost",
            "Cuisines",
            "Total Ratings",
            "similarity",
        ]
    ].head(5)


Cuisine_List = [
    "chinese",
    "north indian",
    "fast food",
    "beverages",
    "desserts",
    "biryani",
    "bengali",
    "rolls",
    "mughlai",
    "sandwich",
    "street food",
    "shake",
    "pizza",
    "momos",
    "kebab",
    "continental",
    "burger",
    "bakery",
    "south indian",
    "italian",
    "seafood",
    "ice cream",
    "mishti",
    "asian",
    "cafe",
]

Feature_list = ["Homedelivery", "Indoorseating", "Takeway", "VegOnly"]
Price_list = ["cheapeats", "midrange", "expensive"]
Area_List = [
    "",
    "New Town",
    "Chinar Park",
    "Baguihati",
    "Behala",
    "Sector 5",
    "Tollygunge",
    "Dum Dum",
    "Kestopur",
    "Jadavpur",
    "Sector 1",
    "Kasba",
    "Park Circus Area",
    "Ballygunge",
    "Garia",
    "Park Street Area",
    "Bangur",
    "Picnic Garden",
    "New Alipore",
    "Kaikhali",
    "Kankurgachi",
    "Naktala",
    "Bhawanipur",
    "Southern Avenue",
    "Prince Anwar Shah Road",
    "Topsia",
]


tfidf = pickle.load(open("Vectorizer.pkl", "rb"))
model = pickle.load(open("Model.pkl", "rb"))

st.set_page_config(layout="wide")
st.title("Restaurant Recommendation")
st.divider()
st.header("About the App")
st.write("This app recommends top 5 restaurants in kolkata with your choice")
st.divider()

# Inputs
st.subheader("Choose Cuisine")
input_cuisine = st.multiselect("inputCusine", (Cuisine_List), label_visibility="hidden")

st.subheader("Choose Features")
input_Features = st.multiselect(
    "inputFeatures", (Feature_list), label_visibility="hidden"
)

st.subheader("Choose Price Range")
input_PriceRange = st.multiselect(
    "inputPriceRange", (Price_list), label_visibility="hidden"
)

st.subheader("Choose Area")
input_Area = st.selectbox("inputArea", (Area_List), label_visibility="hidden")


if st.button("Recommend"):

    if input_Area == "":
        st.warning("Please select a Area of your choice")
    else:
        result = RecommendRestaurant(
            input_cuisine, input_Features, input_PriceRange, input_Area
        )
        st.write(result)
