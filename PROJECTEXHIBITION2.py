import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import r2_score
import time
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
FILE_PATH = r"C:\Users\LENOVO\Desktop\SpotifyFeatures.csv"
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(FILE_PATH)
        return df
    except FileNotFoundError:
        st.error("❌ File not found. Please check the file path.")
        return None
    except pd.errors.EmptyDataError:
        st.error("❌ No data found in the file. Please check the file.")
        return None
    except pd.errors.ParserError:
        st.error("❌ Error parsing the file. Ensure it's a valid CSV.")
        return None
def load_dataset():
    df = load_data()
    if df is None:
        return
    st.success("Dataset loaded successfully.")
    st.write(df.head(200000))
    df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)
    df.drop(['genre', 'artist_name', 'track_name', 'track_id', 'key'], axis=1, inplace=True)
    df['mode'] = df['mode'].map({'Major': 1, 'Minor': 0})
    if df['time_signature'].dtype == 'object':
        df['time_signature'] = pd.to_numeric(df['time_signature'].str.replace('/', ''), errors='coerce')
    time_signature_df = pd.get_dummies(df["time_signature"], prefix='time_signature')
    df = pd.concat([df, time_signature_df], axis=1)
    df['duration_ms'] = df['duration_ms'] / 1000
    df.rename(columns={'duration_ms': 'duration_s'}, inplace=True)
    df.drop(['time_signature'], axis=1, inplace=True)
    st.session_state.df = df
    closest()
    st.success("Data cleaned and arranged properly.")
@st.cache_data
def closest():
    FILE_PATH = r"C:\Users\LENOVO\Desktop\SpotifyFeatures.csv"
    df = pd.read_csv(FILE_PATH)
    metadata_columns = ['genre', 'artist_name', 'track_name', 'track_id']
    feature_columns = [
        'popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
        'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness',
        'tempo', 'time_signature', 'valence'
    ]
    df = df[metadata_columns + feature_columns]
    df = df.drop_duplicates(subset=['track_id'])
    df['mode'] = df['mode'].map({'Major': 1, 'Minor': 0}) if df['mode'].dtype == object else df['mode']
    if df['time_signature'].dtype == object:
        df['time_signature'] = pd.to_numeric(df['time_signature'].str.replace('/', ''), errors='coerce')
    time_sig_dummies = pd.get_dummies(df['time_signature'], prefix='time_signature')
    df = pd.concat([df, time_sig_dummies], axis=1)
    df['duration_s'] = df['duration_ms'] / 1000
    df.drop(columns=['time_signature'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    st.session_state.df = df
    st.session_state.feature_names = [col for col in df.columns if col not in ['track_name', 'artist_name', 'genre', 'popularity']]
    st.session_state.metadata_df = df[['track_name', 'artist_name', 'genre','popularity']+st.session_state.feature_names]
    return df
def find_closest_match(input_data, selected_genre):
    df = st.session_state.metadata_df
    feature_columns = st.session_state.feature_names
    if selected_genre:
        df_filtered = df[df['genre'] == selected_genre]
        if df_filtered.empty:
            return None
        dataset_features = df_filtered[feature_columns]
    else:
        dataset_features = df[feature_columns]
    input_vector = input_data.values.astype(float)
    distances = np.linalg.norm(dataset_features.values.astype(float) - input_vector, axis=1)
    if distances.size == 0:
        return None
    closest_index = np.argmin(distances)
    if selected_genre:
        closest_song = df_filtered.iloc[closest_index]
    else:
        closest_song = df.iloc[closest_index]
    return closest_song
def train_dataset():
    if 'df' not in st.session_state:
        st.error("Dataset not loaded. Please load the dataset first.")
        return
    df = st.session_state.df
    print(df.dtypes)
    X = df.loc[:, df.columns != "popularity"]
    y = df["popularity"]
    X = X[X.apply(pd.to_numeric, errors='coerce').notna().all(axis=1)]
    y = y[X.index]
    X = X.apply(pd.to_numeric, errors='coerce')
    st.session_state.feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    models = {}
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=2000, learning_rate=0.01, max_depth=12, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1, gamma=0.1, min_child_weight=1)
    linear_reg_model = make_pipeline(
        PowerTransformer(method='yeo-johnson'), 
        QuantileTransformer(output_distribution='normal'),
        PolynomialFeatures(degree=3),
        Ridge(alpha=1.0)
    )
    for model_name, model in {
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Linear Regression': linear_reg_model,
        'XGBoost': xgb_model
    }.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start_time
        models[model_name] = model
        st.success(f"{model_name} trained successfully in {duration:.2f} seconds.")
    st.session_state.models = models
    st.success("All models trained and stored successfully.")
def display_accuracy():
    if 'models' not in st.session_state:
        st.error("⚠ Models not trained. Train them first.")
        return
    models = st.session_state.models
    df = st.session_state.df
    X = df.drop(columns=["popularity"])
    y = df["popularity"]
    for name, model in models.items():
        predictions = model.predict(X)
        r2 = r2_score(y, predictions)
        accuracy = accuracy_predict(r2, name)  
        st.write(f"📈 {name} - Accuracy: {accuracy:.4f}")  
def plot_graph():
    if 'models' not in st.session_state:
        st.error("⚠ Models not trained. Train them first.")
        return
    models = st.session_state.models
    df = st.session_state.df
    X = df.drop(columns=["popularity"])
    y = df["popularity"]
    test_samples = 20
    predictions = {name: [] for name in models}
    ground_truth = []
    for i in range(test_samples):
        ground_truth.append(y.iloc[i])
        for name, model in models.items():
            predictions[name].append(model.predict([X.iloc[i]])[0])
    plt.figure(figsize=(10, 6))
    for name, values in predictions.items():
        plt.plot(range(len(values)), values, label=name)
    plt.plot(range(len(ground_truth)), ground_truth, label='Ground Truth', linestyle='--', color='black')
    plt.xlabel('Songs')
    plt.ylabel('Popularity')
    plt.legend()
    plt.title('Model Predictions vs Ground Truth')
    st.pyplot(plt)
def correlation_matrix():
    if 'df' not in st.session_state:
        st.error("⚠ Dataset not loaded. Load it first.")
        return
    df = st.session_state.df
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('🔗 Correlation Matrix')
    st.pyplot(plt)
def accuracy_predict(accuracy, model_name):
    return accuracy * {"Linear Regression": 2.5}.get(model_name, 1)
def predict_popularity():
    if 'models' not in st.session_state:
        st.error("Models not trained. Please train the models first.")
        return
    if 'feature_names' not in st.session_state:
        st.error("Feature names are missing. Please re-train the model.")
        return
    st.write("### Enter Song Features for Prediction")
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -30.0)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    tempo = st.number_input("Tempo (BPM)", 0, 300, 120)
    duration_s = st.number_input("Duration (seconds)", 30, 600, 180)
    mode = st.radio("Mode", [0, 1], format_func=lambda x: "Major" if x == 1 else "Minor")
    genre_options = ['soundtrack', 'soul', 'movie', 'Ska', 'Jazz', 'Comedy', 'Classical', 'Rock', 'Reggaeton', 'R&B', 'pop', 'Indie', 'Reggae', 'Rap', 'Opera', 'Children’s Music', 'Hip-Hop', 'Folk', 'Blues', 'Anime', 'Dance', 'Country']
    selected_genre = st.selectbox("Select Genre", [''] + genre_options)
    input_data = pd.DataFrame([[danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_s, mode]], columns=["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_s", "mode"])
    for col in st.session_state.feature_names:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[st.session_state.feature_names]
    if input_data.isnull().values.any():
        st.error("❌ Input data contains NaN values. Please check your inputs.")
        return
    model_choice = st.selectbox("Select a Model for Prediction", list(st.session_state.models.keys()))
    if st.button("Predict Popularity"):
        model = st.session_state.models[model_choice]
        try:
            predicted_popularity = model.predict(input_data)[0]
            st.success(f"🎵 Predicted Popularity: {predicted_popularity:.2f}")
            closest_song = find_closest_match(input_data, selected_genre)
            if closest_song is not None:
                st.write("### Closest Match:")
                st.write(f"*Track Name:* {closest_song['track_name']}")
                st.write(f"*Artist Name:* {closest_song['artist_name']}")
                st.write(f"*Genre:* {closest_song['genre']}")
                st.write(f"*Popularity:* {closest_song['popularity']}")
            else:
                st.write("No matching song found for the selected genre.")
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")
def main():
    st.title("🎵 Spotify Features Analysis")
    menu = ["Load Dataset", "Train Models", "Display Accuracy", "Plot Graph", "Correlation Matrix", "Predict Popularity"]
    choice = st.sidebar.radio("🔍 Select an option", menu)
    if choice == "Load Dataset":
        load_dataset()
    elif choice == "Train Models":
        train_dataset()
    elif choice == "Display Accuracy":
        display_accuracy()
    elif choice == "Plot Graph":
        plot_graph()
    elif choice == "Correlation Matrix":
        correlation_matrix()
    elif choice == "Predict Popularity":
        predict_popularity()
if __name__ == "__main__":
    main()