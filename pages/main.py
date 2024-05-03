import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

st.title('Discover Spotify and Billboard Top Tracks')
st.markdown("The year 2023 has seen the emergence of a plethora of great songs, which in turn have received their own rankings on different platforms based on different evaluation criteria. Let's take a closer look at the most streamed songs of 2023 on Spotify as well as Billboard's best songs of the year.\n")
st.markdown("What kind of musical features do these songs possess? And how do these features respond to streams and popularity? Which features are most important? Let's find them out.")
st.caption('See the sidebar on the left and choose a ranking!')


# Read all data
data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')
data3 = pd.read_csv('data3.csv')
data4 = pd.read_csv('data4.csv')
data5 = pd.read_csv('data5.csv')
data6 = pd.read_csv('data6.csv')
data7 = pd.read_csv('data7.csv')
data8 = pd.read_csv('data8.csv')


# Separate the first column into Artists and Tracks
data1['Artist'] = data1['Artist and Title'].apply(lambda x: x.split(' - ')[0])
data1['Title'] = data1['Artist and Title'].apply(lambda x: x.split(' - ')[1])
data1_artist_list = data1['Artist'].tolist()
data1_title_list = data1['Title'].tolist()

data5['Artist'] = data5['Artist and Title'].apply(lambda x: x.split(' - ')[0])
data5['Title'] = data5['Artist and Title'].apply(lambda x: x.split(' - ')[1])
data5_artist_list = data5['Artist'].tolist()
data5_title_list = data5['Title'].tolist()

data6['Artist'] = data6['Artist and Title'].apply(lambda x: x.split(' - ')[0])
data6['Title'] = data6['Artist and Title'].apply(lambda x: x.split(' - ')[1])
data6_artist_list = data6['Artist'].tolist()
data6_title_list = data6['Title'].tolist()

data7['Artist'] = data7['Artist and Title'].apply(lambda x: x.split(' - ')[0])
data7['Title'] = data7['Artist and Title'].apply(lambda x: x.split(' - ')[1])
data7_artist_list = data7['Artist'].tolist()
data7_title_list = data7['Title'].tolist()

data8['Artist'] = data8['Artist and Title'].apply(lambda x: x.split(' - ')[0])
data8['Title'] = data8['Artist and Title'].apply(lambda x: x.split(' - ')[1])
data8_artist_list = data8['Artist'].tolist()
data8_title_list = data8['Title'].tolist()


# Add data 2&4 info into new lists
data2_artists = data2['Artists'].tolist()
data2_tracks = data2['Tracks'].tolist()
data2_danceability = data2['danceability'].tolist()
data2_energy = data2['energy'].tolist()
data2_key = data2['key'].tolist()
data2_loudness = data2['loudness'].tolist()
data2_mode = data2['mode'].tolist()
data2_speechiness = data2['speechiness'].tolist()
data2_acousticness = data2['acousticness'].tolist()
data2_instrumentalness = data2['instrumentalness'].tolist()
data2_liveness = data2['liveness'].tolist()
data2_valence = data2['valence'].tolist()
data2_track_popularity = data2['track_popularity'].tolist()
data2_artist_popularity = data2['artist_popularity'].tolist()

data4_artists = data4['Artists'].tolist()
data4_tracks = data4['Tracks'].tolist()
data4_danceability = data4['danceability'].tolist()
data4_energy = data4['energy'].tolist()
data4_key = data4['key'].tolist()
data4_loudness = data4['loudness'].tolist()
data4_mode = data4['mode'].tolist()
data4_speechiness = data4['speechiness'].tolist()
data4_acousticness = data4['acousticness'].tolist()
data4_instrumentalness = data4['instrumentalness'].tolist()
data4_liveness = data4['liveness'].tolist()
data4_valence = data4['valence'].tolist()
data4_track_popularity = data4['track_popularity'].tolist()
data4_artist_popularity = data4['artist_popularity'].tolist()


# Clean the tracks' title
def remove_brackets(text):
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r"[‘’“”\"]", "'", text)
    text = text.rstrip()
    return text

data1_title_list = [remove_brackets(track) for track in data1_title_list]
data5_title_list = [remove_brackets(track) for track in data5_title_list]
data7_title_list = [remove_brackets(track) for track in data7_title_list]
data8_title_list = [remove_brackets(track) for track in data8_title_list]
data2_tracks = [remove_brackets(track) for track in data2_tracks]
data4_tracks = [remove_brackets(track) for track in data4_tracks]

data2_tracks_streams = [track for track in data2_tracks if track in data1_title_list or track in data5_title_list or track in data6_title_list or track in data7_title_list or track in data8_title_list]


# Add Daily Streams for data2
daily_Spotify = []

data_title_lists = [data1_title_list, data5_title_list, data6_title_list, data7_title_list, data8_title_list]
data_sets = [data1, data5, data6, data7, data8]

for track in data2_tracks_streams:
    for idx, data_title_list in enumerate(data_title_lists):
        if track in data_title_list:
            daily_Spotify.append(data_sets[idx]['Daily'][data_title_list.index(track)])
            break


# For compicated ones do manually
daily_Spotify.insert(10, 2604452)
daily_Spotify.insert(12, 1255496)
daily_Spotify.insert(21, 1941158)
daily_Spotify.insert(23, 579069)
daily_Spotify.insert(24, 928643)
daily_Spotify.insert(29,1071674)
daily_Spotify.insert(35, 1899639)
daily_Spotify.insert(41, 1286871)
daily_Spotify.insert(45,1460119)

data2['Daily Streams'] = daily_Spotify


# Add Daily Streams for data4
data4_tracks_streams = [track for track in data4_tracks if track in data1_title_list or track in data5_title_list or track in data6_title_list or track in data7_title_list or track in data8_title_list]
data4_daily_Spotify = []

for track in data4_tracks_streams:
    for idx, data_title_list in enumerate(data_title_lists):
        if track in data_title_list:
            data4_daily_Spotify.append(data_sets[idx]['Daily'][data_title_list.index(track)])
            break


# For compicated ones do manually
data4_daily_Spotify.insert(3, 1071717)
data4_daily_Spotify.insert(22, 288749)
data4_daily_Spotify.insert(39, 1255496)
data4_daily_Spotify.insert(43, 579069)
data4_daily_Spotify.insert(45, 399040)

data4['Daily Streams'] = data4_daily_Spotify


# Spotify aanlysis
feature_1 = data2.describe()


# find all features
features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']


# The most important features for data2
X = data2.drop(['Artists', 'Tracks', 'track_popularity', 'artist_popularity', 'Daily Streams'], axis=1)
y_popularity = data2['track_popularity']
y_streams = data2['Daily Streams']

model_popularity = RandomForestRegressor()
model_popularity.fit(X, y_popularity)

model_streams = RandomForestRegressor()
model_streams.fit(X, y_streams)

feature_importance_popularity = model_popularity.feature_importances_
feature_importance_streams = model_streams.feature_importances_

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.barh(X.columns, feature_importance_popularity)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Track Popularity')

plt.subplot(1, 2, 2)
plt.barh(X.columns, feature_importance_streams)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Daily Streams')
plt.tight_layout()
plt.savefig('feature_importance.png') # save as png


# The most important features for data4
X = data4.drop(['Artists', 'Tracks', 'track_popularity', 'artist_popularity', 'Daily Streams'], axis=1)
y_popularity = data4['track_popularity']
y_streams = data4['Daily Streams']

model_popularity = RandomForestRegressor()
model_popularity.fit(X, y_popularity)

model_streams = RandomForestRegressor()
model_streams.fit(X, y_streams)

feature_importance_popularity = model_popularity.feature_importances_
feature_importance_streams = model_streams.feature_importances_

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.barh(X.columns, feature_importance_popularity)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Track Popularity')

plt.subplot(1, 2, 2)
plt.barh(X.columns, feature_importance_streams)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Daily Streams')
plt.tight_layout()
plt.savefig('feature_importance4.png') # save as png


# Billboard analysis
feature_2 = data4.describe()


# common tracks analysis
common_tracks = [track for track in data2_tracks if track in data4_tracks]
filter_condition = data4['Tracks'].isin(common_tracks)
filtered_result = data4[filter_condition].reset_index(drop=True)

# The most important features for common tracks
X = filtered_result.drop(['Artists', 'Tracks', 'track_popularity', 'artist_popularity', 'Daily Streams'], axis=1)
y_popularity = filtered_result['track_popularity']
y_streams = filtered_result['Daily Streams']

model_popularity = RandomForestRegressor()
model_popularity.fit(X, y_popularity)

model_streams = RandomForestRegressor()
model_streams.fit(X, y_streams)

feature_importance_popularity = model_popularity.feature_importances_
feature_importance_streams = model_streams.feature_importances_

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.barh(X.columns, feature_importance_popularity)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Track Popularity')

plt.subplot(1, 2, 2)
plt.barh(X.columns, feature_importance_streams)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Daily Streams')
plt.tight_layout()
plt.savefig('feature_importance_common.png') # save as png

# Normalize data
scaler = MinMaxScaler()

# preparation for track popularity
X_data2 = data2.iloc[:,2:]
X_data4 = data4.iloc[:,2:]

X_data2 = X_data2.drop(['artist_popularity', 'Daily Streams'], axis=1)
X_data2_normal = scaler.fit_transform(X_data2)
X_data2 = pd.DataFrame(X_data2_normal, columns = X_data2.columns)

X_data4 = X_data4.drop(['artist_popularity', 'Daily Streams'], axis=1)
X_data4_normal = scaler.fit_transform(X_data4)
X_data4 = pd.DataFrame(X_data4_normal, columns = X_data4.columns)

correlation2 = X_data2.corr()
correlation4 = X_data4.corr()


# preparation for daily streams
X_data2_2 = data2.iloc[:,2:]
X_data4_2 = data4.iloc[:,2:]
X_data2_2 = X_data2_2.drop(['track_popularity', 'artist_popularity'], axis=1)
X_data4_2 = X_data4_2.drop(['track_popularity', 'artist_popularity'], axis=1)
correlation2_2 = X_data2_2.corr()
correlation4_2 = X_data4_2.corr()


# Create a sidebar to choose ranking
ranking = st.sidebar.multiselect('Choose your Ranking:', ['Spotify', 'Billboard'], default = ['Spotify'])

# specify each choice on sidebar
if st.sidebar.button('Submit'):
    if 'Spotify' in ranking and 'Billboard' in ranking:
        st.subheader('Tracks exist in Spotify and Billboard Rankings')
        tab1, tab2, tab3 = st.tabs(['Data', 'Chart','Conclusion'])
        tab1.write(filtered_result)

        tab1.caption("Common tracks shared by Spotify and Billboard rankings along with their musical features, artist/song popularity and daily streams")

        tab1.caption('- danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.')
        tab1.caption('- energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.')
        tab1.caption('- key: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.')
        tab1.caption('- loudness: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.')
        tab1.caption('- mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.')
        tab1.caption('- speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.')
        tab1.caption('- acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.')
        tab1.caption('- instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.')
        tab1.caption('- liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.')
        tab1.caption('- valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).')
        tab1.caption('- Track popularity: The popularity of the track. The value will be between 0 and 100, with 100 being the most popular. The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity. Note: the popularity value may lag actual popularity by a few days: the value is not updated in real time.')
        tab1.caption("- Track popularity: The popularity of the artist. The value will be between 0 and 100, with 100 being the most popular. The artist's popularity is calculated from the popularity of all the artist's tracks.")
        tab1.caption('- Daily Streams: Daily plays of a song')
        tab1.caption('Descriptions above see Data Source 2 link')

        with tab2:
            st.write('Using a portion of the overlapping tracks to test the conclusions we made when comparing the two rankings because they are representative.')
            st.subheader('Feature Importance')
            tab2.image('feature_importance_common.png')
            tab2.write('The two graphs above show that the most important factor for tracks appearing in both rankings, for both Track Popularity and Daily Streams, is speechiness. This is in line with the conclusions we drew from doing the comparisons, but another important factor loudness seeams not fit here.')
        with tab3:
            tab3.subheader('Conclusion')
            tab3.write("- Spotify and Billboard's rankings in 2023 share similar trends and some same important factors.")
            tab3.write('- Track Popularity and Daily Streams performance for Spotify and Billboard ranking tracks are generally consistent, and one of the most important of the many musical features is speechiness.')
            tab3.write("- As a result of sharing a portion of the songs that are both present, we'll find that Spotify streams have an effect on Billboard rankings, but not just because of this, other factors are needed (such as radio, iTunes, and other data to analyze), but overall they have many similarities.")

    elif 'Spotify' in ranking:
        st.subheader('2023 Spotify most streamed tracks')
        tab1, tab2, tab3 = st.tabs(['Data', 'Chart', 'Result'])
        tab1.subheader('Data')
        tab1.write(data2)
        tab1.caption("Spotify TOP 50 most played songs in 2023 along with their musical features, artist/song popularity and daily streams")

        tab1.caption('- danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.')
        tab1.caption('- energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.')
        tab1.caption('- key: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.')
        tab1.caption('- loudness: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.')
        tab1.caption('- mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.')
        tab1.caption('- speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.')
        tab1.caption('- acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.')
        tab1.caption('- instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.')
        tab1.caption('- liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.')
        tab1.caption('- valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).')
        tab1.caption('- Track popularity: The popularity of the track. The value will be between 0 and 100, with 100 being the most popular. The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity. Note: the popularity value may lag actual popularity by a few days: the value is not updated in real time.')
        tab1.caption("- Track popularity: The popularity of the artist. The value will be between 0 and 100, with 100 being the most popular. The artist's popularity is calculated from the popularity of all the artist's tracks.")
        tab1.caption('- Daily Streams: Daily plays of a song')
        tab1.caption('Descriptions above see Data Source 2 link')


        # Features
        tab2.subheader('Descriptive Statistics')
        tab2.write(feature_1)
        tab2.write("Let's start by looking at observing the basic descriptive statistics of this data set.")
        tab2.write('One of the more representative features is the mean, where we can see that the mean of danceability is 0.6626, the mean of energy is 0.6607, the mean of key is 5, and so on. These data can be compared to the same type of data from Billboard.')
        
        tab2.divider()
        tab2.write("For a song's reputation, we have 'Track Popularity' and 'Daily Streams' as two indicators to refer to, while other music features can be variables that have an impact on them, so we can deal with 'Track Popularity' and 'Daily Streams' as the dependent variables respectively.")


        # Correlation
        with tab2:
            tab2.divider()
            tab2.subheader('Correlation for Track Popularity')
            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            axes = axes.flatten()
            for i, feature in enumerate(features):
                sns.scatterplot(x=feature, y='track_popularity', data=data2, ax=axes[i])
                axes[i].set_title(f'{feature.capitalize()} vs Track Popularity')
                axes[i].set_xlabel(feature.capitalize())
                axes[i].set_ylabel('Track Popularity')
            axes[-1].axis('off')
            plt.tight_layout()
            st.pyplot(fig)

            tab2.subheader('Correlation for Daily Streams')

            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            axes = axes.flatten()
            for i, feature in enumerate(features):
                sns.scatterplot(x=feature, y='Daily Streams', data=data2, ax=axes[i])
                axes[i].set_title(f'{feature.capitalize()} vs Daily Streams')
                axes[i].set_xlabel(feature.capitalize())
                axes[i].set_ylabel('Track Popularity')
            axes[-1].axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.write("Unfortunately, we can't directly draw a connection between individual factors and Track Popularity/Daily Streams from the graphs above.")
            st.divider()


            # Heatmap for track popularity
            st.subheader('Correlation Heatmap for Track Popularity')
            st.write('Next we try to use heatmap to observe the associations.')
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(X_data2_2.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            st.subheader('Variables Heatmap')
            track_popularity_correlation = correlation2['track_popularity']
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(track_popularity_correlation.drop('track_popularity').to_frame().T, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
           
            st.write('Heatmap can be used to look at the connection between factors and dependent variables, as well as factors and factors. The information in these two graphs is consistent and it can be concluded that for Track Popularity, loudness and speechiness have the largest positive relationship with it.')

            tab2.divider()


            # Heatmap for daily streams
            st.subheader('Correlation Heatmap for Daily Streams')
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(X_data2.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            st.subheader('Variables Heatmap')
            track_popularity_correlation = correlation2_2['Daily Streams']
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(track_popularity_correlation.drop('Daily Streams').to_frame().T, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            
            st.write('Similarly, for Daily Streams, mode and speechiness have the largest positive relationship with it.')
            st.write('Also I was able to get some additional information such as a positive relationship between loudness and energy, and also between valence and danceability, which is a proof of the correlation between the factors.')

            st.divider()


            # Feature Importance
            st.subheader('Feature Importance')
            tab2.write("Then how do we know the importance of each factor? Let's try Random Forest Regressor!")   
            tab2.image('feature_importance.png')
            tab2.write("By using the best split strategy, on the left we can see that in the Spotify data, for Track Popularity, speechiness and valence have the greatest importance, followed by liveness and danceability, and on the right we can see that for Daily Streams, danceability, followed by speechiness, has the greatest importance, while the other factors are about similar.")
            

        # Observation Result
        tab3.write('- Spotify')
        tab3.write('Factors impacting Track Popularity: loudness, speechiness, valence')
        tab3.write('Factors impacting Daily Streams: mode, speechiness, danceability')
        tab3.write('- Billboard')
        tab3.write('Factors impacting Track Popularity: loudness, speechiness, valence, key')
        tab3.write('Factors impacting Daily Streams: speechiness, loudness, energy')
        tab3.divider()
        tab3.subheader('Any catch?')
        tab3.write('- From the tables of features of the two sets of data, most of them are similar (especially the means), which can be a rough indication of possessing similar trends in general.')
        tab3.write('- From the Scatter Plots, we cannot make any conclusion, and the trends in plots for Track Populairty and Daily Streams are not the same.')
        tab3.write('- From the heatmaps and feature importance graphs, we can argue on the whole that the most important musical characteristics of the tracks in the two lists are loudness and speechiness because they appear frequently in two analyses of the data at the the same time.')
        tab3.write("- By comparison alone, I think it's fair to say that the 2023 Spotify and Billboard rankings share similar trends and the same important factors.")


    elif 'Billboard' in ranking:
        st.subheader('2023 Billboard best tracks')
        tab1, tab2, tab3= st.tabs(['Data', 'Chart', 'Result'])
        tab1.subheader('Data')
        tab1.write(data4)

        tab1.caption("Billboard Year-end hot 50 songs in 2023 along with their musical features, artist/song popularity and daily streams")

        tab1.caption('- danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.')
        tab1.caption('- energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.')
        tab1.caption('- key: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.')
        tab1.caption('- loudness: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.')
        tab1.caption('- mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.')
        tab1.caption('- speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.')
        tab1.caption('- acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.')
        tab1.caption('- instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.')
        tab1.caption('- liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.')
        tab1.caption('- valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).')
        tab1.caption('- Track popularity: The popularity of the track. The value will be between 0 and 100, with 100 being the most popular. The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity. Note: the popularity value may lag actual popularity by a few days: the value is not updated in real time.')
        tab1.caption("- Track popularity: The popularity of the artist. The value will be between 0 and 100, with 100 being the most popular. The artist's popularity is calculated from the popularity of all the artist's tracks.")
        tab1.caption('- Daily Streams: Daily plays of a song')
        tab1.caption('Descriptions above see Data Source 2 link')


        # Features
        tab2.subheader('Descriptive Statistics')
        tab2.write(feature_2)

        tab2.write("Let's start by looking at observing the basic descriptive statistics of this data set.")
        tab2.write('One of the more representative features is the mean, where we can see that the mean of danceability is 0.6504, the mean of energy is 0.6233, the mean of key is 5.42, and so on. These data can be compared to the same type of data from Spotify.')
        
        tab2.divider()
        tab2.write("For a song's reputation, we have 'Track Popularity' and 'Daily Streams' as two indicators to refer to, while other music features can be variables that have an impact on them, so we can deal with 'Track Popularity' and 'Daily Streams' as the dependent variables respectively.")
        tab2.write('If one wants to understand the association between each factor and the dependent variable, one starts with simple scatter plots between each factor and Track Popularity/Daily Stremas.')


        # Correlation
        with tab2:
            tab2.divider()
            tab2.subheader('Correlation for Track Popularity')
            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            axes = axes.flatten()
            for i, feature in enumerate(features):
                sns.scatterplot(x=feature, y='track_popularity', data=data4, ax=axes[i])
                axes[i].set_title(f'{feature.capitalize()} vs Track Popularity')
                axes[i].set_xlabel(feature.capitalize())
                axes[i].set_ylabel('Track Popularity')
            axes[-1].axis('off')
            plt.tight_layout()
            st.pyplot(fig)

            tab2.subheader('Correlation for Daily Streams')

            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            axes = axes.flatten()
            for i, feature in enumerate(features):
                sns.scatterplot(x=feature, y='Daily Streams', data=data4, ax=axes[i])
                axes[i].set_title(f'{feature.capitalize()} vs Daily Streams')
                axes[i].set_xlabel(feature.capitalize())
                axes[i].set_ylabel('Track Popularity')
            axes[-1].axis('off')
            plt.tight_layout()
            st.pyplot(fig)

            st.write("Unfortunately, we can't directly draw a connection between individual factors and Track Popularity/Daily Streams from the graphs above.")
            st.divider()

            # Heatmap for track popularity
            st.subheader('Correlation Heatmap for Track Popularity')
            st.write('Next we try to use heatmap to observe the associations.')
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(X_data4.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            st.subheader('Variables Heatmap')
            track_popularity_correlation = correlation4['track_popularity']
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(track_popularity_correlation.drop('track_popularity').to_frame().T, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            st.write('Heatmap can be used to look at the connection between factors and dependent variables, as well as factors and factors. The information in these two graphs is consistent and it can be concluded that for Track Popularity, loudness, followed by key and speechiness have the largest positive relationship with it.')
            tab2.divider()


            # Heatmap for daily streams
            st.subheader('Correlation Heatmap for Daily Streams')
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(X_data4_2.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            st.subheader('Variables Heatmap')
            track_popularity_correlation = correlation4_2['Daily Streams']
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(track_popularity_correlation.drop('Daily Streams').to_frame().T, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            st.write('Similarly, for Daily Streams, loudness, followed by energy has the largest positive relationship with it.')
            st.write('Also I was able to get some additional information such as a positive relationship between loudness and energy, and also between valence and energy, which is a proof of the correlation between the factors.')
            st.divider()


            # Feature Importance
            st.subheader('Feature Importance')
            tab2.write("Then how do we know the importance of each factor? Let's try Random Forest Regressor!")   
            tab2.image('feature_importance4.png')
            tab2.write("By using the best split strategy, on the left we can see that in the Billboard data, for Track Popularity, speechiness has the greatest importance, followed by valence, energy and loudness; and on the right we can see that for Daily Streams, loudness has the greatest importance, while the other factors are not that important.")
        

        # Observation Result
        tab3.write('- Spotify')
        tab3.write('Factors impacting Track Popularity: loudness, speechiness, valence')
        tab3.write('Factors impacting Daily Streams: mode, speechiness, danceability')
        tab3.write('- Billboard')
        tab3.write('Factors impacting Track Popularity: loudness, speechiness, valence, key')
        tab3.write('Factors impacting Daily Streams: speechiness, loudness, energy')
        tab3.divider()
        tab3.subheader('Any catch?')
        tab3.write('- From the tables of features of the two sets of data, most of them are similar (especially the means), which can be a rough indication of possessing similar trends in general.')
        tab3.write('- From the Scatter Plots, we cannot make any conclusion, and the trends in plots for Track Populairty and Daily Streams are not the same.')
        tab3.write('- From the heatmaps and feature importance graphs, we can argue on the whole that the most important musical characteristics of the tracks in the two lists are loudness and speechiness because they appear frequently in two analyses of the data at the the same time.')
        tab3.write("- By comparison alone, I think it's fair to say that the 2023 Spotify and Billboard rankings share similar trends and the same important factors.")