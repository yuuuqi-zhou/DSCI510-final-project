import streamlit as st

st.title('Data sources')
st.subheader('Data Source 1:')
st.markdown('URL: https://kworb.net/spotify/songs_2023.html, https://kworb.net/spotify/songs_2022.html, https://kworb.net/spotify/songs_2021.html, https://kworb.net/spotify/songs_2020.html, https://kworb.net/spotify/songs_2019.html')
st.markdown('Brief description of data: Data for Spotify top streamed songs for each year are included here, giving a clear picture of total streaming and daily streaming.\n')

st.subheader('Data Source 2:')
st.markdown('APi: https://developer.spotify.com/documentation')
st.markdown("Brief description of data: I use this Web Api (specifically using Get Playlist, Get Artists, Get Tracks and Get Track's Audio Features) to get detailed information about a playlist containing the 50 most streamed songs released in 2023, such as artist, musical features like danceability, key, and acousticness, and track/artist popularity, etc. I also get similar information about the Billboard 2023 top tracks mentioned in the next data source using the same way.")

st.subheader('Data Source 3:')
st.markdown('URL: https://www.billboard.com/charts/year-end/hot-100-songs/')
st.markdown("Brief description of data: Here's the Billboard 2023 top 100 tracks by combining several different criteria including Spotify plays, and I scrape the information of these songs from this website.")