import streamlit as st

st.set_page_config(
    page_title = 'Multipage App',
)

# Create content for the homepage
st.title('Instruction')
st.markdown('Name: Yuqi Zhou \n USC-ID: 1421-7070-96')
st.markdown('DSCI 510 Final Project')

st.subheader('How to use this webapp?')
st.markdown('➭ Question 1-3 on Homepage')
st.markdown('➭ Data sources and and their descriptions on Data Sources page')
st.markdown('➭ The key analysis section on the main page')
st.markdown('- On the main page, you can select the track information data of the specific ranking you need to view in the left sidebar, and selecting them at the same time means that you can view the track information data of the two rankings that are common to both charts. (Remember to click Submit!)')
st.markdown("- After selecting the rankings, you can view the detailed data, chart and analysis result under the 'Data', 'Chart' and 'Result' tabs respectively. When selecting two rankings at the same time, in addition to the above, you can also view the final conclusions. (So it is recommended to follow the reading order of selecting the separate rankings first, and then the two rankings!)")
st.markdown('- Descriptions about my graphs and analysis will be on main page.')
st.markdown('➭ Question 4-8 on Q&A')

st.subheader('What does this webapp research?')
st.markdown("➭ I'm focusing on the Spotify Most Streamed Songs list and the Billboard Songs Ranking in 2023, and I claim that the songs on both lists share similar musical features and trends, and that Spotify streaming has some effect on Billboard's rankings.")

st.subheader('Any major "gotchas"?')
st.markdown("➭ Daily streamer don't have the best representation of a song's play; the best data should be streams for a fixed period of time after release, but it's harder to obtain.")
st.markdown('➭ Some of the various musical features may be correlated with each other, not independent, so that may have impact on the overall judgment. At the same time, these musical characteristics may not include all the factors that may affect streaming/popularity, all of which may introduce inaccuracy into our study.')
        
# Create a hint for selection
st.sidebar.success('Select a page above.')