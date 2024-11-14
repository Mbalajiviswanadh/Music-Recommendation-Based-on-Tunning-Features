import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import requests
import urllib.parse
from datetime import datetime
from torch import cosine_similarity

load_dotenv()


st.set_page_config(page_title="Music Recommendations from your Playlists", page_icon="🎵")

# Spotify API Configuration
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REDIRECT_URI = 'https://mbalajiviswanadh-music-recommendation-based-on-tun-music-huubkk.streamlit.app/'
AUTH_URL = 'https://accounts.spotify.com/authorize'
TOKEN_URL = 'https://accounts.spotify.com/api/token'
API_BASE_URL = 'https://api.spotify.com/v1/'


# Initialize session state for tokens and other details
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.expires_at = None
    st.session_state.user_info = None
    st.session_state.selected_playlist_id = None
    st.session_state.playlists = []
    st.session_state.songs = []
    st.session_state.music_df = None

# Function to get Spotify authorization URL
def get_auth_url():
    scope = 'user-read-private user-read-email user-top-read playlist-read-private'
    params = {
        'client_id': CLIENT_ID,
        'response_type': 'code',
        'redirect_uri': REDIRECT_URI,
        'scope': scope,
        'show_dialog': True
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

# Function to get or refresh access token
def get_access_token():
    if st.session_state.access_token and datetime.now().timestamp() < st.session_state.expires_at:
        return st.session_state.access_token

    # Refresh token if access token expired
    if st.session_state.refresh_token:
        refresh_data = {
            'grant_type': 'refresh_token',
            'refresh_token': st.session_state.refresh_token,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET
        }
        response = requests.post(TOKEN_URL, data=refresh_data)
        if response.status_code == 200:
            new_token_info = response.json()
            st.session_state.access_token = new_token_info['access_token']
            st.session_state.expires_at = datetime.now().timestamp() + new_token_info['expires_in']
            return st.session_state.access_token
        else:
            st.error("Failed to refresh token, please log in again.")
            return None
    else:
        st.error("No refresh token available. Please log in.")
        return None

# Function to retrieve user info (profile picture and name)
def fetch_user_info():
    access_token = get_access_token()
    if access_token:
        headers = {'Authorization': f"Bearer {access_token}"}
        response = requests.get(API_BASE_URL + 'me', headers=headers)
        if response.status_code == 200:
            st.session_state.user_info = response.json()
        else:
            st.error("Failed to fetch user info.")

# Function to get user playlists
def fetch_user_playlists():
    access_token = get_access_token()
    if access_token:
        headers = {'Authorization': f"Bearer {access_token}"}
        response = requests.get(API_BASE_URL + 'me/playlists', headers=headers)
        if response.status_code == 200:
            playlists = response.json()['items']
            st.session_state.playlists = [{'id': p['id'], 'name': p['name'], 'image': p['images'][0]['url'] if p['images'] else None} for p in playlists]
        else:
            st.error("Failed to fetch playlists.")

# Function to get songs from a selected playlist
def fetch_playlist_songs(playlist_id):
    access_token = get_access_token()
    if access_token:
        headers = {'Authorization': f"Bearer {access_token}"}
        response = requests.get(API_BASE_URL + f'playlists/{playlist_id}/tracks', headers=headers)
        
        if response.status_code == 200:
            tracks = response.json()['items']
            
            # Check if playlist is empty
            if not tracks:
                st.warning("No songs in playlist")
                st.session_state.songs = []
                st.session_state.music_df = None
                return
            
            # Extract track IDs and basic info
            track_data = []
            track_ids = []
            
            for t in tracks:
                if t['track'] and t['track']['id']:  # Check if track exists and has ID
                    track_ids.append(t['track']['id'])
                    track_data.append({
                        'id': t['track']['id'],
                        'name': t['track']['name'],
                        'artists': ', '.join([artist['name'] for artist in t['track']['artists']]),
                        'album_name': t['track']['album']['name'],
                        'release_date': t['track']['album']['release_date'],
                        'image': t['track']['album']['images'][0]['url'] if t['track']['album']['images'] else None
                    })
            
            if not track_data:
                st.warning("No valid songs found in playlist")
                st.session_state.songs = []
                st.session_state.music_df = None
                return
            
            # Store the simplified track data for display
            st.session_state.songs = track_data
            
            try:
                # Get audio features and track info for recommendations
                audio_features = get_audio_features(track_ids, access_token)
                tracks_info = get_tracks_info(track_ids, access_token)
                
                # Create DataFrame for recommendations
                df = pd.DataFrame(track_data)
                df = df.rename(columns={
                    'id': 'Track ID',
                    'name': 'Track Name',
                    'artists': 'Artists',
                    'album_name': 'Album Name',
                    'release_date': 'Release Date',
                    'image': 'Image'
                })
                
                # Create audio features DataFrame with proper error handling
                audio_features_df = pd.DataFrame([af for af in audio_features if af is not None])
                features_to_keep = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                  'speechiness', 'acousticness', 'instrumentalness', 
                                  'liveness', 'valence', 'tempo']
                
                # Check if all required features are present
                if not all(feature in audio_features_df.columns for feature in features_to_keep):
                    missing_features = [f for f in features_to_keep if f not in audio_features_df.columns]
                    st.error(f"Missing audio features: {', '.join(missing_features)}")
                    st.session_state.music_df = None
                    return
                
                # Concatenate only if we have matching indices
                if len(df) == len(audio_features_df):
                    df = pd.concat([df, audio_features_df[features_to_keep]], axis=1)
                else:
                    st.error("Mismatch between track data and audio features")
                    st.session_state.music_df = None
                    return
                
                # Add popularity
                popularity_data = {t['id']: t['popularity'] for t in tracks_info}
                df['Popularity'] = df['Track ID'].map(popularity_data)
                
                # Store in session state
                st.session_state.music_df = df
                
            except Exception as e:
                st.error(f"Error processing playlist data: {str(e)}")
                st.session_state.music_df = None
        else:
            st.error("Failed to fetch songs from playlist.")
            st.session_state.music_df = None


# Function to get song recommendations based on selected song
def get_audio_features(track_ids, access_token):
    headers = {'Authorization': f"Bearer {access_token}"}
    audio_features = []
    
    # Process track IDs in chunks of 100 (Spotify API limit)
    for i in range(0, len(track_ids), 100):
        chunk = track_ids[i:i + 100]
        response = requests.get(f"{API_BASE_URL}audio-features?ids={','.join(chunk)}", headers=headers)
        if response.status_code == 200:
            audio_features.extend(response.json()['audio_features'])
    
    return audio_features

def get_tracks_info(track_ids, access_token):
    headers = {'Authorization': f"Bearer {access_token}"}
    tracks_info = []
    
    # Process track IDs in chunks of 50 (Spotify API limit)
    for i in range(0, len(track_ids), 50):
        chunk = track_ids[i:i + 50]
        response = requests.get(f"{API_BASE_URL}tracks?ids={','.join(chunk)}", headers=headers)
        if response.status_code == 200:
            tracks_info.extend(response.json()['tracks'])
    
    return tracks_info


def calculate_weighted_popularity(release_date):
    try:
        # Handle different date formats
        try:
            release_date = datetime.strptime(release_date, '%Y-%m-%d')
        except ValueError:
            release_date = datetime.strptime(release_date, '%Y')
        
        time_span = datetime.now() - release_date
        weight = 1 / (time_span.days + 1)
        return weight
    except:
        return 0.5  # Default weight if there's an error

def get_hybrid_recommendations(input_song_name, num_recommendations=5):
    if st.session_state.music_df is None or input_song_name not in st.session_state.music_df['Track Name'].values:
        return []

    try:
        # Prepare features for content-based filtering
        features_to_use = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                          'speechiness', 'acousticness', 'instrumentalness', 
                          'liveness', 'valence', 'tempo']
        
        # Normalize features
        scaler = MinMaxScaler()
        music_features = st.session_state.music_df[features_to_use].values
        music_features_scaled = scaler.fit_transform(music_features)
        
        # Get input song index
        input_song_index = st.session_state.music_df[st.session_state.music_df['Track Name'] == input_song_name].index[0]
        
        # Calculate similarity scores using numpy instead of torch
        input_features = music_features_scaled[input_song_index].reshape(1, -1)
        similarity_scores = np.dot(input_features, music_features_scaled.T) / (
            np.linalg.norm(input_features) * np.linalg.norm(music_features_scaled, axis=1)
        )
        
        # Get similar song indices
        similar_song_indices = similarity_scores.argsort()[0][::-1][1:num_recommendations + 5]
        
        # Get recommendations
        recommendations = st.session_state.music_df.iloc[similar_song_indices].copy()
        
        # Calculate weighted popularity
        recommendations['WeightedPopularity'] = recommendations.apply(
            lambda x: x['Popularity'] * calculate_weighted_popularity(x['Release Date']), axis=1
        )
        
        # Sort by weighted popularity and get top recommendations
        recommendations = recommendations.sort_values('WeightedPopularity', ascending=False)
        recommendations = recommendations.head(num_recommendations)
        
        return recommendations['Track ID'].tolist()
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []

# Modify the fetch_song_recommendations function to use the new hybrid system
def fetch_song_recommendations(track_id):
    if not st.session_state.music_df is None:
        # Get the song name from track_id
        song_name = st.session_state.music_df[st.session_state.music_df['Track ID'] == track_id]['Track Name'].iloc[0]
        
        # Get recommendations using hybrid system
        recommended_track_ids = get_hybrid_recommendations(song_name)
        
        # Fetch full track information for the recommendations
        access_token = get_access_token()
        if access_token and recommended_track_ids:
            headers = {'Authorization': f"Bearer {access_token}"}
            tracks_info = get_tracks_info(recommended_track_ids, access_token)
            return tracks_info
    return []

# Streamlit app layout
st.title("Spotify Music Recommendation App")

if st.button("Login with Spotify"):
    auth_url = get_auth_url()
    st.write(f"[Login Here]({auth_url})")



# Handling Spotify OAuth callback
if 'access_token' in st.session_state and st.session_state.access_token:
    # User is logged in, so proceed to fetch user information and playlists
    # st.success("You are already logged in!")
    fetch_user_info()
    fetch_user_playlists()
else:
    # If the code exists in the query parameters, proceed with authentication
    if "code" in st.query_params.to_dict():
        code = st.query_params["code"]
        req_body = {
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': REDIRECT_URI,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET
        }
        # Add a spinner and loading message when the user clicks the login button
        with st.spinner('Logging in...'):
            response = requests.post(TOKEN_URL, data=req_body)
            
            if response.status_code == 200:
                # Authentication successful, store the tokens in session state
                token_info = response.json()
                st.session_state.access_token = token_info['access_token']
                st.session_state.refresh_token = token_info['refresh_token']
                st.session_state.expires_at = datetime.now().timestamp() + token_info['expires_in']
                
                # Show success message after authentication
                st.success("Successfully authenticated with Spotify!")
                
                # Fetch user information and playlists after successful login
                fetch_user_info()
                fetch_user_playlists()
            else:
                # Authentication failed, show error message
                st.error("Authentication failed. Please try again.")
    else:
        # If there's no code in the query parameters, show a message to log in
        st.write("Please log in using Spotify to continue.")
        
        
st.write("")  
if st.session_state.access_token:
    if st.session_state.user_info:
        col1, col2 = st.columns([1,3])
        with col1:
            st.markdown(
                f"""
                <style>
                    .responsive-profile-image {{
                        display: inline-block;
                        border-radius: 50%;
                        overflow: hidden;
                        border: 2px solid lightgreen;  
                        width: 25vw; 
                        height: 25vw; 
                        max-width: 100px; 
                        max-height: 100px;
                    }}
                    .responsive-profile-image img {{
                        width: 100%;
                        height: 100%;
                        object-fit: cover;
                    }}
                </style>
                <div class="responsive-profile-image">
                    <img src="{st.session_state.user_info['images'][0]['url']}" alt="Profile Image">
                </div>
                """,
                unsafe_allow_html=True
            )


        with col2:
            st.markdown(
                f"""
                <div>
                    Hello, <span style='color: lightgreen; font-size: 20px;'>{st.session_state.user_info['display_name']}</span><br>
                    <span style='color: white; font-size: 16px;'>{st.session_state.user_info['email']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
   
            
            
    st.markdown("___")
    # Playlist dropdown
    playlist_names = [p['name'] for p in st.session_state.playlists]
    selected_playlist = st.selectbox("Choose a playlist", playlist_names if playlist_names else ["No playlists available"])

    if selected_playlist:
        selected_playlist_id = next(p['id'] for p in st.session_state.playlists if p['name'] == selected_playlist)
        st.session_state.selected_playlist_id = selected_playlist_id
        
        with st.spinner("Loading playlist tracks..."):
            fetch_playlist_songs(selected_playlist_id)

        # Display playlist cover image
        playlist_image = next(p['image'] for p in st.session_state.playlists if p['id'] == selected_playlist_id)
        # if playlist_image:
        #     st.image(playlist_image, width=60)
        if playlist_image:
            col1, col2 = st.columns([1, 3])  
            with col1:
                st.image(playlist_image, width=160) 
            with col2:
                # Display the song name
                st.markdown(f"Selected Playlist: <span style='color: lightgreen;'>**{selected_playlist}**</span>", unsafe_allow_html=True)


        if st.session_state.songs:
            st.success(f"Loaded {len(st.session_state.songs)} tracks from the playlist!")
    
            # Song dropdown for selected playlist
            st.subheader("Get Recommendations")
            song_names = [song.get('name', 'Unknown') for song in st.session_state.songs]
            selected_song = st.selectbox("Choose a song", song_names if song_names else ["No songs available"])

            if selected_song:
                selected_song_index = next((i for i, song in enumerate(st.session_state.songs) if song.get('name') == selected_song), None)
                if selected_song_index is not None:
                    selected_song_id = st.session_state.songs[selected_song_index]['id']
                    song_image = st.session_state.songs[selected_song_index].get('image', None)
                    
                    if song_image:
                        col1, col2 = st.columns([1, 3])  
                        with col1:
                            st.image(song_image, width=160)
                        with col2:
                            st.markdown(f"Selected Song: <span style='color: lightgreen;'>**{selected_song}**</span>", unsafe_allow_html=True)

                    # Fetch and display song recommendations with embedded players
                    with st.spinner("Getting recommendations..."):
                        recommendations = fetch_song_recommendations(selected_song_id)
                    
                    if recommendations:
                        st.subheader("Recommended Songs")
                        for rec in recommendations:
                            with st.expander(f"🎵 {rec.get('name', 'Unknown')} - {', '.join(artist['name'] for artist in rec['artists'])}"):
                                embed_url = f"https://open.spotify.com/embed/track/{rec['id']}"
                                st.markdown(
                                    f'<iframe style="border-radius:12px" src="{embed_url}" '
                                    'width="100%" height="152" frameBorder="0" allowfullscreen="" '
                                    'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" '
                                    'loading="lazy"></iframe>',
                                    unsafe_allow_html=True
                                )
