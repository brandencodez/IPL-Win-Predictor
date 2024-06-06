import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import random

# Load the model pipeline
pipe = pickle.load(open('pipe.pkl','rb'))

# Load IPL matches data
@st.cache_data
def load_matches_data():
    return pd.read_csv('matches.csv')

matches_data = load_matches_data()

# Load IPL deliveries data
@st.cache_data
def load_deliveries_data():
    return pd.read_csv('deliveries.csv')

deliveries_data = load_deliveries_data()

# Define the teams and cities
teams = {
    'Sunrisers Hyderabad': 'orange',
    'Mumbai Indians': 'blue',
    'Royal Challengers Bangalore': 'red',
    'Kolkata Knight Riders': 'purple',
    'Kings XI Punjab': 'silver',
    'Chennai Super Kings': 'yellow',
    'Rajasthan Royals': 'pink',
    'Delhi Capitals': 'darkblue'
}

cities = sorted(['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru'])

# Commentary phrases
commentary_phrases = [
    "What an intense match we have here!",
    "Both teams are giving it their all!",
    "The match is finely balanced!",
    "The excitement is building up in the stadium!",
    "This match could go down to the wire!",
    "The pressure is on for both batting and bowling sides!",
    "The chase is on for the batting team!",
    "The bowling team is looking for quick wickets!",
    "The outcome of this match hangs in the balance!",
    "The fans are in for a nail-biting finish!",
    "This match has all the ingredients of a classic!",
    "Who will emerge victorious in this thrilling encounter?",
    "The middle overs are crucial for both teams.",
    "The powerplay overs have set the tone for the innings.",
    "The fielding team needs to tighten up their fielding.",
    "The atmosphere is electric in the final overs!",
    "The fans are witnessing a classic encounter!",
    "The pressure is getting to the players!",
    "The batsmen are taking the attack to the bowlers!",
    "The chase is on for the batting side!"
]

# Streamlit app layout
st.image('74508790.png')
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', list(teams.keys()))

with col2:
    bowling_team = st.selectbox('Select the bowling team', list(teams.keys()))

selected_city = st.selectbox('Cities', cities)

target = st.number_input('Target', min_value=0)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)

with col4:
    wickets = st.number_input('Wickets', min_value=0, max_value=9)

with col5:
    overs = st.number_input('Overs completed', min_value=0, max_value=20)

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - overs * 6
    wickets = 10 - wickets
    crr = min(score / overs, 6)  # Limiting CRR to 6 runs per over
    rrr = min(runs_left * 6 / balls_left, 6)  # Limiting RRR to 6 runs per over

    # Randomly select a commentary phrase
    selected_commentary = random.choice(commentary_phrases)

    # Display the commentary
    st.write(selected_commentary)

    # Display the message
    message = f"{batting_team} needs {runs_left} runs to win in {balls_left} balls."
    st.write(message)

    df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})
    result = pipe.predict_proba(df)
    r_1 = round(result[0][0] * 100)
    r_2 = round(result[0][1] * 100)
    st.header('Winning Probability')
    st.header(f"{batting_team}  : {r_2} %")
    st.header(f"{bowling_team}  : {r_1} %")

    # Plot winning probability bar chart
    fig, ax = plt.subplots()
    teams_list = [batting_team, bowling_team]
    probabilities = [r_2, r_1]
    team_colors = [teams[team] for team in teams_list]
    ax.bar(teams_list, probabilities, color=team_colors)
    ax.set_ylabel('Winning Probability (%)')
    ax.set_title('Winning Probability')
    st.pyplot(fig)

    # Display head-to-head results
    st.subheader('Head-to-Head Results')
    head_to_head_matches = matches_data[((matches_data['team1'] == batting_team) & (matches_data['team2'] == bowling_team)) | ((matches_data['team1'] == bowling_team) & (matches_data['team2'] == batting_team))]
    if not head_to_head_matches.empty:
        st.write(head_to_head_matches[['team1', 'team2', 'winner', 'win_by_wickets', 'win_by_runs',
'venue', 'date']])

        # Calculate win count for each team
        team1_wins = len(head_to_head_matches[head_to_head_matches['winner'] == batting_team])
        team2_wins = len(head_to_head_matches[head_to_head_matches['winner'] == bowling_team])

        st.subheader('Head-to-Head Win Counts')
        st.write(f"{batting_team}: {team1_wins} wins")
        st.write(f"{bowling_team}: {team2_wins} wins")
    else:
        st.write("No head-to-head matches found between the selected teams.")

    # Display recent match results for the selected teams
    st.subheader('Recent Match Results')
    recent_matches_batting = matches_data[((matches_data['team1'] == batting_team) | (matches_data['team2'] == batting_team)) & (matches_data['winner'].notnull())].tail(5)
    recent_matches_bowling = matches_data[((matches_data['team1'] == bowling_team) | (matches_data['team2'] == bowling_team)) & (matches_data['winner'].notnull())].tail(5)

    st.write(f"Recent matches played by {batting_team}:")
    st.write(recent_matches_batting[['team1', 'team2', 'winner',  'win_by_wickets', 'win_by_runs','venue', 'date']])

    st.write(f"Recent matches played by {bowling_team}:")
    st.write(recent_matches_bowling[['team1', 'team2', 'winner', 'win_by_wickets', 'win_by_runs', 'venue', 'date']])
