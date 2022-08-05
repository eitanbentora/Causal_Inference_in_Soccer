import sqlite3
import pandas as pd
from datetime import datetime
import numpy as np
from consts import WINS_QUERY, MATCH_DF_QUERY, OBSERVED_LEAGUE_ID, BETTING_COLS, IDENTIFICATION_COLS, \
    LEAGUE_CONSTANT_COLS, CONST_SEASON_COLS, SEASON_COLS, PLAYER_ATT_QUERY


def find_closest_past_time_from_time_list(find_time, time_list, return_list):
    deltas = np.array([(time - find_time).days for time in time_list])
    if (deltas > 0).all():
        return return_list[0]
    neg_deltas = deltas[deltas <= 0]
    closest_delta = neg_deltas[(np.abs(neg_deltas) <= min(np.abs(neg_deltas)))][0]
    return return_list[np.argwhere(deltas == closest_delta)[0]][0]


def get_player_att_dict(connection, seasons):
    query = PLAYER_ATT_QUERY
    player_att_df = pd.read_sql_query(query, connection)
    format_date_str = "%Y-%m-%d %H:%M:%S"
    player_att_df['date'] = player_att_df['date'].apply(lambda x: datetime.strptime(x, format_date_str))
    player_att_df = player_att_df[player_att_df['date'] >= datetime(year=2006, month=1, day=1)]
    # season_names = seasons
    # seasons_dates = np.array([datetime(year=int(season.split('/')[0]), month=1, day=1) for season in seasons])
    # player_att_df['season'] = player_att_df['date'].apply(lambda x: find_closest_season(x, season_names, seasons_dates))
    return player_att_df[['player_api_id', 'date', 'overall_rating']]


def team_score(row, player_att_df, court):
    player_id_cols = [col for col in row.index if f'{court}_player' in col and 'X' not in col and 'Y' not in col]
    score = 0
    for player_col in player_id_cols:
        player_df = player_att_df[player_att_df['player_api_id'] == row[player_col]]
        if len(player_df) == 0:
            continue
        season = row['season_date']
        evaluation_times = player_df['date'].to_numpy()
        scores = player_df['overall_rating'].to_numpy()
        score += find_closest_past_time_from_time_list(season, evaluation_times, scores)
    return score / len(player_id_cols)


def get_match_df(connection):
    query = MATCH_DF_QUERY
    match_df = pd.read_sql_query(query, connection)
    format_date_str = "%Y-%m-%d %H:%M:%S"
    betting_cols = BETTING_COLS
    match_df['date'] = match_df['date'].apply(lambda x: datetime.strptime(x, format_date_str))
    seasons = match_df.season.unique()
    match_df['season_date'] = match_df['season'].apply(lambda x: datetime(year=int(x.split('/')[0]), month=1, day=1))
    player_att_df = get_player_att_dict(connection, seasons)
    print("a")
    match_df['home_team_rating'] = match_df.apply(lambda x: team_score(x, player_att_df, 'home'), axis=1)
    print("b")
    match_df['away_team_rating'] = match_df.apply(lambda x: team_score(x, player_att_df, 'away'), axis=1)
    print("c")
    match_df['goal_diff'] = match_df['home_team_goal'] - match_df['away_team_goal']
    identification_cols = IDENTIFICATION_COLS
    cols = identification_cols + betting_cols
    match_df = match_df[cols]
    for res in ['H', 'D', 'A']:
        match_df[f'mean_bet_{res}'] = match_df[[col for col in betting_cols if col[-1] == res]].mean(axis=1)
    match_df = match_df[identification_cols + [f'mean_bet_{res}' for res in ['H', 'D', 'A']]]
    return match_df


def get_season_goals_df(match_df):
    season_goals_df = []
    for team_id in match_df.home_team_api_id.unique():
        team_df = match_df[(match_df['home_team_api_id'] == team_id) | (match_df['away_team_api_id'] == team_id)]
        team_df['season_goals'] = team_df.apply(
            lambda x: x['home_team_goal'] if x['home_team_api_id'] == team_id else x['away_team_goal'], axis=1)
        season_team_df = team_df.groupby(['season', 'league_id']).agg({'season_goals': 'sum'}).reset_index()
        season_team_df['row_team_api_id'] = team_id
        season_goals_df.append(season_team_df)
    return pd.concat(season_goals_df)


def get_wins_df(connection):
    wins = pd.read_sql_query(WINS_QUERY, connection)
    wins['league_points'] = 3 * wins['wins'] + wins['draws']
    return wins


def get_season_df(connection, match_df):
    wins = get_wins_df(connection)
    season_goals_df = get_season_goals_df(match_df)
    return wins.merge(season_goals_df, on=['season', 'row_team_api_id', 'league_id'])


def add_season_df_to_match_df(match_df, season_df):
    match_df = match_df.copy()
    season_df = season_df.copy()
    season_cols = SEASON_COLS
    home_season_df = season_df.copy().rename({col: f"{col}_home" for col in season_cols}, axis=1)
    home_season_df = home_season_df.rename({'row_team_api_id': 'home_team_api_id'}, axis=1)
    away_season_df = season_df.copy().rename({col: f"{col}_away" for col in season_cols}, axis=1)
    away_season_df = away_season_df.rename({'row_team_api_id': 'away_team_api_id'}, axis=1)
    match_df = match_df.merge(home_season_df, on=['league_id', 'season', 'home_team_api_id'])
    match_df = match_df.merge(away_season_df, on=['league_id', 'season', 'away_team_api_id'])
    return leave_one_league(match_df)


def leave_one_league(df, league_id=OBSERVED_LEAGUE_ID):
    return df[df['league_id'] == league_id].copy().sort_values('date').reset_index(drop=True)


def get_row_team_data(team_df):
    cols_dict = {}
    for result in SEASON_COLS:
        cols_dict[f'row_team_{result}'] = team_df.apply(
            lambda row: row[f'{result}_home'] if row['home'] else row[f'{result}_away'], axis=1)
    cols_dict['row_team_goal'] = team_df.apply(
        lambda row: row['home_team_goal'] if row['home'] else row['away_team_goal'], axis=1)
    row_team_data = pd.DataFrame.from_dict(cols_dict)
    return row_team_data


def get_rival_data(team_df):
    cols_dict = {}
    cols_dict['rival_team_api_id'] = team_df.apply(
            lambda row: row['home_team_api_id'] if not row['home'] else row['away_team_api_id'], axis=1)
    for result in SEASON_COLS:
        cols_dict[f'rival_{result}'] = team_df.apply(
            lambda row: row[f'{result}_home'] if not row['home'] else row[f'{result}_away'], axis=1)
    cols_dict['rival_team_goal'] = team_df.apply(
        lambda row: row['home_team_goal'] if not row['home'] else row['away_team_goal'], axis=1)
    rival_team_data = pd.DataFrame.from_dict(cols_dict)
    return rival_team_data


def get_match_data(team_df):
    cols_dict = {}
    cols = ['match_api_id', 'date', 'home'] + LEAGUE_CONSTANT_COLS
    for col in cols:
        cols_dict[col] = team_df[col]
    cols_dict['goal_diff'] = team_df.apply(lambda row: row['goal_diff'] if row['home'] else -row['goal_diff'], axis=1)
    cols_dict['mean_bet_row_team'] = team_df.apply(
        lambda row: row['mean_bet_H'] if row['home'] else row['mean_bet_A'], axis=1)
    cols_dict['mean_bet_rival'] = team_df.apply(
        lambda row: row['mean_bet_A'] if row['home'] else row['mean_bet_H'], axis=1)
    cols_dict['mean_bet_draw'] = team_df['mean_bet_D']
    match_data = pd.DataFrame.from_dict(cols_dict)
    return match_data


def get_team_df(df, team_id):
    team_df = df[(df['home_team_api_id'] == team_id) | (df['away_team_api_id'] == team_id)]
    team_df['home'] = df['home_team_api_id'] == team_id
    row_team_data = get_row_team_data(team_df)
    rival_team_data = get_rival_data(team_df)
    match_results_data = get_match_data(team_df)
    row_team_data = pd.concat((match_results_data, row_team_data, rival_team_data), axis=1)
    return row_team_data


def join_consecutive_games(season_df, team_id):
    season_df = season_df.sort_values('date')
    non_const_cols = [col for col in season_df.columns if col not in CONST_SEASON_COLS]
    next_game = season_df.iloc[1:].copy().set_index(season_df.index[:-1], drop=True)[non_const_cols]
    prev_game = season_df.iloc[:-1].copy()[non_const_cols]
    next_game = next_game.rename({col: f'next_{col}' for col in next_game.columns}, axis=1)
    prev_game = prev_game.rename({col: f'prev_{col}' for col in prev_game.columns}, axis=1)
    consecutive_games_df = pd.concat((season_df[CONST_SEASON_COLS].iloc[:-1], prev_game, next_game), axis=1)
    consecutive_games_df['row_team'] = team_id
    consecutive_games_df = consecutive_games_df[['row_team'] + list(consecutive_games_df.columns[:-1])]
    return consecutive_games_df


def add_next_game_columns(df):
    consecutive_games_dfs = []
    team_ids = df['home_team_api_id'].unique()
    for team_id in team_ids:
        team_df = get_team_df(df, team_id)
        for season, season_df in team_df.groupby('season'):
            consecutive_games_df = join_consecutive_games(season_df, team_id)
            consecutive_games_dfs.append(consecutive_games_df)
    return pd.concat(consecutive_games_dfs, axis=0).sort_values('prev_date')


def main():
    connection = sqlite3.connect('data/database.sqlite')
    match_df = get_match_df(connection)
    season_df = get_season_df(connection, match_df)
    full_match_df = add_season_df_to_match_df(match_df, season_df)
    consecutive_games_df = add_next_game_columns(full_match_df)
    print(consecutive_games_df.columns)
    consecutive_games_df.to_pickle('data/consecutive_games_df.pkl')


if __name__ == '__main__':
    main()
