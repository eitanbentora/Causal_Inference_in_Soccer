import sqlite3
import pandas as pd
from datetime import datetime

from consts import WINS_QUERY, MATCH_DF_QUERY, OBSERVED_LEAGUE_ID, BETTING_COLS, IDENTIFICATION_COLS, \
    LEAGUE_CONSTANT_COLS, CONST_SEASON_COLS


def get_match_df(connection):
    query = MATCH_DF_QUERY
    match_df = pd.read_sql_query(query, connection)
    format_date_str = "%Y-%m-%d %H:%M:%S"
    betting_cols = BETTING_COLS
    match_df['date'] = match_df['date'].apply(lambda x: datetime.strptime(x, format_date_str))
    match_df['goal_diff'] = match_df['home_team_goal'] - match_df['away_team_goal']
    identification_cols = IDENTIFICATION_COLS
    cols = identification_cols + betting_cols
    match_df = match_df[cols]
    for res in ['H', 'D', 'A']:
        match_df[f'mean_bet_{res}'] = match_df[[col for col in betting_cols if col[-1] == res]].mean(axis=1)
    match_df = match_df[identification_cols + [f'mean_bet_{res}' for res in ['H', 'D', 'A']]]
    return match_df


def get_wins_df(connection):
    wins = pd.read_sql_query(WINS_QUERY, connection)
    wins['league_points'] = 3 * wins['wins'] + wins['draws']
    return wins


def add_wins_to_match_df(match_df, wins_df):
    match_df = match_df.copy()
    wins_df = wins_df.copy()
    win_cols = ['wins', 'draws', 'league_points']
    wins_home = wins_df.copy().rename({col: f"{col}_home" for col in win_cols}, axis=1)
    wins_away = wins_df.copy().rename({col: f"{col}_away" for col in win_cols}, axis=1)
    wins_away = wins_away.rename({'home_team_api_id': 'away_team_api_id'}, axis=1)
    match_df = match_df.merge(wins_home, on=['league_id', 'season', 'home_team_api_id'])
    match_df = match_df.merge(wins_away, on=['league_id', 'season', 'away_team_api_id'])
    return leave_one_league(match_df)


def leave_one_league(df, league_id=OBSERVED_LEAGUE_ID):
    return df[df['league_id'] == league_id].copy().sort_values('date').reset_index(drop=True)


def get_row_team_data(team_df):
    cols_dict = {}
    for result in ['wins', 'draws', 'league_points']:
        cols_dict[f'row_team_{result}'] = team_df.apply(
            lambda row: row[f'{result}_home'] if row['home'] else row[f'{result}_away'], axis=1)
    row_team_data = pd.DataFrame.from_dict(cols_dict)
    return row_team_data


def get_rival_data(team_df):
    cols_dict = {}
    cols_dict['rival_team_api_id'] = team_df.apply(
            lambda row: row['home_team_api_id'] if not row['home'] else row['away_team_api_id'], axis=1)
    for result in ['wins', 'draws', 'league_points']:
        cols_dict[f'rival_{result}'] = team_df.apply(
            lambda row: row[f'{result}_home'] if not row['home'] else row[f'{result}_away'], axis=1)
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
    wins_df = get_wins_df(connection)
    full_match_df = add_wins_to_match_df(match_df, wins_df)
    consecutive_games_df = add_next_game_columns(full_match_df)
    consecutive_games_df.to_pickle('data/consecutive_games_df.pkl')


if __name__ == '__main__':
    main()
