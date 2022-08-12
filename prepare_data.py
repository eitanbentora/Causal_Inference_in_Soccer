import sqlite3
import pandas as pd
from datetime import datetime
import numpy as np
from consts import WINS_QUERY, MATCH_DF_QUERY, OBSERVED_LEAGUE_ID, BETTING_COLS, IDENTIFICATION_COLS, \
    LEAGUE_CONSTANT_COLS, CONST_SEASON_COLS, SEASON_COLS, PLAYER_ATT_QUERY, FORMAT_DATE_STR


def find_closest_past_time_from_time_list(find_time, time_list, return_list):
    deltas = np.array([(pd.Timestamp(time) - find_time).days for time in time_list])
    if (deltas > 0).all():
        return return_list[0]
    neg_deltas = deltas[deltas <= 0]
    closest_delta = neg_deltas[(np.abs(neg_deltas) <= min(np.abs(neg_deltas)))][0]
    return return_list[np.argwhere(deltas == closest_delta)[0]][0]


def get_player_att_dict(connection):
    # Return overall ratings of players since 2006
    player_att_df = pd.read_sql_query(PLAYER_ATT_QUERY, connection)
    player_att_df['date'] = player_att_df['date'].apply(lambda x: datetime.strptime(x, FORMAT_DATE_STR))
    player_att_df = player_att_df[player_att_df['date'] >= datetime(year=2006, month=1, day=1)]
    return player_att_df[['player_api_id', 'date', 'overall_rating']]


def team_score(row, player_att_df, court):
    player_id_cols = [col for col in row.index if f'{court}_player' in col and 'X' not in col and 'Y' not in col]
    players_score_list = []
    for player_col in player_id_cols:
        player_df = player_att_df[player_att_df['player_api_id'] == row[player_col]]
        if len(player_df) == 0:
            continue
        season = row['season_date']
        evaluation_times = player_df['date'].to_numpy()
        scores = player_df['overall_rating'].to_numpy()
        players_score_list.append(find_closest_past_time_from_time_list(season, evaluation_times, scores))
    team_scores = (np.min(players_score_list), np.max(players_score_list), np.mean(players_score_list),
                   np.std(players_score_list), np.median(players_score_list))
    return team_scores


def get_match_df(connection):
    match_df = pd.read_sql_query(MATCH_DF_QUERY, connection)
    match_df['date'] = match_df['date'].apply(lambda x: datetime.strptime(x, FORMAT_DATE_STR))
    match_df['season_date'] = match_df['season'].apply(lambda x: datetime(year=int(x.split('/')[0]), month=1, day=1))
    player_att_df = get_player_att_dict(connection)

    print("Evaluating home team players' stats in all matches...")
    match_df[[f'home_team_{x}_squad_rating' for x in ['min', 'max', 'mean', 'std', 'median']]] = \
        match_df.apply(lambda x: team_score(x, player_att_df, 'home'), axis=1, result_type='expand')

    print("Evaluating away team players' stats in all matches...")
    match_df[[f'away_team_{x}_squad_rating' for x in ['min', 'max', 'mean', 'std', 'median']]] = \
        match_df.apply(lambda x: team_score(x, player_att_df, 'away'), axis=1, result_type='expand')

    match_df['goal_diff'] = match_df['home_team_goal'] - match_df['away_team_goal']
    # Get only ids, bets and relevant info about matches (goals and ratings):
    match_df = match_df[IDENTIFICATION_COLS + BETTING_COLS]
    # Average all betting sites to a single odd for each outcome per match:
    for res in ['H', 'D', 'A']:
        match_df[f'mean_bet_{res}'] = match_df[[col for col in BETTING_COLS if col[-1] == res]].mean(axis=1)
    match_df = match_df[IDENTIFICATION_COLS + [f'mean_bet_{res}' for res in ['H', 'D', 'A']]]
    return match_df


def get_season_goals_df(match_df):
    # Total amount of goals per season for each team
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
    # Wins and league points info
    wins = pd.read_sql_query(WINS_QUERY, connection)
    wins['league_points'] = 3 * wins['wins'] + wins['draws']
    return wins


def get_season_df(connection, match_df):
    # Win, league points and season info (total goals for each team)
    wins = get_wins_df(connection)
    season_goals_df = get_season_goals_df(match_df)
    season_df = wins.merge(season_goals_df, on=['season', 'row_team_api_id', 'league_id'])
    # season_df['season'] = season_df['season'].apply(lambda s: '/'.join([str(int(y) + 1) for y in s.split('/')]))
    return season_df


def add_season_df_to_match_df(match_df, season_df):
    match_df = match_df.copy()
    season_df = season_df.copy()
    home_season_df = season_df.copy().rename({col: f"{col}_home" for col in SEASON_COLS}, axis=1)
    home_season_df = home_season_df.rename({'row_team_api_id': 'home_team_api_id'}, axis=1)
    away_season_df = season_df.copy().rename({col: f"{col}_away" for col in SEASON_COLS}, axis=1)
    away_season_df = away_season_df.rename({'row_team_api_id': 'away_team_api_id'}, axis=1)
    match_df = match_df.merge(home_season_df, on=['league_id', 'season', 'home_team_api_id'])
    match_df = match_df.merge(away_season_df, on=['league_id', 'season', 'away_team_api_id'])
    return leave_one_league(match_df)


def leave_one_league(df, league_id=OBSERVED_LEAGUE_ID):
    return df[df['league_id'] == league_id].copy().sort_values('date').reset_index(drop=True)


def get_row_team_data(team_df):
    cols_dict = {}
    for result in SEASON_COLS:
        cols_dict[f'row_team_{result}'] = get_feature_for_row_team(team_df, f'{result}_home', f'{result}_away')
    cols_dict['row_team_goal'] = get_feature_for_row_team(team_df, 'home_team_goal', 'away_team_goal')
    row_team_data = pd.DataFrame.from_dict(cols_dict)
    return row_team_data


def get_rival_data(team_df):
    cols_dict = {}
    cols_dict['rival_team_api_id'] = get_feature_for_rival_team(team_df, 'home_team_api_id', 'away_team_api_id')
    for result in SEASON_COLS:
        cols_dict[f'rival_{result}'] = get_feature_for_rival_team(team_df, f'{result}_home', f'{result}_away')
    cols_dict['rival_team_goal'] = get_feature_for_rival_team(team_df, 'home_team_goal', 'away_team_goal')
    rival_team_data = pd.DataFrame.from_dict(cols_dict)
    return rival_team_data


def get_feature_for_row_team(team_df, home_feature, away_feature):
    return team_df.apply(lambda row: row[home_feature] if row['home'] else row[away_feature], axis=1)


def get_feature_for_rival_team(team_df, home_feature, away_feature):
    return team_df.apply(lambda row: row[home_feature] if not row['home'] else row[away_feature], axis=1)


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
    for measure in ['min', 'max', 'mean', 'std', 'median']:
        home_feature = f'home_team_{measure}_squad_rating'
        away_feature = f'away_team_{measure}_squad_rating'
        cols_dict[f'row_team_{measure}_squad_rating'] = get_feature_for_row_team(team_df, home_feature, away_feature)
        cols_dict[f'rival_team_{measure}_squad_rating'] = get_feature_for_rival_team(team_df, home_feature, away_feature)
    match_data = pd.DataFrame.from_dict(cols_dict)
    return match_data


def get_team_df(df, team_id):
    # Take all matches in which team 'team_id' participates
    team_df = df[(df['home_team_api_id'] == team_id) | (df['away_team_api_id'] == team_id)]
    team_df['home'] = df['home_team_api_id'] == team_id
    # For each of these matches, get seasonal data for each participant (team_id and rival team)
    row_team_data = get_row_team_data(team_df)
    rival_team_data = get_rival_data(team_df)
    # For each match also get general data (betting odds and ratings)
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
        # team_df - info about team_id matches: team_id and rival seasonal info, and match info (odds and ratings)
        team_df = get_team_df(df, team_id)
        # for each season, for each match add info about next match of team_id
        for season, season_df in team_df.groupby('season'):
            consecutive_games_df = join_consecutive_games(season_df, team_id)
            consecutive_games_dfs.append(consecutive_games_df)
    return pd.concat(consecutive_games_dfs, axis=0).sort_values('prev_date')


def main():
    connection = sqlite3.connect('data/database.sqlite')
    # match_df - goals, ratings and betting odds info for each match in a certain league
    match_df = get_match_df(connection)
    # season_df - wins, league points and season info (total goals for each team)
    season_df = get_season_df(connection, match_df)
    # Merge match and season dataframes
    full_match_df = add_season_df_to_match_df(match_df, season_df)
    # consecutive_games_df - match info per team, including team and rival seasonal info, odds and ratings,
    # and team's next match info
    consecutive_games_df = add_next_game_columns(full_match_df)
    # print(consecutive_games_df.columns)
    consecutive_games_df.to_pickle('data/consecutive_games_df.pkl')


if __name__ == '__main__':
    main()
