MATCH_DF_QUERY = '''
      SELECT * 
      FROM Match
    '''
WINS_QUERY = """
      SELECT season, home_team_api_id, SUM(home_win + away_win) as wins, SUM(draw) as draws, league_id
      FROM
        (
        SELECT MATCH.id, MATCH.home_team_api_id, Match.season, t1.team_long_name as home_team, t2.team_long_name as away_team,
          (CASE WHEN home_team_goal > away_team_goal THEN 1 ELSE 0 END) AS home_win,
          (CASE WHEN home_team_goal = away_team_goal THEN 1 ELSE 0 END) AS draw,
          (CASE WHEN home_team_goal < away_team_goal THEN 1 ELSE 0 END) AS away_win, league_id
        FROM MATCH 
        JOIN TEAM as t1 ON t1.team_api_id = MATCH.home_team_api_id 
        JOIN TEAM as t2 ON t2.team_api_id = MATCH.away_team_api_id)
        GROUP BY home_team, season
        ORDER BY season DESC, wins DESC
      """

OBSERVED_LEAGUE_ID = 1729

BETTING_COLS = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD',
                'PSA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD',
                'BSA']

IDENTIFICATION_COLS = ['league_id', 'season', 'date', 'match_api_id', 'home_team_api_id', 'away_team_api_id',
                       'goal_diff']

LEAGUE_CONSTANT_COLS = ['season', 'league_id']

CONST_SEASON_COLS = LEAGUE_CONSTANT_COLS + ['row_team_wins', 'row_team_draws', 'row_team_league_points']
