import argparse
import os
import re
import csv
import datetime
import pytz
import calendar
from itertools import accumulate
import pickle

from functools import partial
from functools import reduce

from ortools.sat.python import cp_model

from utils import *

import requests


DEFAULT_DAILY_MATCHES = [
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (2, 2, 2),
    (2, 2, 2),
    (4, 4, 2),
]


def set_consecutive_days(match_days, nc=1):
    consecutive_days = []
    for i in range(len(match_days)-nc):
        cdays = [i, ]
        for j in range(i+1, i+1+nc):
            ddiff = match_days[j][-1] - match_days[i][-1]
            if (ddiff <= nc):
                cdays.append(j)
            else:
                break
        if len(cdays) > 1:
            consecutive_days.append(cdays)

    return consecutive_days


def set_matchdays(num_matches, initial=[]):

    print("# of matches to be assigned: {}".format(num_matches))

    match_days = initial

    min_matches_per_week = sum([mi for (mi, mx, _) in DEFAULT_DAILY_MATCHES])
    max_matches_per_week = sum([mx for (mi, mx, _) in DEFAULT_DAILY_MATCHES])

    print(
        "Min and Max Matches per Week {} {}".format(
            min_matches_per_week, max_matches_per_week
        )
    )

    min_initial = sum([mi for (mi, mx, _) in match_days])
    max_initial = sum([mx for (mi, mx, _) in match_days])

    num_matches_left_max = int(num_matches - max_initial)
    num_full_weeks = num_matches_left_max // max_matches_per_week
    print("Number full weeks: {}".format(num_full_weeks))

    for i in range(num_full_weeks):
        match_days += DEFAULT_DAILY_MATCHES

    tmp = sum([mx for (mi, mx, _) in match_days])
    rm_matches_max = num_matches - tmp

    tmp = sum([mi for (mi, mx, _) in match_days])
    rm_matches_min = num_matches - tmp

    print("# of remaining matches (min): {}".format(rm_matches_min))
    print("# of remaining matches (max): {}".format(rm_matches_max))

    while rm_matches_min > 0:
        for day in DEFAULT_DAILY_MATCHES:
            if rm_matches_min < 1:
                break
            if rm_matches_max > 0:
                mtchs = min(rm_matches_min, day[0])
                match_days += [(mtchs, day[1], day[2])]
            else:
                mtchs = min(rm_matches_min, day[1])
                match_days += [(0, mtchs, day[2])]
            rm_matches_min -= mtchs
            rm_matches_max -= mtchs

    tmp = sum([mx for (mi, mx, _) in match_days])
    print("# of matches(max): {}".format(tmp))

    tmp = sum([mi for (mi, mx, _) in match_days])
    print("# of matches(min): {}".format(tmp))

    return match_days


def check_max_home_stand(x):
    n1 = 0
    n2 = 0
    max_home_stand = 2
    m = max_home_stand
    for e in x:
        n1 = n1 + e*n1
        n2 = n2 - e*n2
        if n1 > m or n2 < -m:
            return False
    return True


def get_scheduled_fixtures(solver, fixtures, start_date, match_days):
    timezone = pytz.timezone('Asia/Kolkata')
    startdate = datetime.datetime.strptime(start_date, "%d/%m/%Y")
    startdate = timezone.localize(startdate)
    fixed_matches = []
    for (day, fd) in enumerate(fixtures):
        for (home, fh) in enumerate(fd):
            for (away, fixture) in enumerate(fh):
                if solver.Value(fixture):
                    Date = startdate + \
                        datetime.timedelta(days=match_days[day][-1])
                    fixed_matches += [(Date, home, away,
                                       match_days[day][-1]), ]
    return fixed_matches


def screen_dump_results(scheduled_games, teams):

    prev_date = ""
    d2_time = [(7, 0), (17, 30)]
    d4_time = [(7, 0), (8, 0), (16, 30), (17, 30)]
    scdl = []
    ds = []

    print("")
    print("")
    print("-------- Fixture -----------")
    for (d, item) in enumerate(scheduled_games):

        df = {}
        df['date'] = item[0]
        df['day'] = d
        df['home'] = teams[item[1]]['pk']
        df['homeAbbr'] = teams[item[1]]['abbr']
        df['away'] = teams[item[2]]['pk']
        df['awayAbbr'] = teams[item[2]]['abbr']
        df['ground'] = teams[item[1]]['home_ground']

        if prev_date != df['date']:
            if len(ds) == 4:
                d_time = d4_time
            elif len(ds) == 2:
                d_time = d2_time

            for (t, it) in enumerate(ds):
                it['date'] += datetime.timedelta(hours=d_time[t]
                                                 [0], minutes=d_time[t][1])
                scdl.append(it)
            ds = []
        ds.append(df)
        if not prev_date == df['date']:
            print("")
            print("")
            print(
                "---> {0: >12} ".format(item[0].strftime("%b. %d %a")))

        print("               {0: >4} x {1: <4} ".format(
            df['homeAbbr'], df['awayAbbr']))
        prev_date = df['date']

    if len(ds) == 4:
        d_time = d4_time
    elif len(ds) == 2:
        d_time = d2_time
    for (t, it) in enumerate(ds):
        it['date'] += datetime.timedelta(hours=d_time[t]
                                         [0], minutes=d_time[t][1])
        scdl.append(it)

    with open("fixture.dat", "wb") as fp:
        pickle.dump(scdl, fp)


def report_results(solver, status, fixtures, start_date, match_days, teams, time_limit=None):

    if status == cp_model.INFEASIBLE:
        print('INFEASIBLE')
        return status

    if status == cp_model.UNKNOWN:
        print('Not enough time allowed to compute a solution')
        print('Add more time using the --timelimit command line option')
        return status

    print('Optimal objective value: %i' % solver.ObjectiveValue())

    scheduled_games = get_scheduled_fixtures(
        solver, fixtures, start_date, match_days)

    screen_dump_results(scheduled_games, teams)

    if status != cp_model.OPTIMAL and solver.WallTime() >= time_limit:
        print('Please note that solver reached maximum time allowed %i.' % time_limit)
        print('A better solution than %i might be found by adding more time using the --timelimit command line option' %
              solver.ObjectiveValue())


def model_matches(teams_d, initial=[], max_home_stand=2, donematches=[]):

    model = cp_model.CpModel()
    num_teams = len(teams_d)
    grnd_pk = list(set([item['home_ground'] for item in teams_d]))

    grounds_d = {}
    for (i, pk) in enumerate(grnd_pk):
        grounds_d[pk] = i

    num_matches = (num_teams * (num_teams - 1))
    num_matches_half_season = num_matches // 2
    num_grounds = len(grounds_d)
    grounds = range(num_grounds)
    teams = range(num_teams)

    print("# of half season matches: {}".format(num_matches))
    daily_matches = set_matchdays(num_matches, initial=initial)
    print(daily_matches)

    match_days = []
    for (daynum, (mi, mx, numgrnd)) in enumerate(daily_matches):
        if mi+mx > 0:
            match_days += [(mi, mx, numgrnd, daynum)]

    print(match_days)

    num_match_days = len(match_days)
    matchdays = range(num_match_days)

    num_days_half_season = 0
    nmatches = 0
    for day in match_days:
        nmatches += day[0]
        if nmatches > num_matches_half_season:
            nmatches -= day[0]
            break
        num_days_half_season += 1

    print("first half season days: {} ({} matches)".format(
        num_days_half_season, nmatches))

    print("# of match days: %i" % (num_match_days))

    consec_days = set_consecutive_days(match_days, nc=1)

    fixtures = daily_fixtures(model, num_teams, num_match_days)

    ground_fixture = []
    for day in matchdays:
        name_prefix = 'day %i, ' % (day)
        ground_fixture.append(
            [model.NewBoolVar(name_prefix+'ground %i' % ground) for ground in grounds])

    home_fixture = []
    away_fixture = []
    for day in matchdays:
        name_prefix = 'day %i, ' % (day)
        home_fixture.append(
            [model.NewBoolVar(name_prefix+'home %i' % i) for i in teams])
        away_fixture.append(
            [model.NewBoolVar(name_prefix+'away %i' % i) for i in teams])

    # Link match fixtures to ground fixtures and home fixtures
    for d in matchdays:
        for i in teams:
            for j in teams:
                if teams_d[i]['home_ground'] != teams_d[j]['home_ground']:
                    ng = grounds_d[teams_d[i]['home_ground']]
                    model.AddImplication(
                        fixtures[d][i][j],
                        ground_fixture[d][ng])
                    ng = grounds_d[teams_d[j]['home_ground']]
                    model.AddImplication(
                        fixtures[d][i][j],
                        ground_fixture[d][ng].Not())
                elif i != j:
                    ng = grounds_d[teams_d[i]['home_ground']]
                    model.AddImplication(
                        fixtures[d][i][j],
                        ground_fixture[d][ng])

                model.AddImplication(
                    fixtures[d][i][j], home_fixture[d][i])
                model.AddImplication(
                    fixtures[d][i][j], home_fixture[d][j].Not())
                model.AddImplication(
                    fixtures[d][i][j], away_fixture[d][j])
                model.AddImplication(
                    fixtures[d][i][j], away_fixture[d][i].Not())

    # forbid playing self
    [model.Add(fixtures[d][i][i] == 0) for d in matchdays for i in teams]

    # minimum and maximum number of matches per day
    # maximum grounds where matches are happening in day
    print("Rule: Minimum and Maximum number of Matches for each day")
    print("Rule: Maximum number of Ground on which matches are held on any day")
    for (d, (minM, maxM, maxG, daynum)) in enumerate(match_days):
        model.Add(sum([fixtures[d][i][j]
                       for i in teams for j in teams if i != j]) >= minM)
        model.Add(sum([fixtures[d][i][j]
                       for i in teams for j in teams if i != j]) <= maxM)
        model.Add(sum([ground_fixture[d][i] for i in grounds]) <= maxG)

    # For any team no more than one match per # consecutive days
    print("Rule: For any team no more than one match per 2 consecutive days")
    for batch in consec_days:
        for i in teams:
            model.Add(sum([fixtures[d][i][j] + fixtures[d][j][i]
                           for j in teams if i != j for d in batch]) <= 1)

    print("Rule: A match only once in season")
    for i in teams:
        for j in teams:
            if i != j:
                model.Add(sum([fixtures[d][i][j] for d in matchdays]) == 1)

    first_leg_days = range(num_days_half_season)
    # Either home or away for the half season
    # Here more constrains will come as the first leg matches will be over
    print("Rule: Either A vs B or B vs A in a half Season")
    for i in teams:
        for j in teams:
            if i != j:
                model.Add(sum([fixtures[d][i][j] + fixtures[d][j][i]
                               for d in first_leg_days]) == 1)

    print("Rule: No more than 2 consecutive home or away matches for a team")

    for t in teams:
        model.Add(sum([home_fixture[d][t] for d in first_leg_days]) <= 6)
        model.Add(sum([away_fixture[d][t] for d in first_leg_days]) <= 6)

    for donematch in donematches:
        d, home, away, date = donematch
        model.Add(fixtures[d][home][away] == 1)
        print('({}) {} - {}x{}'.format(d, date,
                                       teams_d[home]['abbr'], teams_d[away]['abbr']))

    return (model, fixtures, match_days)


def solve_model(model, time_limit=None, num_cpus=None, debug=False):
    # run the solver
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = num_cpus
    status = solver.Solve(model)
    print("Solve status: %s" % solver.StatusName(status))
    print("Statistics")
    print("  - conflicts : %i" % solver.NumConflicts())
    print("  - branches  : %i" % solver.NumBranches())
    print("  - wall time : %f s" % solver.WallTime())
    return (solver, status)


def get_team_id(teams, pk):
    for i, team in enumerate(teams):
        if team['pk'] == pk:
            return i
    return None


def main():
    url = "https://vleague.in/en/fixture/api/fixtureinput/?format=json"
    resp = requests.get(url=url)
    data = resp.json()
    print("# of Team: %i" % (len(data)))

    teams = []
    for item in data:
        teams += [item, ]

    print(teams)

    url = "https://vleague.in/en/fixture/api/matchinput/?format=json"
    resp = requests.get(url=url)
    data = resp.json()
    print("# of Fixed matches: %i" % (len(data)))

    dates = {}
    for item in data:
        date = datetime.datetime.strptime(item['date'], '%Y-%m-%dT%H:%M:%S%z')
        item['date'] = date
        dt = date.strftime('%Y-%m-%d')
        li = dates.get(dt, [])
        li.append(item)
        dates[dt] = li

    tmp = sorted(dates.keys())
    sdate = datetime.datetime.strptime(tmp[0], '%Y-%m-%d')
    edate = datetime.datetime.strptime(tmp[-1], '%Y-%m-%d')
    dt = datetime.timedelta(days=1)
    date = sdate

    initial = []
    day = 0
    donematches = []
    while date <= edate:
        tstamp = date.strftime('%Y-%m-%d')
        li = dates.get(tstamp, [])
        nm = len(li)
        initial.append((nm, nm, 2))
        date += dt
        for match in li:
            home = get_team_id(teams, match['home'])
            away = get_team_id(teams, match['away'])
            donematches.append((day, home, away, tstamp))
        if nm > 0:
            day += 1

    cpu = cpu_guess_and_gripe(6)
    # set up the model
    start_date = "30/01/2021"

    time_limit = 600
    (model, fixtures, match_days) = model_matches(
        teams, initial=initial, donematches=donematches)
    (solver, status) = solve_model(model, time_limit=time_limit, num_cpus=cpu)
    report_results(solver, status, fixtures, start_date,
                   match_days, teams, time_limit)


if __name__ == "__main__":
    main()
