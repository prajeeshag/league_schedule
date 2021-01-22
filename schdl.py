
import argparse
import os
import re
import csv
from functools import partial
from functools import reduce

from ortools.sat.python import cp_model

DEFAULT_DAILY_MATCHES = [
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (2, 2),
    (2, 2),
    (2, 4),
]

DEFAULT_NUM_GROUNDS = [0, 0, 0, 0, 2, 2, 2]

TEAMS = (
    ('YFC', 'BSK'),
    ('KBFC', 'SRV'),
    ('BTZ', 'RFC'),
    ('RGM', 'BFC'),
    ('TTM', 'TMFC'),
    'FCC', 'SAR'
)


def set_matchdays(num_matches, initial=[]):

    print("# of matches to be assigned: {}".format(num_matches))

    match_days = initial

    min_matches_per_week = sum(
        [mi for (mi, mx) in DEFAULT_DAILY_MATCHES)
    max_matches_per_week = sum(
        [mx for (mi, mx) in DEFAULT_DAILY_MATCHES)
    print("Min and Max Matches per Week {} {}".format(
        min_matches_per_week, max_matches_per_week))

    min_initial = sum([mi for (mi, mx) in match_days])
    max_initial = sum([mx for (mi, mx) in match_days])

    num_matches_left_max = int(num_matches - max_initial)
    num_full_weeks = num_matches_left_max//max_matches_per_week
    print("Number full weeks: {}".format(num_full_weeks))

    for i in range(num_full_weeks):
        match_days += DEFAULT_DAILY_MATCHES

    tmp = sum([mx for (mi, mx) in match_days])
    rm_matches_max = num_matches - tmp

    tmp = sum([mi for (mi, mx) in match_days])
    rm_matches_min = num_matches - tmp

    print("# of remaining matches (min): {}".format(rm_matches_min))
    print("# of remaining matches (max): {}".format(rm_matches_max))

    while(rm_matches_min > 0):
        for day in DEFAULT_DAILY_MATCHES:
            if rm_matches_min < 1:
                break
            if rm_matches_max > 0:
                mtchs = min(rm_matches_min, day[0])
                match_days += [(mtchs, day[1])]
            else:
                mtchs = min(rm_matches_min, day[1])
                match_days += [(0, mtchs)]
            rm_matches_min -= mtchs
            rm_matches_max -= mtchs

    tmp = sum([mx for (mi, mx) in match_days])
    print("# of matches(max): {}".format(tmp))

    tmp = sum([mi for (mimn, mxmn) in match_days])
    print("# of matches(min): {}".format(tmp))

    return match_days


def model_matches():

    model = cp_model.CpModel()

    num_teams = 14
    num_matches = (num_teams*(num_teams-1))//2
    print("# of half season matches: {}".format(num_matches))

    daily_matches_weekly = [
        (0, 0),  # Monday
        (0, 0),  # Tuesday
        (0, 0),  # Wednesday
        (2, 2),  # Thursday
        (2, 2),  # Friday
        (2, 2),  # Saturday
        (2, 2),  # Sunday
    ]

    teams = range(num_teams)
    daily_matches = []

    for i in range(11):
        daily_matches += daily_matches_weekly

    daily_matches += daily_matches_weekly[:4]
    daily_matches += [(1, 1)]

    print(sum([i for (i, j) in daily_matches]))

    num_match_days = sum([i > 0 for (i, j) in daily_matches])

    print("# of match days: %i" % (num_match_days))

    match_days = []
    for (daynum, (minM, maxM)) in enumerate(daily_matches):
        if minM > 0:
            match_days += [(minM, maxM, daynum)]

    fixtures = daily_fixtures(
        model, num_teams, num_match_days)

    matchdays = range(num_match_days)
    # forbid playing self
    [model.Add(fixtures[d][i][i] == 0)
     for d in matchdays for i in teams]

    # minimum and maximum number of matches per day
    for (d, (minM, maxM, daynum)) in enumerate(match_days):
        model.Add(sum([fixtures[d][i][j]
                       for i in teams
                       for j in teams
                       if i != j]) >= minM)
        model.Add(sum([fixtures[d][i][j]
                       for i in teams
                       for j in teams
                       if i != j]) <= maxM)

    # For any team no more than one match per day
    # Note: this should be modified to include the rest day criteria
    for d in range(num_match_days-1):
        for i in teams:
            model.Add(sum([fixtures[d][i][j] +
                           fixtures[d][j][i] +
                           fixtures[d+1][i][j] +
                           fixtures[d+1][j][i]
                           for j in teams if i != j]) <= 1)

    # Either home or away for the half season
    for i in teams:
        for j in teams:
            if i < j:
                model.Add(sum([fixtures[d][i][j] + fixtures[d][j][i]
                               for d in matchdays]) == 1)

    return (model, fixtures)


def solve_model(model,
                time_limit=None,
                num_cpus=None,
                debug=False):
    # run the solver
    solver = cp_model.CpSolver()
    # solver.parameters.max_time_in_seconds = time_limit
    # solver.parameters.log_search_progress = debug
    solver.parameters.num_search_workers = num_cpus

    # solution_printer = SolutionPrinter() # since we stop at first
    # solution, this isn't really
    # necessary I think
    status = solver.Solve(model)
    print('Solve status: %s' % solver.StatusName(status))
    print('Statistics')
    print('  - conflicts : %i' % solver.NumConflicts())
    print('  - branches  : %i' % solver.NumBranches())
    print('  - wall time : %f s' % solver.WallTime())
    return (solver, status)


def main():
    """Entry point of the program."""
    parser = argparse.ArgumentParser(
        description='Solve sports league match play assignment problem')

    parser.add_argument('--csv', type=str, dest='csv', default='output.csv',
                        help='A file to dump the team assignments.  Default is output.csv')

    parser.add_argument('--timelimit', type=int, dest='time_limit', default=300,
                        help='Maximum run time for solver, in seconds.  Default is 300 seconds.')

    parser.add_argument('--cpu', type=int, dest='cpu',
                        help='Number of workers (CPUs) to use for solver.  Default is 6 or number of CPUs available, whichever is lower')

    parser.add_argument('--debug', action='store_true',
                        help="Turn on some print statements.")

    args = parser.parse_args()

    cpu = cpu_guess_and_gripe(args.cpu)

    # set up the model
    (model, fixtures) = model_matches()
    (solver, status) = solve_model(model, args.time_limit, cpu, args.debug)
    report_results(solver, status, fixtures, args.time_limit, args.csv)


if __name__ == '__main__':
    date = datetime.
    mi = 0
    mx = 0
    initial = [(2, 2), (4, 4)]
    daily_matches = set_matchdays(66, initial=initial)
    for day in daily_matches:
        print(date.strftime("%d/%m")+" " +
          calendar.day_abbr[date.weekday()]+" - ", day, mi, mx)
    date += datetime.timedelta(days=1)
