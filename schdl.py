import argparse
import os
import re
import csv
import datetime
import calendar
import itertools


from functools import partial
from functools import reduce

from ortools.sat.python import cp_model

from utils import *

DEFAULT_DAILY_MATCHES = [
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (2, 2, 2),
    (2, 2, 2),
    (2, 4, 2),
]

TEAMS = ("YFC", "BSK", "KBFC", "SRV", "BTZ", "RFC",
         "RGM", "BFC", "TTM", "TMFC", "FCC", "SAR")
GROUND_ID = (0,     0,      1,     1,     2,     2,
             3,     3,     4,      4,     5,     6)


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


def model_matches(num_teams=12, num_grounds=7, initial=[]):

    model = cp_model.CpModel()
    initial = [(2, 2, 1), (4, 4, 2)]
    num_matches = (num_teams * (num_teams - 1)) // 2

    grounds = range(num_grounds)

    print("# of half season matches: {}".format(num_matches))

    teams = range(num_teams)
    daily_matches = set_matchdays(num_matches, initial=initial)
    print(sum([i for (i, j) in daily_matches]))

    match_days = []
    for (daynum, (mi, mx, numgrnd)) in enumerate(daily_matches):
        if mi+mx > 0:
            match_days += [(mi, mx, numgrnd, daynum)]

    num_match_days = len(match_days)
    matchdays = range(num_match_days)

    print("# of match days: %i" % (num_match_days))

    consec_days = set_consecutive_days(match_days, nc=1)

    fixtures = daily_fixtures(model, num_teams, num_match_days)

    ground_fixture = []
    for day in matchdays:
        name_prefix = 'day %i, ' % (day)
        ground_fixture.append(
            [model.NewBoolVar(name_prefix+'ground %i' % ground) for ground in grounds])

    # Link match fixtures to ground fixtures
    for d in matchdays:
        for i in teams:
            for j in teams:
                if GROUND_ID[i] != GROUND_ID[j]:
                    model.AddImplication(
                        fixtures[d][i][j], ground_fixture[d][GROUND_ID[i]])
                    model.AddImplication(
                        fixtures[d][i][j], ground_fixture[d][GROUND_ID[j]].Not())
                elif i != j:
                    model.AddImplication(
                        fixtures[d][i][j], ground_fixture[d][GROUND_ID[i]])

    # forbid playing self
    [model.Add(fixtures[d][i][i] == 0) for d in matchdays for i in teams]

    # minimum and maximum number of matches per day
    # maximum grounds where matches are happening in day
    for (d, (minM, maxM, maxG, daynum)) in enumerate(match_days):
        model.Add(sum([fixtures[d][i][j]
                       for i in teams for j in teams if i != j]) >= minM)
        model.Add(sum([fixtures[d][i][j]
                       for i in teams for j in teams if i != j]) <= maxM)
        model.Add(sum([ground_fixture[d][i] for i in grounds]) <= maxG)

    # For any team no more than one match per # consecutive days
    for batch in consec_days:
        for i in teams:
            model.Add(sum([fixtures[d][i][j] + fixtures[d][j][i]
                           for j in teams if i != j for d in batch]) <= 1)

    # Either home or away for the half season
    # Here more constrains will come as the first leg matches will be over
    for i in teams:
        for j in teams:
            if i < j:
                model.Add(sum([fixtures[d][i][j] + fixtures[d][j][i]
                               for d in matchdays]) == 1)

    return (model, fixtures)


def solve_model(model, time_limit=None, num_cpus=None, debug=False):
    # run the solver
    solver = cp_model.CpSolver()
    # solver.parameters.max_time_in_seconds = time_limit
    # solver.parameters.log_search_progress = debug
    solver.parameters.num_search_workers = num_cpus

    # solution_printer = SolutionPrinter() # since we stop at first
    # solution, this isn't really
    # necessary I think
    status = solver.Solve(model)
    print("Solve status: %s" % solver.StatusName(status))
    print("Statistics")
    print("  - conflicts : %i" % solver.NumConflicts())
    print("  - branches  : %i" % solver.NumBranches())
    print("  - wall time : %f s" % solver.WallTime())
    return (solver, status)


def main():
    """Entry point of the program."""
    parser = argparse.ArgumentParser(
        description="Solve sports league match play assignment problem"
    )

    parser.add_argument(
        "--csv",
        type=str,
        dest="csv",
        default="output.csv",
        help="A file to dump the team assignments.  Default is output.csv",
    )

    parser.add_argument(
        "--timelimit",
        type=int,
        dest="time_limit",
        default=300,
        help="Maximum run time for solver, in seconds.  Default is 300 seconds.",
    )

    parser.add_argument(
        "--cpu",
        type=int,
        dest="cpu",
        help="Number of workers (CPUs) to use for solver.  Default is 6 or number of CPUs available, whichever is lower",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Turn on some print statements."
    )

    args = parser.parse_args()

    cpu = cpu_guess_and_gripe(args.cpu)

    # set up the model
    (model, fixtures) = model_matches()
    (solver, status) = solve_model(model, args.time_limit, cpu, args.debug)
    report_results(solver, status, fixtures, args.time_limit, args.csv)


if __name__ == "__main__":
    date = datetime.datetime.strptime("30/01/2021", "%d/%m/%Y")
    main()
