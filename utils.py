
import argparse
import os
import re
import csv
from functools import partial
from functools import reduce

from ortools.sat.python import cp_model


def opponent_fixtures(model, num_teams, day, home_team):
    name_prefix = 'fixture: day %i, home %i, ' % (day, home_team)
    return [model.NewBoolVar(name_prefix+'away %i' % away) for away in range(num_teams)]


def home_fixtures(model, num_teams, day):
    opp_fix = partial(opponent_fixtures, model=model,
                      num_teams=num_teams, day=day)
    result = list(map(lambda x: opp_fix(home_team=x), list(range(num_teams))))
    return result


def daily_thing(fn, model, num_teams, num_days):
    fixed = partial(fn, model=model, num_teams=num_teams)
    result = list(map(lambda x: fixed(day=x), list(range(num_days))))
    return result


def daily_fixtures(model, num_teams, num_days):
    return daily_thing(home_fixtures,
                       model=model, num_teams=num_teams, num_days=num_days)


def create_at_home_array(model, num_teams, day):
    name_prefix = 'at_home: day %i, ' % day
    return [model.NewBoolVar(name_prefix+'home %i' % home) for home in range(num_teams)]


def daily_at_home(model, num_teams, num_days):
    return daily_thing(create_at_home_array,
                       model=model, num_teams=num_teams, num_days=num_days)


def screen_dump_results(scheduled_games):
    for row in scheduled_games:
        [print('%s=%i,' % (k, v), end=' ') for (k, v) in row.items()]
        print()


def get_scheduled_fixtures(solver, fixtures):
    fixed_matches = [{'day': day+1, 'home': home+1, 'away': away+1, }
                     for (day, fd) in enumerate(fixtures)
                     for (home, fh) in enumerate(fd)
                     for (away, fixture) in enumerate(fh)
                     if solver.Value(fixture)]
    return list(fixed_matches)


def report_results(solver, status, fixtures, time_limit=None, csv=None):

    if status == cp_model.INFEASIBLE:
        print('INFEASIBLE')
        return status

    if status == cp_model.UNKNOWN:
        print('Not enough time allowed to compute a solution')
        print('Add more time using the --timelimit command line option')
        return status

    print('Optimal objective value: %i' % solver.ObjectiveValue())

    scheduled_games = get_scheduled_fixtures(solver, fixtures)

    screen_dump_results(scheduled_games)

    if status != cp_model.OPTIMAL and solver.WallTime() >= time_limit:
        print('Please note that solver reached maximum time allowed %i.' % time_limit)
        print('A better solution than %i might be found by adding more time using the --timelimit command line option' %
              solver.ObjectiveValue())


def cpu_guess_and_gripe(cpu):

    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = 1

    if ncpu == 1:
        try:
            ncpu = os.cpu_count()
        except AttributeError:
            ncpu = 1

    print('# of cpus available: {}'.format(ncpu))

    if not cpu:
        cpu = min(6, ncpu)
    print('Setting number of search workers to %i' % cpu)

    if cpu > ncpu:
        print('You asked for %i workers to be used, but the os only reports %i CPUs available.  This might slow down processing' % (cpu, ncpu))

    if cpu != 6:
        if cpu < ncpu:
            print('Using %i workers, but there are %i CPUs available.  You might get faster results by using the command line option --cpu %i, but be aware ORTools CP-SAT solver is tuned to 6 CPUs' % (cpu, ncpu, ncpu))

        if cpu > 6:
            print(
                'Using %i workers.  Be aware ORTools CP-SAT solver is tuned to 6 CPUs' % cpu)

    return cpu


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
    main()
