
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


