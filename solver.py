import sys
from itertools import product, filterfalse, zip_longest
from functools import reduce

DIGITS = '123456789'
ROWS = 'ABCDEFGHI'
COLS = DIGITS


def cross(a: str, b:str) -> list:
    return [i + j for i in a for j in b]


def read(fname: str) -> list:
    with open(fname, 'r') as f:
        line = f.read().replace('\n', '').split()
        constraints = [int(i) for i in line]
    return constraints


def parse_constraints(con_list: list) -> None:
    """Parse the input file into constraints as a list of list of numbers. """
    length = len(con_list)
    assert length % 4 == 0, "Input length must be a multiple of 4."

    global size
    size = length // 4
    assert all(i in range(1, size + 1) for i in con_list), \
        "Unexpected constraint value detected."
    
    global constraints
    constraints = [con_list[i:i+size] for i in range(0, length, size)]


def generate_squares() -> list:
    """Create a global a list of squares in the game."""
    global squares
    squares = cross(ROWS[:size], COLS[:size])


def generate_unitlist() -> list:
    """Create a global list of units in the game."""
    rows = ROWS[:size]
    cols = COLS[:size]
    global unitlist
    unitlist = ([cross(rows, c) for c in cols]  # group by column
                + [cross(r, cols) for r in rows]    # group by rows
                )


def generate_units() -> dict:
    """Create a global dictionary of {squares: unitlist}"""
    global units
    units = dict((s, [u for u in unitlist if s in u]) for s in squares)


def generate_peers() -> None:
    """Create a global dictionary of {squares: peers}."""
    global peers
    peers = dict((s, set(sum(units[s], [])) - set([s])) for s in squares)


def generate_candidates() -> dict:
    """Generate an all candidates for all squares."""
    digits = DIGITS[:size]
    return dict((s, digits) for s in squares)


def assign(values: dict, s: str, d: str) -> dict:
    """Eliminate all the other values (except d) from values[s] and propagate.
    Return values, except return False if a contradiction is detected."""
    other_values = values[s].replace(d, '') # values other than d
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False


def eliminate(values: dict, s: str, d: str) -> dict:
    """Eliminate d from values[s]; propagate when values or places <= 2.
    Return values, except return False if a contradiction is detected."""
    if d not in values[s]:
        return values   # value d is already eliminated
    
    values[s] = values[s].replace(d, '')  # remove value d
    
    if len(values[s]) == 0:
        return False
    
    # if square s reduced to only one value d2
    if len(values[s]) == 1:
        if not check_all(values, s):
            return False
        
        # then eleminate d2 from peers
        d2 = values[s]
        if not all(eliminate(values, s2, d2)
                    for s2 in peers[s]):
            return False
    
    # if unit u is reduced to one place for value d, assign d to that location
    for u in units[s]:
        dplaces = [s2 for s2 in u if d in values[s2]]
        if len(dplaces) == 0:
            return False    # contradiction: no place for value d
        elif len(dplaces) == 1:
            # d can only be in one place in that unit. Assign it to the square
            if not assign(values, dplaces[0], d):
                return False
    
    return values


def apply_edge_clue(values: dict) -> dict:
    """Remove candidates from values based on edge clues (constraints).
    Values from N - c + 2 + d up to N can be removed.
    where N = size of the game
          c = a given clue / constraint
          d = index along the row / column
    """
    if not values:
        return False
    
    N = size
    rows = ROWS[:N]
    cols = COLS[:N]
    
    # apply top constraints
    for i, c in enumerate(constraints[0]):
        for d in range(N):
            for v in range(N - c + 2 + d, N + 1):
                if not eliminate(values, rows[d] + cols[i], str(v)):
                    return False
    
    # apply bottom constraints
    for i, c in enumerate(constraints[1]):
        for d in range(N):
            for v in range(N - c + 2 + d, N + 1):
                if not eliminate(values, rows[-d - 1] + cols[i], str(v)):
                    return False
    
    # apply left constraints
    for i, c in enumerate(constraints[2]):
        for d in range(N):
            for v in range(N - c + 2 + d, N + 1):
                if not eliminate(values, rows[i] + cols[d], str(v)):
                    return False
    
    # apply right constraints
    for i, c in enumerate(constraints[3]):
        for d in range(N):
            for v in range(N - c + 2 + d, N + 1):
                if not eliminate(values, rows[i] + cols[-d - 1], str(v)):
                    return False
    
    return values


def check_all(values: dict, s: str):
    """Check constraints along the 4 directions.
    
    Return True if the square or any of its peers have multiple values.
    Return False if constraint is violated.
    """
    rows = ROWS[:size]
    cols = COLS[:size]
    col_mem = [rows[i] + s[1] for i in range(size)]
    row_mem = [s[0] + cols[i] for i in range(size)]
    
    return all([
        check(values, col_mem, constraints[0][cols.index(s[1])]),
        check(values, col_mem[::-1], constraints[1][cols.index(s[1])]),
        check(values, row_mem, constraints[2][rows.index(s[0])]),
        check(values, row_mem[::-1], constraints[3][rows.index(s[0])])
    ])


def check(values: dict, squares: list, limit: int) -> bool:
    seen = 0
    max_val = 0
    
    for s in squares:
        # If there more than 1 value in the square, the unit is unsolve. Skip
        if len(values[s]) > 1:
            return True
        
        if not values[s]:
            return False
        
        # if see a higher skyscraper, update the seen count and max_value
        val = int(values[s][0])
        if (val > max_val):
            max_val = val
            seen += 1
        
        # if limit is exceeded, return False
        if seen > limit:
            return False
    
    return seen == limit

def check_str(s: str, limit: int) -> bool:
    """Check that a string satisfy a constraint."""
    seen = 0
    max_val = 0
    
    for c in s:
        # if see a higher skyscraper, update the seen count and max_value
        val = int(c)
        if (val > max_val):
            max_val = val
            seen += 1
        
        # if limit is exceeded, return False
        if seen > limit:
            return False
    
    return seen == limit

def check_str_both(s: str, limit1: int, limit2) -> bool:
    return all([check_str(s, limit1), check_str(s[::-1], limit2)])

def solve(values: dict) -> None:
    search(values)

def search(values: dict) -> None:
    "Using depth-first search and propagation, try all possible values."
    
    if values is False:
        return False    # invalid input
    
    squares = values.keys()
    if all(len(values[s]) == 1 for s in squares):
        display(values) # solved and display
        # return False    # discard the current solution and try to find next solution
        return values   # solved
    
    # choose unfilled square s with the fewest possibility
    n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    for d in values[s]:
        v = search(filter_sequence(assign(values.copy(), s, d)))
        if v:
            return v
    return False

def display(values):
    "Display these values as a 2-D grid."
    if values is False:
        raise ValueError("'values' is not a valid input")
    squares = values.keys()
    width = 2 + max(len(values[s]) for s in squares)  # maximum width required
    line = '+'.join(['-' * width ] * size)
    for r in ROWS[:size]:
        for c in COLS[:size]:
            print(values[r+c].center(width), end='')
            if c != str(size):
                print("|", end="")
        print()
        print(line)
    print()


def filter_sequence_one(values: dict, unit: list, limit1: int, limit2: int
        ) -> None:
    # generate all combinations based an a given unit
    combinations = product(*[values[s] for s in unit])
    # only yields those with complete set of characters
    complete = filterfalse(lambda x: len(set(x)) != size, combinations)
    # only yield those that comply
    comply = filterfalse(
        lambda x: not check_str_both(''.join(x), limit1, limit2), complete)
    # print("comply", list(comply))
    candidates = [''.join(sorted(set(l))) for l in zip_longest(*comply)]
    
    for s, c in zip(unit, candidates):
        for d in values[s]:
            if d not in c:
                eliminate(values, s, d)
    
    return values

def filter_sequence(values):
    """Find the unit with lowest number of combinations and apply sequence
    filter to the unit. Loop until there is no change.
    """
    if not values:
        return False
    
    while True:
        try:
            n, u = min((combo(u), i)
                        for i, u in enumerate(unitlist) if combo(u) > 1)
        except ValueError:
            break
        
        i, j = divmod(u, size)
        copy = filter_sequence_one(
            values.copy(), unitlist[u],
            constraints[2*i][j], constraints[2*i + 1][j])
        
        # there has been no update
        if copy == values:
            break
        else:
            values = copy
    
    return values

def test():
    parse_constraints([3, 1, 2, 2, 1, 3, 2, 2, 2, 3, 2, 1, 2, 2, 1, 3])
    assert constraints == [[3, 1, 2, 2], [1, 3, 2, 2], [2, 3, 2, 1], [2, 2, 1, 3]]
    assert size == 4
    
    generate_squares()
    assert len(squares) == 16
    
    generate_unitlist()
    assert len(unitlist) == 8
    
    generate_units()
    assert all(len(units[s]) == 2 for s in squares)
    assert units["C2"] == [['A2', 'B2', 'C2', 'D2'],
                           ['C1', 'C2', 'C3', 'C4']]
    
    generate_peers()
    assert peers['C2'] == set(['A2', 'B2', 'D2', 'C1', 'C3', 'C4'])
    
    values = generate_candidates()
    assert all(len(values[s]) == 4 for s in squares) == True


def combo(unit: list) -> int:
    prod = lambda x, y: x * y
    return reduce(prod, (len(values[s]) for s in unit))


if __name__ == '__main__':
    # test()
    parse_constraints([int(i) for i in sys.argv[1].split(' ')])
    generate_squares()
    generate_unitlist()
    generate_units()
    generate_peers()
    values = generate_candidates()
    values = apply_edge_clue(values)
    values = filter_sequence(values)
    solve(values)
    if not values:
        print("Error: No solution found")