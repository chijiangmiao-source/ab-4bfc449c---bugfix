"""Exact maximum-reachability solver for finite controllable models (MDPs)
over rational transition probabilities.

The maximal rescue (reachability) probability vector of a finite MDP is the
least fixed point of the Bellman optimality operator

    (T v)(s) = max_a  sum_{s'} p(s, a, s') * v(s')

with v(rescued) = 1 and v >= 0.  That least fixed point is the unique
componentwise-minimal point of the feasible polytope

    v(s) >= sum_{s'} p(s, a, s') * v(s')   for every controllable s, action a
    v(r) >= 1                              for every rescued terminal r
    v(s) >= 0                              for every state s

so it is obtained *exactly* by solving the linear program

    minimise  sum_s v(s)   subject to the constraints above.

This module solves that LP with a two-phase simplex method carried out
entirely in ``fractions.Fraction`` arithmetic (Bland's pivoting rule, so
termination is guaranteed).  No floating point, no random simulation, no
enumeration of policies and no truncated value iteration is involved: the
simplex method is an exact algebraic algorithm that ends after finitely many
rational pivots with the provably optimal solution.

States that can never reach any terminal (rescued or lost) -- e.g. closed
loops with no exit -- are additionally identified by an exact graph
reachability pass over the support of the transition distributions.
"""

from fractions import Fraction


class SolverError(Exception):
    """Raised on internal solver inconsistencies (infeasible/unbounded LP).

    The reachability LP is always feasible (v == 1 everywhere satisfies every
    constraint) and bounded below by 0, so this signals a bug, not bad input.
    """


def _simplex_min(c, A, b):
    """Solve ``min c.x`` subject to ``A x >= b`` and ``x >= 0`` exactly.

    ``c``, ``A`` and ``b`` must contain :class:`fractions.Fraction` values.
    Returns the optimal solution vector as a list of Fractions.
    """
    n = len(c)
    m = len(A)
    if m == 0:
        return [Fraction(0)] * n

    A = [[Fraction(x) for x in row] for row in A]
    b = [Fraction(x) for x in b]
    for i in range(m):
        if b[i] < 0:
            A[i] = [-x for x in A[i]]
            b[i] = -b[i]

    # Equality form:  A x - slack = b,  plus one artificial variable per row
    # to bootstrap phase 1.  Column layout:
    #   [0, n)             structural variables x
    #   [n, n + m)         slack variables
    #   [n + m, n + 2m)    artificial variables
    width = n + 2 * m
    rhs = width  # index of the right-hand-side column
    tableau = []
    for i in range(m):
        row = [Fraction(0)] * (width + 1)
        for j in range(n):
            row[j] = A[i][j]
        row[n + i] = Fraction(-1)
        row[n + m + i] = Fraction(1)
        row[rhs] = b[i]
        tableau.append(row)
    basis = [n + m + i for i in range(m)]
    artificial_from = n + m

    def pivot(row_ix, col):
        prow = tableau[row_ix]
        inv = Fraction(1) / prow[col]
        tableau[row_ix] = [x * inv for x in prow]
        for i in range(len(tableau)):
            if i != row_ix:
                factor = tableau[i][col]
                if factor:
                    tableau[i] = [x - factor * y
                                  for x, y in zip(tableau[i], tableau[row_ix])]
        basis[row_ix] = col

    def optimize(costs, allowed):
        # Bland's rule on both entering and leaving variables: terminates.
        while True:
            rows = len(tableau)
            cost_b = [costs[basis[i]] for i in range(rows)]
            enter = -1
            for j in range(width):
                if j not in allowed:
                    continue
                # reduced cost r_j = c_B . T[:, j] - c_j ; r_j > 0 improves
                acc = -costs[j]
                for i in range(rows):
                    if cost_b[i]:
                        acc += cost_b[i] * tableau[i][j]
                if acc > 0:
                    enter = j
                    break
            if enter < 0:
                return
            leave = -1
            best_ratio = None
            for i in range(rows):
                coeff = tableau[i][enter]
                if coeff > 0:
                    ratio = tableau[i][rhs] / coeff
                    if (best_ratio is None or ratio < best_ratio
                            or (ratio == best_ratio and basis[i] < basis[leave])):
                        best_ratio = ratio
                        leave = i
            if leave < 0:
                raise SolverError("reachability LP is unbounded")
            pivot(leave, enter)

    # ---- Phase 1: minimise the sum of artificial variables ----
    cost1 = [Fraction(0)] * width
    for j in range(artificial_from, width):
        cost1[j] = Fraction(1)
    optimize(cost1, set(range(width)))
    phase1_obj = sum(cost1[basis[i]] * tableau[i][rhs]
                     for i in range(len(tableau)))
    if phase1_obj != 0:
        raise SolverError("reachability LP is infeasible")

    # Drive artificial variables out of the basis; drop redundant rows.
    i = 0
    while i < len(tableau):
        if basis[i] >= artificial_from:
            col = -1
            for j in range(artificial_from):
                if tableau[i][j] != 0:
                    col = j
                    break
            if col >= 0:
                pivot(i, col)
                i += 1
            else:
                del tableau[i]
                del basis[i]
        else:
            i += 1

    # ---- Phase 2: minimise the true objective; artificials never re-enter ----
    cost2 = [Fraction(0)] * width
    for j in range(n):
        cost2[j] = Fraction(c[j])
    optimize(cost2, set(range(artificial_from)))

    x = [Fraction(0)] * n
    for i in range(len(tableau)):
        if basis[i] < n:
            x[basis[i]] = tableau[i][rhs]
    return x


def non_terminating_states(states, terminals, actions):
    """Return the sorted list of states that can never reach ANY terminal.

    A state can reach a terminal if a path exists in the support graph, whose
    edges are every transition with a strictly positive probability under some
    action.  States that cannot are trapped forever in non-terminal closed
    loops (they are rescued with probability 0 and lost with probability 0).
    """
    terminal_set = set(terminals)
    successors = {s: set() for s in states}
    for state, state_actions in actions.items():
        for dist in state_actions.values():
            for target, prob in dist.items():
                if prob > 0:
                    successors[state].add(target)
    can_reach = set(terminal_set)
    changed = True
    while changed:
        changed = False
        for s in states:
            if s not in can_reach and successors.get(s) & can_reach:
                can_reach.add(s)
                changed = True
    return sorted(s for s in states if s not in can_reach)


def solve_reachability(states, start, rescued, lost, actions):
    """Solve one audit model exactly.

    ``states``   ordered list of unique state identifiers (validated already)
    ``start``    the starting state identifier
    ``rescued``  collection of rescued terminal identifiers (value 1)
    ``lost``     collection of lost terminal identifiers (value 0)
    ``actions``  ``{state: {action: {target: Fraction}}}`` for controllable
                 states only

    All non-terminal state values are solved *together* in one global LP, so
    mutual feedback (A -> B -> A), direct self loops (A -> A) and lost
    closed loops are handled exactly and the result is independent of the
    order in which states, actions or transition entries were recorded.

    Returns a dict of exact results (Fractions, not strings):
      ``maxRescueProbability``  value of the start state
      ``stateValues``           {state: Fraction} for every state
      ``optimalActions``        {state: canonical action} per controllable state
      ``certificates``          per-state itemised certificates
      ``nonTerminatingStates``  sorted list of states that never reach a terminal
    """
    rescued_set = set(rescued)
    lost_set = set(lost)
    terminals = rescued_set | lost_set
    controllable = [s for s in states if s not in terminals]
    index = {s: i for i, s in enumerate(controllable)}
    n = len(controllable)

    # Global least-fixed-point LP over every controllable state at once:
    #   minimise sum_s v(s)
    #   s.t.  v(s) >= sum_{s'} p(s,a,s') v(s')   per controllable s, action a
    #         v(s) >= 0
    # Terminal successors enter the right-hand side: rescued -> 1, lost -> 0.
    rows = []
    rhs = []
    for s in controllable:
        for action_id in sorted(actions[s]):
            row = [Fraction(0)] * n
            required = Fraction(0)
            row[index[s]] = Fraction(1)
            for target, probability in actions[s][action_id].items():
                if target in rescued_set:
                    required += probability
                elif target in lost_set:
                    pass
                else:
                    row[index[target]] -= probability
            rows.append(row)
            rhs.append(required)

    solution = _simplex_min([Fraction(1)] * n, rows, rhs)
    values = {s: Fraction(1) for s in rescued}
    values.update({s: Fraction(0) for s in lost})
    for s, x in zip(controllable, solution):
        values[s] = x

    certificates = []
    optimal_actions = {}
    for s in controllable:
        entries = []
        for action_id in sorted(actions[s]):
            expected = Fraction(0)
            for target, probability in actions[s][action_id].items():
                expected += probability * values[target]
            entries.append({
                "action": action_id,
                "expectedValue": expected,
                "optimal": expected == values[s],
            })
        # Canonical choice: among ALL optimal actions, the smallest identifier.
        chosen = min(e["action"] for e in entries if e["optimal"])
        for e in entries:
            e["selected"] = e["action"] == chosen
        optimal_actions[s] = chosen
        certificates.append({
            "state": s,
            "value": values[s],
            "selectedAction": chosen,
            "actions": entries,
        })

    return {
        "start": start,
        "maxRescueProbability": values[start],
        "stateValues": values,
        "optimalActions": optimal_actions,
        "certificates": certificates,
        "nonTerminatingStates": non_terminating_states(states, terminals, actions),
    }
