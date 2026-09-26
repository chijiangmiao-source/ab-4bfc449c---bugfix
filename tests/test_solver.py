"""Unit tests for the exact rational reachability solver."""

import unittest
from fractions import Fraction as F

from app.solver import _simplex_min, non_terminating_states, solve_reachability


def model(states, start, rescued, lost, actions):
    return solve_reachability(states, start, rescued, lost, actions)


class SimplexTest(unittest.TestCase):
    def test_trivial_constraint(self):
        # min x s.t. x >= 1  ->  x = 1
        self.assertEqual(_simplex_min([F(1)], [[F(1)]], [F(1)]), [F(1)])

    def test_componentwise_minimum(self):
        # min x + y s.t. x - y >= 0, y >= 1  ->  (1, 1)
        x = _simplex_min([F(1), F(1)],
                         [[F(1), F(-1)], [F(0), F(1)]],
                         [F(0), F(1)])
        self.assertEqual(x, [F(1), F(1)])

    def test_zero_solution(self):
        # min x + y s.t. x - y >= 0, y - x >= 0  ->  (0, 0)
        x = _simplex_min([F(1), F(1)],
                         [[F(1), F(-1)], [F(-1), F(1)]],
                         [F(0), F(0)])
        self.assertEqual(x, [F(0), F(0)])

    def test_no_constraints(self):
        self.assertEqual(_simplex_min([F(1), F(1)], [], []), [F(0), F(0)])


class MutualFeedbackTest(unittest.TestCase):
    """Deep-sea buoy relays with mutual feedback: s0 commits into relayA,
    relayA stays or hands over to relayB with probability 1/2 each, relayB
    falls back to relayA or completes the rescue with probability 1/2 each.
    The model keeps earning another attempt, so the exact maximum rescue
    probability of every controllable state is 1 -- never 1/2."""

    ACTIONS = {
        "s0": {"commit": {"relayA": F(1)}},
        "relayA": {"retry": {"relayA": F(1, 2), "relayB": F(1, 2)}},
        "relayB": {"retry": {"relayA": F(1, 2), "rescued": F(1, 2)}},
    }

    def result_for(self, states, actions=None):
        return model(states=states, start="s0", rescued=["rescued"],
                     lost=["lost"], actions=actions or self.ACTIONS)

    def test_values_are_exactly_one(self):
        result = self.result_for(["s0", "relayA", "relayB", "rescued", "lost"])
        self.assertEqual(result["maxRescueProbability"], F(1))
        for s in ("s0", "relayA", "relayB", "rescued"):
            self.assertEqual(result["stateValues"][s], F(1))
        self.assertEqual(result["stateValues"]["lost"], F(0))

    def test_certificates_match_state_values(self):
        result = self.result_for(["s0", "relayA", "relayB", "rescued", "lost"])
        certs = {c["state"]: c for c in result["certificates"]}
        self.assertEqual(set(certs), {"s0", "relayA", "relayB"})
        for s, expected_action in (("s0", "commit"), ("relayA", "retry"),
                                   ("relayB", "retry")):
            cert = certs[s]
            self.assertEqual(cert["value"], F(1))
            self.assertEqual(cert["selectedAction"], expected_action)
            (entry,) = cert["actions"]
            self.assertEqual(entry["action"], expected_action)
            self.assertEqual(entry["expectedValue"], F(1))
            self.assertTrue(entry["optimal"])
            self.assertTrue(entry["selected"])

    def test_relays_still_terminating(self):
        # Every relay can still reach the rescued terminal, so none of them
        # may be reported as non-terminating.
        result = self.result_for(["s0", "relayA", "relayB", "rescued", "lost"])
        self.assertEqual(result["nonTerminatingStates"], [])

    def test_declaration_order_irrelevant(self):
        reference = self.result_for(["s0", "relayA", "relayB",
                                     "rescued", "lost"])
        shuffled_actions = {
            "relayB": {"retry": {"rescued": F(1, 2), "relayA": F(1, 2)}},
            "s0": {"commit": {"relayA": F(1)}},
            "relayA": {"retry": {"relayB": F(1, 2), "relayA": F(1, 2)}},
        }
        variants = [
            (["lost", "rescued", "relayB", "relayA", "s0"], None),
            (["relayB", "s0", "lost", "relayA", "rescued"], None),
            (["s0", "relayA", "relayB", "rescued", "lost"], shuffled_actions),
        ]
        ref_certs = {c["state"]: c for c in reference["certificates"]}
        for states, actions in variants:
            result = self.result_for(states, actions)
            self.assertEqual(result["stateValues"], reference["stateValues"])
            self.assertEqual(result["maxRescueProbability"],
                             reference["maxRescueProbability"])
            self.assertEqual(result["optimalActions"],
                             reference["optimalActions"])
            self.assertEqual(result["nonTerminatingStates"],
                             reference["nonTerminatingStates"])
            got_certs = {c["state"]: c for c in result["certificates"]}
            self.assertEqual(got_certs, ref_certs)


class CombinedModelTest(unittest.TestCase):
    """One model mixing mutual feedback, a direct self-loop, a lost closed
    loop and fractional transitions: exact fractions, canonical actions and
    recomputable certificates must all hold at once."""

    def setUp(self):
        self.result = model(
            states=["s0", "relayA", "relayB", "spin", "trap1", "trap2",
                    "gamble", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                # mutual feedback between the two relays
                "s0": {"commit": {"relayA": F(1)}},
                "relayA": {"retry": {"relayA": F(1, 2), "relayB": F(1, 2)}},
                "relayB": {"retry": {"relayA": F(1, 2), "rescued": F(1, 2)}},
                # direct self-loop with a fractional exit
                "spin": {"wait": {"spin": F(2, 3), "rescued": F(1, 3)}},
                # closed loop with no exit to any terminal
                "trap1": {"drift": {"trap2": F(1)}},
                "trap2": {"drift": {"trap1": F(1)}},
                # fractional choice with a dominated action
                "gamble": {
                    "bold": {"rescued": F(2, 3), "lost": F(1, 3)},
                    "meek": {"rescued": F(1, 3), "lost": F(2, 3)},
                },
            },
        )

    def test_exact_values(self):
        values = self.result["stateValues"]
        self.assertEqual(values["s0"], F(1))
        self.assertEqual(values["relayA"], F(1))
        self.assertEqual(values["relayB"], F(1))
        self.assertEqual(values["spin"], F(1))   # retries until the 1/3 exit
        self.assertEqual(values["trap1"], F(0))
        self.assertEqual(values["trap2"], F(0))
        self.assertEqual(values["gamble"], F(2, 3))
        self.assertEqual(self.result["maxRescueProbability"], F(1))

    def test_canonical_actions(self):
        self.assertEqual(self.result["optimalActions"]["gamble"], "bold")
        self.assertEqual(self.result["optimalActions"]["spin"], "wait")

    def test_certificates_recomputable(self):
        certs = {c["state"]: c for c in self.result["certificates"]}
        gamble = {a["action"]: a for a in certs["gamble"]["actions"]}
        self.assertEqual(gamble["bold"]["expectedValue"], F(2, 3))
        self.assertTrue(gamble["bold"]["optimal"])
        self.assertEqual(gamble["meek"]["expectedValue"], F(1, 3))
        self.assertFalse(gamble["meek"]["optimal"])
        spin = {a["action"]: a for a in certs["spin"]["actions"]}
        self.assertEqual(spin["wait"]["expectedValue"], F(1))

    def test_only_pure_closed_loop_nonterminating(self):
        self.assertEqual(self.result["nonTerminatingStates"],
                         ["trap1", "trap2"])


class CertainRescueTest(unittest.TestCase):
    """A model where rescue is certain: exact value 1 via 1/2 + 1/2."""

    def setUp(self):
        self.result = model(
            states=["s0", "s1", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {"push": {"s1": F(1, 2), "rescued": F(1, 2)}},
                "s1": {"push": {"rescued": F(1)}},
            },
        )

    def test_exact_probability_one(self):
        self.assertEqual(self.result["maxRescueProbability"], F(1))
        self.assertEqual(self.result["stateValues"]["s0"], F(1))
        self.assertEqual(self.result["stateValues"]["s1"], F(1))
        self.assertEqual(self.result["stateValues"]["rescued"], F(1))
        self.assertEqual(self.result["stateValues"]["lost"], F(0))

    def test_optimal_actions(self):
        self.assertEqual(self.result["optimalActions"],
                         {"s0": "push", "s1": "push"})

    def test_no_nonterminating_states(self):
        self.assertEqual(self.result["nonTerminatingStates"], [])


class ThirdsTest(unittest.TestCase):
    """1/3 and 2/3 transitions must yield exact thirds in the values."""

    def setUp(self):
        self.result = model(
            states=["s0", "s1", "s2", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {"commit": {"s1": F(2, 3), "lost": F(1, 3)}},
                "s1": {"retry": {"rescued": F(1, 2), "s1": F(1, 2)}},
                "s2": {"call": {"rescued": F(1, 3), "lost": F(2, 3)}},
            },
        )

    def test_exact_fractions(self):
        values = self.result["stateValues"]
        self.assertEqual(values["s1"], F(1))          # 1/2 + 1/2*v  =>  v = 1
        self.assertEqual(values["s0"], F(2, 3))       # 2/3 * 1
        self.assertEqual(values["s2"], F(1, 3))
        self.assertEqual(self.result["maxRescueProbability"], F(2, 3))

    def test_certificate_expected_values_exact(self):
        cert = {c["state"]: c for c in self.result["certificates"]}
        s2_actions = {a["action"]: a for a in cert["s2"]["actions"]}
        self.assertEqual(s2_actions["call"]["expectedValue"], F(1, 3))


class TieActionTest(unittest.TestCase):
    """Equal-value actions: canonical choice is the smallest identifier and
    the certificate shows exactly equal expected values."""

    def setUp(self):
        self.result = model(
            states=["s0", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {
                    "zulu": {"rescued": F(1, 2), "lost": F(1, 2)},
                    "alpha": {"rescued": F(1, 2), "lost": F(1, 2)},
                    "mike": {"rescued": F(1, 4), "lost": F(3, 4)},
                },
            },
        )

    def test_canonical_lexicographic_choice(self):
        self.assertEqual(self.result["optimalActions"]["s0"], "alpha")

    def test_certificate(self):
        (cert,) = self.result["certificates"]
        self.assertEqual(cert["state"], "s0")
        self.assertEqual(cert["value"], F(1, 2))
        self.assertEqual(cert["selectedAction"], "alpha")
        entries = {a["action"]: a for a in cert["actions"]}
        # equal expected values for the tied optimal actions
        self.assertEqual(entries["alpha"]["expectedValue"], F(1, 2))
        self.assertEqual(entries["zulu"]["expectedValue"], F(1, 2))
        self.assertTrue(entries["alpha"]["optimal"])
        self.assertTrue(entries["zulu"]["optimal"])
        self.assertTrue(entries["alpha"]["selected"])
        self.assertFalse(entries["zulu"]["selected"])
        # dominated action excluded with its (smaller) expectation visible
        self.assertEqual(entries["mike"]["expectedValue"], F(1, 4))
        self.assertFalse(entries["mike"]["optimal"])
        self.assertFalse(entries["mike"]["selected"])


class ClosedLoopTest(unittest.TestCase):
    """A closed loop that can never reach any terminal, plus a start state
    that is forced into it: zero rescue probability everywhere."""

    def setUp(self):
        self.result = model(
            states=["s0", "loop1", "loop2", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {"dive": {"loop1": F(1)}},
                "loop1": {"drift": {"loop2": F(1)}},
                "loop2": {"drift": {"loop1": F(1)}},
            },
        )

    def test_zero_rescue_probability(self):
        self.assertEqual(self.result["maxRescueProbability"], F(0))
        for s in ("s0", "loop1", "loop2"):
            self.assertEqual(self.result["stateValues"][s], F(0))

    def test_closed_loop_identified(self):
        self.assertEqual(self.result["nonTerminatingStates"],
                         ["loop1", "loop2", "s0"])

    def test_canonical_actions_still_reported(self):
        self.assertEqual(self.result["optimalActions"],
                         {"s0": "dive", "loop1": "drift", "loop2": "drift"})


class LoopWithExitTest(unittest.TestCase):
    """A loop with an exit to the lost terminal is NOT non-terminating, but
    its rescue probability is still exactly zero."""

    def test(self):
        result = model(
            states=["s0", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {
                    "spin": {"s0": F(1)},
                    "sink": {"lost": F(1)},
                },
            },
        )
        self.assertEqual(result["maxRescueProbability"], F(0))
        self.assertEqual(result["nonTerminatingStates"], [])
        # both actions are optimal (expected 0 == value 0); smallest id wins
        self.assertEqual(result["optimalActions"]["s0"], "sink")


class ChoiceTest(unittest.TestCase):
    """The solver must pick the action with the strictly larger exact
    expectation, including through multi-step paths."""

    def test_better_action_wins(self):
        result = model(
            states=["s0", "s1", "rescued", "lost"],
            start="s0",
            rescued=["rescued"],
            lost=["lost"],
            actions={
                "s0": {
                    "direct": {"rescued": F(1, 3), "lost": F(2, 3)},
                    "via": {"s1": F(1)},
                },
                "s1": {"push": {"rescued": F(3, 4), "lost": F(1, 4)}},
            },
        )
        self.assertEqual(result["maxRescueProbability"], F(3, 4))
        self.assertEqual(result["optimalActions"]["s0"], "via")
        cert = {c["state"]: c for c in result["certificates"]}["s0"]
        entries = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(entries["direct"]["expectedValue"], F(1, 3))
        self.assertFalse(entries["direct"]["optimal"])
        self.assertEqual(entries["via"]["expectedValue"], F(3, 4))
        self.assertTrue(entries["via"]["optimal"])

    def test_start_on_terminal(self):
        result = model(
            states=["s0", "rescued"],
            start="rescued",
            rescued=["rescued"],
            lost=[],
            actions={"s0": {"go": {"rescued": F(1)}}},
        )
        self.assertEqual(result["maxRescueProbability"], F(1))


class NonTerminatingHelperTest(unittest.TestCase):
    def test_pure_graph(self):
        self.assertEqual(
            non_terminating_states(
                ["a", "b", "t"],
                {"t"},
                {"a": {"x": {"b": F(1)}}, "b": {"x": {"a": F(1)}}},
            ),
            ["a", "b"],
        )


if __name__ == "__main__":
    unittest.main()
