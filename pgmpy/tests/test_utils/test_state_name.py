import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import Inference, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


class TestStateNameInit(unittest.TestCase):
    def setUp(self):
        self.sn2 = {
            "grade": ["A", "B", "F"],
            "diff": ["high", "low"],
            "intel": ["poor", "good", "very good"],
        }
        self.sn1 = {
            "speed": ["low", "medium", "high"],
            "switch": ["on", "off"],
            "time": ["day", "night"],
        }

        self.sn2_no_names = {"grade": [0, 1, 2], "diff": [0, 1], "intel": [0, 1, 2]}
        self.sn1_no_names = {"speed": [0, 1, 2], "switch": [0, 1], "time": [0, 1]}

        self.phi1 = DiscreteFactor(["speed", "switch", "time"], [3, 2, 2], np.ones(12))
        self.phi2 = DiscreteFactor(
            ["speed", "switch", "time"], [3, 2, 2], np.ones(12), state_names=self.sn1
        )

        self.cpd1 = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
        )
        self.cpd2 = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
            state_names=self.sn2,
        )

        student = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])
        diff_cpd = TabularCPD("diff", 2, [[0.2], [0.8]])
        intel_cpd = TabularCPD("intel", 2, [[0.3], [0.7]])
        grade_cpd = TabularCPD(
            "grade",
            3,
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.8, 0.8, 0.8, 0.8]],
            evidence=["diff", "intel"],
            evidence_card=[2, 2],
        )
        student.add_cpds(diff_cpd, intel_cpd, grade_cpd)
        self.model1 = Inference(student)
        self.model2 = Inference(student)

    def test_factor_init_statename(self):
        self.assertEqual(self.phi1.state_names, self.sn1_no_names)
        self.assertEqual(self.phi2.state_names, self.sn1)

    def test_cpd_init_statename(self):
        self.assertEqual(self.cpd1.state_names, self.sn2_no_names)
        self.assertEqual(self.cpd2.state_names, self.sn2)


class StateNameDecorator(unittest.TestCase):
    def setUp(self):
        self.sn2 = {
            "grade": ["A", "B", "F"],
            "diff": ["high", "low"],
            "intel": ["poor", "good", "very good"],
        }
        self.sn1 = {
            "speed": ["low", "medium", "high"],
            "switch": ["on", "off"],
            "time": ["day", "night"],
        }

        self.phi1 = DiscreteFactor(["speed", "switch", "time"], [3, 2, 2], np.ones(12))
        self.phi2 = DiscreteFactor(
            ["speed", "switch", "time"], [3, 2, 2], np.ones(12), state_names=self.sn1
        )

        self.cpd1 = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
        )
        self.cpd2 = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
            state_names=self.sn2,
        )

        student = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])
        student_state_names = DiscreteBayesianNetwork(
            [("diff", "grade"), ("intel", "grade")]
        )

        diff_cpd = TabularCPD("diff", 2, [[0.2], [0.8]])
        intel_cpd = TabularCPD("intel", 2, [[0.3], [0.7]])
        grade_cpd = TabularCPD(
            "grade",
            3,
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.8, 0.8, 0.8, 0.8]],
            evidence=["diff", "intel"],
            evidence_card=[2, 2],
        )

        diff_cpd_state_names = TabularCPD(
            variable="diff",
            variable_card=2,
            values=[[0.2], [0.8]],
            state_names={"diff": ["high", "low"]},
        )
        intel_cpd_state_names = TabularCPD(
            variable="intel",
            variable_card=2,
            values=[[0.3], [0.7]],
            state_names={"intel": ["poor", "good", "very good"]},
        )
        grade_cpd_state_names = TabularCPD(
            "grade",
            3,
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.8, 0.8, 0.8, 0.8]],
            evidence=["diff", "intel"],
            evidence_card=[2, 2],
            state_names=self.sn2,
        )

        student.add_cpds(diff_cpd, intel_cpd, grade_cpd)
        student_state_names.add_cpds(
            diff_cpd_state_names, intel_cpd_state_names, grade_cpd_state_names
        )

        self.model_no_state_names = VariableElimination(student)
        self.model_with_state_names = VariableElimination(student_state_names)

    def test_assignment_statename(self):
        req_op1 = [
            [("speed", "low"), ("switch", "on"), ("time", "night")],
            [("speed", "low"), ("switch", "off"), ("time", "day")],
        ]
        req_op2 = [
            [("speed", 0), ("switch", 0), ("time", 1)],
            [("speed", 0), ("switch", 1), ("time", 0)],
        ]

        self.assertEqual(self.phi1.assignment([1, 2]), req_op2)
        self.assertEqual(self.phi2.assignment([1, 2]), req_op1)

    def test_factor_reduce_statename(self):
        phi = DiscreteFactor(
            ["speed", "switch", "time"], [3, 2, 2], np.ones(12), state_names=self.sn1
        )
        phi.reduce([("speed", "medium"), ("time", "day")])
        self.assertEqual(phi.variables, ["switch"])
        self.assertEqual(phi.cardinality, [2])
        np_test.assert_array_equal(phi.values, np.array([1, 1]))

        phi = DiscreteFactor(
            ["speed", "switch", "time"], [3, 2, 2], np.ones(12), state_names=self.sn1
        )
        phi = phi.reduce([("speed", "medium"), ("time", "day")], inplace=False)
        self.assertEqual(phi.variables, ["switch"])
        self.assertEqual(phi.cardinality, [2])
        np_test.assert_array_equal(phi.values, np.array([1, 1]))

        phi = DiscreteFactor(["speed", "switch", "time"], [3, 2, 2], np.ones(12))
        phi.reduce([("speed", 1), ("time", 0)])
        self.assertEqual(phi.variables, ["switch"])
        self.assertEqual(phi.cardinality, [2])
        np_test.assert_array_equal(phi.values, np.array([1, 1]))

        phi = DiscreteFactor(["speed", "switch", "time"], [3, 2, 2], np.ones(12))
        phi = phi.reduce([("speed", 1), ("time", 0)], inplace=False)
        self.assertEqual(phi.variables, ["switch"])
        self.assertEqual(phi.cardinality, [2])
        np_test.assert_array_equal(phi.values, np.array([1, 1]))

    def test_reduce_cpd_statename(self):
        cpd = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
            state_names=self.sn2,
        )
        cpd.reduce([("diff", "high")])
        self.assertEqual(cpd.variable, "grade")
        self.assertEqual(cpd.variables, ["grade", "intel"])
        np_test.assert_array_equal(
            cpd.get_values(),
            np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.8, 0.8, 0.8]]),
        )

        cpd = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
        )
        cpd.reduce([("diff", 0)])
        self.assertEqual(cpd.variable, "grade")
        self.assertEqual(cpd.variables, ["grade", "intel"])
        np_test.assert_array_equal(
            cpd.get_values(),
            np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.8, 0.8, 0.8]]),
        )

        cpd = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
            state_names=self.sn2,
        )
        cpd = cpd.reduce([("diff", "high")], inplace=False)
        self.assertEqual(cpd.variable, "grade")
        self.assertEqual(cpd.variables, ["grade", "intel"])
        np_test.assert_array_equal(
            cpd.get_values(),
            np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.8, 0.8, 0.8]]),
        )

        cpd = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
        )
        cpd = cpd.reduce([("diff", 0)], inplace=False)
        self.assertEqual(cpd.variable, "grade")
        self.assertEqual(cpd.variables, ["grade", "intel"])
        np_test.assert_array_equal(
            cpd.get_values(),
            np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.8, 0.8, 0.8]]),
        )

    def test_inference_query_statename(self):
        inf_op1 = self.model_with_state_names.query(
            ["grade"], evidence={"intel": "poor"}
        )
        inf_op2 = self.model_no_state_names.query(["grade"], evidence={"intel": 0})
        req_op = DiscreteFactor(
            ["grade"],
            [3],
            np.array([0.1, 0.1, 0.8]),
            state_names={"grade": ["A", "B", "F"]},
        )
        self.assertEqual(inf_op1, req_op)
        self.assertEqual(inf_op1, req_op)

        inf_op1 = self.model_with_state_names.map_query(
            ["grade"], evidence={"intel": "poor"}
        )
        inf_op2 = self.model_no_state_names.map_query(["grade"], evidence={"intel": 0})
        req_op1 = {"grade": "F"}
        req_op2 = {"grade": 2}

        self.assertEqual(inf_op1, req_op1)
        self.assertEqual(inf_op2, req_op2)

    def test_add_state_names(self):
        # Test string state names taking precedence over numeric ones
        numeric_states = DiscreteFactor(
            ["speed"], [3], np.ones(3), state_names={"speed": [0, 1, 2]}
        )
        string_states = DiscreteFactor(
            ["speed"], [3], np.ones(3), state_names={"speed": ["low", "medium", "high"]}
        )

        # Make a copy to test in both directions
        numeric_states_copy = DiscreteFactor(
            ["speed"], [3], np.ones(3), state_names={"speed": [0, 1, 2]}
        )

        # String states should take precedence
        numeric_states.add_state_names(string_states)
        self.assertEqual(numeric_states.state_names["speed"], ["low", "medium", "high"])

        # Test the opposite direction - string states should still take precedence
        string_states.add_state_names(numeric_states_copy)
        self.assertEqual(string_states.state_names["speed"], ["low", "medium", "high"])

        # Test conflicting string state names
        states1 = DiscreteFactor(
            ["switch"], [2], np.ones(2), state_names={"switch": ["on", "off"]}
        )
        states2 = DiscreteFactor(
            ["switch"], [2], np.ones(2), state_names={"switch": ["high", "low"]}
        )

        # Should raise a ValueError due to conflict
        with self.assertRaises(ValueError):
            states1.add_state_names(states2)

        # Test merging non-conflicting state names for different variables
        factor1 = DiscreteFactor(
            ["speed"], [3], np.ones(3), state_names={"speed": ["low", "medium", "high"]}
        )
        factor2 = DiscreteFactor(
            ["switch"], [2], np.ones(2), state_names={"switch": ["on", "off"]}
        )

        # Should merge without conflict
        factor1.add_state_names(factor2)
        self.assertEqual(factor1.state_names["speed"], ["low", "medium", "high"])
        self.assertEqual(factor1.state_names["switch"], ["on", "off"])
