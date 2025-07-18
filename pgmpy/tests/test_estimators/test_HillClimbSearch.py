import unittest

import numpy as np
import pandas as pd

from pgmpy.estimators import K2, ExpertKnowledge, HillClimbSearch
from pgmpy.models import DiscreteBayesianNetwork


class TestHillClimbEstimatorDiscrete(unittest.TestCase):
    def setUp(self):
        self.rand_data = pd.DataFrame(
            np.random.randint(0, 5, size=(int(1e4), 2)), columns=list("AB")
        )
        self.rand_data["C"] = self.rand_data["B"]
        self.est_rand = HillClimbSearch(self.rand_data)
        k2score = K2(self.rand_data)
        self.score_rand = k2score.local_score
        self.score_structure_prior = k2score.structure_prior_ratio

        self.model1 = DiscreteBayesianNetwork()
        self.model1.add_nodes_from(["A", "B", "C"])
        self.model1_possible_edges = set(
            [(u, v) for u in self.model1.nodes() for v in self.model1.nodes()]
        )

        self.model2 = self.model1.copy()
        self.model2.add_edge("A", "B")
        self.model2_possible_edges = set(
            [(u, v) for u in self.model2.nodes() for v in self.model2.nodes()]
        )

        # link to dataset: "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )
        self.titanic_data1 = self.titanic_data[
            ["Survived", "Sex", "Pclass", "Age", "Embarked"]
        ]
        self.est_titanic1 = HillClimbSearch(self.titanic_data1)
        self.score_titanic1 = K2(self.titanic_data1).local_score

        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]
        self.est_titanic2 = HillClimbSearch(self.titanic_data2)
        self.score_titanic2 = K2(self.titanic_data2).local_score

    def test_legal_operations(self):
        model2_legal_ops = list(
            self.est_rand._legal_operations(
                model=self.model2,
                score=self.score_rand,
                structure_score=self.score_structure_prior,
                tabu_list=set(),
                max_indegree=float("inf"),
                required_edges=set(),
                forbidden_edges=set(),
            )
        )
        model2_legal_ops_ref = [
            (("+", ("C", "A")), -28.15602208305154),
            (("+", ("A", "C")), -28.155467430966382),
            (("+", ("C", "B")), 7636.947544933631),
            (("+", ("B", "C")), 7937.805375579936),
            (("-", ("A", "B")), 28.155467430966382),
            (("flip", ("A", "B")), -0.0005546520851567038),
        ]
        self.assertSetEqual(
            set([op for op, score in model2_legal_ops]),
            set([op for op, score in model2_legal_ops_ref]),
        )

    def test_legal_operations_forbidden_required(self):
        model2_legal_ops_bl = list(
            self.est_rand._legal_operations(
                model=self.model2,
                score=self.score_rand,
                structure_score=self.score_structure_prior,
                tabu_list=set(),
                max_indegree=float("inf"),
                forbidden_edges=set([("A", "B"), ("A", "C"), ("C", "A"), ("C", "B")]),
                required_edges=set(),
            )
        )
        model2_legal_ops_bl_ref = [
            ("+", ("B", "C")),
            ("-", ("A", "B")),
            ("flip", ("A", "B")),
        ]
        self.assertSetEqual(
            set([op for op, score in model2_legal_ops_bl]), set(model2_legal_ops_bl_ref)
        )

        model2_legal_ops_wl = list(
            self.est_rand._legal_operations(
                model=self.model2,
                score=self.score_rand,
                structure_score=self.score_structure_prior,
                tabu_list=set(),
                max_indegree=float("inf"),
                forbidden_edges=set([("B", "C"), ("C", "B"), ("B", "A")]),
                required_edges=set(),
            )
        )
        model2_legal_ops_wl_ref = [
            ("+", ("A", "C")),
            ("+", ("C", "A")),
            ("-", ("A", "B")),
        ]
        self.assertSetEqual(
            set([op for op, score in model2_legal_ops_wl]), set(model2_legal_ops_wl_ref)
        )

    def test_legal_operations_titanic(self):
        start_model = DiscreteBayesianNetwork(
            [("Survived", "Sex"), ("Pclass", "Age"), ("Pclass", "Embarked")]
        )
        all_possible_edges = set(
            [(u, v) for u in start_model.nodes() for v in start_model.nodes()]
        )
        legal_ops = self.est_titanic1._legal_operations(
            model=start_model,
            score=self.score_titanic1,
            structure_score=self.score_structure_prior,
            tabu_list=[],
            max_indegree=float("inf"),
            forbidden_edges=set(),
            required_edges=set(),
        )
        self.assertEqual(len(list(legal_ops)), 20)

        tabu_list = [
            ("-", ("Survived", "Sex")),
            ("-", ("Survived", "Pclass")),
            ("flip", ("Age", "Pclass")),
        ]
        legal_ops_tabu = self.est_titanic1._legal_operations(
            model=start_model,
            score=self.score_titanic1,
            structure_score=self.score_structure_prior,
            tabu_list=tabu_list,
            max_indegree=float("inf"),
            forbidden_edges=set(),
            required_edges=set(),
        )
        self.assertEqual(len(list(legal_ops_tabu)), 18)

        legal_ops_indegree = self.est_titanic1._legal_operations(
            model=start_model,
            score=self.score_titanic1,
            structure_score=self.score_structure_prior,
            tabu_list=[],
            max_indegree=1,
            forbidden_edges=set(),
            required_edges=set(),
        )
        self.assertEqual(len(list(legal_ops_indegree)), 11)

        legal_ops_both = self.est_titanic1._legal_operations(
            model=start_model,
            score=self.score_titanic1,
            structure_score=self.score_structure_prior,
            tabu_list=tabu_list,
            max_indegree=1,
            forbidden_edges=set(),
            required_edges=set(),
        )

        legal_ops_both_ref = {
            ("+", ("Embarked", "Survived")): 10.050632580087495,
            ("+", ("Survived", "Pclass")): 41.8886804654893,
            ("+", ("Age", "Survived")): -23.635716036430722,
            ("+", ("Pclass", "Survived")): 41.81314459373152,
            ("+", ("Sex", "Pclass")): 4.772261678791324,
            ("-", ("Pclass", "Age")): 11.546515590730905,
            ("-", ("Pclass", "Embarked")): -32.17148283253266,
            ("flip", ("Pclass", "Embarked")): 3.3563814191275583,
            ("flip", ("Survived", "Sex")): 0.0397370279797542,
        }
        self.assertSetEqual(
            set([op for op, score in legal_ops_both]), set(legal_ops_both_ref)
        )
        for op, score in legal_ops_both:
            self.assertAlmostEqual(score, legal_ops_both_ref[op])

    def test_estimate_rand(self):
        est1 = self.est_rand.estimate(scoring_method="k2", show_progress=False)
        self.assertSetEqual(set(est1.nodes()), set(["A", "B", "C"]))
        self.assertTrue(
            list(est1.edges()) == [("B", "C")] or list(est1.edges()) == [("C", "B")]
        )

        est2 = self.est_rand.estimate(
            scoring_method="k2",
            start_dag=DiscreteBayesianNetwork([("A", "B"), ("A", "C")]),
            show_progress=False,
        )
        self.assertTrue(
            list(est2.edges()) == [("B", "C")] or list(est2.edges()) == [("C", "B")]
        )

        expert_knowledge = ExpertKnowledge(required_edges=[("B", "C")])
        est3 = self.est_rand.estimate(
            scoring_method="k2", expert_knowledge=expert_knowledge, show_progress=False
        )
        self.assertTrue([("B", "C")] == list(est3.edges()))

    def test_estimate_titanic(self):
        self.assertSetEqual(
            set(
                self.est_titanic2.estimate(
                    scoring_method="k2", show_progress=False
                ).edges()
            ),
            set([("Survived", "Pclass"), ("Sex", "Pclass"), ("Sex", "Survived")]),
        )

        expert_knowledge = ExpertKnowledge(required_edges=[("Pclass", "Survived")])
        est_edges = self.est_titanic2.estimate(
            scoring_method="k2", expert_knowledge=expert_knowledge, show_progress=False
        ).edges()
        self.assertTrue(("Pclass", "Survived") in est_edges)

        temporal_knowledge = ExpertKnowledge(
            temporal_order=[["Pclass", "Sex"], ["Survived"]]
        )
        est_edges = self.est_titanic2.estimate(
            expert_knowledge=temporal_knowledge, show_progress=False
        ).edges()
        self.assertTrue(
            est_edges
            <= set(
                [
                    ("Sex", "Survived"),
                    ("Sex", "Pclass"),
                    ("Pclass", "Sex"),
                    ("Pclass", "Survived"),
                ]
            )
        )

    def test_no_legal_operation(self):
        data = pd.DataFrame(
            [
                [1, 0, 0, 1, 0, 0, 1, 1, 0],
                [1, 0, 1, 0, 0, 1, 0, 1, 0],
                [1, 0, 0, 0, 0, 1, 0, 1, 1],
                [1, 1, 0, 1, 0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 1, 0, 0],
            ],
            columns=list("ABCDEFGHI"),
        )
        est = HillClimbSearch(data)
        expert_knowledge = ExpertKnowledge(
            required_edges=[("A", "B"), ("B", "C")],
            forbidden_edges=[(u, v) for u in data.columns for v in data.columns],
        )
        best_model = est.estimate(
            scoring_method="k2", expert_knowledge=expert_knowledge
        )

    def test_estimate(self):
        for score in ["k2", "bdeu", "bds", "bic-d", "aic-d"]:
            dag = self.est_rand.estimate(scoring_method=score, show_progress=False)
            dag = self.est_titanic1.estimate(scoring_method=score, show_progress=False)

    def test_search_space(self):
        adult_data = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")

        search_space = [
            ("Age", "Education"),
            ("Education", "HoursPerWeek"),
            ("Education", "Income"),
            ("HoursPerWeek", "Income"),
            ("Age", "Income"),
        ]

        expert_knowledge = ExpertKnowledge(search_space=search_space)

        est = HillClimbSearch(adult_data)

        dag = est.estimate(
            scoring_method="k2",
            expert_knowledge=expert_knowledge,
            show_progress=False,
        )
        # assert if dag is a subset of search_space
        for edge in dag.edges():
            self.assertIn(edge, search_space)

    def tearDown(self):
        del self.rand_data
        del self.est_rand
        del self.model1
        del self.titanic_data
        del self.titanic_data1
        del self.titanic_data2
        del self.est_titanic1
        del self.est_titanic2


class TestHillClimbEstimatorGaussian(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv", index_col=0
        )

    def test_estimate(self):
        est = HillClimbSearch(self.data)
        for score in ["aic-g", "bic-g"]:
            dag = est.estimate(scoring_method=score, show_progress=False)


class TestHillClimbEstimatorMixed(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0
        )
        self.data["A_cat"] = self.data.A_cat.astype("category")
        self.data["B_cat"] = self.data.B_cat.astype("category")
        self.data["C_cat"] = self.data.C_cat.astype("category")
        self.data["B_int"] = self.data.B_int.astype("category")

    def test_estimate(self):
        est = HillClimbSearch(self.data)
        dag = est.estimate(scoring_method="ll-cg")
