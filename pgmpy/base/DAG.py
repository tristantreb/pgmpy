#!/usr/bin/env python3

import itertools
from os import PathLike
from typing import Hashable, Iterable, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import UndirectedGraph
from pgmpy.global_vars import logger
from pgmpy.independencies import Independencies
from pgmpy.utils.parser import parse_dagitty, parse_lavaan


class DAG(nx.DiGraph):
    """
    Base class for all Directed Graphical Models.

    Each node in the graph can represent either a random variable, `Factor`,
    or a cluster of random variables. Edges in the graph represent the
    dependencies between these.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is
        created. The data can be an edge list or any Networkx graph object.

    Examples
    --------
    Create an empty DAG with no nodes and no edges

    >>> from pgmpy.base import DAG
    >>> G = DAG()

    G can be grown in several ways:

    **Nodes:**

    Add one node at a time:

    >>> G.add_node(node='a')

    Add the nodes from any container (a list, set or tuple or the nodes
    from another graph).

    >>> G.add_nodes_from(nodes=['a', 'b'])

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> G.add_edge(u='a', v='b')

    a list of edges,

    >>> G.add_edges_from(ebunch=[('a', 'b'), ('b', 'c')])

    If some edges connect nodes not yet in the model, the nodes
    are added automatically. There are no errors when adding
    nodes or edges that already exist.

    **Shortcuts:**

    Many common graph features allow python syntax for speed reporting.

    >>> 'a' in G     # check if node in graph
    True
    >>> len(G)  # number of nodes in graph
    3
    """

    def __init__(
        self,
        ebunch: Optional[Iterable[tuple[Hashable, Hashable]]] = None,
        latents: set[Hashable] = set(),
        lavaan_str: Optional[list[str]] = None,
        dagitty_str: Optional[list[str]] = None,
    ):
        if lavaan_str:
            ebunch, latents, err_corr, _ = parse_lavaan(lavaan_str)
            if err_corr:
                logger.warning(
                    f"Residual correlations {err_corr} are ignored in DAG. Use the SEM class to keep them."
                )
        elif dagitty_str:
            ebunch, latents = parse_dagitty(dagitty_str)

        super(DAG, self).__init__(ebunch)
        self.latents = set(latents)
        cycles = []
        try:
            cycles = list(nx.find_cycle(self))
        except nx.NetworkXNoCycle:
            pass
        else:
            out_str = "Cycles are not allowed in a DAG."
            out_str += "\nEdges indicating the path taken for a loop: "
            out_str += "".join([f"({u},{v}) " for (u, v) in cycles])
            raise ValueError(out_str)

    @classmethod
    def from_lavaan(
        cls,
        string: Optional[str] = None,
        filename: Optional[str | PathLike] = None,
    ) -> "DAG":
        """
        Initializes a `DAG` instance using lavaan syntax.

        Parameters
        ----------
        string: str (default: None)
            A `lavaan` style multiline set of regression equation representing the model.
            Refer http://lavaan.ugent.be/tutorial/syntax1.html for details.

        filename: str (default: None)
            The filename of the file containing the model in lavaan syntax.

        Examples
        --------
        """
        if filename:
            with open(filename, "r") as f:
                lavaan_str = f.readlines()
        elif string:
            lavaan_str = string.split("\n")
        else:
            raise ValueError("Either `filename` or `string` need to be specified")

        return cls(lavaan_str=lavaan_str)

    @classmethod
    def from_dagitty(cls, string=None, filename=None) -> "DAG":
        """
        Initializes a `DAG` instance using DAGitty syntax.

        Parameters
        ----------
        string: str (default: None)
            A `DAGitty` style multiline set of regression equation representing the model.
            Refer https://www.dagitty.net/manual-3.x.pdf#page=3.58 and
            https://github.com/jtextor/dagitty/blob/7a657776dc8f5e5ba4e323edb028e2c2aaf29327/gui/js/dagitty.js#L3417

        filename: str (default: None)
            The filename of the file containing the model in DAGitty syntax.

        Examples
        --------
        """
        if filename:
            with open(filename, "r") as f:
                dagitty_str = f.readlines()
        elif string:
            dagitty_str = string.split("\n")
        else:
            raise ValueError("Either `filename` or `string` need to be specified")

        return cls(dagitty_str=dagitty_str)

    def add_node(
        self, node: Hashable, weight: Optional[float] = None, latent: bool = False
    ):
        """
        Adds a single node to the Graph.

        Parameters
        ----------
        node: str, int, or any hashable python object.
            The node to add to the graph.

        weight: int, float
            The weight of the node.

        latent: boolean (default: False)
            Specifies whether the variable is latent or not.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_node(node='A')
        >>> sorted(G.nodes())
        ['A']

        Adding a node with some weight.

        >>> G.add_node(node='B', weight=0.3)

        The weight of these nodes can be accessed as:

        >>> G.nodes['B']
        {'weight': 0.3}
        >>> G.nodes['A']
        {'weight': None}
        """

        # Check for networkx 2.0 syntax
        if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], dict):
            node, attrs = node
            if attrs.get("weight", None) is not None:
                attrs["weight"] = weight
        else:
            attrs = {"weight": weight}

        if latent:
            self.latents.add(node)

        super(DAG, self).add_node(node, weight=weight)

    def add_nodes_from(
        self,
        nodes: Iterable[Hashable],
        weights: Optional[list[float] | tuple[float]] = None,
        latent: Sequence[bool] | bool = False,
    ):
        """
        Add multiple nodes to the Graph.

        **The behviour of adding weights is different than in networkx.

        Parameters
        ----------
        nodes: iterable container
            A container (list, dict, set) of nodes (str, int or any hashable python
            object).

        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the variable at index i.

        latent: bool, list, tuple (default=False)
            A container of boolean. The value at index i tells whether the
            node at index i is latent or not.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_nodes_from(nodes=['A', 'B', 'C'])
        >>> G.nodes()
        NodeView(('A', 'B', 'C'))

        Adding nodes with weights:

        >>> G.add_nodes_from(nodes=['D', 'E'], weights=[0.3, 0.6])
        >>> G.nodes['D']
        {'weight': 0.3}
        >>> G.nodes['E']
        {'weight': 0.6}
        >>> G.nodes['A']
        {'weight': None}
        """
        nodes = list(nodes)

        if isinstance(latent, bool):
            latent = [latent] * len(nodes)

        if weights:
            if len(nodes) != len(weights):
                raise ValueError(
                    "The number of elements in nodes and weights" "should be equal."
                )
            for index in range(len(nodes)):
                self.add_node(
                    node=nodes[index], weight=weights[index], latent=latent[index]
                )
        else:
            for index in range(len(nodes)):
                self.add_node(node=nodes[index], latent=latent[index])

    def add_edge(self, u: Hashable, v: Hashable, weight: Optional[int | float] = None):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u, v : nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_nodes_from(nodes=['Alice', 'Bob', 'Charles'])
        >>> G.add_edge(u='Alice', v='Bob')
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob')])

        When the node is not already present in the graph:

        >>> G.add_edge(u='Alice', v='Ankur')
        >>> G.nodes()
        NodeView(('Alice', 'Ankur', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Alice', 'Ankur')])

        Adding edges with weight:

        >>> G.add_edge('Ankur', 'Maria', weight=0.1)
        >>> G.edge['Ankur']['Maria']
        {'weight': 0.1}
        """
        super(DAG, self).add_edge(u, v, weight=weight)

    def add_edges_from(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]],
        weights: list[float] | tuple[float] | None = None,
    ):
        """
        Add all the edges in ebunch.

        If nodes referred in the ebunch are not already present, they
        will be automatically added. Node names can be any hashable python
        object.

        **The behavior of adding weights is different than networkx.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u, v).

        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the edge at index i.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_nodes_from(nodes=['Alice', 'Bob', 'Charles'])
        >>> G.add_edges_from(ebunch=[('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Bob', 'Charles')])

        When the node is not already in the model:

        >>> G.add_edges_from(ebunch=[('Alice', 'Ankur')])
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles', 'Ankur'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Bob', 'Charles'), ('Alice', 'Ankur')])

        Adding edges with weights:

        >>> G.add_edges_from([('Ankur', 'Maria'), ('Maria', 'Mason')],
        ...                  weights=[0.3, 0.5])
        >>> G.edge['Ankur']['Maria']
        {'weight': 0.3}
        >>> G.edge['Maria']['Mason']
        {'weight': 0.5}

        or

        >>> G.add_edges_from([('Ankur', 'Maria', 0.3), ('Maria', 'Mason', 0.5)])
        """
        ebunch = list(ebunch)

        if weights:
            if len(ebunch) != len(weights):
                raise ValueError(
                    "The number of elements in ebunch and weights" "should be equal"
                )
            for index in range(len(ebunch)):
                self.add_edge(ebunch[index][0], ebunch[index][1], weight=weights[index])
        else:
            for edge in ebunch:
                if len(edge) == 2:
                    self.add_edge(edge[0], edge[1])
                else:
                    self.add_edge(edge[0], edge[1], edge[2])

    def get_parents(self, node: Hashable):
        """
        Returns a list of parents of node.

        Throws an error if the node is not present in the graph.

        Parameters
        ----------
        node: string, int or any hashable python object.
            The node whose parents would be returned.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[('diff', 'grade'), ('intel', 'grade')])
        >>> G.get_parents(node='grade')
        ['diff', 'intel']
        """
        return list(self.predecessors(node))

    def moralize(self):
        """
        Removes all the immoralities in the DAG and creates a moral
        graph (UndirectedGraph).

        A v-structure X->Z<-Y is an immorality if there is no directed edge
        between X and Y.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[('diff', 'grade'), ('intel', 'grade')])
        >>> moral_graph = G.moralize()
        >>> moral_graph.edges()
        EdgeView([('intel', 'grade'), ('intel', 'diff'), ('grade', 'diff')])
        """
        moral_graph = UndirectedGraph()
        moral_graph.add_nodes_from(self.nodes())
        moral_graph.add_edges_from(self.to_undirected().edges())

        for node in self.nodes():
            moral_graph.add_edges_from(
                itertools.combinations(self.get_parents(node), 2)
            )

        return moral_graph

    def get_leaves(self):
        """
        Returns a list of leaves of the graph.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> graph = DAG([('A', 'B'), ('B', 'C'), ('B', 'D')])
        >>> graph.get_leaves()
        ['C', 'D']
        """
        return [node for node, out_degree in self.out_degree_iter() if out_degree == 0]

    def out_degree_iter(self, nbunch=None, weight=None):
        return iter(self.out_degree(nbunch, weight))

    def in_degree_iter(self, nbunch=None, weight=None):
        return iter(self.in_degree(nbunch, weight))

    def get_roots(self):
        """
        Returns a list of roots of the graph.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> graph = DAG([('A', 'B'), ('B', 'C'), ('B', 'D'), ('E', 'B')])
        >>> graph.get_roots()
        ['A', 'E']
        """
        return [
            node for node, in_degree in dict(self.in_degree()).items() if in_degree == 0
        ]

    def get_children(self, node: Hashable):
        """
        Returns a list of children of node.
        Throws an error if the node is not present in the graph.

        Parameters
        ----------
        node: string, int or any hashable python object.
            The node whose children would be returned.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> g = DAG(ebunch=[('A', 'B'), ('C', 'B'), ('B', 'D'),
                                      ('B', 'E'), ('B', 'F'), ('E', 'G')])
        >>> g.get_children(node='B')
        ['D', 'E', 'F']
        """
        return list(self.successors(node))

    def get_independencies(
        self, latex=False, include_latents=False
    ) -> Independencies | list[str]:
        """
        Computes independencies in the DAG, by checking minimal d-seperation.

        Parameters
        ----------
        latex: boolean
            If latex=True then latex string of the independence assertion
            would be created.

        include_latents: boolean
            If True, includes latent variables in the independencies. Otherwise,
            only generates independencies on observed variables.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> chain = DAG([('X', 'Y'), ('Y', 'Z')])
        >>> chain.get_independencies()
        (X \u27c2 Z | Y)
        """
        nodes = set(self.nodes())
        if not include_latents:
            nodes -= self.latents

        independencies = Independencies()
        for x, y in itertools.combinations(nodes, 2):
            if not self.has_edge(x, y) and not self.has_edge(y, x):
                minimal_separator = self.minimal_dseparator(
                    start=x, end=y, include_latents=include_latents
                )
                if minimal_separator is not None:
                    independencies.add_assertions([x, y, minimal_separator])

        independencies = independencies.reduce()

        if not latex:
            return independencies
        else:
            return independencies.latex_string()

    def local_independencies(
        self, variables: list[Hashable] | tuple[Hashable, ...] | str
    ):
        """
        Returns an instance of Independencies containing the local independencies
        of each of the variables.

        Parameters
        ----------
        variables: str or array like
            variables whose local independencies are to be found.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> student = DAG()
        >>> student.add_edges_from([('diff', 'grade'), ('intel', 'grade'),
        >>>                         ('grade', 'letter'), ('intel', 'SAT')])
        >>> ind = student.local_independencies('grade')
        >>> ind
        (grade \u27c2 SAT | diff, intel)
        """

        independencies = Independencies()
        for variable in (
            variables if isinstance(variables, (list, tuple)) else [variables]
        ):
            non_descendents = (
                set(self.nodes())
                - {variable}
                - set(nx.dfs_preorder_nodes(self, variable))
            )
            parents = set(self.get_parents(variable))
            if non_descendents - parents:
                independencies.add_assertions(
                    [variable, non_descendents - parents, parents]
                )
        return independencies

    def is_iequivalent(self, model: "DAG"):
        """
        Checks whether the given model is I-equivalent

        Two graphs G1 and G2 are said to be I-equivalent if they have same skeleton
        and have same set of immoralities.

        Parameters
        ----------
        model : A DAG object, for which you want to check I-equivalence

        Returns
        --------
        I-equivalence: boolean
            True if both are I-equivalent, False otherwise

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_edges_from([('V', 'W'), ('W', 'X'),
        ...                   ('X', 'Y'), ('Z', 'Y')])
        >>> G1 = DAG()
        >>> G1.add_edges_from([('W', 'V'), ('X', 'W'),
        ...                    ('X', 'Y'), ('Z', 'Y')])
        >>> G.is_iequivalent(G1)
        True

        """
        if not isinstance(model, DAG):
            raise TypeError(
                f"Model must be an instance of DAG. Got type: {type(model)}"
            )

        if (self.to_undirected().edges() == model.to_undirected().edges()) and (
            self.get_immoralities() == model.get_immoralities()
        ):
            return True
        return False

    def get_immoralities(self) -> dict[Hashable, list[tuple[Hashable, Hashable]]]:
        """
        Finds all the immoralities in the model
        A v-structure X -> Z <- Y is an immorality if there is no direct edge between X and Y .

        Returns
        -------
        Immoralities: set
            A set of all the immoralities in the model

        Examples
        ---------
        >>> from pgmpy.base import DAG
        >>> student = DAG()
        >>> student.add_edges_from([('diff', 'grade'), ('intel', 'grade'),
        ...                         ('intel', 'SAT'), ('grade', 'letter')])
        >>> student.get_immoralities()
        {('diff', 'intel')}
        """
        immoralities = dict()
        for node in self.nodes():
            parent_pairs = []
            for parents in itertools.combinations(self.predecessors(node), 2):
                if not self.has_edge(parents[0], parents[1]) and not self.has_edge(
                    parents[1], parents[0]
                ):
                    parent_pairs.append(tuple(sorted(parents)))
            immoralities[node] = parent_pairs
        return immoralities

    def is_dconnected(
        self,
        start: Hashable,
        end: Hashable,
        observed: Optional[Sequence[Hashable]] = None,
        include_latents=False,
    ):
        """
        Returns True if there is an active trail (i.e. d-connection) between
        `start` and `end` node given that `observed` is observed.

        Parameters
        ----------
        start, end : int, str, any hashable python object.
            The nodes in the DAG between which to check the d-connection/active trail.

        observed : list, array-like (optional)
            If given the active trail would be computed assuming these nodes to
            be observed.

        include_latents: boolean (default: False)
            If true, latent variables are return as part of the active trail.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> student = DAG()
        >>> student.add_nodes_from(['diff', 'intel', 'grades', 'letter', 'sat'])
        >>> student.add_edges_from([('diff', 'grades'), ('intel', 'grades'), ('grades', 'letter'),
        ...                         ('intel', 'sat')])
        >>> student.is_dconnected('diff', 'intel')
        False
        >>> student.is_dconnected('grades', 'sat')
        True
        """
        if (
            end
            in self.active_trail_nodes(
                variables=start, observed=observed, include_latents=include_latents
            )[start]
        ):
            return True
        else:
            return False

    def minimal_dseparator(
        self, start: Hashable, end: Hashable, include_latents=False
    ) -> set[Hashable]:
        """
        Finds the minimal d-separating set for `start` and `end`.

        Parameters
        ----------
        start: node
            The first node.

        end: node
            The second node.

        include_latents: boolean (default: False)
            If true, latent variables are consider for minimal d-seperator.

        Examples
        --------
        >>> dag = DAG([('A', 'B'), ('B', 'C')])
        >>> dag.minimal_dseparator(start='A', end='C')
        {'B'}

        References
        ----------
        [1] Algorithm 4, Page 10: Tian, Jin, Azaria Paz, and Judea Pearl. Finding minimal d-separators. Computer Science Department, University of California, 1998.
        """
        if (end in self.neighbors(start)) or (start in self.neighbors(end)):
            raise ValueError(
                "No possible separators because start and end are adjacent"
            )
        an_graph = self.get_ancestral_graph([start, end])
        separator = set(
            itertools.chain(self.predecessors(start), self.predecessors(end))
        )

        if not include_latents:
            # If any of the parents were latents, take the latent's parent
            while len(separator.intersection(self.latents)) != 0:
                separator_copy = separator.copy()
                for u in separator:
                    if u in self.latents:
                        separator_copy.remove(u)
                        separator_copy.update(set(self.predecessors(u)))
                separator = separator_copy

        # Remove the start and end nodes in case it reaches there while removing latents.
        separator.difference_update({start, end})

        # If the initial set is not able to d-separate, no d-separator is possible.
        if an_graph.is_dconnected(start, end, observed=separator):
            return None

        # Go through the separator set, remove one element and check if it remains
        # a dseparating set.
        minimal_separator = separator.copy()

        for u in separator:
            if not an_graph.is_dconnected(start, end, observed=minimal_separator - {u}):
                minimal_separator.remove(u)

        return minimal_separator

    def get_markov_blanket(self, node: Hashable) -> list[Hashable]:
        """
        Returns a markov blanket for a random variable. In the case
        of Bayesian Networks, the markov blanket is the set of
        node's parents, its children and its children's other parents.

        Returns
        -------
        Markov Blanket: list
            List of nodes in the markov blanket of `node`.

        Parameters
        ----------
        node: string, int or any hashable python object.
              The node whose markov blanket would be returned.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> G = DAG([('x', 'y'), ('z', 'y'), ('y', 'w'), ('y', 'v'), ('u', 'w'),
                               ('s', 'v'), ('w', 't'), ('w', 'm'), ('v', 'n'), ('v', 'q')])
        >>> G.get_markov_blanket('y')
        ['s', 'w', 'x', 'u', 'z', 'v']
        """
        children = self.get_children(node)
        parents = self.get_parents(node)
        blanket_nodes = children + parents
        for child_node in children:
            blanket_nodes.extend(self.get_parents(child_node))
        blanket_nodes = set(blanket_nodes)
        blanket_nodes.discard(node)
        return list(blanket_nodes)

    def active_trail_nodes(
        self,
        variables: list[Hashable] | Hashable,
        observed: Optional[
            Hashable | list[Hashable] | tuple[Hashable, Hashable]
        ] = None,
        include_latents=False,
    ) -> dict[Hashable, set[Hashable]]:
        """
        Returns a dictionary with the given variables as keys and all the nodes reachable
        from that respective variable as values.

        Parameters
        ----------
        variables: str or array like
            variables whose active trails are to be found.

        observed : List of nodes (optional)
            If given the active trails would be computed assuming these nodes to be
            observed.

        include_latents: boolean (default: False)
            Whether to include the latent variables in the returned active trail nodes.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> student = DAG()
        >>> student.add_nodes_from(['diff', 'intel', 'grades'])
        >>> student.add_edges_from([('diff', 'grades'), ('intel', 'grades')])
        >>> student.active_trail_nodes('diff')
        {'diff': {'diff', 'grades'}}
        >>> student.active_trail_nodes(['diff', 'intel'], observed='grades')
        {'diff': {'diff', 'intel'}, 'intel': {'diff', 'intel'}}

        References
        ----------
        Details of the algorithm can be found in 'Probabilistic Graphical Model
        Principles and Techniques' - Koller and Friedman
        Page 75 Algorithm 3.1
        """
        observed_list: list[Hashable] | tuple[Hashable, Hashable]
        if observed:
            if isinstance(observed, set):
                observed = list(observed)

            observed_list = (
                observed if isinstance(observed, (list, tuple)) else [observed]
            )
        else:
            observed_list = []
        ancestors_list = self._get_ancestors_of(observed_list)

        # Direction of flow of information
        # up ->  from parent to child
        # down -> from child to parent

        active_trails = {}
        for start in variables if isinstance(variables, list) else [variables]:
            visit_list = set()
            visit_list.add((start, "up"))
            traversed_list = set()
            active_nodes = set()
            while visit_list:
                node, direction = visit_list.pop()
                if (node, direction) not in traversed_list:
                    if node not in observed_list:
                        active_nodes.add(node)
                    traversed_list.add((node, direction))
                    if direction == "up" and node not in observed_list:
                        for parent in self.predecessors(node):
                            visit_list.add((parent, "up"))
                        for child in self.successors(node):
                            visit_list.add((child, "down"))
                    elif direction == "down":
                        if node not in observed_list:
                            for child in self.successors(node):
                                visit_list.add((child, "down"))
                        if node in ancestors_list:
                            for parent in self.predecessors(node):
                                visit_list.add((parent, "up"))
            if include_latents:
                active_trails[start] = active_nodes
            else:
                active_trails[start] = active_nodes - self.latents

        return active_trails

    def _get_ancestors_of(
        self, nodes: str | tuple[Hashable, Hashable] | Iterable[Hashable]
    ) -> set[Hashable]:
        """
        Returns a dictionary of all ancestors of all the observed nodes including the
        node itself.

        Parameters
        ----------
        nodes: string, list-type
            name of all the observed nodes

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> model = DAG([('D', 'G'), ('I', 'G'), ('G', 'L'),
        ...                        ('I', 'L')])
        >>> model._get_ancestors_of('G')
        {'D', 'G', 'I'}
        >>> model._get_ancestors_of(['G', 'I'])
        {'D', 'G', 'I'}
        """
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]

        for node in nodes:
            if node not in self.nodes():
                raise ValueError(f"Node {node} not in graph")

        ancestors_list = set()
        for node in nodes:
            ancestors_list.update(nx.ancestors(self, node))

        ancestors_list.update(nodes)
        return ancestors_list

    def to_pdag(self):
        """
        Returns the CPDAG (Completed Partial DAG) of the DAG representing the equivalence class that the given DAG belongs to.

        Returns
        -------
        CPDAG: pgmpy.base.PDAG
            An instance of pgmpy.base.PDAG representing the CPDAG of the given DAG.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([('A', 'B'), ('B', 'C'), ('C', 'D')])
        >>> pdag = dag.to_pdag()
        >>> pdag.directed_edges
        {('A', 'B'), ('B', 'C'), ('C', 'D')}

        References
        ----------
        [1] Chickering, David Maxwell. "Learning equivalence classes of Bayesian-network structures." Journal of machine learning research 2.Feb (2002): 445-498. Figure 4 and 5.
        """
        # Perform a topological sort on the nodes
        topo_order = list(nx.topological_sort(self))
        node_order = {node: i for i, node in enumerate(topo_order)}

        # Initialize edge ordering
        i = 0
        edge_order = {}
        unordered_edges = set(self.edges())

        # While there are unordered edges
        while unordered_edges:
            # Find lowest ordered node with unordered edges incident into it
            nodes_with_unordered_edges = {edge[1] for edge in unordered_edges}
            y = min(nodes_with_unordered_edges, key=lambda x: node_order[x])

            # Find highest ordered node for which x->y is not ordered
            unordered_edges_into_y = {edge for edge in unordered_edges if edge[1] == y}
            x = max(
                (edge[0] for edge in unordered_edges_into_y),
                key=lambda x: node_order[x],
            )

            # Label x->y with order i
            edge_order[(x, y)] = i
            i += 1
            unordered_edges.remove((x, y))

        # Label every edge as "unknown"
        edge_labels = {edge: "unknown" for edge in self.edges()}

        # While there are edges labeled "unknown"
        while any(label == "unknown" for label in edge_labels.values()):
            # Let x -> y be the lowest ordered edge that is labeled "unknown"
            unknown_edges = [
                (edge, edge_order[edge])
                for edge, label in edge_labels.items()
                if label == "unknown"
            ]
            x, y = min(unknown_edges, key=lambda x: x[1])[0]

            # Check compelled parents
            compelled_parents = [
                w for w in self.get_parents(x) if edge_labels.get((w, x)) == "compelled"
            ]
            for w in compelled_parents:
                if not self.has_edge(w, y):
                    # Label x -> y and every edge incident into y with "compelled"
                    edge_labels[(x, y)] = "compelled"
                    for z in self.get_parents(y):
                        if edge_labels.get((z, y)) == "unknown":
                            edge_labels[(z, y)] = "compelled"
                    break
                else:
                    # Label w -> y with "compelled"
                    edge_labels[(w, y)] = "compelled"

            # Check for v-structures
            if edge_labels.get((x, y)) != "compelled":
                v_structure_exists = False
                for z in self.get_parents(y):
                    if z != x and not self.has_edge(z, x):
                        v_structure_exists = True
                        break

                if v_structure_exists:
                    # Label x -> y and all "unknown" edges incident into y with "compelled"
                    edge_labels[(x, y)] = "compelled"
                    for z in self.get_parents(y):
                        if edge_labels.get((z, y)) == "unknown":
                            edge_labels[(z, y)] = "compelled"
                else:
                    # Label x -> y and all "unknown" edges incident into y with "reversible"
                    edge_labels[(x, y)] = "reversible"
                    for z in self.get_parents(y):
                        if edge_labels.get((z, y)) == "unknown":
                            edge_labels[(z, y)] = "reversible"

        # Create PDAG with directed and undirected edges
        directed_edges = [
            edge for edge, label in edge_labels.items() if label == "compelled"
        ]
        undirected_edges = [
            edge for edge, label in edge_labels.items() if label == "reversible"
        ]

        return PDAG(
            directed_ebunch=directed_edges,
            undirected_ebunch=undirected_edges,
            latents=self.latents,
        )

    def do(
        self,
        nodes: Hashable | Iterable[Hashable] | tuple[Hashable, Hashable],
        inplace=False,
    ):
        """
        Applies the do operator to the graph and returns a new DAG with the
        transformed graph.

        The do-operator, do(X = x) has the effect of removing all edges from
        the parents of X and setting X to the given value x.

        Parameters
        ----------
        nodes : list, array-like
            The names of the nodes to apply the do-operator for.

        inplace: boolean (default: False)
            If inplace=True, makes the changes to the current object,
            otherwise returns a new instance.

        Returns
        -------
        Modified DAG: pgmpy.base.DAG
            A new instance of DAG modified by the do-operator

        Examples
        --------
        Initialize a DAG

        >>> graph = DAG()
        >>> graph.add_edges_from([('X', 'A'),
        ...                       ('A', 'Y'),
        ...                       ('A', 'B')])
        >>> # Applying the do-operator will return a new DAG with the desired structure.
        >>> graph_do_A = graph.do('A')
        >>> # Which we can verify is missing the edges we would expect.
        >>> graph_do_A.edges
        OutEdgeView([('A', 'B'), ('A', 'Y')])

        References
        ----------
        Causality: Models, Reasoning, and Inference, Judea Pearl (2000). p.70.
        """
        dag = self if inplace else self.copy()

        if isinstance(nodes, (str, int)):
            nodes = [nodes]
        else:
            nodes = list(nodes)

        if not set(nodes).issubset(set(self.nodes())):
            raise ValueError(
                f"Nodes not found in the model: {set(nodes) - set(self.nodes)}"
            )

        for node in nodes:
            parents = list(dag.predecessors(node))
            for parent in parents:
                dag.remove_edge(parent, node)
        return dag

    def get_ancestral_graph(self, nodes: Iterable[Hashable]):
        """
        Returns the ancestral graph of the given `nodes`. The ancestral graph only
        contains the nodes which are ancestors of at least one of the variables in
        node.

        Parameters
        ----------
        node: iterable
            List of nodes whose ancestral graph needs to be computed.

        Returns
        -------
        Ancestral Graph: pgmpy.base.DAG

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([('A', 'C'), ('B', 'C'), ('D', 'A'), ('D', 'B')])
        >>> anc_dag = dag.get_ancestral_graph(nodes=['A', 'B'])
        >>> anc_dag.edges()
        OutEdgeView([('D', 'A'), ('D', 'B')])
        """
        return self.subgraph(nodes=self._get_ancestors_of(nodes=nodes))

    def to_daft(
        self,
        node_pos: str | dict[Hashable, tuple[int, int]] = "circular",
        latex=True,
        pgm_params={},
        edge_params={},
        node_params={},
    ):
        """
        Returns a daft (https://docs.daft-pgm.org/en/latest/) object which can be rendered for
        publication quality plots. The returned object's render method can be called to see the plots.

        Parameters
        ----------
        node_pos: str or dict (default: circular)
            If str: Must be one of the following: circular, kamada_kawai, planar, random, shell, sprint,
                spectral, spiral. Please refer: https://networkx.org/documentation/stable//reference/drawing.html#module-networkx.drawing.layout for details on these layouts.

            If dict should be of the form {node: (x coordinate, y coordinate)} describing the x and y coordinate of each
            node.

            If no argument is provided uses circular layout.

        latex: boolean
            Whether to use latex for rendering the node names.

        pgm_params: dict (optional)
            Any additional parameters that need to be passed to `daft.PGM` initializer.
            Should be of the form: {param_name: param_value}

        edge_params: dict (optional)
            Any additional edge parameters that need to be passed to `daft.add_edge` method.
            Should be of the form: {(u1, v1): {param_name: param_value}, (u2, v2): {...} }

        node_params: dict (optional)
            Any additional node parameters that need to be passed to `daft.add_node` method.
            Should be of the form: {node1: {param_name: param_value}, node2: {...} }

        Returns
        -------
        Daft object: daft.PGM object
            Daft object for plotting the DAG.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([('a', 'b'), ('b', 'c'), ('d', 'c')])
        >>> dag.to_daft(node_pos={'a': (0, 0), 'b': (1, 0), 'c': (2, 0), 'd': (1, 1)})
        <daft.PGM at 0x7fc756e936d0>
        >>> dag.to_daft(node_pos="circular")
        <daft.PGM at 0x7f9bb48c5eb0>
        >>> dag.to_daft(node_pos="circular", pgm_params={'observed_style': 'inner'})
        <daft.PGM at 0x7f9bb48b0bb0>
        >>> dag.to_daft(node_pos="circular",
        ...             edge_params={('a', 'b'): {'label': 2}},
        ...             node_params={'a': {'shape': 'rectangle'}})
        <daft.PGM at 0x7f9bb48b0bb0>
        """
        try:
            from daft import PGM
        except ImportError as e:
            raise ImportError(
                e.msg
                + ". Package daft required. Please visit: https://docs.daft-pgm.org/en/latest/ for installation instructions."
            ) from None

        if isinstance(node_pos, str):
            supported_layouts = {
                "circular": nx.circular_layout,
                "kamada_kawai": nx.kamada_kawai_layout,
                "planar": nx.planar_layout,
                "random": nx.random_layout,
                "shell": nx.shell_layout,
                "spring": nx.spring_layout,
                "spectral": nx.spectral_layout,
                "spiral": nx.spiral_layout,
            }
            if node_pos not in supported_layouts.keys():
                raise ValueError(
                    "Unknown node_pos argument. Please refer docstring for accepted values"
                )
            else:
                node_pos = supported_layouts[node_pos](self)
        elif isinstance(node_pos, dict):
            for node in self.nodes():
                if node not in node_pos.keys():
                    raise ValueError(f"No position specified for {node}.")
        else:
            raise ValueError(
                "Argument node_pos not valid. Please refer to the docstring."
            )

        daft_pgm = PGM(**pgm_params)
        for node in self.nodes():
            try:
                extra_params = node_params[node]
            except KeyError:
                extra_params = dict()

            if latex:
                daft_pgm.add_node(
                    node,
                    rf"${node}$",
                    node_pos[node][0],
                    node_pos[node][1],
                    observed=True,
                    **extra_params,
                )
            else:
                daft_pgm.add_node(
                    node,
                    f"{node}",
                    node_pos[node][0],
                    node_pos[node][1],
                    observed=True,
                    **extra_params,
                )

        for u, v in self.edges():
            try:
                extra_params = edge_params[(u, v)]
            except KeyError:
                extra_params = dict()
            daft_pgm.add_edge(u, v, **extra_params)

        return daft_pgm

    @staticmethod
    def get_random(
        n_nodes=5,
        edge_prob=0.5,
        node_names: Optional[list[Hashable]] = None,
        latents=False,
        seed: Optional[int] = None,
    ) -> "DAG":
        """
        Returns a randomly generated DAG with `n_nodes` number of nodes with
        edge probability being `edge_prob`.

        Parameters
        ----------
        n_nodes: int
            The number of nodes in the randomly generated DAG.

        edge_prob: float
            The probability of edge between any two nodes in the topologically
            sorted DAG.

        node_names: list (default: None)
            A list of variables names to use in the random graph.
            If None, the node names are integer values starting from 0.

        latents: bool (default: False)
            If True, includes latent variables in the generated DAG.

        seed: int (default: None)
            The seed for the random number generator.

        Returns
        -------
        Random DAG: pgmpy.base.DAG
            The randomly generated DAG.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> random_dag = DAG.get_random(n_nodes=10, edge_prob=0.3)
        >>> random_dag.nodes()
        NodeView((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
        >>> random_dag.edges()
        OutEdgeView([(0, 6), (1, 6), (1, 7), (7, 9), (2, 5), (2, 7), (2, 8), (5, 9), (3, 7)])
        """
        # Step 1: Generate a matrix of 0 and 1. Prob of choosing 1 = edge_prob
        gen = np.random.default_rng(seed=seed)
        adj_mat = gen.choice(
            [0, 1], size=(n_nodes, n_nodes), p=[1 - edge_prob, edge_prob]
        )

        # Step 2: Use the upper triangular part of the matrix as adjacency.
        if node_names is None:
            node_names = list([f"X_{i}" for i in range(n_nodes)])

        adj_pd = pd.DataFrame(
            np.triu(adj_mat, k=1), columns=node_names, index=node_names
        )
        nx_dag = nx.from_pandas_adjacency(adj_pd, create_using=nx.DiGraph)

        dag = DAG(nx_dag)
        dag.add_nodes_from(node_names)

        if latents:
            dag.latents = set(
                gen.choice(dag.nodes(), gen.integers(low=0, high=len(dag.nodes())))
            )
        return dag

    def to_graphviz(self):
        """
        Retuns a pygraphviz object for the DAG. pygraphviz is useful for
        visualizing the network structure.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> model = get_example_model('alarm')
        >>> model.to_graphviz()
        <AGraph <Swig Object of type 'Agraph_t *' at 0x7fdea4cde040>>
        >>> model.draw('model.png', prog='neato')
        """
        return nx.nx_agraph.to_agraph(self)

    def fit(self, data, estimator=None, state_names=[], n_jobs=1, **kwargs) -> "DAG":
        """
        Estimates the CPD for each variable based on a given data set.

        Parameters
        ----------
        data: pandas DataFrame object
            DataFrame object with column names identical to the variable names of the network.
            (If some values in the data are missing the data cells should be set to `numpy.nan`.
            Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

        estimator: Estimator class
            One of:
            - MaximumLikelihoodEstimator (default)
            - BayesianEstimator: In this case, pass 'prior_type' and either 'pseudo_counts'
            or 'equivalent_sample_size' as additional keyword arguments.
            See `BayesianEstimator.get_parameters()` for usage.
            - ExpectationMaximization

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states
            that the variable can take. If unspecified, the observed values
            in the data set are taken to be the only possible states.

        n_jobs: int (default: 1)
            Number of threads/processes to use for estimation. Using n_jobs > 1
            for small models or datasets might be slower.

        Returns
        -------
        Fitted Model: DiscreteBayesianNetwork
            Returns a DiscreteBayesianNetwork object with learned CPDs.
            The DAG structure is preserved, and parameters (CPDs) are added.
            This allows the DAG to represent both the structure and the parameters of a Bayesian Network.

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.base import DAG
        >>> data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        >>> model = DAG([('A', 'C'), ('B', 'C')])
        >>> fitted_model = model.fit(data)
        >>> fitted_model.get_cpds()
        [<TabularCPD representing P(A:2) at 0x17945372c30>,
        <TabularCPD representing P(B:2) at 0x17945a19760>,
        <TabularCPD representing P(C:2 | A:2, B:2) at 0x17944f42690>]
        """
        from pgmpy.estimators import BaseEstimator, MaximumLikelihoodEstimator
        from pgmpy.models import DiscreteBayesianNetwork

        if isinstance(self, DiscreteBayesianNetwork):
            bn = self
        else:
            bn = DiscreteBayesianNetwork(self.edges())
            bn.add_nodes_from(self.nodes())

        if estimator is None:
            estimator = MaximumLikelihoodEstimator
        else:
            if not issubclass(estimator, BaseEstimator):
                raise TypeError("Estimator object should be a valid pgmpy estimator.")

        _estimator = estimator(
            bn,
            data,
            state_names=state_names,
        )
        cpds_list = _estimator.get_parameters(n_jobs=n_jobs, **kwargs)
        bn.add_cpds(*cpds_list)
        return bn

    def _variable_name_contains_non_string(self):
        """
        Checks if the variable names contain any non-string values. Used only for CausalInference class.
        """
        for node in list(self.nodes()):
            if not isinstance(node, str):
                return (node, type(node))
        return False

    def copy(self):
        dag = DAG(ebunch=self.edges(), latents=self.latents)
        dag.add_nodes_from(self.nodes())
        return dag

    def edge_strength(self, data, edges=None):
        """
        Computes the strength of each edge in `edges`. The strength is bounded
        between 0 and 1, with 1 signifying strong effect.

        The edge strength is defined as the effect size measure of a
        Conditional Independence test using the parents as the conditional set.
        The strength quantifies the effect of edge[0] on edge[1] after
        controlling for any other influence paths. We use a residualization-based
        CI test[1] to compute the strengths.

        Interpretation:
        - The strength is the Pillai's Trace effect size of partial correlation.
        - Measures the strength of linear relationship between the residuals.
        - Works for any mixture of categorical and continuous variables.
        - The value is bounded between 0 and 1:
        - Strength close to 1 → strong dependence.
        - Strength close to 0 → conditional independence.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset to compute edge strengths on.

        edges : tuple, list, or None (default: None)
            - None: Compute for all DAG edges.
            - Tuple (X, Y): Compute for edge X → Y.
            - List of tuples: Compute for selected edges.

        Returns
        -------
        dict
            Dictionary mapping edges to their strength values.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork as LGBN
        >>> # Create a linear Gaussian Bayesian network
        >>> linear_model = LGBN([("X", "Y"), ("Z", "Y")])
        >>> # Create CPDs with specific beta values
        >>> x_cpd = LinearGaussianCPD(variable="X", beta=[0], std=1)
        >>> y_cpd = LinearGaussianCPD(variable="Y", beta=[0, 0.4, 0.6], std=1, evidence=["X", "Z"])
        >>> z_cpd = LinearGaussianCPD(variable="Z", beta=[0], std=1)
        >>> # Add CPDs to the model
        >>> linear_model.add_cpds(x_cpd, y_cpd, z_cpd)
        >>> # Simulate data from the model
        >>> data = linear_model.simulate(n_samples=int(1e4))
        >>> # Create DAG and compute edge strengths
        >>> dag = DAG([("X", "Y"), ("Z", "Y")])
        >>> strengths = dag.edge_strength(data)
        {('X', 'Y'): np.float64(0.14587166611282304),
         ('Z', 'Y'): np.float64(0.25683780900125613)}

        References
        ----------
        [1] Ankan, Ankur, and Johannes Textor. "A simple unified approach to testing high-dimensional conditional independences for categorical and ordinal data." Proceedings of the AAAI Conference on Artificial Intelligence.
        """

        from pgmpy.estimators.CITests import pillai_trace

        # If edges is None, compute for all edges in the DAG
        if edges is None:
            edges_to_compute = list(self.edges())
        # If edges is a single edge tuple
        elif isinstance(edges, tuple) and len(edges) == 2:
            edges_to_compute = [edges]
        # If edges is a list of edge tuples
        elif isinstance(edges, list) and all(
            isinstance(edge, tuple) and len(edge) == 2 for edge in edges
        ):
            edges_to_compute = edges
        else:
            raise ValueError(
                "edges parameter must be either None, a 2-tuple (X, Y), or a list of 2-tuples [(X1, Y1), (X2, Y2), ...]"
            )

        strengths = {}
        skipped_edges = []

        for edge in edges_to_compute:
            x, y = edge

            # Get parents of x and y using get_parents instead of predecessors
            pa_Y = self.get_parents(y)

            # Check if either x or y is a latent node
            if (
                x in self.latents
                or y in self.latents
                or any(parent in self.latents for parent in pa_Y)
            ):
                skipped_edges.append(edge)
                continue

            # Combine parents for conditioning set (excluding x and y themselves)
            conditioning_set = set(pa_Y) - {x, y}

            # Run CI test and get effect size
            effect_size, _ = pillai_trace(
                X=x, Y=y, Z=list(conditioning_set), data=data, boolean=False
            )

            # Store the edge strength
            strengths[edge] = effect_size

            # store the values in the graph as well
            self.edges[edge]["strength"] = effect_size

        if skipped_edges:
            logger.warning(
                f"Skipped computing strengths for edges involving latent variables: {skipped_edges}. "
                "Use CausalInference class for advanced causal effect estimation."
            )

        return strengths


class PDAG(nx.DiGraph):
    """
    Class for representing PDAGs (also known as CPDAG). PDAGs are the equivalence classes of
    DAGs and contain both directed and undirected edges.

    Note: In this class, undirected edges are represented using two edges in both direction i.e.
    an undirected edge between X - Y is represented using X -> Y and X <- Y.
    """

    def __init__(
        self,
        directed_ebunch: list[tuple[Hashable, Hashable]] = [],
        undirected_ebunch: list[tuple[Hashable, Hashable]] = [],
        latents: Iterable[Hashable] = [],
    ):
        """
        Initializes a PDAG class.

        Parameters
        ----------
        directed_ebunch: list, array-like of 2-tuples
            List of directed edges in the PDAG.

        undirected_ebunch: list, array-like of 2-tuples
            List of undirected edges in the PDAG.

        latents: list, array-like
            List of nodes which are latent variables.

        Returns
        -------
        An instance of the PDAG object.

        Examples
        --------
        """
        self.latents = set(latents)
        self.directed_edges = set(directed_ebunch)
        self.undirected_edges = set(undirected_ebunch)

        super(PDAG, self).__init__(
            self.directed_edges.union(self.undirected_edges).union(
                set([(Y, X) for (X, Y) in self.undirected_edges])
            )
        )

    def all_neighbors(self, node):
        """
        Returns a set of all neighbors of a node in the PDAG. This includes both directed and undirected edges.

        Parameters
        ----------
        node: any hashable python object
            The node for which to get the neighboring nodes.

        Returns
        -------
        set: A set of neighboring nodes.

        Examples
        --------
        >>> from pgmpy.base import PDAG
        >>> pdag = PDAG(directed_ebunch=[('A', 'C'), ('D', 'C')], undirected_ebunch=[('B', 'A'), ('B', 'D')])
        >>> pdag.all_neighbors('A')
        {'B', 'C'}
        """
        return {x for x in self.successors(node)} | {x for x in self.predecessors(node)}

    def directed_children(self, node):
        """
        Returns a set of children of node such that there is a directed edge from `node` to child.
        """
        return {x for x in self.successors(node) if (node, x) in self.directed_edges}

    def directed_parents(self, node):
        """
        Returns a set of parents of node such that there is a directed edge from the parent to `node`.
        """
        return {x for x in self.predecessors(node) if (x, node) in self.directed_edges}

    def has_directed_edge(self, u, v):
        """
        Returns True if there is a directed edge u -> v in the PDAG.
        """
        if (u, v) in self.directed_edges:
            return True
        else:
            return False

    def has_undirected_edge(self, u, v):
        """
        Returns True if there is an undirected edge u - v in the PDAG.
        """
        if (u, v) in self.undirected_edges or (v, u) in self.undirected_edges:
            return True
        else:
            return False

    def undirected_neighbors(self, node):
        """
        Returns a set of neighboring nodes such that all of them have an undirected edge with `node`.

        Parameters
        ----------
        node: any hashable python object
            The node for which to get the undirected neighboring nodes.

        Returns
        -------
        set: A set of neighboring nodes.

        Examples
        --------
        >>> from pgmpy.base import PDAG
        >>> pdag = PDAG(directed_ebunch=[('A', 'C'), ('D', 'C')], undirected_ebunch=[('B', 'A'), ('B', 'D')])
        >>> pdag.undirected_neighbors('A')
        {'B'}
        """
        return {var for var in self.successors(node) if self.has_edge(var, node)}

    def is_adjacent(self, u, v):
        """
        Returns True if there is an edge between u and v. This can be either of u - v, u -> v, or u <- v.
        """
        if (u, v) in self.edges or (v, u) in self.edges:
            return True
        else:
            return False

    def copy(self):
        """
        Returns a copy of the object instance.

        Returns
        -------
        Copy of PDAG: pgmpy.dag.PDAG
            Returns a copy of self.
        """
        pdag = PDAG(
            directed_ebunch=list(self.directed_edges.copy()),
            undirected_ebunch=list(self.undirected_edges.copy()),
            latents=self.latents,
        )
        pdag.add_nodes_from(self.nodes())
        return pdag

    def _directed_graph(self):
        """
        Returns a subgraph containing only directed edges.
        """
        dag = nx.DiGraph(self.directed_edges)
        dag.add_nodes_from(self.nodes())
        return dag

    def orient_undirected_edge(self, u, v, inplace=False):
        """
        Orients an undirected edge u - v as u -> v.

        Parameters
        ----------
        u, v: Any hashable python objects
            The node names.

        inplace: boolean (default=False)
            If True, the PDAG object is modified inplace, otherwise a new modified copy is returned.

        Returns
        -------
        None or pgmpy.base.PDAG: The modified PDAG object.
            If inplace=True, returns None and the object itself is modified.
            If inplace=False, returns a PDAG object.
        """

        if inplace:
            pdag = self
        else:
            pdag = self.copy()

        # Remove the edge for undirected_edges.
        if (u, v) in pdag.undirected_edges:
            pdag.undirected_edges.discard((u, v))
        elif (v, u) in pdag.undirected_edges:
            pdag.undirected_edges.discard((v, u))
        else:
            raise ValueError(f"Undirected Edge {u} - {v} not present in the PDAG.")

        # Remove the inverse edge from the graph
        pdag.remove_edge(v, u)

        # Add the edge to directed_edges.
        pdag.directed_edges.add((u, v))

        if not inplace:
            return pdag

    def _check_new_unshielded_collider(self, u, v):
        """
        Tests if orienting an undirected edge u - v as u -> v creates new unshielded V-structures in the PDAG.

        Checks whether v has any directed parents other than u that are not adjacent to u.

        Returns
        -------
        True, if the orientation u -> v would lead to creation of a new V-structure.
        False, if no new V-structures are formed.
        """
        for node in self.directed_parents(v):
            if (node != u) and (not self.is_adjacent(u, node)):
                return True
        return False

    def apply_meeks_rules(self, apply_r4=False, inplace=False, debug=False):
        """
        Applies the Meek's rules to orient the undirected edges of a PDAG to return a CPDAG.

        Parameters
        ----------
        apply_r4: boolean (default=False)
            If True, applies Rules 1 - 4 of Meek's rules.
            If False, applies only Rules 1 - 3.

        inplace: boolean (default=False)
            If True, the PDAG object is modified inplace, otherwise a new modified copy is returned.

        debug: boolean (default=False)
            If True, prints the rules being applied to the PDAG.

        Returns
        -------
        None or pgmpy.base.PDAG: The modified PDAG object.
            If inplace=True, returns None and the object itself is modified.
            If inplace=False, returns a PDAG object.

        Examples
        --------
        >>> from pgmpy.base import PDAG
        >>> pdag = PDAG(directed_ebunch=[('A', 'B')], undirected_ebunch=[('B', 'C'), ('C', 'B')])
        >>> pdag.apply_meeks_rules()
        >>> pdag.directed_edges
        {('A', 'B'), ('B', 'C')}
        """
        if inplace:
            pdag = self
        else:
            pdag = self.copy()

        changed = True
        while changed:
            changed = False

            # Rule 1: If X -> Y - Z and
            #            (X not adj Z) and
            #            (adding Y -> Z doesn't create cycle) and
            #            (adding Y -> Z doesn't create an unshielded collider) =>  Y → Z
            for y in pdag.nodes():
                # Select x's such that there are directed edges x -> y.
                for x in pdag.directed_parents(y):
                    for z in pdag.undirected_neighbors(y):
                        if (
                            (not pdag.is_adjacent(x, z))
                            and (not pdag._check_new_unshielded_collider(y, z))
                            and (not nx.has_path(pdag._directed_graph(), z, y))
                        ):
                            pdag.orient_undirected_edge(y, z, inplace=True)
                            changed = True
                            if debug:
                                logger.info(
                                    f"Applying Rule 1: {x} -> {y} - {z} => {x} -> {y} -> {z}"
                                )

            # Rule 2: If X -> Z -> Y  and X - Y =>  X → Y
            for z in pdag.nodes():
                xs = pdag.directed_parents(z)
                ys = pdag.directed_children(z)

                for x in xs:
                    for y in ys:
                        if pdag.has_undirected_edge(x, y):
                            pdag.orient_undirected_edge(x, y, inplace=True)
                            changed = True
                            if debug:
                                logger.info(
                                    f"Applying Rule 2: {x} -> {z} -> {y} and {x} - {y} => {x} -> {y}"
                                )

            # Rule 3: If X - {Y, Z, W} and {Z, Y} -> W => X -> W
            for x in pdag.nodes():
                undirected_nbs = pdag.undirected_neighbors(x)

                if len(undirected_nbs) < 3:
                    continue

                for y, z, w in itertools.permutations(undirected_nbs, 3):
                    if pdag.has_directed_edge(y, w) and pdag.has_directed_edge(z, w):
                        pdag.orient_undirected_edge(x, w, inplace=True)
                        changed = True
                        if debug:
                            logger.info(
                                f"Applying Rule 3: {x} - {y}, {z}, {w}; {y}, {z} -> {w} => {x} -> {w}"
                            )
                        break

            # Rule 4: If d -> c -> b & a - {b, c, d} and b not adj d => a -> b
            if apply_r4:
                for c in pdag.nodes():
                    directed_graph = pdag._directed_graph()
                    for b in pdag.directed_children(c):
                        for d in pdag.directed_parents(c):
                            if b == d or pdag.is_adjacent(b, d):
                                continue  # b adjacent d => rule not applicable

                            # find nodes a that are undirected neighbor to b, d, and directed or undirected neighbor to c
                            cand = set(pdag.undirected_neighbors(b)).intersection(
                                pdag.all_neighbors(c),
                                pdag.undirected_neighbors(d),
                            )
                            for a in cand:
                                pdag.orient_undirected_edge(a, b, inplace=True)
                                changed = True
                                break
        if not inplace:
            return pdag

    def to_dag(self) -> "DAG":
        """
        Returns one possible DAG which is represented using the PDAG.

        Returns
        -------
        pgmpy.base.DAG: Returns an instance of DAG.

        Examples
        --------
        >>> pdag = PDAG(
        ... directed_ebunch=[("A", "B"), ("C", "B")],
        ... undirected_ebunch=[("C", "D"), ("D", "A")],
        ... )
        >>> dag = pdag.to_dag()
        >>> print(dag.edges())
        OutEdgeView([('A', 'B'), ('C', 'B'), ('D', 'C'), ('A', 'D')])

        References
        ----------
        [1] Dor, Dorit, and Michael Tarsi. "A simple algorithm to construct a consistent extension of a partially oriented graph." Technicial Report R-185, Cognitive Systems Laboratory, UCLA (1992): 45.
        """
        # Add required edges if it doesn't form a new v-structure or an opposite edge
        # is already present in the network.
        dag = DAG()
        # Add all the nodes and the directed edges
        dag.add_nodes_from(self.nodes())
        dag.add_edges_from(self.directed_edges)
        dag.latents = self.latents

        pdag = self.copy()
        while pdag.number_of_nodes() > 0:
            # find node with (1) no directed outgoing edges and
            #                (2) the set of undirected neighbors is either empty or
            #                    undirected neighbors + parents of X are a clique
            found = False
            for X in pdag.nodes():
                directed_outgoing_edges = set(pdag.successors(X)) - set(
                    pdag.predecessors(X)
                )
                undirected_neighbors = set(pdag.successors(X)) & set(
                    pdag.predecessors(X)
                )
                neighbors_are_clique = all(
                    (
                        pdag.has_edge(Y, Z)
                        for Z in pdag.predecessors(X)
                        for Y in undirected_neighbors
                        if not Y == Z
                    )
                )

                if not directed_outgoing_edges and (
                    not undirected_neighbors or neighbors_are_clique
                ):
                    found = True
                    # add all edges of X as outgoing edges to dag
                    for Y in pdag.predecessors(X):
                        dag.add_edge(Y, X)
                    pdag.remove_node(X)
                    break

            if not found:
                logger.warning(
                    "PDAG has no faithful extension (= no oriented DAG with the "
                    + "same v-structures as PDAG). Remaining undirected PDAG edges "
                    + "oriented arbitrarily."
                )
                for X, Y in pdag.edges():
                    if not dag.has_edge(Y, X):
                        try:
                            dag.add_edge(X, Y)
                        except ValueError:
                            pass
                break
        return dag

    def to_graphviz(self) -> object:
        """
        Retuns a pygraphviz object for the DAG. pygraphviz is useful for
        visualizing the network structure.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> model = get_example_model('alarm')
        >>> model.to_graphviz()
        <AGraph <Swig Object of type 'Agraph_t *' at 0x7fdea4cde040>>
        """
        return nx.nx_agraph.to_agraph(self)
