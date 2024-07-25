import ast
import numpy as np
import uuid

class TypeHintRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.returns = None
        if hasattr(node, 'args') and node.args.args:
            for arg in node.args.args:
                arg.annotation = None
        return node

    def visit_AnnAssign(self, node):
        return ast.Assign(
            targets=[node.target],
            value=node.value,
            lineno=node.lineno,
            end_lineno=node.end_lineno,
            col_offset=node.col_offset,
            end_col_offset=node.end_col_offset,
        )

class CleanupVisitor(ast.NodeTransformer):
    def visit(self, node):
        for attr in ['uid', 'parents', 'parent']:
            if hasattr(node, attr):
                delattr(node, attr)
        super().visit(node)
        return node

class ParentageVisitor(ast.NodeTransformer):
    def __init__(self, remove_ast_nodes):
        self.remove_ast_nodes = remove_ast_nodes
        self.parent = None
        self.edges = []

    def visit(self, node):
        name = type(node).__name__
        if name in self.remove_ast_nodes:
            return None
        node.uid = int(uuid.uuid4().node)
        node.parent = self.parent
        if not hasattr(node, 'parents'):
            node.parents = []
        node.parents.append(node.parent)
        self.parent = node
        super().visit(node)
        if isinstance(node, ast.AST):
            self.parent = node.parent
        return node

class GraphVisitor(ast.NodeVisitor):
    def __init__(self):
        self.edges = []
        self.vertices = {}

    def visit(self, node):
        if node.parent is not None:
            self.process_node(node)
        else:
            self.vertices[node.uid] = type(node).__name__
        self.generic_visit(node)

    def process_node(self, node):
        name = type(node).__name__
        if len(node.parents) == 1:
            self.vertices[node.uid] = name
            self.edges.append((node.parent.uid, node.uid))
            return
        for parent in node.parents:
            uid = node.uid + parent.uid
            if uid not in self.vertices:
                self.vertices[uid] = name
                self.edges.append((parent.uid, uid))

def preprocess(tree, removals=[]):
    for visitor in [
        TypeHintRemover(),
        CleanupVisitor(),
        ParentageVisitor(removals),
    ]:
        update = visitor.visit(tree)
        if update:
            tree = update
    return tree

def graph(code, removals=[]):
    tree = ast.parse(code)
    tree = preprocess(tree, removals + ["Load", "Store", "alias", "Import", "ImportFrom"])
    visitor = GraphVisitor()
    visitor.visit(tree)
    vertices = {(key,): (value,) for key, value in visitor.vertices.items()}
    edges = [((source,), (dest,)) for source, dest in visitor.edges]
    return vertices, edges

def lift(vertices, edges, type):
    graph = set()
    v = dict()
    for source, middle in edges:
        for m, destination in edges:
            if middle == m:
                es = source + (middle[-1],)
                ed = middle + (destination[-1],)
                graph.add((es, ed))
                v[es] = type(source) + (type(middle)[-1],)
                v[ed] = type(middle) + (type(destination)[-1],)
    return v, graph

def markov(vertices, edges, type):
    vertypes = set(type(vertice) for vertice in vertices if type(vertice))
    chain_edges = set()
    for vertype in vertypes:
        targets = set(vd for vs, vd in edges if type(vs) and type(vs) == vertype)
        tartypes = set(type(target) for target in targets if type(target))
        for tartype in tartypes:
            weight = len([target for target in targets if type(target) and type(target) == tartype]) / len(targets)
            chain_edges.add((vertype, tartype, weight))
    return vertypes, chain_edges

def adjacency(edges, nodes):
    n = len(nodes)
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for vs, vd, weight in edges:
                if vs == nodes[i] and vd == nodes[j]:
                    adjacency[i][j] = weight
                    break
    return adjacency

def vector(adjacency):
    return adjacency.reshape((len(adjacency) ** 2,))

def vectorize(codes, graph, markov):
    chains = []
    nodes = set()
    done = 1
    for code in codes:
        if not done % 500:
            print(f'Processed {done} snippets...')
        vertices, edges = graph(code)
        vertices, edges = markov(vertices, edges)
        chains.append(edges)
        nodes |= vertices
        done += 1
    nodes = sorted(list(nodes))
    yield nodes
    done = 1
    for edges in chains:
        if not done % 100:
            print(f'Vectorized {done} chains...')
        matrix = adjacency(edges, nodes)
        yield vector(matrix)
        done += 1
