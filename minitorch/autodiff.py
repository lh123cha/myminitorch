from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
import queue

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # f'(x) = f(x+\epsilon)-f(x-\epsilon)/2*\epsilon
    args = [i for i in vals]
    args[arg] += epsilon
    f1 = f(*args)
    args[arg] -= 2*epsilon
    f2 = f(*args)
    return (f1-f2)/(2*epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    Visited = []
    result = []

    def visit(n: Variable):
        if n.is_constant():
            return
        if n.unique_id in Visited:
            return
        if not n.is_leaf():
            for input in n.history.inputs:
                visit(input)
        Visited.append(n.unique_id)
        result.insert(0, n)

    visit(variable)
    return result
    # q = queue.Queue()
    # res = []
    # Visited = []
    # q.put(variable)
    # Visited.append(variable.unique_id)
    # res.insert(0,variable)
    # while not q.empty():
    #     temp = q.get()
    #     if temp.is_constant():
    #         break
    #     if temp.unique_id in Visited:
    #         break
    #     if not temp.is_leaf():
    #         for t in temp.parents:
    #             q.put(t)
    #     Visited.append(temp.unique_id)
    #     res.insert(0,temp)
    # return res
    # raise NotImplementedError('Need to implement for Task 1.4')


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    compute_graph = topological_sort(variable=variable)
    node_to_grad = {}
    node_to_grad[variable.unique_id]=deriv
    #获得计算图
    for node in compute_graph:
        if node.is_leaf():
            continue
        if node.unique_id in node_to_grad.keys():
            grade = node_to_grad[node.unique_id]
        #获得Variable v_i的inputs
        inputs = node.chain_rule(grade)
        for var,item in inputs:
            #backward()就相当于是v_i*\frac{\partial v_i}{\partial v_k}
            if var.is_leaf():
                var.accumulate_derivative(item)
                continue
            if var.unique_id in node_to_grad.keys():
                #append v_ki to node_to_grad[k]
                node_to_grad[var.unique_id] +=item
            else:
                node_to_grad[var.unique_id] = item


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
