from engine import Value
import random
import graphviz

def zero_grad(node : Value):
    node.grad = 0
    for n in node._prev :
        zero_grad(n)

def draw_diagram(node : Value):
    g = graphviz.Digraph('G', filename="graph.svg")
    g.attr(rankdir='LR')
    g.attr(bgcolor='#1a1a2e')
    g.attr('node', fontname='Helvetica', fontsize='10')
    g.attr('edge', color='#4a4a6a', penwidth='1.5')

    visited : set[int] = set()

    def value_label(n : Value):
        return f"{n.data:.4f}\ngrad: {n.grad:.4f}"

    def dfs(n : Value):
        nid = str(id(n))
        if id(n) not in visited:
            visited.add(id(n))

            if n._op:
                # Operation node: small circle with operator symbol
                op_nid = nid + "_op"
                g.node(op_nid, n._op,
                       shape='circle', width='0.4', height='0.4',
                       style='filled', fillcolor='#e94560', fontcolor='white',
                       fontsize='14', fixedsize='true')
                # Result node coming out of the operation
                g.node(nid, value_label(n),
                       shape='circle', width='1.0', height='1.0',
                       style='filled', fillcolor='#533483', fontcolor='white',
                       fixedsize='true')
                g.edge(op_nid, nid, color='#e94560', penwidth='2')

                for child in n._prev:
                    dfs(child)
                    g.edge(str(id(child)), op_nid, color='#4a4a6a', penwidth='1.5')
            else:
                # Leaf node (input or weight)
                g.node(nid, value_label(n),
                       shape='circle', width='1.0', height='1.0',
                       style='filled', fillcolor='#0f3460', fontcolor='white',
                       fixedsize='true')

    dfs(node)
    g.view()
            



if __name__ == "__main__":
    # Lets try to create a simple neural network from what we have
    # AND Neural Net
    dataset = [
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    
    #  a---ha
    #   \ /  \
    #    X    > Output
    #   / \  /
    #  b---hb

    # Inputs
    a = Value(0)
    b = Value(0)
    # Weights
    w_a_ha = Value.random()
    w_a_hb = Value.random()

    w_b_hb = Value.random()
    w_b_ha = Value.random()

    #First hidden layer biases
    h_a_bias = Value.random()
    h_b_bias = Value.random()

    h_a = (a * w_a_ha) + (b * w_b_ha) + h_a_bias
    h_b = (b * w_b_hb) + (a * w_a_hb) + h_b_bias

    # Last node
    w_ha_out = Value.random()
    w_hb_out = Value.random()
    b_out = Value.random()

    out = (h_a * w_ha_out) + (h_b * w_hb_out) + b_out

    out.backward()
    draw_diagram(out)
    print("Cacagrad first run")

