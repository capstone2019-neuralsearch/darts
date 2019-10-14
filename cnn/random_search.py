from genotypes import Genotype, PRIMITIVES
from random import choice
from visualize import plot

# For visualization on windows...
import os
os.environ["PATH"] += os.pathsep + "C:/Users/Julien/Miniconda3/envs/capstone/Library/bin/graphviz"

usable_primitives = PRIMITIVES[1:]

def generate_random_genotype(n_units=4, operations=None):
    """ Generates a random genotype.

    input:
        - n_units: Number of nodes in the cell
        - operations: List of all operations to sample from. All primitives
        except None as defined in genotypes.py by default.
    
    output:
        - Random genotype with n_units units. All operations are uniformly
        randomly sampled from operations. Input nodes are randomly sampled from
        all nodes except the end of the edge one.
    """

    if operations is None:
        operations = usable_primitives
    
    # all_chosen_ops[0] = Normal, all_chosen_ops[1] = Reduce
    all_chosen_ops = [[], []]

    def random_op(i):
        # An operation that goes to i can't start from i itself
        op = choice(operations)
        starting_point = choice([j for j in range(n_units) if j != i + 2])
        return op, starting_point

    for i in range(n_units):
        normal_op1, normal_starting_point1 = random_op(i)
        normal_op2, normal_starting_point2 = random_op(i)
        reduce_op1, reduce_starting_point1 = random_op(i)
        reduce_op2, reduce_starting_point2 = random_op(i)

        all_chosen_ops[0] += [(normal_op1, normal_starting_point1),
                              (normal_op2, normal_starting_point2)]
        all_chosen_ops[1] += [(reduce_op1, reduce_starting_point1),
                              (reduce_op2, reduce_starting_point2)]

    random_genotype = Genotype(
        normal=all_chosen_ops[0],
        normal_concat=range(2, 2 + n_units),  # TODO: Check this is correct.
        reduce=all_chosen_ops[1],
        reduce_concat=range(2, 2 + n_units),
    )

    return random_genotype


if __name__ == "__main__":
    if not os.path.exists("random_genotypes"):
        os.mkdir("random_genotypes")
    for i in range(5):
        genotype = generate_random_genotype(n_units=4)
        with open("random_genotypes/random_genotype%i.txt" % i, "w") as f:
            f.write(str(genotype))
        plot(genotype.normal, "random_genotypes/random_genotype%i_normal" % i)
        plot(genotype.reduce, "random_genotypes/random_genotype%i_reduce" % i)
