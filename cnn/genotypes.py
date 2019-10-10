from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

# MSE 2019-10-06: Alias CIFAR10 to DARTS; this architecture is really the best discovered for Cifar-10
CIFAR_10 = DARTS_V2

# MSE 2019-10-05: copy the best discovered architecture from NAS for mnist and fashion-mnist
# The way to get these is to look at the log.txt in the directory, e.g. search-EXP-20191004-160833
# train_search.py periodically logs out the best genotype discovered.  copy / paste the last one here.

# Best architecture for mnist
MNIST = Genotype(
	normal=[
	('max_pool_3x3', 0), 
	('skip_connect', 1), 
	('max_pool_3x3', 0), 
	('sep_conv_3x3', 2), 
	('sep_conv_3x3', 3), 
	('sep_conv_3x3', 2), 
	('sep_conv_5x5', 4), 
	('sep_conv_3x3', 3)], 
	normal_concat=range(2, 6), 
	reduce=[
	('avg_pool_3x3', 0), 
	('avg_pool_3x3', 1), 
	('avg_pool_3x3', 0), 
	('avg_pool_3x3', 1), 
	('avg_pool_3x3', 0), 
	('dil_conv_5x5', 3), 
	('avg_pool_3x3', 0), 
	('dil_conv_5x5', 4)], 
	reduce_concat=range(2, 6))
	
# Best architecture for fashion-mnist
FASHION_MNIST = Genotype(
	normal=[
	('skip_connect', 0), 
	('sep_conv_5x5', 1), 
	('dil_conv_5x5', 1), 
	('skip_connect', 0), 
	('sep_conv_5x5', 3), 
	('sep_conv_3x3', 0), 
	('sep_conv_3x3', 4), 
	('sep_conv_5x5', 1)], 
	normal_concat=range(2, 6), 
	reduce=[
	('avg_pool_3x3', 0), 
	('dil_conv_3x3', 1), 
	('avg_pool_3x3', 0), 
	('skip_connect', 2), 
	('avg_pool_3x3', 0), 
	('dil_conv_5x5', 2), 
	('avg_pool_3x3', 0), 
	('skip_connect', 2)], 
	reduce_concat=range(2, 6))

# Best architecture for graphene
GRAPHENE = Genotype(
	normal=[
	('avg_pool_3x3', 1), 
	('skip_connect', 0), 
	('avg_pool_3x3', 2), 
	('skip_connect', 1), 
	('avg_pool_3x3', 3), 
	('avg_pool_3x3', 2), 
	('avg_pool_3x3', 3), 
	('skip_connect', 4)], 
	normal_concat=range(2, 6), 
	reduce=[
	('skip_connect', 0), 
	('skip_connect', 1), 
	('sep_conv_5x5', 2), 
	('sep_conv_3x3', 0), 
	('avg_pool_3x3', 3), 
	('avg_pool_3x3', 2), 
	('sep_conv_5x5', 4), 
	('sep_conv_3x3', 2)], 
	reduce_concat=range(2, 6))