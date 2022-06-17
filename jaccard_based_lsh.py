import sys
import time
import csv

from pyspark import SparkContext
from random import sample
from itertools import combinations


def get_characteristic_matrix(raw_rdd):
    """
    characteristic matrix [(business_id, {user_id1, user_id2}), ..]
    :param raw_rdd:
    :return: key-value pairs of business ids (string) and user ids (set)
    """
    return raw_rdd.map(lambda s: (s[1], s[0])).groupByKey().mapValues(set)


def generate_hash_params(m, n):
    """
    :param m: number of bins / unique users
    :param n: number of hash functions
    :return: hash params
    """
    hash_parameters = dict()
    # hash_parameters['m'] = m
    hash_parameters['a'] = sample(range(1, m), k=n)
    hash_parameters['b'] = sample(range(1, m), k=n)
    return hash_parameters


def get_minhash_signature(char_mat_row, users_dict, hash_parameters, m, n):
    """
    generate minhash signature (list) for every business to create signature matrix
    :param char_mat_row:
    :param users_dict:
    :param hash_parameters:
    :param m: num of bins
    :param n: num of hash functions
    :return: (business_id, {user_id1_minhash_sig, user_id2_minhash_sig})
    """
    # sig_mat:  [('3MntE_HWbNNoyiLGxywjYA', [0, 0, 3]), ('xVEtGucSRLk5pxxN0t4i6g', [0, 2, 2]), ('ZBfp0AT1NOE0ULg3EIYTSA', [0, 0, 1]), ('gTw6PENNGl68ZPUpYWP50A', [0, 1, 0]), ('iAuOpYDfOTuzQ6OPpEiGwA', [0, 0, 0]), ('5j7BnXXvlS69uLVHrY9Upw', [0, 1, 2]), ('jUYp798M93Mpcjys_TTgsQ', [0, 1, 4])]
    # minhash_sig = [min(((hash_parameters['a'][i] * users_dict[user] + hash_parameters['b'][i]) % hash_parameters['p'][i]) % m for user in char_mat_row[1]) for i in range(n)]

    minhash_sig = [min((hash_parameters['a'][i] * users_dict[user] + hash_parameters['b'][i]) % m
                       for user in char_mat_row[1]) for i in range(n)]
    return char_mat_row[0], minhash_sig


def get_banded_signature(s, b, r):
    """
    banding technique
    :param s: row of signature matrix
    :param b: number of bands
    :param r: number of rows per band
    :return: banded signature matrix
    """
    bands = []
    for i in range(0, b):
        bands.append(((i, tuple(s[1][i*r: i*r+r])), [s[0]]))
    return bands


def get_candidate_pairs(s):
    return combinations(sorted(s), 2)


def get_jaccard_similarity(char_mat_map, s):
    users_1 = char_mat_map[s[0]]
    users_2 = char_mat_map[s[1]]
    jaccard_similarity = float(len(users_1 & users_2) / len(users_1 | users_2))
    return s, jaccard_similarity


def write_output(output_file_name, candidate_pairs):
    with open(output_file_name, "w+", newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['business_id_1', ' business_id_2', ' similarity'])
        for candidate in candidate_pairs:
            writer.writerow([str(candidate[0][0]), str(candidate[0][1]), candidate[1]])


def execute_task1():
    if len(sys.argv) > 1:
        input_file_name = sys.argv[1]
        output_file_name = sys.argv[2]
    else:
        input_file_name = './data_small/yelp_train.csv'
        # input_file_name = './HW3StudentData/yelp_train.csv'
        output_file_name = './output/output_task1.csv'

    sc = SparkContext('local[*]', 'Task 1')
    # remove logs
    sc.setLogLevel("OFF")
    # read data rdd
    raw_rdd = sc.textFile(input_file_name)
    # remove first line
    first_line = raw_rdd.first()
    raw_rdd = raw_rdd.filter(lambda s: s != first_line).map(lambda s: s.split(","))

    # 1. create characteristic matrix
    char_mat = get_characteristic_matrix(raw_rdd)
    print("char_mat: ", char_mat.collect())

    # create dictionary of users: {user_id1: index, ..}
    users_dict = raw_rdd.map(lambda s: s[0]).distinct().zipWithIndex().collectAsMap()
    print('users_dict: ', users_dict)

    # 2. hash function: f(x) = ((ax + b) % p) % m or f(x) = (ax + b) % m
    n = 2     # number of hash functions = 100
    m = len(users_dict)      # number of bins
    hash_parameters = generate_hash_params(m, n)
    print('hash parameters: ', hash_parameters)

    # 3. signature matrix
    sig_mat = char_mat.map(lambda s: get_minhash_signature(s, users_dict, hash_parameters, m, n))
    print('sig_mat: ', sig_mat.collect())

    # 4. divide matrix into b bands and r rows: b*r = n
    r = 2       # number of rows per band
    b = n // r      # number of bands
    sig_mat_banding = sig_mat.flatMap(lambda s: get_banded_signature(s, b, r))\
        .reduceByKey(lambda x, y: x+y).filter(lambda s: len(s[1]) > 1)
    print('sig_mat_banding: ', sig_mat_banding.collect())
    # sig_mat_banding:  [((0, (1, 0)), ['xVEtGucSRLk5pxxN0t4i6g', 'iAuOpYDfOTuzQ6OPpEiGwA'])]

    # 5. find candidate pairs
    candidate_pairs = sig_mat_banding.flatMap(lambda s: get_candidate_pairs(s[1])).distinct()
    print('candidate_pairs: ', candidate_pairs.collect())
    # candidate_pairs:  [('xVEtGucSRLk5pxxN0t4i6g', 'iAuOpYDfOTuzQ6OPpEiGwA')]
    similarity_threshold = 0.5
    char_mat_map = char_mat.collectAsMap()
    candidate_pairs = candidate_pairs.map(lambda s: get_jaccard_similarity(char_mat_map, s))\
        .filter(lambda s: s[1] >= similarity_threshold).collect()
    print('candidate_pairs: ', candidate_pairs)
    candidate_pairs.sort()
    print('candidate_pairs: ', candidate_pairs)

    # 6. write to output file
    write_output(output_file_name, candidate_pairs)


if __name__ == '__main__':
    start_time = time.time()
    execute_task1()
    print('Execution time: ', time.time() - start_time)
