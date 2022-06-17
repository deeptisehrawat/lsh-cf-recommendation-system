import csv
import math
import sys
import time
import numpy as numpy

from pyspark import SparkContext


def preprocess_data(sc, train_file_name, test_file_name):
    """
    preprocess data for prediction
    :param sc: spark context
    :param train_file_name:
    :param test_file_name:
    :return: useful dictionaries
    """
    raw_rdd_train = sc.textFile(train_file_name)  # read data rdd
    first_line_train = raw_rdd_train.first()  # remove first line
    raw_rdd_train = raw_rdd_train.filter(lambda s: s != first_line_train).map(lambda s: s.split(","))
    raw_rdd_test = sc.textFile(test_file_name)  # read data rdd
    first_line_test = raw_rdd_test.first()  # remove first line
    raw_rdd_test = raw_rdd_test.filter(lambda s: s != first_line_test).map(lambda s: s.split(","))

    # create dictionary of businesses: {business_id1: {user_id1, user_id2}, ..}
    business_user_dict = raw_rdd_train.map(lambda s: (s[1], s[0])).groupByKey().mapValues(set).collectAsMap()
    # create dictionary of users: {user_id1: {business_id1, business_id2}, ..}
    user_business_dict = raw_rdd_train.map(lambda s: (s[0], s[1])).groupByKey().mapValues(set).collectAsMap()

    business_user_rating_dict = raw_rdd_train.map(lambda s: (s[1], (s[0], s[2]))).groupByKey().mapValues(set).collect()
    business_user_rating_dict = {i: dict(j) for i, j in business_user_rating_dict}

    # business average ratings
    zero_value = (0, 0)
    business_average_dict = raw_rdd_train \
        .map(lambda s: (s[1], float(s[2]))) \
        .aggregateByKey(zero_value,
                        lambda curr_sc, next_sc: (curr_sc[0] + next_sc, curr_sc[1] + 1),
                        lambda curr_sc, part_sc: (curr_sc[0] + part_sc[0], curr_sc[1] + part_sc[1])) \
        .mapValues(lambda s: s[0] / s[1]) \
        .collectAsMap()

    user_average_dict = raw_rdd_train \
        .map(lambda s: (s[0], float(s[2]))) \
        .aggregateByKey(zero_value,
                        lambda curr_sc, next_sc: (curr_sc[0] + next_sc, curr_sc[1] + 1),
                        lambda curr_sc, part_sc: (curr_sc[0] + part_sc[0], curr_sc[1] + part_sc[1])) \
        .mapValues(lambda s: s[0] / s[1]) \
        .collectAsMap()

    return raw_rdd_test, business_user_dict, user_business_dict, business_user_rating_dict, business_average_dict, \
        user_average_dict


def find_pearson_similarity(ratings_1, ratings_2):
    """
    find pearson similarity
    :param ratings_1: ratings of business 1
    :param ratings_2: ratings of business 2
    :return: pearson similarity
    """
    if len(ratings_1) == 0 or len(ratings_2) == 0:
        return 0

    ratings_1 = numpy.array(ratings_1, dtype=float)
    ratings_2 = numpy.array(ratings_2, dtype=float)
    avg_1 = numpy.sum(ratings_1) / ratings_1.size
    avg_2 = numpy.sum(ratings_2) / ratings_2.size
    ratings_avg_1 = ratings_1 - avg_1
    ratings_avg_2 = ratings_2 - avg_2

    # element wise product of both arrays
    numerator = numpy.sum(numpy.multiply(ratings_avg_1, ratings_avg_2))
    # squared sum of all elements
    denominator = numpy.sqrt(numpy.sum(numpy.square(ratings_avg_1))) * numpy.sqrt(
        numpy.sum(numpy.square(ratings_avg_2)))
    return 0 if denominator == 0 else numerator / denominator


pearson_dict = dict()


def get_prediction(s, business_user_dict, user_business_dict, business_user_rating_dict, business_average_dict,
                   user_average_dict):
    """
    generate prediction for a user_id, business_id using pearson similarity
    :param s: test input (user_id, business_id)
    :param business_user_dict:
    :param user_business_dict:
    :param business_user_rating_dict:
    :param business_average_dict:
    :param user_average_dict:
    :return: prediction
    """
    neighborhood_size = 15
    default_rating = 3
    default_similarity = 0.2
    sim_arr = [1, 0.75, 0.5, 0.25, 0]
    # user_id, business_id, true_rating = s  # test user_id, business_id
    user_id, business_id = s  # test user_id, business_id

    if business_id not in business_user_dict:
        return user_id, business_id, user_average_dict[user_id]

    if user_id not in user_business_dict:
        return user_id, business_id, default_rating

    # if user_id not in user_business_dict or business_id not in business_user_dict:
    #     return user_id, business_id, default_rating

    pearson_sims = []
    # make it global
    # pearson_dict = dict()
    businesses = user_business_dict[user_id]  # all user rated businesses

    # 1. find pearson correlation for each business pair:
    for curr_business_id in businesses:
        # check if pearson_similarity already exists
        tup = (business_id, curr_business_id)
        key_business_ids = tuple(sorted(tup))
        if key_business_ids in pearson_dict:
            # print('dict double accessed! woo hoo!!!!')
            pearson_similarity = pearson_dict[key_business_ids]
        else:
            users = business_user_dict[business_id].intersection(business_user_dict[curr_business_id])

            if len(users) < 2:
                # find average rating of businesses
                avg_1 = business_average_dict[business_id]
                avg_2 = business_average_dict[curr_business_id]
                pearson_similarity = sim_arr[round(abs(avg_1 - avg_2))]

            # if len(users) == 0:
            #     pearson_similarity = default_similarity
            # elif len(users) == 1:
            #     (user,) = users
            #     pearson_similarity = sim_arr[round(abs(float(business_user_rating_dict[business_id][user]) -
            #                                            float(business_user_rating_dict[curr_business_id][user])))]

            elif len(users) == 2:
                (user_1, user_2) = users
                sim_1 = sim_arr[round(abs(float(business_user_rating_dict[business_id][user_1]) -
                                          float(business_user_rating_dict[curr_business_id][user_1])))]
                sim_2 = sim_arr[round(abs(float(business_user_rating_dict[business_id][user_2]) -
                                          float(business_user_rating_dict[curr_business_id][user_2])))]
                pearson_similarity = (sim_1 + sim_2) / 2
            else:
                ratings_1 = []
                ratings_2 = []
                for curr_user_id in users:
                    ratings_1.append(business_user_rating_dict[business_id][curr_user_id])
                    ratings_2.append(business_user_rating_dict[curr_business_id][curr_user_id])

                # find pearson similarity between business 1 and 2
                pearson_similarity = find_pearson_similarity(ratings_1, ratings_2)
            # add similarity to dict
            pearson_dict[key_business_ids] = pearson_similarity

        if pearson_similarity > 0:
            pearson_sims.append((pearson_similarity, business_user_rating_dict[curr_business_id][user_id]))

    # 2. choose top n neighbors
    # pearson_sims.sort(reverse=True)
    pearson_sims = sorted(pearson_sims, key=lambda tupl: -tupl[0])
    pearson_sims = pearson_sims[:neighborhood_size]

    # 3. find prediction using pearson similarity
    numerator = 0
    denominator = 0
    for similarity, rating in pearson_sims:
        numerator += (similarity * float(rating))
        denominator += abs(similarity)
    prediction = default_rating if denominator == 0 else numerator / denominator

    return user_id, business_id, prediction


def write_output(output_file_name, predictions):
    with open(output_file_name, "w+", newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['user_id', ' business_id', ' prediction'])
        for prediction in predictions:
            writer.writerow([str(prediction[0]), str(prediction[1]), prediction[2]])


def calculate_error(predictions, raw_rdd_test):
    rmse = 0
    error_arr = [0, 0, 0, 0, 0]
    for prediction, true_rating in zip(predictions, raw_rdd_test):
        diff = abs(prediction[2] - float(true_rating[2]))
        if 0 <= diff < 1:
            error_arr[0] += 1
        elif diff < 2:
            error_arr[1] += 1
        elif diff < 3:
            error_arr[2] += 1
        elif diff < 4:
            error_arr[3] += 1
        else:
            error_arr[4] += 1
        rmse += (diff ** 2)
    print('rmse: ', math.sqrt(rmse / len(raw_rdd_test)))
    print('error_arr: ', error_arr)


def execute_task2_1():
    if len(sys.argv) > 2:
        train_file_name = sys.argv[1]
        test_file_name = sys.argv[2]
        output_file_name = sys.argv[3]
    else:
        train_file_name = '../HW3StudentData/yelp_train.csv'
        test_file_name = '../HW3StudentData/yelp_val.csv'
        output_file_name = '../output/output_task2_1.csv'

    sc = SparkContext('local[*]', 'Task 2_1')
    # 1. preprocess data
    raw_rdd_test, business_user_dict, user_business_dict, business_user_rating_dict, business_average_dict, \
        user_average_dict = preprocess_data(sc, train_file_name, test_file_name)

    # 2. predict
    predictions = raw_rdd_test \
        .map(lambda s: get_prediction(s, business_user_dict, user_business_dict, business_user_rating_dict,
                                      business_average_dict, user_average_dict)).collect()

    # 3. write to output file
    write_output(output_file_name, predictions)

    # comment 99 and uncomment line 100 to submit
    # calculate_error(predictions, raw_rdd_test.collect())


if __name__ == '__main__':
    start_time = time.time()
    execute_task2_1()
    print('Execution time: ', time.time() - start_time)
