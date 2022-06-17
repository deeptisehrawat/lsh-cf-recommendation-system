import csv
import json
import math
import sys
import time
import numpy as np
import xgboost as xgb

from pyspark import SparkContext
from sklearn.metrics import mean_squared_error


def preprocess_data_item(sc, train_file_name, test_file_name):
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

    return raw_rdd_train, raw_rdd_test, business_user_dict, user_business_dict, business_user_rating_dict, \
        business_average_dict, user_average_dict


def find_pearson_similarity(ratings_1, ratings_2):
    """
    find pearson similarity
    :param ratings_1: ratings of business 1
    :param ratings_2: ratings of business 2
    :return: pearson similarity
    """
    if len(ratings_1) == 0 or len(ratings_2) == 0:
        return 0

    ratings_1 = np.array(ratings_1, dtype=float)
    ratings_2 = np.array(ratings_2, dtype=float)
    avg_1 = np.sum(ratings_1) / ratings_1.size
    avg_2 = np.sum(ratings_2) / ratings_2.size
    ratings_avg_1 = ratings_1 - avg_1
    ratings_avg_2 = ratings_2 - avg_2

    # element wise product of both arrays
    numerator = np.sum(np.multiply(ratings_avg_1, ratings_avg_2))
    # squared sum of all elements
    denominator = np.sqrt(np.sum(np.square(ratings_avg_1))) * np.sqrt(
        np.sum(np.square(ratings_avg_2)))
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
    user_id, business_id, true_rating = s  # test user_id, business_id
    # user_id, business_id = s  # test user_id, business_id

    if business_id not in business_user_dict:
        # return user_id, business_id, user_average_dict[user_id]
        return user_average_dict[user_id]

    if user_id not in user_business_dict:
        # return user_id, business_id, default_rating
        return default_rating

    pearson_sims = []
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

    # return user_id, business_id, prediction
    return prediction


def preprocess_data_model(sc, folder_path):
    """
    preprocess data for prediction
    :param sc: spark context
    :param folder_path:
    :return: useful dictionaries
    """
    # train_file_name = folder_path + '/yelp_train.csv'
    user_file_name = folder_path + '/user.json'
    business_file_name = folder_path + '/business.json'

    # create dictionary of users: {user_id: [param_1, param_2], ..}
    user_dict = sc.textFile(user_file_name).map(json.loads) \
        .map(lambda s: (s['user_id'], (float(s['average_stars']), float(s['review_count']), float(s['useful']),
                                       float(s['fans'])))).collectAsMap()

    # create dictionary of businesses: {business_id: [param_1, param_2], ..}
    business_dict = sc.textFile(business_file_name).map(json.loads) \
        .map(lambda s: (s['business_id'], (float(s['stars']), float(s['review_count'])))).collectAsMap()

    return user_dict, business_dict


def get_features_labels(user_business_rating, user_dict, business_dict):
    """
    get features and labels for training the model and test the model
    :param user_business_rating:
    :param user_dict:
    :param business_dict:
    :return: matrix of features and labels
    """
    user = user_business_rating[0]
    business = user_business_rating[1]
    # present in train and validation, not in test
    rating = -1 if len(user_business_rating) < 3 else user_business_rating[2]
    user_average_stars, user_review_count, useful, fans = (None, None, None, None) if user not in user_dict else user_dict[user]
    business_stars, business_review_count = (None, None) if business not in business_dict else business_dict[business]

    return [user, business, rating, user_average_stars, user_review_count, business_stars, business_review_count, useful, fans]


def write_output(output_file_name, predictions, test_ids):
    with open(output_file_name, "w+", newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['user_id', ' business_id', ' prediction'])
        for test_id, prediction in zip(test_ids, predictions):
            writer.writerow([str(test_id[0]), str(test_id[1]), prediction])


def execute_task2_3():
    if len(sys.argv) > 2:
        folder_path = sys.argv[1]
        test_file_name = sys.argv[2]
        output_file_name = sys.argv[3]
    else:
        # folder_path = './data_small'
        # test_file_name = './data_small/yelp_val.csv'
        # test_file_name = './HW3StudentData/yelp_val_in.csv'
        folder_path = './HW3StudentData'
        test_file_name = './HW3StudentData/yelp_val.csv'
        output_file_name = './output/output_task2_3.csv'

    train_file_name = folder_path + '/yelp_train.csv'

    sc = SparkContext('local[*]', 'Task 2_3')
    # item-item based cf
    # 1. preprocess data
    raw_rdd_train, raw_rdd_test, business_user_dict, user_business_dict, business_user_rating_dict, \
        business_average_dict, user_average_dict = preprocess_data_item(sc, train_file_name, test_file_name)

    # 2. predict
    predictions = raw_rdd_test\
        .map(lambda s: get_prediction(s, business_user_dict, user_business_dict, business_user_rating_dict,
                                      business_average_dict, user_average_dict)).collect()

    # model based
    # 1. preprocess data
    user_dict, business_dict = preprocess_data_model(sc, folder_path)

    # 2. get train and test dataset
    fl_train = np.array(raw_rdd_train.map(lambda s: get_features_labels(s, user_dict, business_dict)).collect())
    fl_test = np.array(raw_rdd_test.map(lambda s: get_features_labels(s, user_dict, business_dict)).collect())

    x_train, x_test, y_train, y_actual = np.array(fl_train[:, 3:], dtype='f'), np.array(fl_test[:, 3:], dtype='f'), \
                                         np.array(fl_train[:, 2], dtype='f'), np.array(fl_test[:, 2], dtype='f')

    # 3. fit model and predict
    xgbr = xgb.XGBRegressor()
    xgbr.fit(x_train, y_train)
    y_predictions = xgbr.predict(x_test)

    predictions = np.array(predictions, dtype='f')
    alpha = 0.1
    hybrid_predictions = alpha * predictions + (1 - alpha) * y_predictions

    # 4. write to output file
    write_output(output_file_name, hybrid_predictions, fl_test[:, :2])

    # comment 102 and uncomment line 103 to submit
    # 5. calculate rmse
    # rms = math.sqrt(mean_squared_error(y_actual, y_predictions))
    rms = mean_squared_error(y_actual, hybrid_predictions, squared=False)
    print('rmse: ', rms)


if __name__ == '__main__':
    start_time = time.time()
    execute_task2_3()
    print('Execution time: ', time.time() - start_time)
