import csv
import json
import math
import sys
import time
import numpy as np
import xgboost as xgb

from pyspark import SparkContext
from sklearn.metrics import mean_squared_error


def preprocess_data(sc, folder_path, test_file_name):
    """
    preprocess data for prediction
    :param sc: spark context
    :param folder_path:
    :param test_file_name:
    :return: useful dictionaries
    """
    train_file_name = folder_path + '/yelp_train.csv'
    user_file_name = folder_path + '/user.json'
    business_file_name = folder_path + '/business.json'

    raw_rdd_train = sc.textFile(train_file_name)  # read data rdd
    first_line_train = raw_rdd_train.first()  # remove first line
    raw_rdd_train = raw_rdd_train.filter(lambda s: s != first_line_train).map(lambda s: s.split(","))
    raw_rdd_test = sc.textFile(test_file_name)  # read data rdd
    first_line_test = raw_rdd_test.first()  # remove first line
    raw_rdd_test = raw_rdd_test.filter(lambda s: s != first_line_test).map(lambda s: s.split(","))

    # business features 'business_star', 'latitude', 'longitude', 'business_review_cnt', and
    # user features; 'user_review_cnt',
    # 'useful', 'cool', 'funny', 'fans', 'user_avg_star' to train the Gradient Boosting model

    # create dictionary of users: {user_id: [param_1, param_2], ..}
    user_dict = sc.textFile(user_file_name).map(json.loads) \
        .map(lambda s: (s['user_id'], (float(s['average_stars']), float(s['review_count']), float(s['useful']),
                                       float(s['fans'])))).collectAsMap()

    # create dictionary of businesses: {business_id: [param_1, param_2], ..}
    business_dict = sc.textFile(business_file_name).map(json.loads) \
        .map(lambda s: (s['business_id'], (float(s['stars']), float(s['review_count'])))).collectAsMap()

    return raw_rdd_train, raw_rdd_test, user_dict, business_dict


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


def execute_task2_2():
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
        output_file_name = './output/output_task2_2.csv'

    sc = SparkContext('local[*]', 'Task 2_2')
    # 1. preprocess data
    raw_rdd_train, raw_rdd_test, user_dict, business_dict = preprocess_data(sc, folder_path, test_file_name)

    # 2. get train and test dataset
    fl_train = np.array(raw_rdd_train.map(lambda s: get_features_labels(s, user_dict, business_dict)).collect())
    fl_test = np.array(raw_rdd_test.map(lambda s: get_features_labels(s, user_dict, business_dict)).collect())

    x_train, x_test, y_train, y_actual = np.array(fl_train[:, 3:], dtype='f'), np.array(fl_test[:, 3:], dtype='f'), \
                                         np.array(fl_train[:, 2], dtype='f'), np.array(fl_test[:, 2], dtype='f')

    # 3. fit model and predict
    xgbr = xgb.XGBRegressor()
    xgbr.fit(x_train, y_train)
    y_predictions = xgbr.predict(x_test)

    # 4. write to output file
    write_output(output_file_name, y_predictions, fl_test[:, :2])

    # 5. calculate rmse
    # rms = math.sqrt(mean_squared_error(y_actual, y_predictions))
    rms = mean_squared_error(y_actual, y_predictions, squared=False)
    print('rmse: ', rms)


if __name__ == '__main__':
    start_time = time.time()
    execute_task2_2()
    print('Execution time: ', time.time() - start_time)
