#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Re-munge data from the FTC Robocall data set
'''

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

import phonenumbers
from phonenumbers import geocoder
from phonenumbers import timezone


def get_time_zone(parsed_number):
    '''
    Pulls out the first timezone given back using the phonenumbers library,
    otherwise just calls it America (based on this dataset).
    '''
    tz = timezone.time_zones_for_number(parsed_number)
    
    if(len(tz) > 2):
        # Got a list of timezones, likely toll free number 
        tz = 'America'
    else:
        tz = tz[0]
        
    return tz


def extract_features(ftc_row):
    '''
    Extracts some chosen features from an FTC Robocall DataFrame row
    '''
    to_parsed =  phonenumbers.parse("+" + ftc_row["TO"], None)
    from_parsed = phonenumbers.parse("+" + ftc_row["FROM"], None)

    # Make fields
    ftc_row["TOVALID"] = phonenumbers.is_valid_number(to_parsed)
    ftc_row["FROMVALID"] = phonenumbers.is_valid_number(from_parsed)
    
    ftc_row["TOSTATE"] = geocoder.description_for_number(to_parsed, "EN")
    ftc_row["FROMSTATE"] = geocoder.description_for_number(from_parsed, "EN")
    
    ftc_row["TOTZ"] = get_time_zone(to_parsed)
    ftc_row["FROMTZ"] = get_time_zone(from_parsed)
    
    ftc_row["TOAREACODE"] = str(to_parsed.national_number)[:3]
    ftc_row["FROMAREACODE"] = str(from_parsed.national_number)[:3]
    
    ftc_row["HOUR"] = ftc_row["DATE/TIME"].hour
    ftc_row["MINUTE"] = ftc_row["DATE/TIME"].minute
    
    # Weekdays are 0-4
    ftc_row["ISWEEKDAY"] = ftc_row["DATE/TIME"].dayofweek in range(5)
    
    # Determined that a call within the first 3 minutes looked like an interesting feature
    ftc_row["WITHIN_THREE_MINUTES"] = ftc_row["MINUTE"] < 3
    
    # FTC's data set 
    ftc_row['LIKELY ROBOCALL'] = ftc_row['LIKELY ROBOCALL'] == 'X'
    
    ftc_row["SAMEAREACODE"] = ftc_row["TOAREACODE"] == ftc_row["FROMAREACODE"]
    
    return ftc_row


# Scoring system for contest
# Not 0-1 loss...
def score(our_predictions, true_results):
    our_score = 0
    for i in range(len(true_results)):
        if (our_predictions[i] == True and true_results[i] == True):
            our_score += 1
        if (our_predictions[i] == True and true_results[i] == False):
            our_score -= 1
    return our_score


# features is only a copy of the dataframe, can't use this
#def label_encode(features, feature_name):
#    feature_encoder = preprocessing.LabelEncoder()
#    features[feature_name] = feature_encoder.fit_transform(features[feature_name])
#    return feature_encoder

def enriched_data_to_features(enriched_data):
    feature_names = [
            "TOTZ",
            "FROMTZ",
            "SAMEAREACODE",
            "WITHIN_THREE_MINUTES",
            "FROMVALID",
            "HOUR",
            #"ISWEEKDAY", # Undecided on whether this will generalize since
                          # training and test data have different weekdays
            
            "NUM_FROM_CALLS",
            "NUM_TO_CALLS",
    ]

    features = enriched_data[feature_names]
    
    # Encode categorical data
    if ("TOTZ" in features):
        totz_encoder = preprocessing.LabelEncoder()
        features["TOTZ"] = totz_encoder.fit_transform(features["TOTZ"])
    if ("FROMTZ" in features):
        fromtz_encoder = preprocessing.LabelEncoder()
        features["FROMTZ"] = fromtz_encoder.fit_transform(features["FROMTZ"])
    if ("HOUR" in features):
        hour_encoder = preprocessing.LabelEncoder()
        features["HOUR"] = hour_encoder.fit_transform(features["HOUR"])
    
    target = enriched_data["LIKELY ROBOCALL"].values
    
    return features, target
    

def train(features, target, c):
    '''
    Trains the robocall classifier using an enriched DataFrame (using the
    enrichment library).

    >>> classifier = train(enriched_data)

    '''

    classifier = RandomForestClassifier(n_estimators=200, 
                                        verbose=0,
                                        n_jobs=-1,
                                        min_samples_split=c,
                                        random_state=1,
                                        oob_score=True)

    classifier.fit(features, target)
    print("Resulting OOB Score: {}".format(classifier.oob_score_))

    return classifier


def massage_dataset(dataset="FTC1.csv"):

    X = pd.read_csv(dataset,
                    parse_dates=["DATE/TIME"],
                    dtype={'TO':str, 'FROM':str})

    # Initial feature extraction
    Y = X.apply(extract_features, axis=1)

    sizes = Y.groupby("FROM").size()

    def get_size(val):
       return sizes[val]

    Y["NUM_FROM_CALLS"] = Y["FROM"].apply(get_size)

    sizes = Y.groupby("TO").size()
    Y["NUM_TO_CALLS"] = Y["TO"].apply(get_size)
    return Y

def main(train="FTC1.csv", test="FTC2.csv"):
    print("Massaging datasets")
    Y = massage_dataset(train)
    print("Set 1 done")
    Z = massage_dataset(test)
    print("Set 2 done")
    
    train_data = Y
    test_data = Z
    
    c = 285 # min samples split
    
    train_features, train_target = enriched_data_to_features(train_data)
    test_features, _ = enriched_data_to_features(test_data)
    
    classifier = train(train_features, train_target, c)
    predictions = classifier.predict(test_features)
    
    X = Z[["FROM", "TO", "DATE/TIME"]]
    
    X["LIKELY ROBOCALL"] = predictions
    
    # Turn back into the format the FTC wanted
    X["LIKELY ROBOCALL"] = X["LIKELY ROBOCALL"].apply(lambda x: "X" if x else "")
    
    X.to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()

