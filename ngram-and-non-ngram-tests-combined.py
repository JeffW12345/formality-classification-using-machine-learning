'''

This module is for tests using a combination of n-grams and other features.

For n-gram only tests, use ngram-only-tests.py.

For non n-gram only tests, use non-ngram-tests-only.

NB The vectorizer that is required will need to be selected. See notes beginning on line 181.

'''
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

corpus = []  # List of sentences in human-readable form (i.e. not in a bag of words representation).
documentClassification = []  # Stored as strings - true = formal, false = informal.
formalityScoreList = []  # Mechanical Turk formality scores.
nonDocumentData = []  # List of lists, each containing a sentence's attribute data.
dataFileFieldNames = []  # The field names from the top of the data spreadsheet.

# This function loads the data from the file and stores it in the data structures shown above.

corpus = []  # List of sentences in human-readable form (i.e. not in a bag of words representation).
documentClassification = []  # Stored as strings - true = formal, false = informal.
formalityScoreList = []  # Mechanical Turk formality scores.
nonDocumentData = []  # List of lists, each containing a sentence's attribute data.
dataFileFieldNames = []  # The field names from the top of the data spreadsheet.

# This function loads the data from the file and stores it in the data structures shown above.
# It is always the first function to be run.


def loadData():
    with open('new_formality_data.csv', encoding='utf-8') as inputFile:
        firstLine = inputFile.readline()
        firstLineAsList = firstLine.split(",")
        # Copy the data file field names into a global list:
        for items in firstLineAsList:
            dataFileFieldNames.append(items)
        # The sentence field is always the final field on the right. Therefore, the sentence index is the number of
        # fields up to and including the one immediately preceding the 'sentence' field.
        sentenceIndex = len(firstLineAsList)-1
        for line in inputFile:
            # Searches through the line for commas, character by character. Stops when 'sentenceIndex' number of commas
            # have been encountered.
            # The document is located to the right of the comma corresponding to index 'sentenceIndex'.
            # Everything to the left of that comma is data relating to the document.
            numCommas = 0
            for character in range(len(line)):
                #  Increments numCommas whenever a comma is encountered in the line.
                if line[character] == ",":
                    numCommas = numCommas + 1
                #  The code below is run when when the number of commas encountered equals the value of 'sentenceIndex'.
                #  When the code below is run, it means that everything on the line to the right of the last comma
                #  encountered is part of the sentence, and not attribute data.
                if numCommas == sentenceIndex:
                    dataExcludingSentence = line[:character]
                    dataExcludingSentenceAsList = dataExcludingSentence.split(",")
                    nonDocumentData.append(dataExcludingSentenceAsList)
                    formalityScore = float(dataExcludingSentenceAsList[2])
                    formalityScoreList.append(formalityScore)
                    # If mechanical Turk formality score >=4 then formal status = true:
                    documentClassification.append(formalityScore >= 4)
                    documentToAdd = line[character + 1:]  # The rest of the current line is comprised of the document
                    documentToAdd.replace('\n', '')  # Removes 'next line' symbol \n from the end of the document
                    # Puts document into a list of Strings:
                    corpus.append(documentToAdd)
                    break  # returns to the outer 'for' loop, so the next line can be processed.
    inputFile.close()
    print("\nNo of records uploaded: ", len(corpus))


# This function takes as its inputs the feature or features to be tested for each sentence, and the
# classifications of the sentences. It performs a machine learning test, and prints statistics relating to the
# test to the console.


def classificationResults(feature, results, featureDescription, classifier):
    #  The two lines below convert the lists passed into the function to arrays.
    X = np.array(feature)
    y = np.array(results)
    #  Splits the data into training and testing sets using 5 split k fold:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
    skf.split(X, y)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    # Fits the data to a model. The model is initially instantiated as SVC so that the definitions of 'classifier' in
    # the 'if' statements below it aren't out of scope of the rest of the module.
    model = SVC(gamma='scale', kernel='linear', probability=True).fit(X_train, y_train)
    if classifier == "Logistic Regression":
        model = LogisticRegression(random_state=0, solver='lbfgs',max_iter=1000).fit(X_train, y_train)
    if classifier == "Multinomial Bayes":
        model = MultinomialNB().fit(X_train, y_train)
    if classifier == "Random forest":
        model = RandomForestClassifier().fit(X_train, y_train)
    # Calls a method to generate a prediction for each sentence, and stores them in a list.
    predictions = model.predict(np.array(X_test))
    # Calculates true positives, true negatives, false positives and false negatives:
    truePositives = 0
    trueNegatives = 0
    falsePositives = 0
    falseNegatives = 0
    numberInList = 0
    for prediction in predictions:
        # Is this a formal sentence which was predicted to be formal?
        if y_test[numberInList] and prediction:
            truePositives = truePositives + 1
        # Is this an informal sentence which was predicted to be informal?
        if not y_test[numberInList] and not prediction:
            trueNegatives = trueNegatives + 1
        # Is this an informal sentence which was predicted to be formal?
        if not y_test[numberInList] and prediction:
            falsePositives = falsePositives + 1
        # Is this a formal sentence which was predicted to be informal?
        if y_test[numberInList] and not prediction:
            falseNegatives = falseNegatives + 1
        numberInList = numberInList + 1
    # Performance metrics
    if (truePositives + trueNegatives + falsePositives + falseNegatives) > 0:
        accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)
    else:
        accuracy = 0
    if (truePositives + falsePositives) > 0:
        precision = truePositives / (truePositives + falsePositives)
    else:
        precision = 0
    if (truePositives + falseNegatives) > 0:
        recall = truePositives / (truePositives + falseNegatives)
    else:
        recall = 0
    if (trueNegatives + falsePositives) > 0:
        fallout = falsePositives / (trueNegatives + falsePositives)  # 'Fallout' is the same as the false positive rate.
    else:
        fallout = 0
    balAccuracy = balanced_accuracy_score(y_test, predictions)
    # Area under roc curve
    y_scores = model.predict_proba(X_test)
    y_scores = y_scores[:, 1]
    rocAreaUnderCurve = roc_auc_score(y_test, y_scores)
    # Console output
    print("\nFeature tested: ", featureDescription)
    print("Classifier: " + classifier, "\n")
    print("Total predictions: ", numberInList)
    print("TRUE POSITIVES: ", truePositives)
    print("FALSE POSITIVES: ", falsePositives)
    print("TRUE NEGATIVES: ", trueNegatives)
    print("FALSE NEGATIVES: ", falseNegatives)
    # Division by zero is illegal, so if the denominator is zero, then 'N/A' is given as the metric's value.
    if accuracy > 0:
        print("Accuracy: %3.2f" % accuracy)
    else:
        print("Accuracy: N/A")
    if precision > 0:
        print("Precision: %3.2f" % precision)
    else:
        print("Precision: N/A")
    if recall > 0:
        print("Recall: %3.2f" % recall)
    else:
        print("Recall: N/A")
    if fallout > 0:
        print("False positive rate: %3.2f" % fallout)
    else:
        print("False positive rate: N/A")
    print("AUC: %3.2f" % rocAreaUnderCurve)
    print("Balanced accuracy: %3.2f" % balAccuracy)

'''
In the function below, the line where corpusVector is instantiated needs to be amended as follows, depending on the
test to be carried out. Remove stop_words='english' if you want to include stop words in the test. 

FOR TF-IDF TESTS:

1. TF-IDF unigram tests:

corpusVector = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))

2. TF-IDF bigram tests

corpusVector = TfidfVectorizer(stop_words='english', ngram_range=(2, 2))

3. TF-IDF trigram tests:

corpusVector = TfidfVectorizer(stop_words='english', ngram_range=(3, 3))

4. TF-IDF unigram and bigram combined tests:

corpusVector = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

5. TF-IDF trigram 1,2,3 tests (unigram, bigram and trigram combined):

corpusVector = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))

FOR NON-BINARY TESTS: 

1. Non-binary unigram tests:

corpusVector = CountVectorizer(binary=False, stop_words='english', ngram_range=(1, 1))

2. Non-binary bigram tests:

corpusVector = CountVectorizer(binary=False, stop_words='english', ngram_range=(2, 2))

3. Non-binary trigram tests:

corpusVector = CountVectorizer(binary=False, stop_words='english', ngram_range=(3, 3))

4. Non-binary unigram and bigram 1,2 tests:

corpusVector = CountVectorizer(binary=False, stop_words='english', ngram_range=(1, 2))

5. Non-binary trigram 1,2,3 tests (unigram, bigram and trigram combined):

corpusVector = CountVectorizer(binary=False, stop_words='english', ngram_range=(1, 3))

FOR BINARY TESTS:

1. Binary unigram tests:

corpusVector = CountVectorizer(binary=True, stop_words='english', ngram_range=(1, 1))

2. Binary bigram tests:

corpusVector = CountVectorizer(binary=True, stop_words='english', ngram_range=(2, 2))

3. Binary trigram tests:

corpusVector = CountVectorizer(binary=True, stop_words='english', ngram_range=(3, 3))

4. Binary unigram and bigram combined tests:

corpusVector = CountVectorizer(binary=True, stop_words='english', ngram_range=(1, 2))

5. Binary trigram 1,2,3 tests:

corpusVector = CountVectorizer(binary=True, stop_words='english', ngram_range=(1, 3))

NB IN THE METHOD BELOW, COMMENT OUT THE CALL TO THE classificationResults FUNCTION FOR ANY CLASSIFIERS THAT 
YOU DON'T REQUIRE RESULTS FOR. 

'''


def testFeaturesIncBagOfWords():
    featureNamesList = []
    # Append features to test to featureNamesList. Uncomment or delete any existing ones not required.
    # The features are the field names as they appear in the first row of the data CSV file.
    featureNamesList.append('Number of adverbs')
    featureNamesList.append('Number of adjectives')
    featureNamesList.append('Number of prepositions')

    # Obtain indexes of features to test and put them in a list
    featureIndexList = []
    for fieldNames in featureNamesList:
        featureIndex = dataFileFieldNames.index(fieldNames)
        featureIndexList.append(featureIndex)
    # Goes through every record in the data file, and adds the relevant data from that record to a list, which is
    # stored in a list of lists (featuresToTestDataList).
    featuresToTestDataList = []  # A list of instances of the variables to test
    numberOfFields = len(featureIndexList)
    for records in nonDocumentData:
        dataThisLine = []
        count = 0
        for references in featureIndexList:
            count = count + 1
            records[references] = float(records[references])
            dataThisLine.append(records[references])
            if count == numberOfFields:
                featuresToTestDataList.append(dataThisLine)
    # Bag of words variables:
    corpusVector = CountVectorizer(binary=True, stop_words='english', ngram_range=(1, 1))  # Amend as applicable
    fittedCorpusVector = corpusVector.fit_transform(corpus)
    corpusVectorAsArray = fittedCorpusVector.toarray()
    # You can add a description of features to be tested in the line below, to keep track of what is being tested.
    featureDescription = "Number of adverbs, adjectives and prepositions plus unigrams"
    results = documentClassification
    # On line below, classifier be changed to 'Logistic Regression', 'Multinomial Bayes' or 'Random forest'.
    classifier = 'Support Vector Machine'
    feature = []
    recordNum = 0
    # Add the line's relevant variable to the end of the bag of words vector for that line and store in feature[].
    for documentBagsOfWords in corpusVectorAsArray:
        feature.append(np.hstack((documentBagsOfWords, featuresToTestDataList[recordNum])))
        recordNum = recordNum + 1
    classificationResults(feature, results, featureDescription, classifier)


# METHOD CALLS

loadData()
testFeaturesIncBagOfWords()
