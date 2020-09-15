'''
This module is purely for tests not involving n-grams.

For tests using purely n-grams, please use ngram-only-tests.py.

For tests involving n-grams and other features combined, using ngram-and-non-ngram-tests-combined.py.

'''
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

corpus = []  # List of sentences in human-readable form (i.e. not in a bag of words representation).
documentClassification = []  # Stored as strings - true = formal, false = informal.
formalityScoreList = []  # Mechanical Turk formality scores.
nonDocumentData = []  # List of lists, each containing a sentence's attribute data.
dataFileFieldNames = []  # The field names from the top of the data spreadsheet.
corpus = []  # List of sentences in human-readable form (i.e. not in a bag of words representation).
documentClassification = []  # Stored as strings - true = formal, false = informal.
formalityScoreList = []  # Mechanical Turk formality scores.
nonDocumentData = []  # List of lists, each containing a sentence's attribute data.
dataFileFieldNames = []  # The field names from the top of the data spreadsheet.
fieldsToSelectFrom = []  # List of features that the user has not selected for the test.
chosenFields = []  # List of features that the user has selected for the test.
classifier = ""  # The classifier to be used.

# This function loads the data from the file and stores it in the data structures


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
    if classifier == "Random Forest":
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


# Puts all feature field names into list 'fieldsToSelectFrom'.

def createFeatureFieldList():
    count = 0
    for fieldName in dataFileFieldNames:
        if count < (len(dataFileFieldNames)-1):
            fieldsToSelectFrom.append(fieldName)
            count = count + 1

# Prints a list of fields that are available (excludes fields already selected by the user)


def printAvailableFields():
    count = 1
    print("\nYou can add the following fields: \n")
    for fieldName in fieldsToSelectFrom:
        print(count, "-", fieldName)
        count = count + 1

# Asks the user to choose the features they want to test. Stores field names in 'chosenFields'.

def askForFeatures():
    featureChoice = ""
    if not chosenFields:  # If no selections yet made by the user.
        printAvailableFields()
        print("\nNo features have been selected yet")
        featureChoice = input("\nPlease choose the number of the feature you wish to add: ")
        if featureChoice.isnumeric():
            featureChoice = int(featureChoice)
            # If a valid selection is made, adds the field name to chosenFields and removes it from fieldsToSelect.
            if 0 <= featureChoice <= len(fieldsToSelectFrom):
                chosenFields.append(fieldsToSelectFrom[featureChoice - 1])
                fieldsToSelectFrom.remove(fieldsToSelectFrom[featureChoice - 1])
                askForFeatures()
            else:
                print("You did not enter a valid number. Please try again.")
                askForFeatures()
        else:
            print("You did not enter a number. Please try again.")
            askForFeatures()
    #  If the user has made at least one selection already
    else:
        printAvailableFields()
        print("You have added the following features: ")
        for fields in chosenFields:
            print(fields)
        printAvailableFields()
        featureChoice = input("Please choose an additional feature or press C to select your classifier: ")
        if featureChoice.isnumeric():
            featureChoice = int(featureChoice)
            # If a valid selection is made, adds the field name to chosenFields and removes it from fieldsToSelect.
            if 0 <= featureChoice <= len(fieldsToSelectFrom):
                chosenFields.append(fieldsToSelectFrom[featureChoice - 1])
                fieldsToSelectFrom.remove(fieldsToSelectFrom[featureChoice - 1])
                askForFeatures()
            else:
                print("You did not enter a valid number. Please try again.")
                askForFeatures()
        # Pressing 'C' exits the function.
        elif featureChoice == "C":
            return
        #  If neither 'C' nor a number entered:
        else:
            print("You did not enter a number. Please try again.")
            askForFeatures()


# Asks the user to select a classifier.


def askForClassifier():
    print("\n The classifiers are: ")
    print("1 - Support Vector Machine")
    print("2 - Logistic Regression")
    print("3 - Multinomial Bayes")
    print("4 - Random Forest")
    classifierChoice = input("\n Please choose a classifier by typing a number between 1 and 4: ")
    if classifierChoice.isnumeric():
        classifierChoice = int(classifierChoice)
        if classifierChoice == 1:
            print("You have selected Support Vector Machine")
            global classifier
            classifier = "Support Vector Machine"
        if classifierChoice == 2:
            print("You have selected Logistic Regression")
            classifier = "Logistic Regression"
        if classifierChoice == 3:
            print("You have selected Multinomial Bayes")
            classifier = "Multinomial Bayes"
        if classifierChoice == 4:
            print("You have selected Random Forest")
            classifier = "Random Forest"
        else:
            print("That was not a valid selection. Please try again.")
            askForClassifier()
    else:
        print("That was not a valid selection. Please try again.")
        askForClassifier()
    print("\n Please choose a classifier")


def setParameters():
    createFeatureFieldList()  # Puts all feature field names into list 'fieldsToSelectFrom'.
    askForFeatures()  # Asks the user to choose the features they want to test. Stores field names in 'chosenFields'.
    askForClassifier()  # Asks the user to choose a classifier. Stores the result in global variable 'classifier'.
    # Obtains the index positions of the features above from the list of field headers, to test and puts the indexes in
    # a newly created list, featureIndexList.
    featureIndexList = []
    for fieldName in chosenFields:
        featureIndex = dataFileFieldNames.index(fieldName)
        featureIndexList.append(featureIndex)
    # Goes through every record in the data file, and adds the relevant data from that record to a list, which is
    # stored in a list of lists (featuresToTestDataList).
    featuresToTestDataList = []  # A list of instances of the variables to test
    numberOfFields = len(featureIndexList)
    for record in nonDocumentData:
        dataThisLine = []
        count = 0
        for references in featureIndexList:
            count = count + 1
            record[references] = float(record[references])
            dataThisLine.append(record[references])
            if count == numberOfFields:
                featuresToTestDataList.append(dataThisLine)
    featureDescription = chosenFields
    results = documentClassification
    feature = featuresToTestDataList
    classificationResults(feature, results, featureDescription, classifier)


# METHOD CALLS

loadData()
setParameters()