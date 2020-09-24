'''

Calculates the McNemar statistic and the corresponding p value from two sets of machine learning test results.

Some of the code below is adapted from code at https://machinelearningmastery.com/mcnemars-test-for-machine-learning/.

'''

from statsmodels.stats.contingency_tables import mcnemar


# The function below asks the user for data relating to the two sets of machine learning test results and assigns the
# data to the correct variable.
# 'nameOfSetOfResults' refers whether the data set is for the first or second set of data.
# 'metricName' refers to true positive, true negative, etc.
#  The function asks the user to try again if a non-numeric value is entered.
def askForData(nameOfSetOfResults, metricName):
    global truePositivesFirstSet, falsePositivesFirstSet, trueNegativesFirstSet, falseNegativesFirstSet, \
        truePositivesSecondSet, falsePositivesSecondSet, trueNegativesSecondSet, falseNegativesSecondSet
    toAsk = "For the " + nameOfSetOfResults + " set of data, how many " + metricName + " are there? "
    userInput = input(toAsk)

    # True positives, first data set
    if nameOfSetOfResults == "first" and metricName == "true positives" and userInput.isnumeric():
        truePositivesFirstSet = int(userInput)
        return
    if nameOfSetOfResults == "first" and metricName == "true positives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(nameOfSetOfResults, metricName)

    # False positives, first data set
    if nameOfSetOfResults == "first" and metricName == "false positives" and userInput.isnumeric():
        falsePositivesFirstSet = int(userInput)
        return
    if nameOfSetOfResults == "first" and metricName == "false positives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(nameOfSetOfResults, metricName)

    # True negatives, first data set
    if nameOfSetOfResults == "first" and metricName == "true negatives" and userInput.isnumeric():
        trueNegativesFirstSet = int(userInput)
        return
    if nameOfSetOfResults == "first" and metricName == "true negatives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(nameOfSetOfResults, metricName)

    # False negatives, first data set
    if nameOfSetOfResults == "first" and metricName == "false negatives" and userInput.isnumeric():
        falseNegativesFirstSet = int(userInput)
        return
    if nameOfSetOfResults == "first" and metricName == "false negatives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(nameOfSetOfResults, metricName)

    # True positives, second data set
    if nameOfSetOfResults == "second" and metricName == "true positives" and userInput.isnumeric():
        truePositivesSecondSet = int(userInput)
        return
    if nameOfSetOfResults == "second" and metricName == "true positives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(nameOfSetOfResults, metricName)

    # False positives, second data set
    if nameOfSetOfResults == "second" and metricName == "false positives" and userInput.isnumeric():
        falsePositivesSecondSet = int(userInput)
        return
    if nameOfSetOfResults == "second" and metricName == "false positives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(nameOfSetOfResults, metricName)

    # True negatives, second data set
    if nameOfSetOfResults == "second" and metricName == "true negatives" and userInput.isnumeric():
        trueNegativesSecondSet = int(userInput)
        return
    if nameOfSetOfResults == "second" and metricName == "true negatives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(nameOfSetOfResults, metricName)

    # False negatives, second data set
    if nameOfSetOfResults == "second" and metricName == "false negatives" and userInput.isnumeric():
        falseNegativesSecondSet = int(userInput)
        return
    if nameOfSetOfResults == "second" and metricName == "false negatives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(nameOfSetOfResults, metricName)


# Function calls to ask user for data input
askForData("first", "true positives")
askForData("first", "false positives")
askForData("first", "true negatives")
askForData("first", "false negatives")
askForData("second", "true positives")
askForData("second", "false positives")
askForData("second", "true negatives")
askForData("second", "false negatives")

#  Console summary of data just entered.

print("\nSUMMARY OF DATA ENTERED")
print("-------------------------")
print("First data set - true positives:", truePositivesFirstSet)
print("First data set - false positives:", falsePositivesFirstSet)
print("First data set - true negatives:", trueNegativesFirstSet)
print("First data set - false negatives:", falseNegativesFirstSet)
print("Second data set - true positives:", truePositivesSecondSet)
print("Second data set - false positives:", falsePositivesSecondSet)
print("Second data set - true negatives:", trueNegativesSecondSet)
print("Second data set - false negatives:", falseNegativesSecondSet)

#  Number of correct and incorrect predictions for each set of results.

# Correct predictions for first set of data
correctFirstSet = int(truePositivesFirstSet) + int(trueNegativesFirstSet)
# Incorrect predictions for first set of data
incorrectFirstSet = int(falsePositivesFirstSet) + int(falseNegativesFirstSet)
# Correct predictions for second set of data
correctSecondSet = int(truePositivesSecondSet) + int(trueNegativesSecondSet)
# Incorrect predictions for second set of data
incorrectSecondSet = int(falsePositivesSecondSet) + int(falseNegativesSecondSet)

print("\nFor the first set of data, there are", correctFirstSet, "correct predictions and", incorrectFirstSet,
      "incorrect predictions.")
print("\nFor the second set of data, there are", correctSecondSet, "correct predictions and", incorrectSecondSet,
      "incorrect predictions.")

# Contingency table
table = [[correctFirstSet + correctSecondSet, correctFirstSet + incorrectSecondSet],
         [incorrectFirstSet + correctSecondSet, incorrectFirstSet + incorrectSecondSet]]

# Calculate mcNemar test
result = mcnemar(table, exact=False, correction=True)

# McNemar stats and interpretation
print('\nMcNemar co-efficient value = %.3f, p-value = %.3f' % (result.statistic, result.pvalue))
alpha = 0.05
if result.pvalue > alpha:
    print('\nThe results ARE NOT significantly different, based on a p value of 5 percent.')
else:
    print('\nThe results ARE significantly different, based on a p value of 5 percent.')
