'''

Calculates the McNemar statistic and states the p-value of the result.

Some of the code below is adapted from httruePositivess://machinelearningmastery.com/mcnemars-test-for-machine-learning/.

'''

from statsmodels.stats.contingency_tables import mcnemar

# The function below asks the user for data relating to machine learning tests and assigns it to the correct variable.
# Asks the user to try again if a non-numeric value is entered.
def askForData(position, description):
    # In variables below, truePositives 1 = number of true positives for the first test, true Positives2 = number of true
    # positives for the second test, etc.
    global truePositives1, falsePositives1, trueNegatives1, falseNegatives1, truePositives2, falsePositives2, \
        trueNegatives2, falseNegatives2

    toAsk = "For the " + position + " set of data, how many " + description + " are there? "
    userInput = input(toAsk)

    # True positives, first data set
    if position == "first" and description == "true positives" and userInput.isnumeric():
        truePositives1 = int(userInput)
        return
    if position == "first" and description == "true positives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(position, description)

    # False positives, first data set
    if position == "first" and description == "false positives" and userInput.isnumeric():
        falsePositives1 = int(userInput)
        return
    if position == "first" and description == "false positives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(position, description)

    # True negatives, first data set
    if position == "first" and description == "true negatives" and userInput.isnumeric():
        trueNegatives1 = int(userInput)
        return
    if position == "first" and description == "true negatives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(position, description)

    # False negatives, first data set
    if position == "first" and description == "false negatives" and userInput.isnumeric():
        falseNegatives1 = int(userInput)
        return
    if position == "first" and description == "false negatives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(position, description)

    # True positives, second data set
    if position == "second" and description == "true positives" and userInput.isnumeric():
        truePositives2 = int(userInput)
        return
    if position == "second" and description == "true positives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(position, description)

    # False positives, second data set
    if position == "second" and description == "false positives" and userInput.isnumeric():
        falsePositives2 = int(userInput)
        return
    if position == "second" and description == "false positives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(position, description)

    # True negatives, second data set
    if position == "second" and description == "true negatives" and userInput.isnumeric():
        trueNegatives2 = int(userInput)
        return
    if position == "second" and description == "true negatives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(position, description)

    # False negatives, second data set
    if position == "second" and description == "false negatives" and userInput.isnumeric():
        falseNegatives2 = int(userInput)
        return
    if position == "second" and description == "false negatives" and not userInput.isnumeric():
        print("You need to enter an integer. Please try again.")
        askForData(position, description)


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
print("First data set - true positives: ")
print("First data set - false positives: ")
print("First data set - true negatives: ")
print("First data set - false negatives: ")
print("Second data set - true positives: ")
print("Second data set - false positives: ")
print("Second data set - true negatives: ")
print("Second data set - false negatives: ")

#  Number of correct and incorrect predictions for each test.

correct1 = int(truePositives1) + int(trueNegatives1)  # Correct predictions for first set of data
incorrect1 = int(falsePositives1) + int(falseNegatives1)  # Incorrect predictions for first set of data
correct2 = int(truePositives2) + int(trueNegatives2)  # Correct predictions for second set of data
incorrect2 = int(falsePositives2) + int(falseNegatives2)  # Incorrect predictions for second set of data

print("\nFor the first set of data, there are", correct1, "correct predictions and", incorrect1,
      "incorrect predictions.")
print("\nFor the second set of data, there are", correct2, "correct predictions and", incorrect2,
      "incorrect predictions.")

# Contingency table
table = [[correct1 + correct2, correct1 + incorrect2],
         [incorrect1 + correct2, incorrect1 + incorrect2]]

# Calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)

# McNemar stats and interpretation
print('\nMcNemar co-efficient value = %.3f, p-value = %.3f' % (result.statistic, result.pvalue))
alpha = 0.05
if result.pvalue > alpha:
    print('\nThe results ARE NOT significantly different, based on a p value of 5 percent.')
else:
    print('\nThe results ARE significantly different, based on a p value of 5 percent.')
