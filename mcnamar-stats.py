'''

Calculates the McNemar statistic and states the p-value of the result.

Code below adapted from https://machinelearningmastery.com/mcnemars-test-for-machine-learning/

'''


from statsmodels.stats.contingency_tables import mcnemar

TP1 = input("For the first set of data, how many true positives are there?")
FP1 = input("For the first set of data, how many false positives are there?")
TN1 = input("For the first set of data, how many true negatives are there?")
FN1 = input("For the first set of data, how many false negatives are there?")

TP2 = input("For the second set of data, how many true positives are there?")
FP2 = input("For the second set of data, how many false positives are there?")
TN2 = input("For the second set of data, how many true negatives are there?")
FN2 = input("For the second set of data, how many false negatives are there?")

correct1 = int(TP1) + int(TN1)
incorrect1 = int(FP1) + int(FN1)
correct2 = int(TP2) + int(TN2)
incorrect2 = int(FP2) + int(FN2)

print("For the first set of data, there are", correct1, "correct predictions and", incorrect1,
      "incorrect predictions.")
print("For the second set of data, there are", correct2, "correct predictions and", incorrect2,
      "incorrect predictions.")

# define contingency table
table = [[correct1+correct2, correct1 + incorrect2],
         [incorrect1+correct2, incorrect1 + incorrect2]]

# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)

# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
    print('\nMcNemar statistic=%.3f, p-value=%.3f.\nThe results ARE NOT significantly different, based on a p value of '
          '5 percent.')
else:
    print('\nMcNemar statistic=%.3f, p-value=%.3f. \nThe results ARE significantly different, based on a p value of '
          '5 percent.')
