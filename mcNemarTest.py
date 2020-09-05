# Code below taken from https://machinelearningmastery.com/mcnemars-test-for-machine-learning/

from statsmodels.stats.contingency_tables import mcnemar

# define contingency table
table = [[2103, 1409],
         [1403, 709]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
    print('McNemar statistic=%.3f, p-value=%.3f . Same proportions of errors (fail to reject H0)' %
          (result.statistic, result.pvalue))
else:
    print('McNemar statistic=%.3f, p-value=%.3f. Different proportions of errors (reject H0)' %
          (result.statistic, result.pvalue))
