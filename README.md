USER NOTES
----------

PROGRAM FILES:

formalityTests.py is used to run formality classification machine learning tests using data uploaded from a data file.

addFields.py reads data from the data file, creates additional data fields and populates them, and then writes both the original data and the new data back to the file.

mcNemarTest.py is used to check if the chances of the differences between two sets of prediction results is due to randomness is less than 5%.

DATASETS:

The original dataset is original_formality_dataset.csv.

The modified dataset, which contains additional fields, is new_formality_data.csv.

addFields.py has been set up to work with original_formality_dataset.csv. test.py has been set up to work with new_formality_data.csv. However, both programs can be made to work with any suitable CSV dataset, by changing the filename.

PROGRAM INTEGRITY TEST FILES FOLDER:

These files were used to test that the programs were working as they should.

Dummy_Data_Before.csv is populated with 20 randomly generated sentences, and the rest of the fields are populated with random numbers..

Dummy_Data_After.csv contains additional data fields and data relating to the aforementioned sentences.

SPREADSHEET:

Formality_Classification_Results is a results spreadsheet containing the results of formality classification prediction tests.

The spreadsheet is organised into sheets, each of which relates to a different category of test.
