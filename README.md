For the final project of my MSc in Software Development at the University of Glasgow, I investigated using machine learning to predict sentence formality 
classifications.

This repository contains:

- My code (which is written in Python and uses scikit-learn), along with a requirements.txt file. 
- The data file I used for the test (the original data file and that I added data fields to), consisting of sentences and fields relating to the sentences.
- Files relating to testing the program, which are stored in the Program integrity test material folder.
- A results spreadsheet.
- Sample console output of the machine learning tests. 

USER NOTES
----------

PROGRAM FILES:

ngram-only-tests is used to run formality classification tests using purely n-grams as the feature. 

non-ngram-only-tests is used to run formality classification tests using purely non n-gram features (such as the number of verbs in each sentence). 

ngram-and-non-ngram-tests-combined is used to run formality classification tests combining n-grams and other features. 

add-fields.py reads data from the data file, creates additional data fields and populates them, and then writes both the original data and the new data 
back to the file.

mcnemar-stats.py is used to check if the chances of the differences between two sets of prediction results is due to randomness is less than 5%.

checkForDuplicateSentences.py - This is located in the 'Program integrity test material' folder and can be used to check for duplicate sentences. 

DATASETS:

The original dataset is original_formality_dataset.csv.

The modified dataset, which contains additional fields that were added by add-fields.py, is new_formality_data.csv.

PROGRAM INTEGRITY TEST FILES FOLDER:

These files were used to test that the programs were working as they should.

Dummy_Data_Before.csv is populated with 20 randomly generated sentences, and the rest of the fields are populated with random numbers..

Dummy_Data_After.csv contains additional data fields and data relating to the aforementioned sentences.

Program integrity tests.doxc - Copies of the output of tests to ensure that the program was running as it should. Includes test print statements. 

checkForDuplicateSentences.py - Described in 'PROGRAM FILES' above. 

RESULTS SPREADSHEET:

Formality_Classification_Results is a results spreadsheet containing the results of formality classification prediction tests.

The spreadsheet is organised into sheets, each of which relates to a different category of test.

SAMPLE CONSOLE OUTPUT:

Sample_Console_Output.docx displays sample console output relating to the three modules responsible for the machine learning tests.
