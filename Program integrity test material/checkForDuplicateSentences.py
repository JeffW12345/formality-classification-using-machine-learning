'''
Checks corpus of sentences for duplicates.
'''
import sys

dataFileFieldNames = []  # The field names from the top of the data spreadsheet.
corpus = []  # List of sentences in human-readable form (i.e. not in a bag of words representation)
fileName = "new_formality_data.csv"


# Checks if the 'fileName' is the correct file name
def checkFileNameCorrect():
    global fileName
    print("The default file name is ", fileName, "\n")
    print("If this is the name of the data file, press enter")
    newFileName = input("Otherwise, please provide the correct name, then press enter: ")
    if newFileName != "":
        fileName = newFileName
        print("\nThank you. The file name has been changed to", fileName)
    else:
        print("\nThank you. You have confirmed that the file name is correct:", fileName)


# Checks if file present. Code for this module adapted from:
# https://stackoverflow.com/questions/5627425/what-is-a-good-way-to-handle-exceptions-when-trying-to-read-a-file-in-python
def checkFilePresent():
    try:
        f = open(fileName, 'rb')
    except OSError:
        print("\nFile not found:", fileName)
        print("Please ensure that the data file is in the same folder as the program file.")
        print("Exiting program.")
        sys.exit()


# This function loads the data from the file and stores it in the data structures shown above.
# It is always the first function to be run.

# This function loads the data from the file and stores it in the data structures
def loadData():
    checkFileNameCorrect()
    checkFilePresent()
    with open(fileName, encoding='utf-8') as inputFile:
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
                    documentToAdd = line[character + 1:]  # The rest of the current line is comprised of the document

                    # Puts document into a list of Strings:
                    corpus.append(documentToAdd)
                    break  # returns to the outer 'for' loop, so the next line can be processed.
    inputFile.close()
    print("\nNo of records uploaded: ", len(corpus))


def testForUniqueSentences():
    tempSentencesList = []
    duplicate = False
    for sentence in corpus:
        if sentence in tempSentencesList:
            print("Duplicate sentence: " + sentence)
            duplicate = True
        tempSentencesList.append(sentence)
    if not duplicate:
        print("\nGood news! No duplicates were found.")


# METHOD CALLS THAT EXECUTE WHENEVER THE PROGRAM IS RUN

loadData()
testForUniqueSentences()