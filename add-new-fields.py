'''
The functions in this file first upload data from the data file, using loadData(). They then create data for new fields,
using extractData(), and then write the original data and the new data back to the original data file, using writeData().

'''

import nltk
import io
import re
from nltk.corpus import stopwords
from nltk.tokenize import SyllableTokenizer
from nltk import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys

# The lists that data loaded from original document will be put into.
corpus = []  # List of sentences in human-readable form (i.e. not vectorized).
dataExcludingDocumentAllRecordsList = []  # List of lists of data excluding the sentence for each record.

# These lists store data for the new fields to be added to the data file.
numAdjectivesList = []
numVerbsList = []
numAdverbsList = []
numConjunctionsList = []
numNounsList = []
numPronounsList = []
numModalVerbsList = []
numPrepositionsList = []
numDeterminerList = []
numCommasList = []
numExclamationMarksList = []
numFullStopsList = []
numQuestionMarksList = []
averageNumSyllableList = []
numInterjectionsList = []
avWordFrequencyList = []  # Average frequency of words in each document
avWordLengthList = []
numStopWordsList = []  # Number of stop words in each document
numProperNounsList = []
numMoreThanSevenCharsList = []
numLessThanFiveCharsList = []
numCapitalisedWordsList = []
VADERSentimentScoreList = []
numOfWordsInTop35 = []
numExistentialTheresList = []
fileName = "original_formality_dataset.csv"


# Checks if the 'fileName' is the correct file name
def checkFileNameCorrect():
    global fileName
    print("The default file name is ", fileName, "\n")
    print("If this is the name of the data file, press enter")
    newFileName = input("Otherwise, please provide the correct name, then press enter")
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


# loadData() uploads data for each record from the existing csv file.
def loadData():
    checkFileNameCorrect()
    checkFilePresent()
    with open(fileName, encoding='utf-8') as inputFile:
        firstLine = inputFile.readline()
        firstLineAsList = firstLine.split(",")

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
                if line[character] == ",":
                    numCommas = numCommas + 1
                if numCommas == sentenceIndex:
                    dataExcludingSentence = line[:character]
                    dataExcludingSentenceThisRecordList = dataExcludingSentence.split(",")  # List containing attributes
                    dataExcludingDocumentAllRecordsList.append(dataExcludingSentenceThisRecordList)

                    # The rest of the current line is comprised of the document:
                    documentToAdd = line[character + 1:]

                    # Removes \n from the end of the sentence:
                    documentToAdd.replace('\n', '')

                    # Puts document into a list of Strings:
                    corpus.append(documentToAdd)
                    break
    inputFile.close()
    print("\nNo of records uploaded: ", len(corpus))


# Returns a list of most common words in the corpus, with rankDepth being the ranking when the words are ordered by
# frequency. Returns a list of words up to and including that ranking.

def getTopWordsInCorpus(rankDepth):
    topWordsList = []
    # Creates a dictionary containing every word in the corpus as a key, and the frequency of occurrence as value.
    wordsInCorpus = {}
    for documents in corpus:
        wordList = documents.split(" ")
        for words in wordList:

            #  Removes punctuation. Code apapted from:
            #  https://stackoverflow.com/questions/875968/how-to-remove-symbols-from-a-string-with-python
            words = re.sub(r'[^\w]', '', words)
            words = words.lower()  # makes lower case
            if words in wordsInCorpus.keys():
                wordsInCorpus[words] = wordsInCorpus[words] + 1
            else:
                wordsInCorpus[words] = 1

    # List of word counts and words in descending order by word count. Line below adapted from:
    # https://careerkarma.com/blog/python-sort-a-dictionary-by-value/
    wordFreqList = sorted(wordsInCorpus.items(), key=lambda x: x[1], reverse=True)

    # Puts the dictionary's keys and values into separate lists
    frequencyKeyList = []  # What is the word?
    frequencyValList = []  # How frequently does the word appear in the corpus?
    for entry in wordFreqList:
        frequencyKeyList.append(entry[0])
        frequencyValList.append(entry[1])
    topValList = []  # List of word frequency values, containing with the top 'rankDepth' number of items
    currentRank = 1
    numInstancesThisVal = 1
    isFirstVal = True
    previousVal = 0

    #  Finds finds the top 'rankDepth' greatest 'word frequency within corpus' values and appends them to topValList.
    for value in frequencyValList:
        if isFirstVal:
            previousVal = value
            isFirstVal = False
            topValList.append(value)
            continue
        if value == previousVal:
            numInstancesThisVal = numInstancesThisVal + 1
            if currentRank <= rankDepth:
                topValList.append(value)
        else:
            currentRank = currentRank + numInstancesThisVal
            previousVal = value
            numInstancesThisVal = 1

            # Add word to topWordsList when its ranking is <= rankDepth and it is different to the previous word.
            if currentRank <= rankDepth:
                topValList.append(value)

    # Appends the top 'rankDepth' words to topWordsList.
    for i in range(len(topValList)):
        topWordsList.append(frequencyKeyList[i])
    return topWordsList


# Returns the number of words in the document which are in the 'topWordsInCorpus' list, which is a list of the most
# frequently occurring words in the corpus of sentences.
def getNumTopWordsInDocument(cleansedWordsList, topWordsInCorpus):
    count = 0
    for words in cleansedWordsList:
        if words.lower() in topWordsInCorpus:
            count = count + 1
    return count


# Calculates the number of each of various grammar types (nouns, verbs, etc) in the document, and appends the relevant
# lists.
def extractGrammar(document):
    documentAsTokens = word_tokenize(document)
    grammarList = nltk.pos_tag(documentAsTokens)
    numAdjectives = numVerbs = numAdverbs = numConjunctions = numNouns = numPronouns = numModalVerbs = 0
    numPrepositions = numDeterminer = numInterjections = numProperNouns = numExistentialTheres = 0
    for entry in grammarList:

        # See https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html for the meanings of the
        # abbreviations below
        if entry[1] == "JJ" or entry[1] == "JJR" or entry[1] == "JJS":
            numAdjectives = numAdjectives + 1
        if entry[1] == "VB" or entry[1] == "VBD" or entry[1] == "VBG" or entry[1] == "VBN" or \
                entry[1] == "VBP" or entry[1] == "VBZ":
            numVerbs = numVerbs + 1
        if entry[1] == "RB" or entry[1] == "RBR" or entry[1] == "RBS" or entry[1] == "WRB":
            numAdverbs = numAdverbs + 1
        if entry[1] == "CC":
            numConjunctions = numConjunctions + 1
        if entry[1] == "NN" or entry[1] == "NNS" or entry[1] == "NNP" or entry[1] == "NNPS":
            numNouns = numNouns + 1
        if entry[1] == "PRP" or entry[1] == "PRP$" or entry[1] == "WP":
            numPronouns = numPronouns + 1
        if entry[1] == "MD":
            numModalVerbs = numModalVerbs + 1
        if entry[1] == "IN" or entry[1] == "TO":
            numPrepositions = numPrepositions + 1
        if entry[1] == "DT" or entry[1] == "WDT":
            numDeterminer = numDeterminer + 1
        if entry[1] == "UH":
            numInterjections = numInterjections + 1
        if entry[1] == "NNP" or entry[1] == "NNPS":
            numProperNouns = numProperNouns + 1
        if entry[1] == "EX":
            numExistentialTheres = numExistentialTheres + 1
    numAdjectivesList.append(numAdjectives)
    numVerbsList.append(numVerbs)
    numAdverbsList.append(numAdverbs)
    numConjunctionsList.append(numConjunctions)
    numNounsList.append(numNouns)
    numPronounsList.append(numPronouns)
    numModalVerbsList.append(numModalVerbs)
    numPrepositionsList.append(numPrepositions)
    numDeterminerList.append(numDeterminer)
    numInterjectionsList.append(numInterjections)
    numProperNounsList.append(numProperNouns)
    numExistentialTheresList.append(numExistentialTheres)


# Calculates how many instances there are of various types of punctuation in each sentence, and appends
# the relevant lists.
def extractPunctuation(document):
    numCommas = numExclamationMarks = numFullStops = numQuestionMarks = 0
    for character in document:
        if character == ",":
            numCommas = numCommas + 1
        if character == "!":
            numExclamationMarks = numExclamationMarks + 1
        if character == ".":
            numFullStops = numFullStops + 1
        if character == "?":
            numQuestionMarks = numQuestionMarks + 1
    numCommasList.append(numCommas)
    numExclamationMarksList.append(numExclamationMarks)
    numFullStopsList.append(numFullStops)
    numQuestionMarksList.append(numQuestionMarks)


# Calculates the average number of syllables per word in the document and appends the relevant list with
# the figure.
# Code for this function adapted from code at:
# https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.sonority_sequencing.SyllableTokenizer
def syllableCount(cleansedWordsList):
    wordCountThisDocument = 0
    totalSyllsThisDocument = 0
    SSP = SyllableTokenizer()
    for words in cleansedWordsList:
        numSyllThisWord = len(SSP.tokenize(words.lower()))
        if numSyllThisWord > 0 and words:  # Run if number of syllables > 0 and list entry not null
            wordCountThisDocument = wordCountThisDocument + 1
            totalSyllsThisDocument = totalSyllsThisDocument + numSyllThisWord
    if wordCountThisDocument > 0:
        averageNumSyllables = totalSyllsThisDocument / wordCountThisDocument
    else:
        averageNumSyllables = "N/A"
    averageNumSyllableList.append(averageNumSyllables)


# How frequently does each word appear in the sentence on average? Appends the answer to the relevant list.
def wordFreqThisDoc(cleansedWordsList):
    listOfFrequencies = []
    wordsProcessed = []
    for word in cleansedWordsList:
        wordCount = 0
        if word in wordsProcessed:
            continue
        else:
            wordsProcessed.append(word)
        for i in range(len(cleansedWordsList)):
            if cleansedWordsList[i] == word:
                wordCount = wordCount + 1
            if i == len(cleansedWordsList) - 1:
                listOfFrequencies.append(wordCount)
    averageWordFrequency = 0
    if len(listOfFrequencies) != 0:
        averageWordFrequency = sum(listOfFrequencies) / len(listOfFrequencies)
    avWordFrequencyList.append(averageWordFrequency)


# Calculates the sentence's average word length, and appends the relevant list with the result.
def avWordLengthThisDoc(cleansedWordsList):
    characterCount = 0
    for words in cleansedWordsList:
        characterCount = characterCount + len(words)
    avWordLength = 0
    if len(cleansedWordsList) != 0:
        avWordLength = characterCount / len(cleansedWordsList)
    avWordLengthList.append(avWordLength)
    # Number of stop words in the document.
    stopWords = stopwords.words('english')
    stopWordCount = 0
    for word in cleansedWordsList:
        if word in stopWords:
            stopWordCount = stopWordCount + 1
    numStopWordsList.append(stopWordCount)


# Returns the number of words greater than 'length' in length within the sentence.
def numWordsGreaterThanLength(cleansedWordsList, length):
    numWords = 0
    for word in cleansedWordsList:
        if len(word) > length:
            numWords = numWords + 1
    return numWords


# Returns the number of words less than 'length' in length within the sentence.
def numWordsLessThanLength(cleansedWordsList, length):
    numWords = 0
    for word in cleansedWordsList:
        if len(word) < length:
            numWords = numWords + 1
    return numWords


#  Calculates the number of capitalised words within the document, and appends the figure to the relevant list.
def numCapitalisedWords(cleansedWordsList):
    numWords = 0
    for word in cleansedWordsList:
        if len(word) > 0:
            if word[0].isupper():
                numWords = numWords + 1
    numCapitalisedWordsList.append(numWords)


'''
VADER document sentiment analysis - Code adapted from code at:
https://github.com/cjhutto/vaderSentiment#citation-information

Produces a score between -1 and 1, where -1 is extreme negative sentiment and 1 is extreme positive sentiment.

It then appends the relevant list with the score. 
'''


def VADERScore(document):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(document)
    compoundScore = vs['compound']
    VADERSentimentScoreList.append(compoundScore)

# extractData() goes through each sentence in the corpus in turn using a for loop, and calls helper methods to
# populate lists with new data for each sentence.


def extractData():
    print("Processing data. This can take a while. Please be patient.")
    # Downloads for NLTK, which is a library that is used for labelling words as nouns, verbs, etc and counting
    # syllables
    nltk.download('cmudict')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    for document in corpus:
        extractGrammar(document)  # Finds number of each of the grammar type and puts the figures into lists.
        extractPunctuation(document)  # Gets number of various punctuation types and puts figures into lists.
        listOfWords = document.split(" ")  # Splits each sentence into a list of its words.

        #  Creates list with punctuation removed from the words. Code for this functionality adapted from code at:
        #  https://stackoverflow.com/questions/875968/how-to-remove-symbols-from-a-string-with-python
        cleansedWordsList = []
        for words in listOfWords:
            words = re.sub(r'[^\w]', '', words)
            cleansedWordsList.append(words)
        syllableCount(cleansedWordsList)  # Average number of syllables per word
        wordFreqThisDoc(cleansedWordsList)  # How frequently does each word appear in the sentence on average?
        avWordLengthThisDoc(cleansedWordsList)  # Average word length. Total character length ex punctuation / num words
        numMoreThanSevenCharsList.append(numWordsGreaterThanLength(cleansedWordsList, 7))  # Num words with > 7 chars
        numLessThanFiveCharsList.append(numWordsLessThanLength(cleansedWordsList, 5))  # Num words with < 5 chars
        numCapitalisedWords(cleansedWordsList)  # Number of capitalised words
        VADERScore(document)  # The sentence's VADER sentiment score
        top35WordsList = getTopWordsInCorpus(35)  # Returns top 35 most common words in corpus.
        numOfWordsInTop35.append(getNumTopWordsInDocument(cleansedWordsList, top35WordsList))


# writeData() writes the original data and the newly created fields/accompanying data to the original data file.
# The new data is stored after the end of original document-related data and before the sentences themselves (the
# sentence field is the rightmost field).
def writeData():
    with open('original_formality_dataset.csv', encoding='utf-8') as inputFile:
        # Get column headers from existing data set and insert new field headers before the sentence field (which is the
        # furthest field on the right).
        existingHeaders = inputFile.readline()
        existingHeadersAsList = existingHeaders.split(",")

        # How many fields (and therefore commas) up to and including the one immediately preceding the 'sentence' field?
        sentenceIndex = len(existingHeadersAsList)-1

        # Additions to existing header.
        newHeaders = ['Number of adjectives', 'Number of verbs', 'Number of adverbs', 'Number of conjuctions',
                      'Number of nouns', 'Number of pronouns', 'Number of modal verbs', 'Number of prepositions',
                      'Number of determiners', 'Number of commas', 'Number of exclamation marks',
                      'Number of full stops', 'Number of question marks', 'Number of existential theres',
                      'Number of proper nouns', 'Number of capitalised words','Number of interjections',
                      'Average number of syllables per word', 'Average word length', 'Number of stop words',
                      'Number of words with > seven characters','Number of words with < 5 characters',
                      'Average word frequency', 'Number of words in 35 most common words in corpus',
                      'VADER sentiment score']
        tempSentenceIndex = sentenceIndex
        for items in newHeaders:
            existingHeadersAsList.insert(tempSentenceIndex, items)
            tempSentenceIndex = tempSentenceIndex + 1
        sep = ","
        allHeaders = sep.join(existingHeadersAsList)
    inputFile.close()
    numberOfDocs = len(corpus)

    # Copy data to file. NB Overwrites previous file, so back it up first.
    with io.open('original_formality_dataset.csv', 'w', encoding='utf8') as outputFile:
        outputFile.write(allHeaders)
        print("Number of records for new file ", numberOfDocs)
        for i in range(numberOfDocs):

            # References to ints are surrounded by [ and ] to prevent the error msg: TypeError: can only concatenate
            # list (not "int") to list
            listOfStrings = dataExcludingDocumentAllRecordsList[i] + [numAdjectivesList[i]] + [numVerbsList[i]] + \
                            [numAdverbsList[i]] + [numConjunctionsList[i]] + [numNounsList[i]] + \
                            [numPronounsList[i]] + [numModalVerbsList[i]] + [numPrepositionsList[i]] + \
                            [numDeterminerList[i]] + [numCommasList[i]] + [numExclamationMarksList[i]] + \
                            [numFullStopsList[i]] + [numQuestionMarksList[i]] + [numExistentialTheresList[i]] + \
                            [numProperNounsList[i]] + [numCapitalisedWordsList[i]] + [numInterjectionsList[i]] + \
                            [averageNumSyllableList[i]] + [avWordLengthList[i]] + [numStopWordsList[i]] + \
                            [numMoreThanSevenCharsList[i]] + [numLessThanFiveCharsList[i]] + \
                            [avWordFrequencyList[i]] + [numOfWordsInTop35[i]] + [VADERSentimentScoreList[i]] + \
                            [corpus[i]]
            allDataThisLine = ','.join(map(str, listOfStrings))  # This line's data as a comma-separated string
            outputFile.write(allDataThisLine)
    print("New data successfully written to file. Program complete.")
    outputFile.close()


# METHOD CALLS THAT EXECUTE WHENEVER THE PROGRAM IS RUN
loadData()
extractData()
writeData()