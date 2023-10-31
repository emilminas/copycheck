#!/usr/bin/env python3
"""
copycheck.py

This module contains multiple functions that identify identical sequences of words between two texts.
------
Method:
1. Read in two inputs representing sample and reference documents,
2. Assign a unique integer to each word (type, not token) and convert both texts into NumPy integer arrays,
3. Get sliding frames across the arrays, comparing them, and convolving them to get matching sequences,
4. Format the input texts to highlight matching sequences.
5. Optionally color code verbatim sequences that are quoted.
------
Created by Emil Minas
October 30, 2023
"""
import re
import numpy as np


def input_text():
    """ Accepts multiple lines of input texts and returns it as a string.

     Output
     ------
     Text input as a string object.
     """
    text = []
    while True:             # Accepts input lines until an EOF character is passed: Ctrl-D (Mac) or Ctrl-Z (Windows)
        try:
            line = input()  # Passes input and assigns it to a string variable.
        except EOFError:
            break
        text.append(line)   # Populates list with strings representing each input line.
    return '\n'.join(text)  # Join the list into one input string.


def process_text(text):
    """ Uses regex substitution to format an input string to separate hyphenated words and normalize punctuation.

    Arguments
    ----------
    text   : input string

    Output
    ------
    A formatted version of the input with spaced hyphens (including m- and n-dashes)
    and normalized apostrophe and quotation marks.
    """
    return re.sub(r'(\w)([—–-])(\w)', '\g<1>\g<2> \g<3>',
                  re.sub(r'(\w) ([—–-]) (\w)', '\g<1>\g<2> \g<3>',
                         re.sub(r'[“”]', '\"',
                                re.sub(r'[’‘]', '\'', text))))


def to_list(text):
    """ Splits a given string by space char while preserving any newline, tab, etc.

    Arguments
    ----------
    text   : input string

    Output
    ------
    A list of words split by whitespace, while preserving other formatting,
    e.g. "\tHello world\n\n!" --> ['\t', 'Hello', 'world', '\n\n', '!']
    ['\tHello', 'world\n\n!']
    """
    words = re.split(' +|([\t\n\r\f\v]+)', text)      # Splits text.
    return [word for word in words if word]           # Filters out None and empty elements.


def get_color(color):
    """ Converts the name of a color to its ANSI emboldened and color start and stop codes.

    Arguments
    ----------
    color   : input string

    Output
    ------
    The correct ANSI formatting start and stop sequences for the given color as a tuple of strings,
    e.g. "red" --> ("\033[41;1m", "033[0m")
    """
    if color == "red":
        start = "\033[41;1m"
    elif color == "green":
        start = "\033[42;1m"
    elif color == "yellow":
        start = "\033[43;1m"
    elif color == "blue":
        start = "\033[44;1m"
    elif color == "magenta":
        start = "\033[45;1m"
    elif color == "cyan":
        start = "\033[46;1m"
    else:
        start = "\033[47;1m"    # Defaults to white.

    stop = "\033[0m"

    return start, stop


def highlight_text(masked_text, color):
    """ Finds and highlights match sequences within a given text.

     Arguments
     ----------
     masked_text   : input list of (word, boolean) tuples where a word is True when masked, i.e. not a match.
     color         : input string specifying desired highlight color.

     Output
     ------
     A copy of the input highlighted for match sequences.
     """
    start, stop = get_color(color)  # Controls the highlighter color start and stop commands.
    return [f'{start}{token}{stop}' if target else token for token, target in masked_text]


def smooth_highlights(highlighted_text):
    """ Removes redundant formatting between highlighted and emboldened words. .

     Arguments
     ----------
     highlighted_text   : input string with each match word highlights.

     Output
     ------
     The same string with excessive formatting in between match words removed.
     """
    # Removes breaks in highlighting between words.
    return re.sub(r'\033\[0m\s(\033\[\d\d;1m)', ' \g<1>', highlighted_text)


def search_arrays(reference_list, sample_list, frame_size_int):
    """ Use a sample to search for matches in a reference of a given frame length.

    Arguments
    ----------
    reference_list      : input 1D array -- reference text represented as an array of integers
    sample_list         : input 1D array -- sample text represented as an array of integers
    frame_size_int      : input integer  -- the minimum threshold for matching consecutive words

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """
    # Convert list of sample and reference lists into NumPy arrays.
    reference = np.asarray(reference_list, dtype=int)
    sample = np.asarray(sample_list, dtype=int)

    # Store sizes of sample, reference, and frame size.
    ref_size, sam_size, frame_size = reference.size, sample.size, frame_size_int
    # Frame template ([0, 1, ..., frame_size])
    frame_range = np.arange(frame_size)

    # Create a 3D tensor (shape: num_sample_frames x 1 x frame_size)
    # of stacked sliding indices across the entire length of sample array.
    sample_frames = sample[np.expand_dims(np.arange(sam_size - frame_size + 1), (1, 2)) + frame_range]

    # Create a 2D tensor (shape: num_reference_frames x frame_size)
    # of sliding indices across the entire length of sample array.
    reference_frames = reference[np.expand_dims(np.arange(ref_size - frame_size + 1), 1) + frame_range]
    # Create a 3D tensor by stacking copies of the 2D array number of sample frame times
    # (shape: num_sample_frames x num_reference_frames x frame_size)
    reference_frames = np.tile(reference_frames, (sample_frames.shape[0], 1, 1))

    # Match the two sequences to obtain boolean matrix of shape: num_sample_frames x num_reference_frames.
    matches = (reference_frames == sample_frames).all(2)

    # Get masks for matching sequences in the sample and reference. 1D arrays of sample size and reference size.
    if matches.any() > 0:
        reference_mask = np.convolve(matches.any(0), np.ones((frame_size), dtype=int)) > 0
        sample_mask = np.convolve(matches.any(1), np.ones((frame_size), dtype=int)) > 0
        return reference_mask, sample_mask
    else:
        return np.full(ref_size, False), np.full(sam_size, False)


def match_verbatim(reference, sample, frame_size):
    """ Find matching sequences between two arrays using NumPy mask arrays.

     Arguments
     ----------
     reference          : input str list  -- the reference text tokenized
     sample             : input str list  -- the sample text tokenized
     frame_size         : input integer   -- the minimum threshold for matching consecutive words
     find_quotes        : input boolean   -- determine if quoted matches should be identified
     reference_color    : input string    -- color coded reference matches
     sample_color       : input string    -- color coded sample matches
     quote_color        : input string    -- color coded quote matches

     Output
     ------
     A pair of A Boolean NumPy arrays acting as masks to identify matching sequences of tokens
     between the reference and sample texts.
     """
    # reference_tokens, sample_tokens = to_list(reference), to_list(sample)  # Input texts to strings of words.
    reference_nums, sample_nums = [], []
    word2num = {}

    for index, token in enumerate(reference):
        word = re.sub(r'\W', '', token.lower())     # Gets the normalized form of each word.
        word2num[word] = word2num.get(word, index)  # Populates a dict: each word key gets a unique integer.
        reference_nums.append(word2num[word])       # Populates a list of integers representing the reference input.

    for token in sample:
        word = re.sub(r'\W', '', token.lower())     # Gets the normalized form of each word.
        try:
            sample_nums.append(word2num[word])      # Populates a dict: each word key gets a unique integer.
        except KeyError:
            sample_nums.append(-1)                  # If the sample word is not in the reference, assigned -1.

    # Get matches: boolean arrays identical in sample and reference sizes acting as masks.
    reference_mask, sample_mask = search_arrays(reference_nums, sample_nums, frame_size)

    return reference_mask, sample_mask


def match_quotes(sample):
    """ Finds sequences of the quotes text and creates a mask array to identify them.

     Arguments
     ----------
     sample  : input string -- The sample text.

     Output
     ------
     A Boolean NumPy array acting as a mask to identify sequences of quoted words.
     """

    # Only attempts to identify quoted sequences if there are an even number of quotation marks in the sample.
    if len(re.findall('\"', sample)) % 2 == 0:

        # Creates a regex pattern that matches all tokens within or directly outside quotation marks.
        # For example: "Apples and oranges," (and also "bananas"). --matches--> ["Apples and oranges,"], ["bananas").]
        quotation = '\S*?\"(.*?)\"\S*'

        # This lambda function uses the above pattern and regex substitution
        # to replace each token in a match with the placeholder token "<quoted>"
        #  For example: "Apples and oranges," (and also "bananas"). --lambda-->
        #               <quoted> <quoted> <quoted> (and also <quoted>
        quotes_found = re.sub(quotation, lambda m: '<quoted> ' * len(to_list(m.group(1))), sample)

        # Returns a Boolean NumPy array that masks unquoted sequences.
        #  For example: "Apples and oranges," (and also "bananas"). --mask-->
        #               [True True True False False True]
        return np.array(to_list(quotes_found)) == '<quoted>'

    else:
        # If there are an odd number of quotation marks, returns a mask that does not identify quotes.
        #  For example: "Apples and oranges, (and also "bananas"). --mask-->
        #               [False False False False False False]
        return np.full(len(to_list(sample)), False)


def layer_masks(m, q):
    """ Find matching sequences between two arrays using NumPy mask arrays.
    Arguments
     ----------
     m  : Boolean array which is True for every match word in the corpus, i.e. match mask.
     q  : Boolean array which is True for every quoted word in the corpus, i.e. quote mask.

     Output
     ------
     mq     : Boolean array True for every quoted sequence that is a match, m *AND* q.
     m_q    : Boolean array True for every match that is not a question sequence, (m *AND* (*NOT* (q))
    """
    mq = np.logical_and(m, q)                   # Uses logical "and" to identify intersection of match and quote.
    m_q = np.logical_and(m, np.logical_not(q))  # Uses logical "and" to identify intersection of match and NOT quote.

    return m_q, mq


def format_text(r_in, s_in, frame, get_quotes, reference_color, sample_color, quote_color):
    """ This is the core function of the module that leverages all other functions to identify and color code
    matching and quoted sequences of tokens between two texts (i.e. reference and sample).

    Arguments
     ----------
     r_in           : raw input string representing the reference text.
     s_in           : raw input string representing the sample text.
     frame          : integer specifying the size of the minimum verbatim sequence to identify (i.e. 11).
     get_quotes     : Boolean specifying whether quotes matches should be color coded.
     reference_color: string spelling out what color the matching sequences in the reference should be highlighted.
     sample_color   : string spelling out what color the matching sequences in the sample should be highlighted.
     quote_color    : string spelling out what color the matching quoted sequences in the sample should be highlighted.

     Output
     ------
     Copies of the input texts with formatting that emboldens and color codes target sequences.
    """

    reference, sample = process_text(r_in), process_text(s_in)  # Normalizes punctuaion in the inputs.
    r_tokens, s_tokens = to_list(reference), to_list(sample)    # Tokenizes the inputs, creating lists of strings.

    # Get Boolean masks identifying matching sequences of tokens (min length of frame) between the two lists.
    r_match_mask, s_match_mask = match_verbatim(r_tokens, s_tokens, frame)

    if r_match_mask.any() and r_match_mask.any():  # If any matches exist.

        # Imposes the mask array on the reference list to create an array of tuples
        # where each token is paired with a Boolean identifying it as a match or not; match tokens get color coded.
        r_out = highlight_text(np.ma.masked_array(r_tokens, r_match_mask).toflex(), reference_color)

        # Get Boolean mask identifying quotes sequences. If the User doesn't want to color code quotes, get dummy mask.
        quote_mask = match_quotes(sample) if get_quotes else np.full(s_match_mask.size, False)

        # Using logical operands, combine masks to get (1) matches excluding quotes and (2) quotes matches.
        matches, quotes = layer_masks(s_match_mask, quote_mask)

        s_out = highlight_text(np.ma.masked_array(s_tokens, quotes).toflex(), quote_color)  # highlight quoted matches
        s_out = highlight_text(np.ma.masked_array(s_out, matches).toflex(), sample_color)  # highlight unquoted matches.

        # Joins the lists of tokens back into strings and uses regex substitution to remove redundant formatting.
        return smooth_highlights(' '.join(r_out)), smooth_highlights(' '.join(s_out))

    else:
        # If no matches were found, do not format the text at all.
        return None, None


if __name__ == "__main__":

    print("\n\n***Enter the reference text. Ctrl-D (Mac) or Ctrl-Z (Windows) to save it.***\n")
    reference_in = input_text()

    print("\n\n***Enter the sample text. Ctrl-D (Mac) or Ctrl-Z (Windows) to save it.***\n")
    sample_in = input_text()

    while True:
        word_count = len(sample_in.split())
        print("\n\n***How many consecutive words is the minimum match?***\n"
              "Default: 11.\n")
        frame_size = input()
        frame_size = int(frame_size) if frame_size and re.match(r'\d+', frame_size) else 11
        if frame_size > word_count:
            print(f'\nThe number you entered is higher than the total word count of {word_count}.\n')
            break

        print("\n\n***Do you want to identify quoted sequences?***\n"
              "Options: y or n. Default: y\n")
        quotes = input()
        find_quotes = False if quotes == 'n' else True

        reference_out, sample_out = format_text(reference_in, sample_in, frame_size, find_quotes,
                                                "yellow", "magenta", "cyan")
        if not reference_out or not sample_out:
            reference_out = reference_in
            sample_out = sample_in

        print(f'\n\n*** Matching sequences in the reference text are highlighted yellow. \n\n{reference_out}')

        print(f'\n\n*** Matching sequences in the sample text are highlighted magenta.')
        if find_quotes:
            print(f'Quoted sequences are highlighted cyan.')
        print(f'\n\n{sample_out}')

        print(f'\n\n*** The sample documents contains {word_count} words, '
              f'which is {round(word_count / len(reference_in.split()) * 100, 2)}% of the reference.\n\n')

        print("\n\n***Press \'enter\' terminate or 'y' to go again.")
        again = input()
        if again:
            print(
                "\n\n***Keep reference?(y/n). "
                "Otherwise, enter the sample text.Ctrl-D (Mac) or Ctrl-Z (Windows) to save it.***\n")
            keep = False if input() == 'n' else True
            reference_in = reference_in if keep else input_text()

            print(
                "\n\n***Keep sample?(y/n). "
                "Otherwise, enter the sample text.Ctrl-D (Mac) or Ctrl-Z (Windows) to save it.***\n")
            keep = False if input() == 'n' else True
            sample_in = sample_in if keep else input_text()
        else:
            break
