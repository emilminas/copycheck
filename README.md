# Copycheck
Copycheck is a module that identifies matching sequences of tokens between two strings. 

## Description
This module was created to address the need to match a verbatim sequence of words that would constitute a copyright violation. Throughout the module, the original document is referred to as the *reference* and the document generated from the reference is called the *sample*. Let's first assume that there is a minimum threshold for the length of a match sequence to constitute plagiarism or a copyright violation, e.g. >10 words. In order to determine this manually, we would have to select words 0-11 from the sample and "control-F" it in the reference. We would then have to shift over one word, now selecting words 1-12, and perform the search again, continuing until we reach words n-11 to n. Copycheck automates this operation by handling it as a pattern-matching problem (very similar to image recognition). 

One of the main considerations in implementing a pattern matcher was minimizing the:
- number of libraries that would have to be imported, and
- complexity of the operation.
  
This was achieved by leveraging vectorized operations in NumPy arrays. Accepting the computational overhead of converting strings to NumPy integer arrays and performing regex-based text processing, unnecessary, repetitive, or loopy operations are avoided down the line. 

## Pipeline
The following is a step-by-step description of the copycheck process:
1. The reference and sample documents are read on-line as input strings and processed to normalize potential variations in punctuation.
2. The strings are tokenized in a fashion that preserves formatting such as newline characters.
3. Each unique word (type) is assigned an integer value in a dictionary, which is used to convert each string into an array of integers.
4. The sample and reference arrays are restructured into overlapping sliding frames of indices, e.g. given a frame size of 11 we get the original arrays at indices  
   [[0, 1, ..., 10, 11]  
   [1, 2, ..., 11, 12]  
   ...  
   [n-11, n-10, ..., n-1, n]].
   
   However, each sample frame is compared to all reference frames. Let's introduce a simple example to illustrate this. We want to compare
   - reference: "The fog of San Francisco"
   - sample: "Hedgehog and the Fog"
     
   Converting them to integers in the order each new unique word is encountered we get
   - reference: [0 1 2 3 4]
   - sample: [5 6 0 1]

   Now let's say we want to match 2-work sequences, so a frame size of two. That means that we have to compare
   - [5 6] with [[0 1]
                 [1 2]
                 [2 3]
                 [3 4]]
   - [6 0] with [[0 1]
                 [1 2]
                 [2 3]
                 [3 4]]
   - [0 1] with [[0 1]
                 [1 2]
                 [2 3]
                 [3 4]]
     
   To achieve this efficiently, we need to move to the 3rd dimension by stacking each of the three frames listed directly above on top of each other to form a cube, so we actually     compare everything at once!
   - [  [ [5 6] ]      
        [ [6 0] ]           
        [ [0 1] ] ]        
     with  
     [ [ [0 1][1 2][2 3][3 4] ]  
       [ [0 1][1 2][2 3][3 4] ]  
       [ [0 1][1 2][2 3][3 4] ] ]
     
   Finally, by comparing the two cubes along their 3rd dimensions and convulving the comparison we attain a Boolean mask array for the reference and a Boolean mask array for the sample that is True for each match token. For our example, we would create:
   - reference: [True True False False False]
   - sample: [False False True True]
6. These masks are imposed on the original tokenized documents and color-coded (formatted to change the background color of the text) if a token is matched with True.
7. Matches that are in quotation marks are optionally color-coded.
8. The module finally prints a formatted version of the reference and sample documents that highlights matching sequences:

## Roadmap
There are three categories of improvement that should be implemented.
1. Further testing and refinement of the existing code.
2. Implementation of named entity recognition (NER) to identify proper nouns in matched sequences. This can be very simply done with NLTK or spaCy, which (I think) uses HMM-based tagging. However, if additional libraries are to be avoided, it is possible to implement a very rudimentary/naive NER leveraging English language capitalization rules with regex. I'm actually in the process of smoothing out a simple regex-based solution, which obviously won't be as good as NLTK, but it's a "quick and dirty" solution. 
3. A frontend should be developed to make the code more user-friendly.

## Final thoughts
I welcome any suggestions on how to improve this code. I welcome pull requests. For major changes or any use of my code for other purposes, please open an issue or contact me. 
  











