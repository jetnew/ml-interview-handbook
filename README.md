# ML Interview Handbook

Made for revising for machine learning interviews.

## ML - Machine Learning

#### L1 & L2 regularization
* L1 regularization is used in Lasso regression, adding absolute magnitude of coefficients to loss function
* L1 regularization performs feature selection by decreasing feature coefficients to zero.
* L1 regularization = lambda * sum(B^2)
* L2 regularization is used in Ridge regression, adding squared magnitude of coefficients to loss function.
* L2 regularization = lambda * sum (|B|)

#### Naive Bayes
* Naive Bayes

#### Precision
* Precision = TP / (TP + FP)
* Precision is employed when the cost of FP is high, e.g. a patient falsely classified positive for heart disease will experience unnecessary stress.

#### Recall
* Recall = TP / (TP + FN)
* Recall is employed when the cost of FN is high, e.g. a patient falsely classified negative for heart disease will be denied treatment.

#### F1 score
* F1 score is the harmonic mean of the precision and recall.
* F1 score = 2 * (precision * recall) / (precision + recall)
* The harmonic mean is the reciprocal of the arithmetic mean of the reciprocals of a given set of observations.
* Harmonic mean(x1, x2, x3) = 3 / [(1/x1) + (1/x2) + (1/x3)]

#### ROC curve
* ROC curve is the Receiver Operating Characteristic curve.
* ROC curve visualises the trade-off between TP rate and FP rate.
* AUROC (or AUC) is the Area Under ROC curve, which measures the performance over all possible classification thresholds.


## DL - Deep Learning

#### ReLU
* ReLU is the Rectified Linear Unit.
* ReLU resolves the Vanishing Gradients problem, where the gradient of the sigmoid activation function tends towards 0 as the value of the sigmoid tends to 0 or 1.

## NLP - Natural Language Processing

#### Stop words
* Stop words, e.g. is, was, are, were, are usually removed during text pre-processing because they are usually not useful for NLP.

#### TF-IDF
* TF-IDF stands for Term Frequency-Inverse Document Frequency.
* TF-IDF indicates the importance of a word in a text dataset, helping with computing numerical statistics about words in a text dataset.
* TF(Term) = Term frequency / Total no. of terms in the document
* IDF(Word) = log_e(Total no. of documents / No. of documents with term)
* When TF * IDF is high, frequency of the term is low, vice versa.

### BoW
* Bag-of-Words is a representation of the vocabulary in a document.
* Bag-of-Words can be a vector, mapping word to word frequency in a document. E.g. [0,1,1,2,1]

#### Stemming
* Stemming removes the suffix from a word to obtain its root word.
* E.g. [running, flying] to [run, fly]

#### Lemmatization
* Lemmatization combines words using suffixes without altering the words' meaning.
* E.g. [quicker, browner, foxes] to [quick, brown, fox]

#### Tokenization
* Tokenization separates text into tokens, e.g. words.

#### NER
* NER stands for Named Entity Recognition.
* NER identifies entities such as the name of a person, place or organization.

#### N-gram
* N-gram is the type of parsing, where N is the no. of words that are parsed at a time.
* E.g. 1-gram: [The, quick, brown, fox], 2-gram: [The quick, quick brown, brown fox], 3-gram [The quick brown, quick brown fox].

#### POS tagging
* POS tagging stands for Parts-Of-Speech tagging.
* POS tagging assigns tags to words, such as nouns, adjectives, verbs.

#### Dependency parsing
* Dependency parsing (or syntactic parsing) assigns a syntactic structure, such as a parse tree.
* Dependency parsing is used in grammar checking.

#### Word similarity
* Word similarity can be measured by cosine distance between word vectors.
* Cosine distance = (A . B) / (||A|| * ||B||)

#### Perplexity
* Perplexity is the exponentiated average negative log-likelihood per token, or the probability distribution of words over the entire text.
* Perplexity can evaluate good language models by assigning a higher probability to the right prediction.
* Perplexity = exp(-log(p(string) / (no. of words/chars + 1 in the string)))

#### Levenshtein distance
* Levenshtein distance is the minimum edit distance (single-character edits) required to transform between words.

## DL for NLP - Deep Learning for NLP

#### LSTM
* LSTM is the Long Short Term Memory network.
* LSTM is a recurrent neural network that avoids the long-term dependency problem of RNNs.

#### Attention
* Attention mechanism enables prediction of an output word by using only relevant parts of the input instead the entire sentence.

#### Self-attention
* Self-attention mechanism relates different positions of the input sequence to compute a representation.

#### Multi-head attention
* Multi-head attention mechanism computes attention multiple times in parallel, then concatenated together.
* Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

#### BERT
* BERT is Bidirectional Encoder Representations from Transformers.
* BERT uses Masked LM (MLM) to perform bidrectional training in models.
* BERT trains by masking 15% of words in a sequence and evaluates the prediction of the masked words.

#### GPT-2
* GPT is Generative Pretrained Transformer (with GPT-2, GPT-3 versions).
