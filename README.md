# Feature Selection For Clickbait Detection

The purpose of this project is to determine which features are useful in predicting clickbait headlines.

The following tools were used:
* [Scikit-learn](https://scikit-learn.org/stable/)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/)
* [spaCy](https://spacy.io/)
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)

## Background

In recent decades, online news media has become a primary source of information. As the internet's information grows rapidly and competition among providers increases, attracting readers' attention is crucial. Clickbait headlines, designed to pique curiosity, have become a popular strategy, though they often disappoint by not delivering as promised. This project focuses on automatically detecting clickbait headlines by classifying them as either clickbait or non-clickbait. This way, online readers can be warned about potential clickbait links, allowing them to make informed decisions before clicking.

## Dataset

This research used the dataset from [here](https://github.com/bhargaviparanjape/clickbait/tree/master/dataset), which included 15,999 clickbait headlines and 16,001 non-clickbait headlines. To balance the data, we randomly selected 10,000 clickbait and 10,000 non-clickbait headlines from the original dataset. To ensure consistent feature extraction, all headlines were converted to lowercase, although this may slightly affect proper noun identification during part-of-speech tagging. However, since nouns and proper nouns weren't used as features, this impact should be negligible.

## Features

Here are the features used in this project. All features, except sentiment analysis, were extracted using spaCy and the en_core_web_lg model. SpaCy was chosen for its speed, particularly with large datasets where computational complexity is a concern. Due to the reliance on POS tagging for most features, spaCy was selected to minimize errors during feature extraction. Stanford CoreNLP was utilized for sentiment analysis.

1. Sentence Structure
  - Number of words in a headline: Clickbait headlines tend to be longer than non-clickbait headlines due to the need to attract readers with interesting phrases and gain access numbers, resulting in the inclusion of longer phrases like "You Won't Believe...". ![Difference in the average number of words between clickbait and non-clickbait headlines](/plots/avg_n_words.png)
  - The average length of a word: Clickbait headlines, despite being longer, typically consist of shorter words compared to non-clickbait headlines, possibly due to the inclusion of informal abbreviated words and a higher usage of specific or technical words in non-clickbait headlines that provide more context. ![Difference in the average word length between clickbait and non-clickbait headlines](/plots/avg_word_len.png)
2. Words and Phrases
  - Percentage of stop words in a headline: Clickbait headlines have a higher occurrence of stop words, constituting over 40% of the content, while non-clickbait headlines use stop words less frequently, below 25%, possibly because non-clickbait headlines prioritize concise yet context-rich content, avoiding words that don't contribute to the readers' understanding of the article's topic. ![Difference in the average stop words usage between clickbait and non-clickbait headlines](/plots/stop_word_pct.png)
  - Inclusion of a number in a headline: Clickbait headlines frequently incorporate numbers, which are approximately twice as prevalent compared to non-clickbait headlines, often utilized for listing purposes, such as "12 Mind-Blowing Ways To Eat Polenta," while non-clickbait headlines sparingly include numbers to provide additional context.

![Percentage of including number, determiner, pronoun, comparative words, superlative words, and one of comparative and superlative words between clickbbait and non-clickbait headlines](/plots/pos_pct.png)

3. Part-Of-Speech Tagging
  - Inclusion of a determiner in a headline: Clickbait headlines exhibit a significantly higher frequency of determiner usage compared to non-clickbait headlines, as determiners do not contribute contextual information, thus non-clickbait headlines prioritize conciseness by omitting them.
  - Inclusion of a pronoun in a headline: There is a significant contrast in pronoun usage between clickbait and non-clickbait articles, with a mere 3% of non-clickbait headlines incorporating pronouns, whereas 55% of clickbait headlines employ pronouns to establish a personal connection with readers and entice them to click the article.
  - Inclusion of a superlative adjective/adverb in a headline: Clickbait headlines make greater use of superlative words to amplify the subject, while non-clickbait articles are less inclined to include them due to their subjective nature, resulting in a higher likelihood of encountering superlatives in clickbait headlines compared to non-clickbait headlines.
  - Inclusion of a comparative adjective/adverb in a headline: Clickbait headlines may employ comparative words more often, albeit with a marginal difference, to enhance their exaggeration, while both clickbait and non-clickbait headlines show a comparable likelihood of featuring comparative words.
  - Inclusion of a superlative and/or comparative adjective/adverb in a headline: To avoid redundancy and capture the presence of comparative or superlative words, a combined feature was created to detect their occurrence in headlines, primarily focusing on clickbait headlines due to their higher likelihood, while considering the usefulness of these features during feature selection.
4. Sentiment Analysis
  - Sentiment score: Research shows that clickbait headlines are more likely to use words with a "Very Positive" sentiment, such as "Inspiring" or "Awesome", which are rarely found in non-clickbait headlines. Sentiment analysis, particularly using Stanford CoreNLP, helps identify the emotional impact of words in headlines and is more effective than word count approaches. The sentiment score range between 0 to 4 where 0 means “Very Negative” and 4 means “Very Positive”. Clickbait headlines often utilize a mix of very positive and negative words to create an enticing teaser, and very few headlines have a sentiment value of 0 for both clickbait and non-clickbait. ![Count of sentences classified in different sentiment values between clickbait and non-clickbait headlines](/plots/avg_sentiment_score.png)

## Feature Selection

Applying Occam's razor principle, we conducted feature selection using XGBoost to choose the most relevant features based on their importance in splitting trees and achieving homogeneous outcomes. Among the features, superlative, comparative, and superlative or comparative did not contribute significantly, and their removal did not impact the initial exploratory performance, leading to their exclusion. `XGBoost Release 1.1.0-SNAPSHOT` was utilized, specifically the `get_score(importance_type='weight')` function, to determine feature importance. ![Feature importance generated by XGBoost. The xaxis depict the features and the y-axis depict the weights of each features](/plots/feature_imp.png)

## Classification Algorithm

Clickbait detection involves binary classification, where headlines are categorized into either clickbait or non-clickbait classes based on the features mentioned earlier.

- Support Vector Machine: To overcome the challenge of high-dimensional data and improve classification accuracy, Support Vector Machine (SVM) with a Radial Basis Function Kernel (RBF) is used to find the best hyperplane that maximizes the distance between classes, while a penalty parameter is utilized to control misclassification. `scikit-learn.svm.SVC` was used to implement the SVM classifier and the default value of the hyper-parameters was used.
- Random Forest: Random Forest (RF) is preferred for large datasets to handle noise and control error internally, as it comprises multiple decision trees, each predicting the observation's class, ensuring uncorrelation through bagging (random selection of observations with replacement) and feature randomness (selecting from random feature subsets). The RF classifier was implemented using `scikit-learn.ensemble.RandomForestClassifier` with the default hyper-parameter values, including 100 trees due to a future warning about the default changing from 10 to 100 trees.
- XGBoost: XGBoost (Extreme Gradient Boosting), a popular algorithm known for its state-of-the-art performance and computational speed, was chosen for this project due to its boosting technique that transforms weak learning models into strong ones. XGBoost has a stop criterion for tree splitting, reducing computation time, and it can learn implicit feature relationships and find optimal split points in the dataset. The XGBClassifier from the `xgboost` library, with default hyper-parameter values, was used to implement the XGBoost classifier

## Evaluation

- Cross-Validation: For a classifier to generalize well, it needs low bias and low variance. Bias represents the difference between average predicted and actual clickbait headlines, while variance shows the data spread. Cross-validation with 10 folds was used to reduce bias and variance, enabling better model performance.
- Accuracy, Precision, Recall, F1-Score: To evaluate the classifiers, accuracy alone is not sufficient, as it assumes equal importance of false positives and false negatives. Therefore, the F1 score, which considers precision and recall, is calculated as a weighted average. Precision is the ratio of true positives to the total number of clickbait headlines, while recall is the ratio of true positives to all clickbait headlines. The classifier's performance would be assessed based on accuracy, precision, recall, and the F1 score, with values above 0.50 indicating good performance.
- Confusion Matrices, ROC, AUC: Confusion matrices provide a summary of performance metrics and help identify common errors, while also serving as the basis for creating ROC curves. ROC curves plot TPR against FPR, and classifiers with higher TPR are considered to perform well. The AUC of the ROC curve indicates the classifier's ability to differentiate clickbait and non-clickbait headlines, with an AUC of 1 indicating perfect classification and 0.5 indicating chance-level performance.

## Results

- Accuracy, Precision, Recall, F1-Score: Table below presents averaged accuracy, precision, recall, and F1-score for each classifier based on cross-validation testing, indicating that the classifiers exhibit similar performance.

    | Classifier | Accuracy | Precision | Recall | F1-Score |
    |------------|----------|-----------|--------|----------|
    | SVM        | 0.868    | 0.886     | 0.844  | 0.865    |
    | RF         | 0.855    | 0.846     | 0.889  | 0.853    |
    | XGBoost    | 0.876    | 0.862     | 0.859  | 0.874    |

- Confusion Matrices: In figure below, the confusion matrices of all classifiers reveal a higher frequency of misclassifying clickbait as non-clickbait compared to misclassifying non-clickbait as clickbait. ![SVM: TP=8456 TN=8912 FP=1088 FN=1544](/plots/cm_svm.png) ![RF: TP=8434 TN=8620 FP=1380 FN=1566](/plots/cm_rbf.png) ![XGBoost: TP=8586 TN=8931 FP=1069 FN=1414](/plots/cm_xgb.png)
- Receiver Operating Characteristic and Area Under this Curve: In figure below, the ROC curves and corresponding AUC values for each classifier indicate similar performance among them, suggesting that the classifier seem to perform fairly well. ![Receiver Operating Characteristic for each of the classifier; SVM (blue), RF (yellow), XGBoost (pink). The x-axis show the false positive rate and the y-axis show the true positive rate. The grey dashed line depicts classification by chance.](/plots/roc_auc.png)

The project aimed to identify useful features for predicting clickbait headlines. Results indicate that superlative and comparative words are not important for classifying clickbaits. Numerical features (word count, word length) contribute more than binary features (pronouns, determiners) suggesting that using POS-tag distribution as features might be more valuable than checking for presence alone. Classification results show effective prediction of clickbait headlines, with F1-scores ranging from 0.85 to 0.87, indicating good classification. The AUC values further support this. However, confusion matrices reveal a higher frequency of misclassifying clickbait as non-clickbait, potentially due to mislabeling of the data instead of the classification technique itself. The dataset used in this project lacked the manually labeled data provided in Chakraborty et al.'s research. Compared to another academid research on this topic, this project's classifiers performed less effectively, potentially due to untuned hyperparameters and the use of fewer features focused on POS-tagging, sentence structure, and sentiment analysis, as opposed to N-grams.