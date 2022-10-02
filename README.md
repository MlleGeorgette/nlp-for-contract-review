# nlp-for-contract-review

There are several approaches to contract review with natural language processing including topical and functional classification. Topical classification involves the classification of clauses in a contract according to their subject matter for example ‘governing law’, ‘termination clause’, ‘confidentiality clause’, ‘data protection’. On the other hand, functional classification categorises contract sentences according to deontic modality.

Deontic modality refers to the way norms and attitudes are expressed in the form of permissions, obligations, and prohibitions.

This project was undertaken for a degree in MSc Applied AI & Data Science at Solent University, Southampton. It involved the manual annotation of contracts to create a small dataset which was used to train various traditional machine learning and neural network models to classify contract sentences using deontic labels. The best performing model was the convolutional neural network which achieved minimal ranking loss of 0.02 and up to 98% precision.

Traditional ML models: Naive Bayes, Logistic Regression, Support Vector Machine (SVM), and SVM with SGD

Neural network models: Convolutional Neural Network (CNN), CNN + law2vec, Long Short-Term Memory Recurrent Neural Network

The code also includes a script for a simple web-based interface, which accepts sentences sourced from outside the corpus and identifies firstly whether the sentence contains a deontic modality. If yes, the application then identifies the deontic modalities (permission, obligation, prohibition) present. This interface was designed using PyWebio.

Data: The dataset was created from contracts which are a part of the Contract Understanding Atticus Dataset, a dataset created by The Atticus Project for use in topical classification tasks. Read more here: https://www.atticusprojectai.org
