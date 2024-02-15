# nlp-for-contract-review

## Attribution Guidelines
Please note that the research paper for this project has now been published in Springer Artificial Intelligence and Law, which can be accessed here: https://link.springer.com/article/10.1007/s10506-023-09379-2#article-info

Kindly attribute any use of the dataset in this repository using the following citation: 

Graham, S.G., Soltani, H. & Isiaq, O. Natural language processing for legal document review: categorising deontic modalities in contracts. Artif Intell Law (2023). https://doi.org/10.1007/s10506-023-09379-2

Please also ensure that proper credit is given to the original paper from which the dataset was derived:

CUAD: Hendrycks D, Burns C, Chen A, Ball S (2021) CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. 35th Conference on Neural Information Processing Systems (NeurIPS 2021) Track on Datasets and Benchmarks https://doi.org/10.48550/arXiv.2103.06268 

While the dataset was created using publicly available documents, individuals using the dataset are responsible for undertaking their own ethics due diligence.

## Research summary
There are several approaches to contract review with natural language processing including topical and functional classification. Topical classification involves the classification of clauses in a contract according to their subject matter for example ‘governing law’, ‘termination clause’, ‘confidentiality clause’, and ‘data protection’. On the other hand, functional classification categorises contract sentences according to deontic modality.

Deontic modality refers to how norms and attitudes are expressed in the form of permissions, obligations, and prohibitions.

This project was undertaken for an MSc Applied AI & Data Science degree at Solent University, Southampton. It involved the manual annotation of contracts to create a small dataset which was used to train various traditional machine learning and neural network models to classify contract sentences using deontic labels. The best performing model was the convolutional neural network which achieved minimal ranking loss of 0.02 and up to 98% precision.

Traditional ML models: Naive Bayes, Logistic Regression, Support Vector Machine (SVM), and SVM with SGD

Neural network models: Convolutional Neural Network (CNN), CNN + law2vec, Long Short-Term Memory Recurrent Neural Network

The code also includes a script for a simple web-based interface, which accepts sentences sourced from outside the corpus and identifies firstly whether the sentence contains a deontic modality. If yes, the application identifies the deontic modalities (permission, obligation, prohibition) present. This interface was designed using PyWebio.

Data: The dataset was created from contracts which are a part of the Contract Understanding Atticus Dataset, a dataset created by The Atticus Project for use in topical classification tasks. Read more here: https://www.atticusprojectai.org
