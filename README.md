# AI-Powered-Mental-Health-Diagnostic-Aid
This repository contains the implementation, experiments, and application code for an AI-driven mental health diagnostic aid that analyzes written language to identify potential indicators of mental health conditions. The system leverages natural language processing (NLP) and machine learning, including Logistic Regression, Support Vector Machines, LSTM networks, and BERT-based transformers, to perform multi-label classification of mental health states such as stress, depression, anxiety, bipolar disorder, and personality disorder.
 
1. Introduction
Mental health conditions are often underdiagnosed due to subjective evaluation and limited clinical assessment time.
This project explores the feasibility of using computational linguistic analysis and supervised learning to support clinicians in early identification of mental health indicators.
The proposed system evaluates linguistic markers extracted from written text and produces structured diagnostic insights, enabling:
•	Early detection of at-risk individuals
•	Preliminary assessment prior to clinical interviews
•	Reduction of diagnostic subjectivity
•	Enhanced consistency and screening efficiency
 
2. Objectives
The study aims to investigate:
  •	Whether linguistic patterns can reliably indicate mental health conditions.
  •	How supervised machine learning and transformer models compare in identifying these indicators.
  •	How effectively a multi-label framework can capture co-occurring mental health symptoms.
  •	Development of an interpretable diagnostic report summarizing sentiment, emotional tones, and key textual cues.

3. System Architecture
The system is composed of six research modules:
  3.1 Data Acquisition
    Textual data was collected from mental health–related sources, including Reddit mental health posts, covering categories such as anxiety, depression, bipolar disorder, personality         disorders, and stress.
  3.2 Preprocessing Pipeline
    •	Lowercasing
    •	Punctuation and noise removal
    •	Tokenization
    •	Lemmatization
    •	Stopword filtering
    •	TF–IDF vectorization
    •	Handling class imbalance via SMOTE
  3.3 Feature Extraction
  The project evaluates both handcrafted and contextual features:
    •	TF–IDF n-grams
    •	Keyword frequency
    •	Sentiment orientation
    •	Contextual embeddings via BERT
  3.4 Machine Learning Models
  Experiments were conducted with four model architectures:
    •	Logistic Regression
    •	Support Vector Machine (SVM)
    •	LSTM Neural Network
    •	BERT Transformer (Fine-Tuned)
  3.5 Evaluation Metrics
  The research employed standard multi-label metrics including:
    •	Micro F1 Score
    •	Macro F1 Score
    •	Subset Accuracy
    •	Hamming Loss
  3.6 Report Generation
  A clinician-oriented report is produced summarizing:
    •	Detected conditions
    •	Confidence scores
    •	Key emotional indicators
    •	Dominant linguistic cues
 
4. Dataset Description
The dataset consists of short written posts that may express one or more mental health conditions simultaneously.
Classes include:
  •	Anxiety
  •	Depression
  •	Stress
  •	Bipolar Disorder
  •	Personality Disorder
The dataset reflects co-occurrence and imbalance, making multi-label classification essential.
 
5. Experimental Results
Model Performance Summary
Model	Overall Accuracy	Key Observations
BERT (Best Performer)	~70%	Strong contextual understanding; high F1; lowest Hamming Loss
SVM	76.7%	Effective for high-dimensional TF–IDF features
Logistic Regression	76.02%	Balanced baseline; interpretable
LSTM	~24%	Underperformed due to data size and imbalance
BERT Highlights
  •	Micro-F1: ~0.73
  •	Macro-F1: ~0.73
  •	Hamming Loss: 0.11
  •	Excellent performance on depression detection
  •	Stable across multiple conditions
BERT demonstrated superior generalization and contextual inference, making it the optimal model for the final system.
 
6. Application Interface
A lightweight Streamlit application (final_app.py) provides:
  •	Real-time mental health prediction from user text
  •	Probability distribution across all five labels
  •	Highlighted linguistic cues and emotional descriptors
  •	A summary report suitable for clinician review
This interface demonstrates the practical deployment potential of the research.

8. Conclusion
The study demonstrates that NLP-based mental health analysis is feasible and effective, particularly with transformer architectures.
The findings indicate:
  •	BERT provides the most robust performance
  •	Linguistic markers contain measurable mental-health-related signals
  •	Multi-label classification captures co-occurring conditions accurately
  •	Automated reporting can support clinicians in early screening
The project lays the groundwork for future research in clinical NLP, explainable AI, and mental health informatics.
 
9. Future Work
Potential research extensions include:
  •	Speech-based emotion recognition integration
  •	Multilingual model adaptation
  •	Advanced explainability mechanisms (e.g., SHAP, LIME, attention maps)
  •	Clinical validation with mental health professionals
  •	Deployment on secure, healthcare-compliant platforms
