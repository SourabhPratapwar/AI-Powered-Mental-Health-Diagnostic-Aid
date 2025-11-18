# AI-Powered Mental Health Diagnostic Aid

## **Abstract**

Mental health disorders such as anxiety, depression, bipolar disorder, personality disorders, and stress are challenging to diagnose due to their subjective nature and overlapping symptoms. This project presents an **AI-powered diagnostic support system** that analyzes written language using **natural language processing (NLP)** and **machine learning (ML)** to identify mental health indicators. The system incorporates text preprocessing, feature extraction, and multi-label classification using Logistic Regression, Support Vector Machine (SVM), Long Short-Term Memory (LSTM), and a fine-tuned **BERT transformer**. Experimental findings show that BERT outperforms classical models, achieving superior accuracy and reliability. A Streamlit-based reporting interface demonstrates the system’s potential utility in clinical pre-screening contexts.

## **I. Introduction**

Accurate and early detection of mental health conditions is essential for effective intervention but remains limited by subjective evaluation. NLP advancements provide an opportunity to systematically analyze written text for emotional and cognitive cues associated with mental health disorders. This project investigates the viability of automated text-based screening and compares multiple ML models for predicting mental health categories.

## **II. Research Objectives**

The objectives of this project are:

1. To analyze linguistic markers present in mental-health-related text data.
2. To compare classical machine learning and transformer-based models for multi-label prediction.
3. To develop a classification system capable of detecting co-occurring mental health conditions.
4. To design a clinician-oriented diagnostic reporting interface.

## **III. System Architecture**

The system consists of six primary modules:

### **A. Data Acquisition**

Text samples were sourced from mental-health-related online communities (e.g., Reddit), labeled across five conditions.

### **B. Preprocessing Module**

* Text cleaning
* Tokenization
* Lemmatization
* Stop-word removal
* TF-IDF vectorization
* SMOTE-based oversampling

### **C. Feature Extraction Module**

* Keyword extraction
* Sentiment indicators
* TF-IDF n-grams
* **BERT contextual embeddings**

### **D. Machine Learning Models**

Evaluated models include:

* Logistic Regression
* SVM
* LSTM
* **BERT Transformer (fine-tuned)**

### **E. Evaluation Metrics**

* Micro & Macro F1 Score
* Subset Accuracy
* Hamming Loss

### **F. Diagnostic Reporting Interface**

A Streamlit application (`final_app.py`) provides condition probabilities, key indicators, and clinical summaries.

## **IV. Dataset Description**

The dataset consists of user-generated text posts labeled with:

* Anxiety
* Depression
* Stress
* Bipolar Disorder
* Personality Disorder

The labels are inherently **multi-label**, reflecting real-world comorbidity among mental health conditions.

## **V. Methodology**

1. **Preprocessing:**
   Standard NLP transformations and TF-IDF vectorization.

2. **Feature Engineering:**
   Statistical features and contextual embeddings via BERT.

3. **Model Training:**
   Comparison of classical ML models with transformer-based architectures.

4. **Model Evaluation:**
   Performance measured using multi-label metrics.

5. **Clinical Output:**
   Model results compiled into interpretable diagnostic summaries.

## **VI. Experimental Results**

### **A. Classical Models**

* **SVM:** 76.7% accuracy
* **Logistic Regression:** 76.02% accuracy
* **LSTM:** 23.71% accuracy (performance degraded due to dataset size/imbalance)

### **B. BERT Model (Best Results)**

* Micro F1 Score: ~0.73
* Macro F1 Score: ~0.73
* Subset Accuracy: ~69.9%
* Hamming Loss: **0.11**

BERT’s contextual understanding of semantics provides significantly stronger performance across all mental health categories.

## **VII. Application Interface**

The Streamlit interface enables:

* User text input
* Real-time multi-label predictions
* Condition probability visualization
* Emotional and linguistic cue analysis
* Summary diagnostic reporting

## **IX. Conclusion**

This study demonstrates that transformer-based NLP, particularly BERT, is highly effective for automated multi-label mental health classification based on linguistic patterns. The system offers a promising foundation for clinical screening support tools, educational research, and digital mental health platforms.

## **X. Future Work**
* Integration of speech-based emotion recognition
* Multilingual dataset expansion
* Inclusion of explainability frameworks (e.g., SHAP, LIME, attention visualization)
* Longitudinal analysis of user emotional trends
* Deployment in healthcare-compliant environments
