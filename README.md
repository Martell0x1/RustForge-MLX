# Hello

- the picture
```md
                    rust-ml
                       │
        ┌──────────────┴──────────────┐
        │                             │
   Mathematics                   Data system
        │                             │
 Linear Algebra                 Dataset
 Probability                   CSV
 Statistics                    Train/test split
 Optimization                 Normalization
        │                             │
        └──────────────┬──────────────┘
                       │
                Classical ML
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   Regression      Classification   Clustering
        │              │              │
     Linear          Logistic         K-Means
     Ridge           KNN              DBSCAN
     Lasso           Naive Bayes      GMM
     Polynomial      Decision Tree
                    Random Forest
                    SVM
                       │
                       ▼
                Model Evaluation
                       │
          ┌────────────┼────────────┐
          │            │            │
        Metrics      CV            Tuning
          │            │            │
       Accuracy     K-Fold        Grid Search
       Precision    Stratified    Random Search
       Recall
       F1
       ROC-AUC
                       │
                       ▼
                 Neural Networks
                 (later)
