# HELOC Loan Approval – ML-Based Decision Support System

Supervised machine learning project that builds and evaluates a binary classification model for initial screening of Home Equity Line of Credit (HELOC) applications. The system is designed as a decision-support component, providing risk-based recommendations and explanations to assist loan officers rather than replacing human judgment.

1. Business Context
Banks process large volumes of HELOC applications that are traditionally reviewed manually by expert loan officers. This process is time-consuming, costly, and difficult to scale.
This project explores whether historical application data can be used to automate initial screening while maintaining regulatory compliance and human oversight.

The system supports two possible outcomes:
- Negative → Application denied
- Positive → Application forwarded to loan officer for review

If denied, the model’s outputs and interpretability artifacts are used to support explanation and future guidance.

2. Problem Formulation
- Task: Binary classification
- Target: Loan outcome / repayment-based risk indicator
- Input: Structured applicant features
- Output: Predicted risk class + probability score
The model is optimized for discriminatory power while allowing threshold tuning based on business risk tolerance.

3. Dataset
- ~10,000 historical HELOC applications
- Structured tabular features describing applicant financial and credit attributes
- Binary label indicating loan outcome / risk status
(Note: Dataset not included in this repository.)

4. Approach
- Data loading and inspection
- Exploratory data analysis (EDA)
- Data cleaning and preprocessing
- Train/ validation split
- Supervised model training
- Model evaluation
- Error analysis and threshold inspection
- Basic interpretability analysis
- The full workflow is implemented in Python.

5. Models Considered
- Logistic Regression
- Tree-based models (e.g., Decision Tree/ Random Forest/ Gradient-based)
Final model selection is based on validation performance and stability rather than maximum complexity.

6. Evaluation Metrics
- ROC-AUC (primary)
- Precision
- Recall
- Accuracy
- Confusion Matrix
ROC-AUC is used as the primary metric to evaluate ranking quality across thresholds. Precision/recall are used to analyze business tradeoffs between false approvals and false rejections.

7. Decision Support Framing
- This system is intentionally not fully automated approval.
Instead:
- Model produces risk probability
- Threshold determines deny vs forward-to-review
- Loan officers retain final authority

This design supports:
- Faster initial screening
- Consistent application of risk logic
- Regulatory-aligned human oversight

10. Key Outputs
- Trained classification model
- Performance metrics
- Confusion matrix
- ROC curve

11. Limitations
- Static historical dataset
- No production deployment pipeline
- No automated model monitoring
- Threshold selection based on offline analysis

12. Future Improvements
- Cross-validation and hyperparameter tuning
- Advanced feature engineering
- SHAP-based explanations
- Drift detection and monitoring
- Streamlit front-end for interactive DSS
