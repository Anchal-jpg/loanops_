# Viva Preparation

## 30 short questions

1. **Project kya karta hai?** Loan application ke approve/reject hone ki probability predict karta hai.
2. **MLOps kyun?** Model ko reproducibly train, deploy aur monitor karne ke liye.
3. **DVC vs Git?** Git code rakhta hai; DVC large data versions rakhta hai.
4. **MLflow ka role?** Runs, metrics, artifacts aur model versions track karta hai.
5. **Registry kya hai?** Deployable model versions ka central record.
6. **Champion alias kya hai?** Current approved model ko stable naam se access karne ka tareeka.
7. **Quality gate?** New model tabhi promote hota hai jab ROC-AUC minimum improvement de.
8. **Pipeline kyun?** Training aur serving preprocessing same rehti hai.
9. **Data drift?** Input feature distribution time ke saath change hona.
10. **Concept drift?** Features aur target ka relationship change hona.
11. **KS test?** Numeric distributions compare karta hai.
12. **Chi-square?** Categorical distributions compare karta hai.
13. **Evidently?** Drift report generate karne ka monitoring tool.
14. **Fallback kyun?** Monitoring demo dependency failure par bhi useful rahe.
15. **ROC-AUC?** Ranking quality measure; 1.0 best, 0.5 random.
16. **F1?** Precision aur recall ka harmonic mean.
17. **Recall important kyun?** Genuine approvals ko miss na karne ke liye.
18. **Class imbalance?** Classes ke unequal counts; balanced weights use kiye.
19. **XGBoost?** Gradient boosted decision trees ka efficient implementation.
20. **SHAP?** Prediction ke feature-level contributions explain karta hai.
21. **Fairness report?** Gender aur Married groups ke approval/recall compare karta hai.
22. **FastAPI?** Typed, fast prediction REST API.
23. **Pydantic?** Request validation aur 422 errors provide karta hai.
24. **Docker?** Same runtime environment package karta hai.
25. **CI?** Push/PR par automated checks.
26. **CD?** Main branch se image build/publish/deploy.
27. **Training-serving skew?** Train aur production preprocessing mismatch.
28. **Random seed?** Reproducible results ke liye fixed value.
29. **Limitations?** Synthetic data aur local filesystem production-grade nahi hain.
30. **Future work?** Cloud artifact store, auth, real feedback labels aur scheduled retraining.

## Five-minute demo

1. Person 1: sample data, validation and DVC pipeline explain kare.
2. Person 2: `python -m src.train` aur MLflow runs/`champion` alias dikhaye.
3. Person 3: FastAPI `/docs`, `/health`, `/predict` aur Streamlit UI dikhaye.
4. Person 4: simulate drift, HTML report, retrain trigger aur tests explain kare.

Backup: pre-generated reports aur local Docker image use kare; filesystem ephemeral hosting limitation clearly bataye.
