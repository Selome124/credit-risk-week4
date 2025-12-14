# Credit Risk Model

End-to-end credit risk modeling project with:
- Data processing
- Model training
- FastAPI inference service
- Dockerized deployment
- CI/CD pipeline

## Run locally
```bash
uvicorn src.api.main:app --reload

Credit Scoring Business Understanding
Basel II, Risk Measurement, and Model Interpretability

The Basel II Capital Accord places strong emphasis on accurate risk measurement, transparency, and governance in credit risk management. Financial institutions are required to justify how credit risk is measured and how capital requirements are determined. As a result, credit scoring models must be interpretable, auditable, and well-documented, not only to satisfy regulators but also to support internal risk management and decision-making. Interpretable models allow risk managers and regulators to understand how input variables influence credit decisions, validate assumptions, detect bias, and ensure consistency with regulatory standards. Poorly documented or opaque models increase model risk and may lead to regulatory penalties or loss of trust.

Need for a Proxy Default Variable and Associated Risks

In this project, a direct and explicit “default” label is not available. Therefore, it is necessary to construct a proxy variable (for example, based on delinquency status, missed payments, or extreme financial behavior) to approximate default behavior. This proxy enables supervised learning and allows the model to identify patterns associated with higher credit risk. However, using a proxy introduces business and model risks, including label noise, misclassification, and potential bias. If the proxy does not accurately represent true default behavior, the model may overestimate or underestimate risk, leading to poor lending decisions, unfair customer treatment, and incorrect capital allocation.

Trade-offs Between Interpretable and Complex Models

There is a fundamental trade-off between model interpretability and predictive performance in regulated financial environments. Simple models such as Logistic Regression with Weight of Evidence (WoE) are highly interpretable, stable, and easy to explain to regulators and business stakeholders. They support clear variable-level insights and align well with traditional scorecard approaches used in banking. However, they may have limited predictive power when relationships are highly nonlinear. In contrast, complex models like Gradient Boosting often achieve higher accuracy and capture nonlinear interactions but suffer from reduced transparency and explainability. In a regulated context, this complexity increases model governance costs, validation effort, and regulatory scrutiny. Therefore, model selection must balance performance gains against explainability, regulatory compliance, and operational risk.

## Task 3 - Feature Engineering

Run feature engineering test:

```bash
python src/test_feature_engineering.py

