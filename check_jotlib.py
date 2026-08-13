import joblib
from sklearn.base import BaseEstimator

model_path = "topic_classifier.joblib"  # 또는 .joblib

# 1. 로드
model = joblib.load(model_path)

# 2. 타입 체크
assert isinstance(model, BaseEstimator), "로드된 객체가 scikit-learn 모델이 아닙니다."

# 3. 학습 여부 체크 (대부분의 estimator 는 fitted_ 속성이나 is_fitted() 제공)
from sklearn.utils.validation import check_is_fitted

try:
    check_is_fitted(model)
    print("✓ 모델이 학습된 상태입니다 (fitted).")
except Exception as e:
    raise RuntimeError(f"모델이 학습되지 않았습니다: {e}")

# 하이퍼파라미터
print("하이퍼파라미터:", model.get_params())

# 학습된 파라미터 예시 (모델 종류에 따라 다름)
if hasattr(model, "coef_"):
    print("계수 shape:", model.coef_.shape)
if hasattr(model, "intercept_"):
    print("절편:", model.intercept_)
if hasattr(model, "feature_names_in_"):
    print("사용된 피처 수:", len(model.feature_names_in_))1