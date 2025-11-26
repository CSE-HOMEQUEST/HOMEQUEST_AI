import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

np.random.seed(42)

# 1. 챌린지 메타 정보 정의 -------------------------------------------------

challenge_meta = pd.DataFrame([
    # id, type, category, cooldown_days
    ("daily_water_2",      "daily",  "health",      0),
    ("daily_walk_6000",    "daily",  "health",      0),
    ("daily_clean_myroom", "daily",  "cleaning",    2),
    ("daily_laundry_sort", "daily",  "laundry",     3),
    ("daily_dish_1",       "daily",  "dishwashing", 0),
    ("daily_trash",        "daily",  "dishwashing", 0),
    ("daily_tv_less",      "daily",  "energy",      0),
    ("daily_stretch",      "daily",  "health",      0),
    ("daily_ventilation",  "daily",  "energy",      0),
    ("daily_robot_clean",  "daily",  "cleaning",    2),
    ("speed_dishwasher",   "speed",  "dishwashing", 0),
    ("speed_laundry_in",   "speed",  "laundry",     3),
    ("speed_robot_first",  "speed",  "cleaning",    2),
    ("speed_trash_first",  "speed",  "dishwashing", 0),
    ("speed_shower_first", "speed",  "health",      0),
    ("speed_stretch_first","speed",  "health",      0),
    ("speed_fridge_first", "speed",  "dishwashing", 0),
    ("speed_laundry_fold", "speed",  "laundry",     3),
    ("speed_water_first",  "speed",  "health",      0),
    ("speed_random_clean", "speed",  "cleaning",    2),
    ("monthly_heating",    "monthly","energy",      0),
    ("monthly_energy",     "monthly","energy",      0),
    ("monthly_steps",      "monthly","health",      0),
    ("monthly_robot_20",   "monthly","cleaning",    0),
    ("monthly_dish_20",    "monthly","dishwashing", 0),
], columns=["challenge_id", "challenge_type", "category", "cooldown_days"])

# 2. 가상 로그 데이터 생성 -------------------------------------------------

n_families = 5
n_users_per_family = 4
family_ids = [f"fam_{i}" for i in range(n_families)]
user_ids = [f"user_{i}" for i in range(n_families * n_users_per_family)]

n_samples = 6000

rows = []
for i in range(n_samples):
    date_idx = np.random.randint(0, 60)  # 지난 60일
    day_of_week = date_idx % 7
    is_weekend = 1 if day_of_week in [5, 6] else 0
    hour_bucket = np.random.choice(["morning", "afternoon", "evening", "night"])
    family = np.random.choice(family_ids)
    user = np.random.choice(user_ids)
    ch = challenge_meta.sample(1).iloc[0]

    # 유저/가족 성향 가상값
    user_success_30d = np.clip(np.random.normal(0.6, 0.15), 0, 1)
    user_daily_success = np.clip(user_success_30d + np.random.normal(0, 0.05), 0, 1)
    user_speed_success = np.clip(user_success_30d + np.random.normal(0, 0.05), 0, 1)
    user_monthly_success = np.clip(user_success_30d + np.random.normal(0, 0.05), 0, 1)

    recent_complete_7d = max(0, int(np.random.poisson(3)))
    recent_skip_7d = max(0, int(np.random.poisson(2)))

    challenge_seen_30d = max(1, int(np.random.poisson(3)))   # 최소 1
    challenge_completed_30d = max(0, int(challenge_seen_30d * np.random.uniform(0.3, 0.8)))
    challenge_user_success_rate = challenge_completed_30d / challenge_seen_30d

    days_since_last_seen = np.random.randint(0, 15)
    days_since_last_completed = np.random.randint(0, 20)

    # 3. label을 만들기 위한 "진짜" 내부 규칙(가상) -----------------------
    base = 0.3
    # 타입별 가중치
    if ch.challenge_type == "daily":
        base += 0.05
    if ch.challenge_type == "speed":
        base += 0.0
    if ch.challenge_type == "monthly":
        base += 0.02

    # 카테고리별 가중치
    if ch.category == "health":
        base += 0.1
    if ch.category == "energy":
        base += 0.05
    if ch.category == "cleaning":
        base -= 0.05
    if ch.category == "laundry":
        base -= 0.03

    # 유저 성향 영향
    base += 0.2 * (user_success_30d - 0.5)
    base += 0.1 * (challenge_user_success_rate - 0.5)

    # 주말에는 데일리/스피드 수행률 약간 상승
    if is_weekend and ch.challenge_type in ["daily", "speed"]:
        base += 0.05

    # 너무 자주 본 챌린지는 피곤해서 수행률 감소
    base -= 0.01 * challenge_seen_30d

    # logit → 확률
    logit = base
    prob = 1 / (1 + np.exp(-logit))
    label = np.random.rand() < prob

    rows.append({
        "family_id": family,
        "user_id": user,
        "date_idx": date_idx,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "hour_bucket": hour_bucket,
        "recent_complete_7d": recent_complete_7d,
        "recent_skip_7d": recent_skip_7d,
        "user_success_30d": user_success_30d,
        "user_daily_success": user_daily_success,
        "user_speed_success": user_speed_success,
        "user_monthly_success": user_monthly_success,
        "challenge_id": ch.challenge_id,
        "challenge_type": ch.challenge_type,
        "category": ch.category,
        "cooldown_days": ch.cooldown_days,
        "challenge_seen_30d": challenge_seen_30d,
        "challenge_completed_30d": challenge_completed_30d,
        "challenge_user_success_rate": challenge_user_success_rate,
        "days_since_last_seen": days_since_last_seen,
        "days_since_last_completed": days_since_last_completed,
        "label": int(label)
    })

data = pd.DataFrame(rows)

# 4. 전처리: 원-핫 인코딩 --------------------------------------------------

feature_cols = [
    "day_of_week", "is_weekend",
    "recent_complete_7d", "recent_skip_7d",
    "user_success_30d", "user_daily_success", "user_speed_success", "user_monthly_success",
    "cooldown_days",
    "challenge_seen_30d", "challenge_completed_30d", "challenge_user_success_rate",
    "days_since_last_seen", "days_since_last_completed",
    "hour_bucket", "challenge_type", "category"
]

X_raw = data[feature_cols].copy()
y = data["label"].values

X_encoded = pd.get_dummies(X_raw, columns=["hour_bucket", "challenge_type", "category"], drop_first=True)

# 5. 시간 기반 train/test 분리 --------------------------------------------

train_mask = data["date_idx"] < 45   # 첫 45일 학습
test_mask = data["date_idx"] >= 45   # 나머지 평가

X_train, X_test = X_encoded[train_mask], X_encoded[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# 6. 모델 학습 ------------------------------------------------------------

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. 성능 평가 ------------------------------------------------------------

y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("테스트 AUC:", round(auc, 4))

# 8. 오늘 추천: 데일리/스피드/먼슬리 각각 1개씩 ---------------------------

today_date_idx = 61
today_day_of_week = today_date_idx % 7
today_is_weekend = 1 if today_day_of_week in [5, 6] else 0
today_hour_bucket = "morning"

# 가족 상태: 최근 행동 로그를 일부 고정값으로 가정
family_state = {
    "recent_complete_7d": 5,
    "recent_skip_7d": 3,
}

user_state = {
    "user_success_30d": 0.65,
    "user_daily_success": 0.7,
    "user_speed_success": 0.6,
    "user_monthly_success": 0.55,
}

# 카테고리별 마지막 수행일(쿨다운 확인용) - 가상 데이터
last_done = {
    "cleaning": today_date_idx - 1,   # 어제 청소 함 -> cleaning 쿨다운 2일이면 오늘 제외
    "dishwashing": today_date_idx - 0, # 오늘 설거지 이미 함 (쿨다운 0이라 상관 없음)
    "laundry": today_date_idx - 2,    # 2일 전 빨래
    "health": today_date_idx - 5,
    "energy": today_date_idx - 10,
}

def build_feature_row(ch_row):
    """오늘 추천 상황에서 단일 챌린지에 대한 feature row 생성"""
    return {
        "day_of_week": today_day_of_week,
        "is_weekend": today_is_weekend,
        "recent_complete_7d": family_state["recent_complete_7d"],
        "recent_skip_7d": family_state["recent_skip_7d"],
        "user_success_30d": user_state["user_success_30d"],
        "user_daily_success": user_state["user_daily_success"],
        "user_speed_success": user_state["user_speed_success"],
        "user_monthly_success": user_state["user_monthly_success"],
        "cooldown_days": ch_row.cooldown_days,
        "challenge_seen_30d":  max(1, int(np.random.poisson(3))),
        "challenge_completed_30d": max(0, int(np.random.poisson(2))),
        "challenge_user_success_rate": np.clip(np.random.uniform(0.3, 0.8), 0, 1),
        "days_since_last_seen": np.random.randint(0, 10),
        "days_since_last_completed": today_date_idx - last_done.get(ch_row.category, today_date_idx),
        "hour_bucket": today_hour_bucket,
        "challenge_type": ch_row.challenge_type,
        "category": ch_row.category,
    }

def apply_cooldown(ch_row):
    """쿨다운 규칙 적용 (cleaning 2일, laundry 3일, dishwashing 0 등)"""
    cd = ch_row.cooldown_days
    cat = ch_row.category
    if cd == 0:
        return True
    last = last_done.get(cat, None)
    if last is None:
        return True
    days_diff = today_date_idx - last
    return days_diff >= cd

def recommend_by_type(ch_type):
    candidates = challenge_meta[challenge_meta["challenge_type"] == ch_type].copy()

    # 쿨다운 필터
    candidates["available"] = candidates.apply(apply_cooldown, axis=1)
    candidates = candidates[candidates["available"] == True]
    if len(candidates) == 0:
        return None

    # 피처 생성
    feat_rows = [build_feature_row(row) for _, row in candidates.iterrows()]
    feat_df = pd.DataFrame(feat_rows)
    feat_encoded = pd.get_dummies(feat_df, columns=["hour_bucket", "challenge_type", "category"], drop_first=True)

    # 학습 때 쓰인 컬럼과 맞추기 (없는 컬럼은 0으로 채우기)
    for col in X_encoded.columns:
        if col not in feat_encoded.columns:
            feat_encoded[col] = 0
    feat_encoded = feat_encoded[X_encoded.columns]

    scores = model.predict_proba(feat_encoded)[:, 1]
    candidates = candidates.reset_index(drop=True)
    candidates["score"] = scores

    best = candidates.sort_values("score", ascending=False).iloc[0]
    return best[["challenge_id", "challenge_type", "category", "score"]]

daily_rec = recommend_by_type("daily")
speed_rec = recommend_by_type("speed")
monthly_rec = recommend_by_type("monthly")

print("\n=== 오늘 추천 결과 (가상) ===")
print("데일리 추천:", daily_rec.to_dict() if daily_rec is not None else None)
print("스피드 추천:", speed_rec.to_dict() if speed_rec is not None else None)
print("먼슬리 추천:", monthly_rec.to_dict() if monthly_rec is not None else None)
