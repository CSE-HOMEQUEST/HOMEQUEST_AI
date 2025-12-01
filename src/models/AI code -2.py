import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

np.random.seed(42)

# -------------------------------------------------------
# 1. 챌린지 메타 정의 (challengeId별 쿨다운 일수 포함)
# -------------------------------------------------------
challenge_meta = pd.DataFrame([
    # challengeId,        category,      mode,     durationType, progressType, deviceType,     cooldown_days
    ("daily_water_2",      "health",      "daily",   "short",      "counter",   "none",         0),
    ("daily_robot_clean",  "cleaning",    "daily",   "short",      "device",    "robot_vacuum", 2),
    ("speed_dishwasher",   "dishwashing", "speed",   "short",      "device",    "dishwasher",   0),
    ("speed_laundry_in",   "laundry",     "speed",   "short",      "device",    "washer",       3),
    ("monthly_heating",    "energy",      "monthly", "long",       "energy",    "ac",           0),
], columns=["challengeId", "category", "mode", "durationType", "progressType", "deviceType", "cooldown_days"])


# -------------------------------------------------------
# 2. 단일 이벤트 생성 유틸 함수
# -------------------------------------------------------
def make_one_event(i, ch, day_idx, eventDate, weekday):
    """하루(day_idx) 기준으로 챌린지 1개 이벤트 생성"""
    eventId = f"evt_{i:05d}"
    familyId = f"fam_{np.random.randint(1,4)}"
    userId = f"user_{np.random.randint(1,6)}"

    challengeId = ch["challengeId"]
    category = ch["category"]
    mode = ch["mode"]
    durationType = ch["durationType"]
    progressType = ch["progressType"]
    deviceType = ch["deviceType"]

    # 시간대
    timeSlot = np.random.choice(["morning", "afternoon", "evening", "night"])
    notif_hour = {"morning":8, "afternoon":14, "evening":20, "night":22}[timeSlot]
    notificationTime = f"{notif_hour:02d}:00:00"

    # 완료 확률을 위한 간단한 가중치 룰 (데모용)
    base = 0.3
    if mode == "daily":
        base += 0.05
    if mode == "monthly":
        base += 0.02
    if category == "health":
        base += 0.1
    if category == "cleaning":
        base -= 0.05
    if category == "laundry":
        base -= 0.03
    if category == "energy":
        base += 0.05
    if weekday in [5, 6] and mode != "monthly":
        base += 0.05

    personalPoints = np.random.choice([2, 4, 6, 8])
    familyPoints = personalPoints * np.random.randint(1, 4)

    # 🔥 난방(energy) 사용량: 지지난달 < 지난달 이 되도록 설계
    energyKwh = 0.0
    if category == "energy":
        if day_idx < 30:
            # 지지난달: 적게 사용
            energyKwh = np.random.uniform(0.5, 1.0)
        else:
            # 지난달: 더 많이 사용
            energyKwh = np.random.uniform(1.5, 3.0)

    # 완료 여부 샘플링
    logit = base
    prob = 1 / (1 + np.exp(-logit))
    completed_flag = np.random.rand() < prob
    completed = int(completed_flag)

    if completed_flag:
        completion_offset = np.random.randint(5, 120)  # 분 단위
        completion_hour = min(23, notif_hour + completion_offset // 60)
        completion_min = completion_offset % 60
        completionTime = f"{completion_hour:02d}:{completion_min:02d}:00"
    else:
        completionTime = ""

    return {
        "eventId": eventId,
        "familyId": familyId,
        "userId": userId,
        "challengeId": challengeId,
        "category": category,
        "mode": mode,
        "durationType": durationType,
        "progressType": progressType,
        "deviceType": deviceType,
        "eventDate": eventDate,
        "day_index": day_idx,          # 날짜 인덱스(0 ~ n_days-1)
        "weekday": weekday,
        "notificationTime": notificationTime,
        "completionTime": completionTime,
        "completed": completed,
        "timeSlot": timeSlot,
        "personalPoints": personalPoints,
        "familyPoints": familyPoints,
        "energyKwh": energyKwh,
    }


# -------------------------------------------------------
# 3. 가상 이벤트 데이터 생성
#    - 하루마다: 데일리 1개 + 스피드 1개 무조건 생성
#    - 먼슬리는 30일에 1번씩만 생성 (월 단위 느낌)
# -------------------------------------------------------
def simulate_events(n_days=60):
    rows = []
    event_idx = 0

    daily_candidates = challenge_meta[challenge_meta["mode"] == "daily"]
    speed_candidates = challenge_meta[challenge_meta["mode"] == "speed"]
    monthly_candidates = challenge_meta[challenge_meta["mode"] == "monthly"]

    for day in range(n_days):
        day_idx = day
        eventDate = f"2025-10-{1 + (day % 30):02d}"
        weekday = day_idx % 7

        # 1) 그날의 데일리 1개 강제 생성
        ch_daily = daily_candidates.sample(1).iloc[0]
        rows.append(make_one_event(event_idx, ch_daily, day_idx, eventDate, weekday))
        event_idx += 1

        # 2) 그날의 스피드 1개 강제 생성
        ch_speed = speed_candidates.sample(1).iloc[0]
        rows.append(make_one_event(event_idx, ch_speed, day_idx, eventDate, weekday))
        event_idx += 1

        # 3) 먼슬리는 30일에 한 번 등장 (0일, 30일)
        if day % 30 == 0:
            ch_monthly = monthly_candidates.sample(1).iloc[0]
            rows.append(make_one_event(event_idx, ch_monthly, day_idx, eventDate, weekday))
            event_idx += 1

    return pd.DataFrame(rows)


events = simulate_events(n_days=60)

# 디버그용: 모드별 이벤트 개수 출력 (시연 설명에도 사용 가능)
print("=== 이벤트 개수 (mode별) ===")
print(events["mode"].value_counts())

# -------------------------------------------------------
# 4. 메인 모델: "완료 여부(completed)" 예측
# -------------------------------------------------------
main_feature_cols = [
    "weekday",
    "personalPoints",
    "familyPoints",
    "energyKwh",
    "timeSlot",
    "mode",
    "category",
    "durationType",
    "progressType",
    "deviceType",
]

X_raw = events[main_feature_cols].copy()
y_main = events["completed"].values

X_main_encoded = pd.get_dummies(
    X_raw,
    columns=["timeSlot", "mode", "category", "durationType", "progressType", "deviceType"],
    drop_first=True
)

# day_index 기준으로 train/test 분리 (앞 2/3는 train, 뒤 1/3은 test)
day_idx_all = events["day_index"].values
max_day = day_idx_all.max()
split_day = int(max_day * 2 / 3)  # 예: 0~39 train, 40~59 test (n_days=60 기준)

train_mask = day_idx_all <= split_day
test_mask = day_idx_all > split_day

X_main_train, X_main_test = X_main_encoded[train_mask], X_main_encoded[test_mask]
y_main_train, y_main_test = y_main[train_mask], y_main[test_mask]

main_model = GradientBoostingClassifier(random_state=42)
main_model.fit(X_main_train, y_main_train)

y_main_pred_proba = main_model.predict_proba(X_main_test)[:, 1]
main_auc = roc_auc_score(y_main_test, y_main_pred_proba)
print("메인 모델 AUC(완료 여부):", round(main_auc, 4))


# -------------------------------------------------------
# 5. 스피드 챌린지: "1시간 이내 완료" 라벨 생성
# -------------------------------------------------------
def time_to_minutes(t):
    if t == "" or t is None:
        return None
    h, m, s = map(int, t.split(":"))
    return h * 60 + m

speed_events = events[events["mode"] == "speed"].copy()
print("스피드 이벤트 개수:", len(speed_events))

speed_events["notif_min"] = speed_events["notificationTime"].apply(time_to_minutes)
speed_events["comp_min"] = speed_events["completionTime"].apply(time_to_minutes)

def calc_duration(row):
    if row["completed"] == 1 and row["comp_min"] is not None and row["notif_min"] is not None:
        return row["comp_min"] - row["notif_min"]
    else:
        return None

speed_events["duration_min"] = speed_events.apply(calc_duration, axis=1)

def within_1h(row):
    if row["completed"] == 1 and row["duration_min"] is not None and row["duration_min"] <= 60:
        return 1
    else:
        return 0

speed_events["completed_within_1h"] = speed_events.apply(within_1h, axis=1)

# -------------------------------------------------------
# 6. 스피드 전용 모델: 전체 데이터로 "1시간 이내 성공 확률" 예측
# -------------------------------------------------------
X_speed_encoded = None
speed_time_model = None

speed_feature_cols = [
    "weekday",
    "timeSlot",
    "category",
    "challengeId",
    "personalPoints",
    "familyPoints",
    "energyKwh",
]

if len(speed_events) == 0:
    print("⚠ 스피드 이벤트가 없습니다. 스피드 모델 미학습.")
else:
    X_speed_raw = speed_events[speed_feature_cols].copy()
    y_speed = speed_events["completed_within_1h"].values

    X_speed_encoded = pd.get_dummies(
        X_speed_raw,
        columns=["timeSlot", "category", "challengeId"],
        drop_first=True
    )

    X_speed_train = X_speed_encoded
    y_speed_train = y_speed

    speed_time_model = GradientBoostingClassifier(random_state=42)
    speed_time_model.fit(X_speed_train, y_speed_train)

    y_speed_pred_proba = speed_time_model.predict_proba(X_speed_train)[:, 1]
    speed_auc = roc_auc_score(y_speed_train, y_speed_pred_proba)
    print("스피드 모델 AUC(1시간 이내 완료, 학습 데이터 기준):", round(speed_auc, 4))


# -------------------------------------------------------
# 7. 🔥 난방 사용량 비교: 저저번달 vs 저번달
#    - day_index < 30  : 저저번달
#    - day_index >= 30 : 저번달
#    - 저번달 난방 사용량이 더 많으면 monthly_heating 등장 조건 만족
# -------------------------------------------------------
energy_events = events[events["category"] == "energy"].copy()

prev_month_usage = energy_events[energy_events["day_index"] < 30]["energyKwh"].sum()
last_month_usage = energy_events[energy_events["day_index"] >= 30]["energyKwh"].sum()

high_heating_usage = last_month_usage > prev_month_usage

print(f"지지난달 난방 사용량: {prev_month_usage:.2f} kWh")
print(f"지난달 난방 사용량:   {last_month_usage:.2f} kWh")
print("난방 절약 챌린지 조건 충족?:", high_heating_usage)


# -------------------------------------------------------
# 8. 쿨다운 정보 (challengeId 기준)
# -------------------------------------------------------
today_date_idx = max_day + 1  # 마지막 날 다음날 = "오늘"

last_done = {
    "daily_robot_clean": today_date_idx - 1,  # 어제 로봇청소기 → 쿨다운 2일이라 오늘 막힘
    "speed_laundry_in":  today_date_idx - 2,  # 이틀 전 빨래 → 쿨다운 3일이라 오늘 막힘
    # 나머지: 수행 기록 없음
}

def is_available(ch_row):
    cd = ch_row["cooldown_days"]
    if cd == 0:
        return True
    last = last_done.get(ch_row["challengeId"], None)
    if last is None:
        return True
    days_diff = today_date_idx - last
    return days_diff >= cd


# -------------------------------------------------------
# 9. 오늘 데일리 / 먼슬리 추천 (메인 모델 + 쿨다운)
#    - 먼슬리는 "지난달 난방 > 지지난달 난방"일 때만 등장
# -------------------------------------------------------
today_weekday = today_date_idx % 7
today_timeSlot_default = "evening"

def build_today_main_feature_row(ch_row, time_slot):
    return {
        "weekday": today_weekday,
        "personalPoints": 4,
        "familyPoints": 12,
        "energyKwh": 1.2 if ch_row["category"] == "energy" else 0.0,
        "timeSlot": time_slot,
        "mode": ch_row["mode"],
        "category": ch_row["category"],
        "durationType": ch_row["durationType"],
        "progressType": ch_row["progressType"],
        "deviceType": ch_row["deviceType"],
    }

def recommend_today_non_speed(mode_filter):
    # 먼슬리 조건 체크
    if mode_filter == "monthly" and not high_heating_usage:
        return None

    candidates = challenge_meta[challenge_meta["mode"] == mode_filter].copy()
    candidates["available"] = candidates.apply(is_available, axis=1)
    candidates = candidates[candidates["available"] == True]
    if len(candidates) == 0:
        return None

    feat_rows = [build_today_main_feature_row(row, today_timeSlot_default)
                 for _, row in candidates.iterrows()]
    feat_df = pd.DataFrame(feat_rows)
    feat_encoded = pd.get_dummies(
        feat_df,
        columns=["timeSlot", "mode", "category", "durationType", "progressType", "deviceType"],
        drop_first=True
    )

    for col in X_main_encoded.columns:
        if col not in feat_encoded.columns:
            feat_encoded[col] = 0
    feat_encoded = feat_encoded[X_main_encoded.columns]

    scores = main_model.predict_proba(feat_encoded)[:, 1]
    candidates = candidates.reset_index(drop=True)
    candidates["score"] = scores

    best = candidates.sort_values("score", ascending=False).iloc[0]
    return best


# -------------------------------------------------------
# 10. 오늘 스피드 챌린지 추천 (챌린지 + timeSlot)
#      - 스피드만 1시간 이내 완료 모델 사용
# -------------------------------------------------------
def recommend_today_speed():
    if speed_time_model is None or X_speed_encoded is None:
        return None

    candidate_slots = ["morning", "afternoon", "evening", "night"]

    speed_challenges = challenge_meta[challenge_meta["mode"] == "speed"].copy()
    speed_challenges["available"] = speed_challenges.apply(is_available, axis=1)
    speed_challenges = speed_challenges[speed_challenges["available"] == True]

    if len(speed_challenges) == 0:
        return None

    candidate_rows = []
    for _, ch in speed_challenges.iterrows():
        for slot in candidate_slots:
            candidate_rows.append({
                "challengeId": ch["challengeId"],
                "category": ch["category"],
                "timeSlot": slot,
                "weekday": today_weekday,
                "personalPoints": 4,
                "familyPoints": 12,
                "energyKwh": 1.0 if ch["category"] == "energy" else 0.0,
            })

    cand_df = pd.DataFrame(candidate_rows)

    feat_speed = pd.get_dummies(
        cand_df,
        columns=["timeSlot", "category", "challengeId"],
        drop_first=True
    )

    for col in X_speed_encoded.columns:
        if col not in feat_speed.columns:
            feat_speed[col] = 0
    feat_speed = feat_speed[X_speed_encoded.columns]

    probs = speed_time_model.predict_proba(feat_speed)[:, 1]
    cand_df["score_within_1h"] = probs

    best_idx = cand_df["score_within_1h"].idxmax()
    best_row = cand_df.loc[best_idx]
    return best_row.to_dict()


# -------------------------------------------------------
# 11. 최종 오늘 추천 출력
# -------------------------------------------------------
daily_best = recommend_today_non_speed("daily")
monthly_best = recommend_today_non_speed("monthly")
speed_best = recommend_today_speed()

print("\n=== 오늘 추천 (쿨다운 + 스피드 1시간 최적 시간대 + 난방 조건) ===")
print("데일리:", daily_best.to_dict() if daily_best is not None else None)
print("먼슬리:", monthly_best.to_dict() if monthly_best is not None else None)
print("스피드 (챌린지 + timeSlot):", speed_best)
