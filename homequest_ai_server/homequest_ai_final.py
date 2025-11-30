import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ALPHA = 0.02  # 추천 횟수 패널티 계수

# 챌린지 메타 정보
challenge_meta = pd.DataFrame(
    [
        ("daily_water_2", "health", "daily", "short", "counter", "none", 0),
        ("daily_robot_clean", "cleaning", "daily", "short", "device", "robot_vacuum", 2),
        ("speed_dishwasher", "dishwashing", "speed", "short", "device", "dishwasher", 0),
        ("speed_laundry_in", "laundry", "speed", "short", "device", "washer", 3),
        ("monthly_heating", "energy", "monthly", "long", "energy", "ac", 0),
    ],
    columns=[
        "challengeId", "category", "mode", "durationType",
        "progressType", "deviceType", "cooldown_days"
    ],
)


def time_to_minutes(t):
    if t is None:
        return None
    if isinstance(t, float) and np.isnan(t):
        return None
    if t == "":
        return None
    h, m, s = map(int, str(t).split(":"))
    return h * 60 + m


def calc_duration(r):
    if r["completed"] == 1 and r["comp_min"] is not None and r["notif_min"] is not None:
        return r["comp_min"] - r["notif_min"]
    return None


def within_1h(r):
    return 1 if (r["completed"] == 1 and r["duration_min"] is not None and r["duration_min"] <= 60) else 0


def minutes_to_timestr(m):
    h, mm = divmod(int(m), 60)
    return f"{h:02d}:{mm:02d}:00"


def is_available(ch, last_done, today):
    cd = ch["cooldown_days"]
    if cd == 0:
        return True
    cid = ch["challengeId"]
    if cid not in last_done:
        return True
    return (today - last_done[cid]) >= cd


def load_recommend_count(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_recommend_count(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_recommendation(events_df: pd.DataFrame, user_id: str = "user_4", top_k: int = 3):
    """
    Firebase Storage에서 가져온 events_df를 기반으로
    최종 추천 + 모델 성능 지표를 계산해서 dict로 반환하는 함수.
    """
    TARGET_USER = user_id
    TOP_K = top_k

    # 추천 횟수 기록 파일 (로컬 json)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    count_path = os.path.join(base_dir, "recommend_counts.json")
    recommend_count = load_recommend_count(count_path)

    # 원본 훼손 방지용 복사
    events = events_df.copy()

    # ===== 1. 메인 완료 여부 예측 모델 =====
    main_cols = [
        "weekday", "personalPoints", "familyPoints", "energyKwh",
        "mode", "category", "durationType", "progressType", "deviceType", "userId"
    ]
    X_raw = events[main_cols].copy()
    y_main = events["completed"].values

    X_main = pd.get_dummies(
        X_raw,
        columns=["mode", "category", "durationType", "progressType", "deviceType", "userId"],
        drop_first=True,
    )

    idx_all = events["day_index"].values
    max_day = idx_all.max()
    split = int(max_day * 2 / 3)
    tr = idx_all <= split
    te = idx_all > split

    main_model = GradientBoostingClassifier()
    main_model.fit(X_main[tr], y_main[tr])

    if te.sum() > 0:
        pred_main = main_model.predict_proba(X_main[te])[:, 1]
        auc_main = float(roc_auc_score(y_main[te], pred_main))
    else:
        auc_main = None

    # ===== 2. 스피드 챌린지 1시간 이내 완료 예측 모델 =====
    speed_events = events[events["mode"] == "speed"].copy()
    speed_events["notif_min"] = speed_events["notificationTime"].apply(time_to_minutes)
    speed_events["comp_min"] = speed_events["completionTime"].apply(time_to_minutes)
    speed_events["duration_min"] = speed_events.apply(calc_duration, axis=1)
    speed_events["completed_within_1h"] = speed_events.apply(within_1h, axis=1)

    sp_cols = [
        "weekday", "notif_min", "category", "challengeId",
        "personalPoints", "familyPoints", "energyKwh", "userId",
    ]
    X_sp_raw = speed_events[sp_cols].copy()
    y_sp = speed_events["completed_within_1h"].values

    X_sp = pd.get_dummies(
        X_sp_raw,
        columns=["category", "challengeId", "userId"],
        drop_first=True,
    )

    speed_model = GradientBoostingClassifier()
    speed_model.fit(X_sp, y_sp)
    pred_sp = speed_model.predict_proba(X_sp)[:, 1]
    auc_sp = float(roc_auc_score(y_sp, pred_sp))

    # ===== 3. 에너지 사용량 증감 판단 =====
    e = events[events["category"] == "energy"]
    prev = e[e["day_index"] < 30]["energyKwh"].sum()
    last = e[e["day_index"] >= 30]["energyKwh"].sum()
    energy_high = bool(last > prev)

    # ===== 4. 오늘 날짜 / 요일 설정 (시뮬레이션) =====
    today = max_day + 1
    # 예시: 최근 수행일 임시 설정 (나중에 데이터 기반으로 바꿔도 됨)
    last_done = {
        "daily_robot_clean": today - 1,
        "speed_laundry_in": today - 2,
    }
    today_weekday = today % 7

    def build_main_row(ch):
        return {
            "weekday": today_weekday,
            "personalPoints": 4,
            "familyPoints": 12,
            "energyKwh": 1.2 if ch["category"] == "energy" else 0.0,
            "mode": ch["mode"],
            "category": ch["category"],
            "durationType": ch["durationType"],
            "progressType": ch["progressType"],
            "deviceType": ch["deviceType"],
            "userId": TARGET_USER,
        }

    def recommend_non_speed(mode: str):
        # monthly는 에너지 사용량이 늘었을 때만 추천
        if mode == "monthly" and not energy_high:
            return None

        cand = challenge_meta[challenge_meta["mode"] == mode].copy()
        cand["available"] = cand.apply(
            lambda r: is_available(r, last_done, today), axis=1
        )
        cand = cand[cand["available"] == True]
        if len(cand) == 0:
            return None

        rows = [build_main_row(r) for _, r in cand.iterrows()]
        df = pd.DataFrame(rows)
        df = pd.get_dummies(
            df,
            columns=["mode", "category", "durationType", "progressType", "deviceType", "userId"],
            drop_first=True,
        )

        # X_main과 컬럼 맞추기
        for c in X_main.columns:
            if c not in df.columns:
                df[c] = 0
        df = df[X_main.columns]

        scores = main_model.predict_proba(df)[:, 1]
        cand = cand.reset_index(drop=True)
        cand["score"] = scores

        cand["freq"] = cand["challengeId"].map(
            lambda cid: recommend_count.get(cid, 0)
        )
        cand["adj_score"] = cand["score"] - ALPHA * cand["freq"]

        cand_top = cand.sort_values("adj_score", ascending=False).head(TOP_K)

        w = np.exp(cand_top["adj_score"])
        p = w / w.sum()
        chosen = cand_top.sample(n=1, weights=p).iloc[0]

        cid = chosen["challengeId"]
        recommend_count[cid] = recommend_count.get(cid, 0) + 1

        return chosen.to_dict()

    def recommend_speed():
        cand = challenge_meta[challenge_meta["mode"] == "speed"].copy()
        cand["available"] = cand.apply(
            lambda r: is_available(r, last_done, today), axis=1
        )
        cand = cand[cand["available"] == True]
        if len(cand) == 0:
            return None

        notif_list = [h * 60 for h in range(6, 23)]
        rows = []
        for _, ch in cand.iterrows():
            for nm in notif_list:
                rows.append({
                    "challengeId": ch["challengeId"],
                    "category": ch["category"],
                    "weekday": today_weekday,
                    "notif_min": nm,
                    "personalPoints": 4,
                    "familyPoints": 12,
                    "energyKwh": 1.0 if ch["category"] == "energy" else 0.0,
                    "userId": TARGET_USER,
                })

        df_meta = pd.DataFrame(rows)
        feat_cols = [
            "weekday", "notif_min", "category", "challengeId",
            "personalPoints", "familyPoints", "energyKwh", "userId",
        ]
        df_feat = df_meta[feat_cols].copy()
        df_feat = pd.get_dummies(
            df_feat,
            columns=["category", "challengeId", "userId"],
            drop_first=True,
        )

        for c in X_sp.columns:
            if c not in df_feat.columns:
                df_feat[c] = 0
        df_feat = df_feat[X_sp.columns]

        probs = speed_model.predict_proba(df_feat)[:, 1]
        df_meta["score"] = probs
        df_meta["challengeId"] = df_meta["challengeId"].astype(str)

        df_meta["freq"] = df_meta["challengeId"].map(
            lambda cid: recommend_count.get(cid, 0)
        )
        df_meta["adj_score"] = df_meta["score"] - ALPHA * df_meta["freq"]

        df_top = df_meta.sort_values("adj_score", ascending=False).head(TOP_K)

        w = np.exp(df_top["adj_score"])
        p = w / w.sum()
        best = df_top.sample(n=1, weights=p).iloc[0].to_dict()
        best["notificationTime"] = minutes_to_timestr(best["notif_min"])

        cid = best["challengeId"]
        recommend_count[cid] = recommend_count.get(cid, 0) + 1

        return best

    daily = recommend_non_speed("daily")
    monthly = recommend_non_speed("monthly")
    speed = recommend_speed()

    save_recommend_count(count_path, recommend_count)

    return {
        "userId": TARGET_USER,
        "daily": daily,
        "monthly": monthly,
        "speed": speed,
        "energyHigh": energy_high,
        "main_auc": auc_main,
        "speed_auc": auc_sp,
    }
