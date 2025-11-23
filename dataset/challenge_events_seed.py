import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =========================================================
# 0. 기본 설정
# =========================================================

START_DATE = datetime(2025, 11, 1)
END_DATE = datetime(2025, 11, 30)

FAMILY_ID_MAIN = "fam_jinjin"

NICK_JINJIN = "진진이"
NICK_DONG = "동동이"
NICK_MOM = "엄마"
NICK_DAD = "아빠"

HIGH_ENERGY_DEVICES = {"heater", "ac", "washer",
                       "dishwasher", "dryer", "robot_cleaner"}


# =========================================================
# 1. 엑셀 불러오기
# =========================================================

users = pd.read_excel("users.xlsx")
families = pd.read_excel("families.xlsx")
challenges = pd.read_excel("challenges.xlsx")

for col in ["difficultyLevel", "basePersonalPoints", "baseFamilyPoints"]:
    if col in challenges.columns:
        challenges[col] = pd.to_numeric(
            challenges[col], errors="coerce").fillna(0).astype(int)

challenges_by_id = {row["challengeId"]                    : row for _, row in challenges.iterrows()}


def get_ch(ch_id: str):
    if ch_id not in challenges_by_id:
        raise ValueError(
            f"challenges.xlsx 안에 {ch_id} 가 없습니다. challengeId를 확인해 주세요.")
    return challenges_by_id[ch_id]


CH_WATER_MORNING = get_ch("ch_health_water_m1")
CH_DISH_SPEED = get_ch("ch_chores_dish_speed1")
CH_ROBOT_RELAY = get_ch("ch_chores_robot_relay")
CH_HEATER_MONTHLY = get_ch("ch_saving_heater_m1")


# =========================================================
# 2. 진진이네 가족 userId 찾기
# =========================================================

family_main = families[families["familyId"] == FAMILY_ID_MAIN]
if family_main.empty:
    print(
        f"경고: families.xlsx에서 familyId={FAMILY_ID_MAIN} 를 찾지 못했어요. familyId 값을 확인해 주세요.")

users_main_family = users[users["familyId"] == FAMILY_ID_MAIN].copy()


def find_user_id_by_nick(nick):
    row = users_main_family[users_main_family["nickName"] == nick]
    if row.empty:
        print(f"경고: nickName={nick} 인 유저를 찾지 못했어요. users.xlsx를 확인해 주세요.")
        return None
    return row.iloc[0]["userId"]


USERID_JINJIN = find_user_id_by_nick(NICK_JINJIN)
USERID_DONG = find_user_id_by_nick(NICK_DONG)
USERID_MOM = find_user_id_by_nick(NICK_MOM)
USERID_DAD = find_user_id_by_nick(NICK_DAD)

FAMILY_USER_IDS = [uid for uid in [USERID_MOM, USERID_DAD,
                                   USERID_DONG, USERID_JINJIN] if uid is not None]


# =========================================================
# 3. 이벤트 생성
# =========================================================

rows = []
event_id = 1

dates = []
cur = START_DATE
while cur <= END_DATE:
    dates.append(cur)
    cur += timedelta(days=1)


def time_to_str(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")


for date in dates:
    date_str = date.strftime("%Y-%m-%d")
    weekday = date.weekday()  # 0=월요일 ~ 6=일요일

    # -----------------------------------------------------
    # (1) 평일: 진진이 아침 물 챌린지
    #  - 알림: 06:55
    #  - 평소엔 잘 완료하되, 화요일(weekday==1)은 실패 확률 높게
    # -----------------------------------------------------
    if weekday < 5 and USERID_JINJIN is not None:
        ch = CH_WATER_MORNING

        noti_dt = datetime(date.year, date.month, date.day, 6, 55, 0)
        # 완료 시각은 7시~7시40분 사이 랜덤
        comp_dt = datetime(date.year, date.month, date.day, 7, 0, 0) + timedelta(
            minutes=int(np.random.choice([0, 10, 20, 30, 40]))
        )

        # 화요일은 바빠서 절반 정도 실패, 나머지 평일은 거의 성공
        if weekday == 1:
            complete_prob = 0.5
        else:
            complete_prob = 0.9

        completed_flag = int(np.random.binomial(1, complete_prob))

        rows.append({
            "eventId": f"ev_{event_id:04d}",
            "familyId": FAMILY_ID_MAIN,
            "userId": USERID_JINJIN,
            "challengeId": ch["challengeId"],
            "category": ch["category"],
            "mode": ch["mode"],
            "durationType": ch["durationType"],
            "progressType": ch["progressType"],
            "deviceType": ch["deviceType"],
            "eventDate": date_str,
            "weekday": weekday,
            "notificationTime": time_to_str(noti_dt),
            "completionTime": time_to_str(comp_dt) if completed_flag else "",
            "completed": completed_flag,
            "timeSlot": "morning",
            "personalPoints": int(ch["basePersonalPoints"] * completed_flag),
            "familyPoints": int(ch["baseFamilyPoints"] * completed_flag),
            "energyKwh": 0.0,
        })
        event_id += 1

    # -----------------------------------------------------
    # (2) 식기세척기 스피드 챌린지
    #  - 짝수 날짜에만 시도
    #  - 1~10일: 20시 알림 (저녁, 성공률 낮음)
    #  - 11~20일: 18시 알림 (퇴근 직후, 좀 더 나아짐)
    #  - 21~30일: 08시 알림 (출근 전, 성공률 높음)
    #  - 요일/시간대에 따라 성공 확률 다르게
    # -----------------------------------------------------
    if date.day % 2 == 0 and USERID_JINJIN is not None:
        ch = CH_DISH_SPEED

        # 알림 시간 단계별로 변경
        if date.day <= 10:
            noti_hour = 20
        elif date.day <= 20:
            noti_hour = 18
        else:
            noti_hour = 8

        noti_dt = datetime(date.year, date.month, date.day, noti_hour, 0, 0)

        # 완료 시각은 알림 후 1~60분 내
        comp_dt = noti_dt + timedelta(minutes=int(np.random.randint(1, 61)))

        # 요일/시간대별 성공 확률 설계
        # 월/화 저녁 20시는 가족이 늦게 들어와서 성공률 낮음
        if noti_hour == 20:
            if weekday in [0, 1]:   # 월, 화
                base_prob = 0.2
            else:
                base_prob = 0.5
        # 18시는 비교적 안정적인 시간대
        elif noti_hour == 18:
            if weekday in [0, 1]:
                base_prob = 0.5
            else:
                base_prob = 0.7
        # 아침 8시는 다 같이 나가기 직전, 오히려 성공률 높음
        else:  # 8시
            if weekday in [5, 6]:   # 주말
                base_prob = 0.8
            else:
                base_prob = 0.9

        completed_flag = int(np.random.binomial(1, base_prob))

        energy_kwh = float(np.random.uniform(0.3, 1.5))  # 식기세척기 사용 에너지

        # timeSlot은 알림 기준
        if noti_hour < 12:
            slot = "morning"
        elif noti_hour < 18:
            slot = "afternoon"
        elif noti_hour < 22:
            slot = "evening"
        else:
            slot = "night"

        rows.append({
            "eventId": f"ev_{event_id:04d}",
            "familyId": FAMILY_ID_MAIN,
            "userId": USERID_JINJIN,
            "challengeId": ch["challengeId"],
            "category": ch["category"],
            "mode": ch["mode"],
            "durationType": ch["durationType"],
            "progressType": ch["progressType"],
            "deviceType": ch["deviceType"],
            "eventDate": date_str,
            "weekday": weekday,
            "notificationTime": time_to_str(noti_dt),
            "completionTime": time_to_str(comp_dt) if completed_flag else "",
            "completed": completed_flag,
            "timeSlot": slot,
            "personalPoints": int(ch["basePersonalPoints"] * completed_flag),
            "familyPoints": int(ch["baseFamilyPoints"] * completed_flag),
            "energyKwh": energy_kwh,
        })
        event_id += 1

    # -----------------------------------------------------
    # (3) 목요일: 로봇청소기 가족 릴레이
    # -----------------------------------------------------
    if weekday == 3 and len(FAMILY_USER_IDS) > 0:
        ch = CH_ROBOT_RELAY
        base_hour = 14
        base_min = 0

        for i, uid in enumerate(FAMILY_USER_IDS):
            noti_dt = datetime(date.year, date.month, date.day,
                               base_hour, base_min, 0) + timedelta(minutes=10 * i)
            comp_dt = noti_dt + \
                timedelta(minutes=int(np.random.randint(5, 21)))
            completed_flag = 1  # 시나리오상 릴레이는 항상 성공한다고 가정

            # 가족 포인트는 첫 사람에게만
            personal_points = 0
            family_points = ch["baseFamilyPoints"] if i == 0 else 0

            rows.append({
                "eventId": f"ev_{event_id:04d}",
                "familyId": FAMILY_ID_MAIN,
                "userId": uid,
                "challengeId": ch["challengeId"],
                "category": ch["category"],
                "mode": ch["mode"],
                "durationType": ch["durationType"],
                "progressType": ch["progressType"],
                "deviceType": ch["deviceType"],
                "eventDate": date_str,
                "weekday": weekday,
                "notificationTime": time_to_str(noti_dt),
                "completionTime": time_to_str(comp_dt),
                "completed": completed_flag,
                "timeSlot": "afternoon",
                "personalPoints": int(personal_points),
                "familyPoints": int(family_points),
                "energyKwh": float(np.random.uniform(0.2, 1.0)),
            })
            event_id += 1

# ---------------------------------------------------------
# (4) 11월 30일: 난방 월간 절약 챌린지 완료
# ---------------------------------------------------------
if USERID_DAD is not None:
    ch = CH_HEATER_MONTHLY
    date = END_DATE
    date_str = date.strftime("%Y-%m-%d")
    weekday = date.weekday()
    noti_dt = datetime(date.year, date.month, date.day, 21, 30, 0)

    # 월간 챌린지는 완료 이벤트 하나만
    completed_flag = 1
    energy_kwh = float(np.random.uniform(5.0, 15.0))  # 누적 절약량 가짜 값

    rows.append({
        "eventId": f"ev_{event_id:04d}",
        "familyId": FAMILY_ID_MAIN,
        "userId": USERID_DAD,
        "challengeId": ch["challengeId"],
        "category": ch["category"],
        "mode": ch["mode"],
        "durationType": ch["durationType"],
        "progressType": ch["progressType"],
        "deviceType": ch["deviceType"],
        "eventDate": date_str,
        "weekday": weekday,
        "notificationTime": time_to_str(noti_dt),
        "completionTime": time_to_str(noti_dt),  # 같은 시각에 완료 처리했다고 가정
        "completed": completed_flag,
        "timeSlot": "night",
        "personalPoints": 0,
        "familyPoints": int(ch["baseFamilyPoints"] * completed_flag),
        "energyKwh": energy_kwh,
    })
    event_id += 1


# =========================================================
# (옵션) 완전 랜덤 챌린지 수행 로그 추가하고 싶을 때
#  -> 지금은 기본 비활성. 필요하면 아래 주석을 풀어서 사용
# =========================================================
"""
for _, user in users_main_family.iterrows():
    for date in dates:
        # 하루에 0~2개 랜덤 이벤트
        n_events = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])

        for _ in range(n_events):
            # 아무 챌린지 하나 랜덤 선택
            ch = challenges.sample(1).iloc[0]

            # 알림 시간: 대표적인 시간대에서 랜덤 선택
            noti_hour = int(np.random.choice([7, 9, 12, 18, 21]))
            noti_min = int(np.random.choice([0, 10, 20, 30, 40, 50]))
            noti_dt = datetime(date.year, date.month, date.day, noti_hour, noti_min, 0)
            weekday = noti_dt.weekday()

            # 난이도에 따라 기본 완료 확률
            diff = int(ch["difficultyLevel"])
            if diff == 1:
                base_prob = 0.85
            elif diff == 2:
                base_prob = 0.65
            else:
                base_prob = 0.45

            # 요일/시간대에 따라 약간 가중치 (예: 월요일 밤은 더 피곤해서 성공률 낮음)
            if noti_hour >= 21 and weekday == 0:  # 월요일 밤
                base_prob *= 0.7
            if noti_hour == 7 and weekday in [5, 6]:  # 주말 아침
                base_prob *= 1.1

            base_prob = max(0.05, min(0.95, base_prob))

            completed_flag = int(np.random.binomial(1, base_prob))

            # 완료 시각은 알림 후 1~60분 내 (미완료면 빈 문자열)
            if completed_flag:
                comp_dt = noti_dt + timedelta(minutes=int(np.random.randint(1, 61)))
                completion_time = time_to_str(comp_dt)
            else:
                completion_time = ""

            # timeSlot은 알림 기준
            if noti_hour < 12:
                slot = "morning"
            elif noti_hour < 18:
                slot = "afternoon"
            elif noti_hour < 22:
                slot = "evening"
            else:
                slot = "night"

            # 포인트 계산
            personal_points = int(ch["basePersonalPoints"] * completed_flag)
            family_points = int(ch["baseFamilyPoints"] * completed_flag)

            # 에너지 사용량 (해당 디바이스일 때만)
            device = ch["deviceType"]
            if device in HIGH_ENERGY_DEVICES:
                energy_kwh = float(np.random.uniform(0.1, 2.0))
            else:
                energy_kwh = 0.0

            rows.append({
                "eventId": f"ev_{event_id:04d}",
                "familyId": user["familyId"],
                "userId": user["userId"],
                "challengeId": ch["challengeId"],
                "category": ch["category"],
                "mode": ch["mode"],
                "durationType": ch["durationType"],
                "progressType": ch["progressType"],
                "deviceType": device,
                "eventDate": date.strftime("%Y-%m-%d"),
                "weekday": weekday,
                "notificationTime": time_to_str(noti_dt),
                "completionTime": completion_time,
                "completed": completed_flag,
                "timeSlot": slot,
                "personalPoints": personal_points,
                "familyPoints": family_points,
                "energyKwh": energy_kwh,
            })
            event_id += 1
"""

# =========================================================
# 4. 저장
# =========================================================

events = pd.DataFrame(rows)
events.to_excel("challenge_events.xlsx", index=False)
events.to_csv("challenge_events.csv", index=False, encoding="utf-8-sig")

print(f"생성된 이벤트 수: {len(events)}")
print("challenge_events.xlsx / challenge_events.csv 생성 완료 ✅")
