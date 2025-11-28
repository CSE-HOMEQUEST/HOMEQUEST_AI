import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

np.random.seed(42)

challenge_meta = pd.DataFrame(
    [
        ("daily_water_2","health","daily","short","counter","none",0),
        ("daily_robot_clean","cleaning","daily","short","device","robot_vacuum",2),
        ("speed_dishwasher","dishwashing","speed","short","device","dishwasher",0),
        ("speed_laundry_in","laundry","speed","short","device","washer",3),
        ("monthly_heating","energy","monthly","long","energy","ac",0),
    ],
    columns=[
        "challengeId","category","mode","durationType","progressType","deviceType","cooldown_days"
    ],
)

def make_one_event(i, ch, day_idx, eventDate, weekday):
    notif_hour = np.random.randint(6, 23)
    notif_minute = np.random.randint(0, 60)
    notificationTime = f"{notif_hour:02d}:{notif_minute:02d}:00"
    base = 0.3
    if ch["mode"]=="daily": base+=0.05
    if ch["mode"]=="monthly": base+=0.02
    if ch["category"]=="health": base+=0.1
    if ch["category"]=="cleaning": base-=0.05
    if ch["category"]=="laundry": base-=0.03
    if ch["category"]=="energy": base+=0.05
    if weekday in [5,6] and ch["mode"]!="monthly": base+=0.05
    energyKwh=0
    if ch["category"]=="energy":
        if day_idx<30: energyKwh=np.random.uniform(0.5,1.0)
        else: energyKwh=np.random.uniform(1.5,3.0)
    prob=1/(1+np.exp(-base))
    completed=int(np.random.rand()<prob)
    if completed:
        offset=np.random.randint(5,120)
        total=notif_hour*60+notif_minute+offset
        total=min(23*60+59,total)
        h,m=divmod(total,60)
        completionTime=f"{h:02d}:{m:02d}:00"
    else:
        completionTime=""
    return {
        "eventId":f"evt_{i:05d}",
        "familyId":f"fam_{np.random.randint(1,4)}",
        "userId":f"user_{np.random.randint(1,6)}",
        "challengeId":ch["challengeId"],
        "category":ch["category"],
        "mode":ch["mode"],
        "durationType":ch["durationType"],
        "progressType":ch["progressType"],
        "deviceType":ch["deviceType"],
        "eventDate":eventDate,
        "day_index":day_idx,
        "weekday":weekday,
        "notificationTime":notificationTime,
        "completionTime":completionTime,
        "completed":completed,
        "personalPoints":np.random.choice([2,4,6,8]),
        "familyPoints":np.random.choice([2,4,6,8])*np.random.randint(1,4),
        "energyKwh":energyKwh
    }

def simulate_events(n_days=60):
    rows=[]
    idx=0
    daily=challenge_meta[challenge_meta["mode"]=="daily"]
    speed=challenge_meta[challenge_meta["mode"]=="speed"]
    monthly=challenge_meta[challenge_meta["mode"]=="monthly"]
    for day in range(n_days):
        d=f"2025-10-{1+(day%30):02d}"
        w=day%7
        rows.append(make_one_event(idx,daily.sample(1).iloc[0],day,d,w)); idx+=1
        rows.append(make_one_event(idx,speed.sample(1).iloc[0],day,d,w)); idx+=1
        if day%30==0:
            rows.append(make_one_event(idx,monthly.sample(1).iloc[0],day,d,w)); idx+=1
    return pd.DataFrame(rows)

def time_to_minutes(t):
    if t=="": return None
    h,m,s=map(int,t.split(":"))
    return h*60+m

def calc_duration(r):
    if r["completed"]==1 and r["comp_min"] is not None and r["notif_min"] is not None:
        return r["comp_min"]-r["notif_min"]
    return None

def within_1h(r):
    return 1 if (r["completed"]==1 and r["duration_min"] is not None and r["duration_min"]<=60) else 0

def minutes_to_timestr(m):
    h,mm=divmod(int(m),60)
    return f"{h:02d}:{mm:02d}:00"

def is_available(ch,last_done,today):
    cd=ch["cooldown_days"]
    if cd==0: return True
    if ch["challengeId"] not in last_done: return True
    return (today-last_done[ch["challengeId"]])>=cd

def main():
    events=simulate_events(60)
    print("=== 이벤트 개수 ===")
    print(events["mode"].value_counts())

    main_cols=[
        "weekday","personalPoints","familyPoints","energyKwh",
        "mode","category","durationType","progressType","deviceType"
    ]
    X_raw=events[main_cols].copy()
    y_main=events["completed"].values
    X_main=pd.get_dummies(
        X_raw,
        columns=["mode","category","durationType","progressType","deviceType"],
        drop_first=True)
    idx_all=events["day_index"].values
    max_day=idx_all.max()
    split=int(max_day*2/3)
    tr=idx_all<=split
    te=idx_all>split
    main_model=GradientBoostingClassifier(random_state=42)
    main_model.fit(X_main[tr],y_main[tr])
    pred=main_model.predict_proba(X_main[te])[:,1]
    print("\n메인 모델 AUC:",round(roc_auc_score(y_main[te],pred),4))

    speed_events=events[events["mode"]=="speed"].copy()
    speed_events["notif_min"]=speed_events["notificationTime"].apply(time_to_minutes)
    speed_events["comp_min"]=speed_events["completionTime"].apply(time_to_minutes)
    speed_events["duration_min"]=speed_events.apply(calc_duration,axis=1)
    speed_events["completed_within_1h"]=speed_events.apply(within_1h,axis=1)

    if len(speed_events)>0:
        sp_cols=[
            "weekday","notif_min","category","challengeId",
            "personalPoints","familyPoints","energyKwh"
        ]
        X_sp_raw=speed_events[sp_cols].copy()
        y_sp=speed_events["completed_within_1h"].values
        X_sp=pd.get_dummies(X_sp_raw,columns=["category","challengeId"],drop_first=True)
        speed_model=GradientBoostingClassifier(random_state=42)
        speed_model.fit(X_sp,y_sp)
        pred2=speed_model.predict_proba(X_sp)[:,1]
        print("스피드 모델 AUC:",round(roc_auc_score(y_sp,pred2),4))
    else:
        speed_model=None
        X_sp=None
        print("스피드 이벤트 없음")

    e=events[events["category"]=="energy"]
    prev=e[e["day_index"]<30]["energyKwh"].sum()
    last=e[e["day_index"]>=30]["energyKwh"].sum()
    high=last>prev

    today=max_day+1
    last_done={
        "daily_robot_clean":today-1,
        "speed_laundry_in":today-2
    }
    tw=today%7

    def build_main_row(ch):
        return {
            "weekday":tw,
            "personalPoints":4,
            "familyPoints":12,
            "energyKwh":1.2 if ch["category"]=="energy" else 0.0,
            "mode":ch["mode"],
            "category":ch["category"],
            "durationType":ch["durationType"],
            "progressType":ch["progressType"],
            "deviceType":ch["deviceType"],
        }

    def recommend_non_speed(mode):
        if mode=="monthly" and not high: return None
        cand=challenge_meta[challenge_meta["mode"]==mode].copy()
        cand["available"]=cand.apply(lambda r: is_available(r,last_done,today),axis=1)
        cand=cand[cand["available"]==True]
        if len(cand)==0: return None
        rows=[build_main_row(r) for _,r in cand.iterrows()]
        df=pd.DataFrame(rows)
        df=pd.get_dummies(df,
            columns=["mode","category","durationType","progressType","deviceType"],
            drop_first=True)
        for c in X_main.columns:
            if c not in df.columns: df[c]=0
        df=df[X_main.columns]
        scores=main_model.predict_proba(df)[:,1]
        cand=cand.reset_index(drop=True)
        cand["score"]=scores
        return cand.sort_values("score",ascending=False).iloc[0]

    def recommend_speed():
        if speed_model is None: return None
        cand=challenge_meta[challenge_meta["mode"]=="speed"].copy()
        cand["available"]=cand.apply(lambda r:is_available(r,last_done,today),axis=1)
        cand=cand[cand["available"]==True]
        if len(cand)==0: return None
        notif_list=[9*60,13*60,18*60,21*60]
        rows=[]
        for _,ch in cand.iterrows():
            for nm in notif_list:
                rows.append({
                    "challengeId":ch["challengeId"],
                    "category":ch["category"],
                    "weekday":tw,
                    "notif_min":nm,
                    "personalPoints":4,
                    "familyPoints":12,
                    "energyKwh":1.0 if ch["category"]=="energy" else 0.0,
                })
        df=pd.DataFrame(rows)
        df=pd.get_dummies(df,columns=["category","challengeId"],drop_first=True)
        for c in X_sp.columns:
            if c not in df.columns: df[c]=0
        df=df[X_sp.columns]
        probs=speed_model.predict_proba(df)[:,1]
        df["score"]=probs
        best=df.loc[df["score"].idxmax()].to_dict()
        best["notificationTime"]=minutes_to_timestr(best["notif_min"])
        return best

    daily=recommend_non_speed("daily")
    monthly=recommend_non_speed("monthly")
    speed=recommend_speed()

    print("\n=== 오늘 추천 ===")
    print("데일리:", daily.to_dict() if daily is not None else None)
    print("먼슬리:", monthly.to_dict() if monthly is not None else None)
    print("스피드:", speed)

if __name__=="__main__":
    main()
