import os
from io import StringIO
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

BASE_CSV = "homequest_simulated_6months.csv"
TARGET_ROWS = 10000
BATCH_SIZE = 100
OUTPUT_CSV = "homequest_simulated_10k.csv"

df_base = pd.read_csv(BASE_CSV)
sample_df = df_base.sample(min(30, len(df_base)), random_state=42)
sample_csv_text = sample_df.to_csv(index=False)

def clean_csv_text(raw_text):
    text = raw_text.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            if "eventId" in p and "," in p:
                text = p
                break
    text = text.replace("csv", "").strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("빈 응답")
    header = lines[0]
    n_cols = header.count(",") + 1
    cleaned = [header]
    for ln in lines[1:]:
        if ln.count(",") + 1 == n_cols:
            cleaned.append(ln)
    if len(cleaned) <= 1:
        raise ValueError("유효한 데이터 없음")
    return "\n".join(cleaned)

def generate_rows(n_rows):
    prompt = f"""
너는 스마트홈 이벤트 로그를 생성하는 CSV 데이터 생성기다.
출력은 CSV 텍스트만 포함해야 하며 다른 문장은 절대 포함하지 마라.

새롭게 생성해야 하는 행 수: {n_rows}

CSV 스키마는 다음과 같다.
eventId,familyId,userId,challengeId,category,mode,durationType,progressType,deviceType,eventDate,day_index,weekday,notificationTime,completionTime,completed,personalPoints,familyPoints,energyKwh

규칙:
1) 첫 줄에는 위 스키마와 동일한 헤더를 넣어라.
2) familyId,userId,challengeId,category,mode,durationType,progressType,deviceType 값은 예시 CSV에 등장하는 값만 사용하라.
3) eventDate는 2025-10-01 이상 2026-03-31 이하의 날짜만 사용하라. 형식은 YYYY-MM-DD.
4) weekday는 eventDate의 실제 요일과 일치하도록 0~6 범위에서 설정하라. 0은 월요일, 6은 일요일이다.
5) notificationTime과 completionTime은 HH:MM:SS 형식으로, completionTime은 notificationTime보다 같거나 늦게 설정하라.
6) completed는 0 또는 1만 사용하라.
7) personalPoints,familyPoints,energyKwh는 예시 CSV의 분포와 비슷한 범위를 유지하라.
8) 기존 예시 행을 복사하지 말고 모두 새로운 행을 생성하라.
9) CSV 중간에 끊기거나 헤더만 있고 본문이 없는 상태가 되지 않도록 완전한 CSV를 생성하라.

아래는 예시 CSV이다. 이 구조와 값의 범위를 따라라.

[예시 시작]
{sample_csv_text}
[예시 끝]

위 예시를 기반으로 새로운 {n_rows}개의 행만 포함하는 CSV 전체를 생성하라.
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=4000
    )
    raw = res.choices[0].message.content or ""
    cleaned = clean_csv_text(raw)
    return pd.read_csv(StringIO(cleaned))

def assign_ids(orig, new):
    df = pd.concat([orig, new], ignore_index=True)
    if "eventDate" in df.columns and "notificationTime" in df.columns:
        df = df.sort_values(by=["eventDate", "notificationTime"])
    df = df.reset_index(drop=True)
    df["eventId"] = df.index.map(lambda i: f"evt_{i:05d}")
    return df

generated_list = []
current = len(df_base)
total_batches = (TARGET_ROWS - len(df_base) + BATCH_SIZE - 1) // BATCH_SIZE
batch_index = 0

while current < TARGET_ROWS:
    batch_index += 1
    to_make = min(BATCH_SIZE, TARGET_ROWS - current)
    print(f"[{batch_index}/{total_batches}] {to_make}행 생성 요청 중... 현재 {current}/{TARGET_ROWS}")
    try:
        new = generate_rows(to_make)
    except Exception as e:
        print("에러 발생:", e)
        break
    generated_list.append(new)
    current += len(new)
    print(f" → 누적 {current}/{TARGET_ROWS}")

if generated_list:
    df_new = pd.concat(generated_list, ignore_index=True)
else:
    df_new = pd.DataFrame(columns=df_base.columns)

df_final = assign_ids(df_base, df_new)
df_final.to_csv(OUTPUT_CSV, index=False)

print("생성 완료:", len(df_final))
print("파일:", OUTPUT_CSV)
