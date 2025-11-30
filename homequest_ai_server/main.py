from fastapi import FastAPI
from google.cloud import storage
from pydantic import BaseModel
import pandas as pd
import io

from homequest_ai_final import run_recommendation

app = FastAPI()

# 🔹 Firebase Storage 설정
SERVICE_ACCOUNT = "firebase-service-account.json"  # 같은 폴더에 있는 서비스 계정 키
BUCKET_NAME = "homequest-dev.firebasestorage.app"
FILE_PATH = "homequest_simulated_6months.csv"


def load_csv_from_storage():
    client = storage.Client.from_service_account_json(SERVICE_ACCOUNT)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(FILE_PATH)

    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))
    return df


print("CSV 로딩 중 (Firebase Storage)...")
df_events = load_csv_from_storage()
print(f"CSV 로딩 완료, 행 개수 = {len(df_events)}")


@app.get("/test")
def test():
    return {
        "rows": len(df_events),
        "preview": df_events.head().to_dict(orient="records"),
    }


class RecommendRequest(BaseModel):
    userId: str = "user_4"
    top_k: int = 3


@app.post("/recommend")
def recommend(req: RecommendRequest):
    result = run_recommendation(
        events_df=df_events,
        user_id=req.userId,
        top_k=req.top_k,
    )
    return result
