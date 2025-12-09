# weekly_ai_report.py
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_weekly_ai_report(data: dict):
    period = data["period"]
    today = data["today_success_count"]
    streaks = data["category_streaks"]
    contributions = data["weekly_contributions"]

    system_prompt = """
너는 사용자의 일주일 습관 데이터를 기반으로
행동 패턴을 3줄로 요약하는 AI 코치야.

출력 규칙:
1. 이번 주 전반적인 흐름 요약 (1줄)
2. 잘 이어지고 있는 패턴 요약 (1줄)
3. 내일 이어가기 좋은 구체적인 제안 (1줄)
- 무조건 3문장
- 한국어만 출력
"""

    user_prompt = f"""
기간: {period['start']} ~ {period['end']}
오늘 성공 횟수: {today}
카테고리별 연속 성공일: {streaks}
주간 기여도(weekly_contributions): {contributions}
"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_completion_tokens=200,
    )

    content = completion.choices[0].message.content.strip()
    lines = [line for line in content.split("\n") if line.strip()]

    while len(lines) < 3:
        lines.append("")

    return {
        "line1": lines[0],
        "line2": lines[1],
        "line3": lines[2],
        "raw": content,
    }
