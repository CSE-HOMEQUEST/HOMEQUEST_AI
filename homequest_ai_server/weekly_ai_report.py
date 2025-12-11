# weekly_ai_report.py
import os
import json
from openai import OpenAI

# Cloud Run의 환경변수에서 API 키를 읽음
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_weekly_ai_report(data: dict):
    """
    주간 습관 데이터를 기반으로 행동 패턴을 3줄로 요약하는 AI 보고서를 생성합니다.

    Args:
        data (dict): 사용자 습관 데이터.
            {
              "period": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
              "today_success_count": int,
              "category_streaks": str,
              "weekly_contributions": str
            }

    Returns:
        dict: 3줄 요약이 포함된 JSON 형식의 결과.
            {
              "line1": "<첫 번째 문장>",
              "line2": "<두 번째 문장>",
              "line3": "<세 번째 문장>"
            }
    """

    # 입력 데이터 파싱
    period = data.get("period", {})
    today = data.get("today_success_count", 0)
    streaks = data.get("category_streaks", "")
    contributions = data.get("weekly_contributions", "")

    # ----------------------------------------------------------------------
    # 시스템 프롬프트 (새로운 규칙 반영)
    # ----------------------------------------------------------------------
    system_prompt = f"""
역할:
너는 사용자의 일주일 습관 데이터를 기반으로 행동 패턴을 3줄로 요약하는 AI 코치야.

일반 규칙:
- 무조건 한국어로만 출력해.
- 항상 정확히 3문장만 생성해.
- 말투는 부드러운 존댓말을 사용하고, “~요”로 자연스럽게 끝나는 문장을 사용해.
- 사용자를 평가하거나 비난하지 말고, 관찰 + 격려 + 제안에 집중해.
- AI, 모델, 데이터 같은 기술적 표현은 쓰지 마.
- 극단적인 표현(“전혀”, “완전히 실패” 등)은 피하고, 쉬어간 흐름을 부드럽게 표현해.

1번 문장(line1) 규칙 – 오늘 + 연속 흐름 요약:
- today_success_count와 category_streaks를 바탕으로,
  오늘 어느 정도로 성공했고, 연속으로 얼만큼 이어졌는지를 간단하게 표현해.
- 한 문장 안에 “오늘의 성공 정도”와 “연속성(스트릭)”이 모두 느껴지도록 써.
- 사용자를 격려해주는 톤이어야 해.

2번 문장(line2) 규칙 – 일주일 패턴 요약:
- weekly_contributions를 바탕으로, 최근 일주일이 어떤 패턴을 보이는지 예리하게 짚어내.
- 숫자를 정확히 세어 언급하기보다는, 패턴의 성격을 한 줄로 간략하게 표현해.

3번 문장(line3) 규칙 – 패턴 기반 미세 행동 전략 제안:
- 2번 문장에서 짚어낸 주간 패턴을 기준으로,
  다음 주에 가장 자연스럽게 이어갈 수 있는 ‘작은 행동 전략’을 한 문장으로 부드럽게 제안해.
- 구체적이지만 부담 없는 제안을 해.

출력 형식:
아래 JSON 형식으로만 출력해.
{{
  "line1": "<첫 번째 문장>",
  "line2": "<두 번째 문장>",
  "line3": "<세 번째 문장>"
}}
- line1, line2, line3에는 각각 한 문장씩만 넣어.
"""

    # ----------------------------------------------------------------------
    # 사용자 프롬프트
    # ----------------------------------------------------------------------
    user_prompt = f"""
기간: {period.get('start', 'N/A')} ~ {period.get('end', 'N/A')}
오늘 성공 횟수: {today}
카테고리별 연속 성공일: {streaks}
주간 기여도(weekly_contributions): {contributions}
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.2,
            # JSON 모드를 사용하여 정확한 출력 형식 강제
            response_format={"type": "json_object"},
            max_tokens=300,  # 안전을 위해 max_completion_tokens를 max_tokens로 변경하고 넉넉하게 설정
        )

        content = completion.choices[0].message.content.strip()

        # JSON 문자열을 파싱
        report_json = json.loads(content)

        # 요청한 필드(line1, line2, line3)가 모두 있는지 확인하고 반환
        return {
            "line1": report_json.get("line1", ""),
            "line2": report_json.get("line2", ""),
            "line3": report_json.get("line3", ""),
        }

    except Exception as e:
        # 에러 발생 시 처리
        print(f"AI 보고서 생성 중 오류 발생: {e}")
        return {
            "line1": "데이터를 분석하는 중 잠시 오류가 발생했어요.",
            "line2": "다음 주에 다시 한번 시도해 주시면 감사하겠습니다.",
            "line3": "사용자님의 꾸준한 노력은 항상 최고예요!",
        }
