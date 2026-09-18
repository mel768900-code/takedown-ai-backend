"""
TAKEDOWN AI - Backend Server (Render.com 상시 구동용)
--------------------------------------------------
- google-genai 최신 라이브러리 사용 (새 형식 API 키 대응)
- 503(서버 혼잡) 자동 재시도 로직 포함
- ngrok 없음: Render.com이 자체 도메인을 줍니다
"""

import os
import json
import re
import tempfile
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

# -----------------------------
# 0. 초기 설정
# -----------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다. Render 대시보드에서 설정해주세요.")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="TAKEDOWN AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_INSTRUCTION = """
너는 세계 최고의 레슬링 감독이자 비디오 분석관이야.
사용자가 올린 태클/메치기 동영상을 정밀하게 분석해줘.

다음 3가지 항목을 반드시 영상 안에서 시간 흐름을 따라가며 체크할 것:
1. 선수의 머리 위치(시선) - 기술 진입/전개 시 머리가 아래로 떨어지는지, 시선이 상대의 어느 지점을 보는지.
2. 엉덩이(골반) 높이 및 중심 이동 - 무게중심이 낮게 유지되는지, 전진/회전 시 중심이 안정적으로 이동하는지.
3. 상대방과의 타이밍/거리 - 진입 타이밍이 상대의 체중 이동 시점과 맞는지, 거리 조절이 적절한지.

분석이 끝나면 반드시 아래 JSON 형식으로만 응답할 것. 다른 설명, 마크다운, 코드블록 기호(```)를 절대 붙이지 말고
순수 JSON 텍스트 하나만 출력할 것:

{
  "score": 85,
  "good": "잘한 점에 대한 구체적인 설명 (2~3문장, 한국어)",
  "bad": "개선이 필요한 점에 대한 구체적인 설명 (2~3문장, 한국어)",
  "recommend": "다음에 연계하면 좋을 기술 또는 보완 운동 추천 (한국어)"
}

score는 0~100 사이의 정수여야 한다. 반드시 위 4개 키(score, good, bad, recommend)만 포함한 JSON을 반환할 것.
"""

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "300"))
ALLOWED_CONTENT_TYPES = {"video/mp4", "video/quicktime", "video/x-m4v", "video/webm"}


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "TAKEDOWN AI 백엔드가 정상 작동 중입니다."}


@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):
    if video.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다: {video.content_type}. mp4/mov/webm만 업로드해주세요."
        )

    contents = await video.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=400,
            detail=f"파일이 너무 큽니다 ({size_mb:.1f}MB). {MAX_UPLOAD_MB}MB 이하로 업로드해주세요."
        )

    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    tmp_path = None
    uploaded_file = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        uploaded_file = client.files.upload(file=tmp_path)

        wait_seconds = 0
        while uploaded_file.state.name == "PROCESSING" and wait_seconds < 120:
            time.sleep(2)
            wait_seconds += 2
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise HTTPException(status_code=502, detail="Gemini 서버에서 영상 처리에 실패했습니다.")

        # ---- 503(혼잡) 자동 재시도 로직 ----
        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[uploaded_file, "이 레슬링 영상을 분석하고 지시된 JSON 형식으로만 답변해줘."],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        temperature=0.4,
                    ),
                )
                break
            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(8)
                    continue
                else:
                    raise

        if response is None:
            raise HTTPException(status_code=503, detail="Gemini 서버가 계속 혼잡합니다. 잠시 후 다시 시도해주세요.")

        result = extract_json(response.text)
        result["score"] = int(max(0, min(100, int(result.get("score", 0)))))
        for key in ("good", "bad", "recommend"):
            result.setdefault(key, "분석 결과를 가져오지 못했습니다.")

        return JSONResponse(content=result)

    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Gemini 응답을 JSON으로 해석할 수 없습니다.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
