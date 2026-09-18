"""
TAKEDOWN AI - Backend Server (안전한 "직접 업로드" 방식)
--------------------------------------------------------
핵심 아이디어:
1. 아이패드가 서버에 "업로드하고 싶어"라고만 요청 (영상은 아직 안 보냄)
2. 서버가 Gemini에 "재개 가능한 업로드 URL"을 요청해서 받음
   -> 이 임시 URL에는 API 키가 필요 없어서, 아이패드에 그대로 넘겨줘도 안전함
3. 아이패드가 그 임시 URL로 영상을 "직접" Gemini에 업로드
   -> 우리 서버(Render)는 영상 자체를 한 바이트도 안 받기 때문에 메모리 문제가 없음!
4. 업로드가 끝나면 아이패드가 서버에 "분석해줘"라고 요청 (영상의 Gemini 파일 이름만 보냄, 아주 작은 텍스트)
5. 서버가 그 파일을 Gemini에게 분석시키고, 결과 JSON을 아이패드에 돌려줌

이 구조는 API 키가 절대 브라우저(index.html)에 노출되지 않으면서도,
Render 무료 플랜(메모리 512MB)에서도 큰 영상을 처리할 수 있게 해줍니다.
"""

import os
import json
import re
import time

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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
GEMINI_UPLOAD_BASE = "https://generativelanguage.googleapis.com/upload/v1beta/files"


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


# ---------------------------------------------------------
# 1단계 API: 아이패드가 "업로드하고 싶다"고 요청하면,
# 서버가 대신 Gemini에게 임시 업로드 주소를 받아서 돌려줍니다.
# 이 임시 주소에는 API 키가 필요 없어서, 그대로 브라우저에 넘겨도 안전합니다.
# ---------------------------------------------------------
class StartUploadRequest(BaseModel):
    file_size: int
    mime_type: str


@app.post("/api/start-upload")
def start_upload(req: StartUploadRequest):
    size_mb = req.file_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=400, detail=f"파일이 너무 큽니다 ({size_mb:.1f}MB). {MAX_UPLOAD_MB}MB 이하로 업로드해주세요.")

    resp = requests.post(
        GEMINI_UPLOAD_BASE,
        headers={
            "x-goog-api-key": GEMINI_API_KEY,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(req.file_size),
            "X-Goog-Upload-Header-Content-Type": req.mime_type,
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": "wrestling_video"}},
        timeout=30,
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini 업로드 준비 중 오류: {resp.text}")

    upload_url = resp.headers.get("x-goog-upload-url")
    if not upload_url:
        raise HTTPException(status_code=502, detail="Gemini로부터 업로드 주소를 받지 못했습니다.")

    # 이 upload_url은 이번 업로드 한 번에만 쓸 수 있는 임시 주소라서 안전합니다 (API 키 불필요).
    return {"upload_url": upload_url}


# ---------------------------------------------------------
# 2단계 API: 아이패드가 위에서 받은 upload_url로 영상을 "직접" Gemini에 업로드한 뒤,
# 그 결과로 받은 file_uri/file_name을 여기로 보내서 "분석해줘"라고 요청합니다.
# 이 단계에서는 아주 작은 텍스트(파일 이름)만 오가기 때문에 서버 메모리 부담이 없습니다.
# ---------------------------------------------------------
class AnalyzeRequest(BaseModel):
    file_name: str  # 예: "files/abc123"


@app.post("/api/analyze")
def analyze_video(req: AnalyzeRequest):
    try:
        uploaded_file = client.files.get(name=req.file_name)

        wait_seconds = 0
        while uploaded_file.state.name == "PROCESSING" and wait_seconds < 120:
            time.sleep(2)
            wait_seconds += 2
            uploaded_file = client.files.get(name=req.file_name)

        if uploaded_file.state.name == "FAILED":
            raise HTTPException(status_code=502, detail="Gemini 서버에서 영상 처리에 실패했습니다.")

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
        try:
            client.files.delete(name=req.file_name)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
