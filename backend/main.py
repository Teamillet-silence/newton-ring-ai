from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel

import requests
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "sk-yeocqzbbkzytjjtzemydtpcunfsqxjqewajjxrqpdcyputgo"

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):

    url = "https://api.siliconflow.cn/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [
            {
                "role": "system",
                "content": """
你是一个专业的牛顿环实验 AI 助手。

要求：

1. 所有数学公式必须使用 LaTeX。

2. 行内公式使用：
$...$

例如：
$\\lambda$
$r_m$

3. 独立公式使用：
$$...$$

例如：
$$
r_m^2 = m\\lambda R
$$

4. 禁止使用：
\\(...)
\\[...\\]

5. 所有物理量解释必须使用 LaTeX。

6. 回答适合大学物理实验教学。
"""
            },
            {
                "role": "user",
                "content": req.message
            }
        ],
        "temperature": 0.7
    }

    response = requests.post(
        url,
        headers=headers,
        json=data
    )

    result = response.json()

    ai_reply = result["choices"][0]["message"]["content"]

    ai_reply = ai_reply.replace("\\(", "$")
    ai_reply = ai_reply.replace("\\)", "$")

    ai_reply = ai_reply.replace("\\[", "$$")
    ai_reply = ai_reply.replace("\\]", "$$")

    return {
        "reply": ai_reply
    }


def _detect_rings(gray):
    """检测牛顿环暗环数——径向平均 + 找波谷"""
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    max_r = int(min(w, h) * 0.45)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 每个半径上的平均亮度（径向平均）
    radial_profile = []
    for r in range(1, max_r):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, 1)
        mean_val = cv2.mean(blur, mask)[0]
        radial_profile.append(mean_val)

    profile = np.array(radial_profile, dtype=np.float32)

    # 平滑
    profile = np.convolve(profile, np.ones(7) / 7, mode="same")

    # 找波谷（暗环 = 亮度局部最低点）
    valleys = []
    for i in range(2, len(profile) - 2):
        if profile[i] < profile[i - 1] and profile[i] < profile[i + 1]:
            # 谷够深才计入
            left = max(profile[i - 2], profile[i - 1])
            right = max(profile[i + 1], profile[i + 2])
            depth = min(left, right) - profile[i]
            if depth > np.std(profile) * 0.3:
                valleys.append(i)

    # 合并相邻的波谷（取较深的）
    merged = []
    for v in valleys:
        if not merged or v - merged[-1] > 3:
            merged.append(v)

    ring_count = max(0, len(merged))
    return ring_count, f"检测到约 {ring_count} 个暗环"


@app.post("/preview-binary")
async def preview_binary(file: UploadFile = File(...)):

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "message": "图片读取失败"}

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 生成径向平均图
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    max_r = int(min(w, h) * 0.45)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    radial_profile = []
    for r in range(1, max_r):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, 1)
        mean_val = cv2.mean(blur, mask)[0]
        radial_profile.append(mean_val)

    profile = np.array(radial_profile, dtype=np.float32)
    profile = np.convolve(profile, np.ones(7) / 7, mode="same")

    # 生成黑白图：波谷位置画黑圈
    binary = np.ones((h, w), dtype=np.uint8) * 255
    valleys = []
    for i in range(2, len(profile) - 2):
        if profile[i] < profile[i - 1] and profile[i] < profile[i + 1]:
            left = max(profile[i - 2], profile[i - 1])
            right = max(profile[i + 1], profile[i + 2])
            depth = min(left, right) - profile[i]
            if depth > np.std(profile) * 0.3:
                valleys.append(i)

    merged = []
    for v in valleys:
        if not merged or v - merged[-1] > 3:
            merged.append(v)
            cv2.circle(binary, (cx, cy), v + 1, 0, 3)

    _, buffer = cv2.imencode(".png", binary)
    return Response(content=buffer.tobytes(), media_type="image/png")


@app.post("/preview-binary")
async def preview_binary(file: UploadFile = File(...)):

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "message": "图片读取失败"}

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    binary = _to_binary(gray)

    _, buffer = cv2.imencode(".png", binary)

    return Response(content=buffer.tobytes(), media_type="image/png")


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "message": "图片读取失败"}

    h, w = image.shape[:2]

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    rings, msg = _detect_rings(gray)

    return {
        "success": True,
        "rings": rings,
        "message": msg,
        "width": w,
        "height": h
    }


@app.get("/")
async def root():

    return {
        "message": "Newton Ring AI Backend Running"
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
