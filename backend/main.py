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


def _compute_radial_profile(gray, cx, cy, max_r):
    """计算径向亮度轮廓——从中心向外每个半径的像素均值"""
    h, w = gray.shape
    y, x = np.indices((h, w))
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    profile = np.zeros(max_r, dtype=np.float32)
    count = np.zeros(max_r, dtype=np.int32)

    # 只算有效范围
    valid = (dist < max_r)
    y_v, x_v = y[valid], x[valid]
    d_v = np.floor(dist[valid]).astype(np.int32)

    np.add.at(profile, d_v, gray[y_v, x_v].astype(np.float32))
    np.add.at(count, d_v, 1)

    count[count == 0] = 1
    profile /= count

    return profile


def _find_valleys(profile):
    """找波谷——亮度局部最小且足够深"""
    profile = np.convolve(profile, np.ones(5) / 5, mode="same")

    valleys = []
    for i in range(2, len(profile) - 2):
        if profile[i] <= profile[i - 1] and profile[i] <= profile[i + 1]:
            # 谷深度 = 两侧峰值取小
            left_peak = max(profile[i - 2], profile[i - 1])
            right_peak = max(profile[i + 1], profile[i + 2])
            depth = min(left_peak, right_peak) - profile[i]
            if depth > np.std(profile) * 0.2:
                valleys.append(i)

    # 合并5px以内的相邻谷（取较深的）
    merged = []
    for v in valleys:
        if merged and v - merged[-1] <= 5:
            prev_idx = merged[-1]
            if profile[v] < profile[prev_idx]:
                merged[-1] = v
        else:
            merged.append(v)

    return merged, profile


def _create_binary_from_valleys(h, w, cx, cy, valley_indices):
    """在谷位置画粗黑圆环"""
    binary = np.ones((h, w), dtype=np.uint8) * 255
    for r in valley_indices:
        cv2.circle(binary, (cx, cy), r, 0, 4)
    return binary


def _detect_rings(gray):
    """检测牛顿环暗环数"""
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    max_r = int(min(w, h) * 0.45)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    profile = _compute_radial_profile(blur, cx, cy, max_r)
    valleys, smoothed = _find_valleys(profile)

    return len(valleys), f"检测到约 {len(valleys)} 个暗环"


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

    h, w = gray.shape
    cx, cy = w // 2, h // 2
    max_r = int(min(w, h) * 0.45)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    profile = _compute_radial_profile(blur, cx, cy, max_r)
    valleys, smoothed = _find_valleys(profile)
    binary = _create_binary_from_valleys(h, w, cx, cy, valleys)

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
