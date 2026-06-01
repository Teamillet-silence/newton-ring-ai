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
    """检测牛顿环暗环数：DoG→Otsu→二值径向轮廓→找峰"""
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    max_r = int(min(w, h) * 0.45)

    # 1) DoG + Otsu 二值化
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur1 = cv2.GaussianBlur(enhanced, (5, 5), 0)
    blur2 = cv2.GaussianBlur(enhanced, (31, 31), 0)
    detail = blur1 - blur2

    detail = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(detail, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 黑环白底
    if binary[cy, cx] == 0:
        binary = 255 - binary

    # 2) 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 3) 二值径向轮廓：每个半径上黑像素占比
    y, x = np.indices((h, w))
    dist = np.floor(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)).astype(np.int32)

    black_count = np.zeros(max_r, dtype=np.int32)
    total_count = np.zeros(max_r, dtype=np.int32)

    valid = (dist >= 0) & (dist < max_r)
    black_pixels = valid & (binary == 0)

    np.add.at(black_count, dist[black_pixels], 1)
    np.add.at(total_count, dist[valid], 1)
    total_count[total_count == 0] = 1

    ratio = black_count.astype(np.float32) / total_count.astype(np.float32)

    # 4) 平滑 + 找峰
    ratio = np.convolve(ratio, np.ones(7) / 7, mode="same")

    # 跳过中心暗斑（~30px）
    min_r = 30
    if max_r <= min_r:
        return 0, "检测到约 0 个暗环"

    thr = np.mean(ratio) + np.std(ratio) * 0.3

    # 找局部峰值
    peaks = []
    for i in range(min_r + 1, max_r - 1):
        if ratio[i] > thr and ratio[i] >= ratio[i - 1] and ratio[i] > ratio[i + 1]:
            peaks.append(i)

    # 合并 5px 内的相邻峰值（保留较高的）
    merged = []
    for p in peaks:
        if merged and p - merged[-1] <= 5:
            if ratio[p] > ratio[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)

    ring_count = len(merged)
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

    h, w = gray.shape
    cx, cy = w // 2, h // 2
    max_r = int(min(w, h) * 0.45)

    # 同 _detect_rings 的前半段
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur1 = cv2.GaussianBlur(enhanced, (5, 5), 0)
    blur2 = cv2.GaussianBlur(enhanced, (31, 31), 0)
    detail = blur1 - blur2
    detail = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(detail, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary[cy, cx] == 0:
        binary = 255 - binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    y, x = np.indices((h, w))
    dist = np.floor(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)).astype(np.int32)

    black_count = np.zeros(max_r, dtype=np.int32)
    total_count = np.zeros(max_r, dtype=np.int32)
    valid = (dist >= 0) & (dist < max_r)
    black_pixels = valid & (binary == 0)
    np.add.at(black_count, dist[black_pixels], 1)
    np.add.at(total_count, dist[valid], 1)
    total_count[total_count == 0] = 1

    ratio = black_count.astype(np.float32) / total_count.astype(np.float32)
    ratio = np.convolve(ratio, np.ones(7) / 7, mode="same")

    # 跳过中心暗斑（~30px）
    min_r = 30
    thr = np.mean(ratio) + np.std(ratio) * 0.3

    peaks = []
    for i in range(min_r + 1, max_r - 1):
        if ratio[i] > thr and ratio[i] >= ratio[i - 1] and ratio[i] > ratio[i + 1]:
            peaks.append(i)

    merged = []
    for p in peaks:
        if merged and p - merged[-1] <= 5:
            if ratio[p] > ratio[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)

    # 画细线圆环
    binary_out = np.ones((h, w), dtype=np.uint8) * 255
    for r in merged:
        cv2.circle(binary_out, (cx, cy), r, 0, 2)

    _, buffer = cv2.imencode(".png", binary_out)
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
