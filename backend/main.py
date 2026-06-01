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


def _binary_to_radial_profile(gray, cx, cy, max_r):
    """DoG+Otsu → 二值 → 径向黑像素占比"""
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

    y, x = np.indices((binary.shape[0], binary.shape[1]))
    dist = np.floor(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)).astype(np.int32)

    black_count = np.zeros(max_r, dtype=np.int32)
    total_count = np.zeros(max_r, dtype=np.int32)

    valid = (dist >= 0) & (dist < max_r)
    np.add.at(black_count, dist[valid & (binary == 0)], 1)
    np.add.at(total_count, dist[valid], 1)
    total_count[total_count == 0] = 1

    ratio = black_count.astype(np.float32) / total_count.astype(np.float32)
    ratio = np.convolve(ratio, np.ones(11) / 11, mode="same")

    return ratio


def _find_rings(ratio, max_r):
    """从径向轮廓中找暗环（跳过中心和外边框），返回每个环的半径"""
    min_r = max(40, int(max_r * 0.08))
    max_r_inner = int(max_r * 0.88)  # 排除外边框 ~12%
    if max_r_inner <= min_r:
        return []

    thr = np.mean(ratio) + np.std(ratio) * 0.25

    # 找局部峰值，相邻峰值间距至少 12px
    peaks = []
    i = min_r
    while i < max_r_inner - 1:
        if ratio[i] > thr and ratio[i] >= ratio[i - 1] and ratio[i] >= ratio[i + 1]:
            # 取局部窗口内的精确峰值
            search_start = max(min_r, i - 5)
            search_end = min(max_r_inner, i + 5)
            peak_idx = search_start + np.argmax(ratio[search_start:search_end])

            # 检查谷底值——峰值必须比两侧谷底深 std*0.25
            left_valley = np.min(ratio[max(min_r, peak_idx - 10):peak_idx])
            right_valley = np.min(ratio[peak_idx:min(max_r_inner, peak_idx + 10)])
            prominence = ratio[peak_idx] - min(left_valley, right_valley)
            if prominence > np.std(ratio) * 0.25:
                peaks.append(peak_idx)

            i = peak_idx + 12
        else:
            i += 1

    return peaks


def _detect_rings(gray):
    """检测牛顿环暗环数"""
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    max_r = int(min(w, h) * 0.45)

    ratio = _binary_to_radial_profile(gray, cx, cy, max_r)
    rings = _find_rings(ratio, max_r)

    return len(rings), f"检测到约 {len(rings)} 个暗环"


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

    ratio = _binary_to_radial_profile(gray, cx, cy, max_r)
    rings = _find_rings(ratio, max_r)

    binary_out = np.ones((h, w), dtype=np.uint8) * 255
    for r in rings:
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
