from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from pydantic import BaseModel

import requests
import cv2
import numpy as np

# =========================
# 创建 FastAPI
# =========================
app = FastAPI()

# =========================
# 允许跨域
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# SiliconFlow API KEY
# =========================
API_KEY = "sk-yeocqzbbkzytjjtzemydtpcunfsqxjqewajjxrqpdcyputgo"

# =========================
# 聊天请求模型
# =========================
class ChatRequest(BaseModel):
    message: str

# =========================
# 聊天接口
# =========================
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

    # latex 修复
    ai_reply = ai_reply.replace("\\(", "$")
    ai_reply = ai_reply.replace("\\)", "$")

    ai_reply = ai_reply.replace("\\[", "$$")
    ai_reply = ai_reply.replace("\\]", "$$")

    return {
        "reply": ai_reply
    }

# =========================
# 上传图片接口（牛顿环专用）
# =========================
def _detect_rings(gray):
    """检测牛顿环暗环数，返回 (rings_count, message)"""
    h, w = gray.shape
    cx, cy = w // 2, h // 2

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 多方向径向扫描取平均
    directions = 12
    all_counts = []

    for i in range(directions):
        angle = 2 * np.pi * i / directions
        dx, dy = np.cos(angle), np.sin(angle)
        profile = []
        max_r = int(min(w, h) * 0.45)

        for r in range(5, max_r):
            x = int(cx + r * dx)
            y = int(cy + r * dy)
            if 0 <= x < w and 0 <= y < h:
                profile.append(blur[y, x])

        if len(profile) < 10:
            continue

        profile = np.array(profile, dtype=np.float32)

        # 归一化去掉整体亮度趋势
        baseline = np.convolve(profile, np.ones(21) / 21, mode="same")
        baseline[:10] = baseline[10]
        baseline[-10:] = baseline[-10]
        normalized = baseline - profile

        # 找波峰（暗环）
        peaks = 0
        for j in range(1, len(normalized) - 1):
            if normalized[j] > normalized[j - 1] and normalized[j] > normalized[j + 1]:
                if normalized[j] > np.std(normalized) * 0.5:
                    peaks += 1

        all_counts.append(peaks)

    if not all_counts:
        return 0, "未检测到暗环"

    # 去掉最高最低再平均，更稳健
    sorted_counts = sorted(all_counts)
    trimmed = sorted_counts[len(sorted_counts) // 4:-len(sorted_counts) // 4] if len(
        sorted_counts) >= 4 else sorted_counts
    ring_count = round(np.mean(trimmed))
    ring_count = max(0, ring_count)

    return ring_count, f"检测到约 {ring_count} 个暗环"


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "message": "图片读取失败"}

    h, w = image.shape[:2]

    # 判断是否是彩色图，如果是，在亮度通道上检测
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

# =========================
# 根路径测试
# =========================
@app.get("/")
async def root():

    return {
        "message": "Newton Ring AI Backend Running"
    }

# =========================
# 启动
# =========================
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )