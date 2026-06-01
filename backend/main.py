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
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "message": "图片读取失败"}

    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    # 灰度 + 高斯模糊
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 自适应阈值（处理颜色深浅不一的问题）
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 5
    )

    # 形态学去噪
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 从中心向多个方向扫描，取平均值
    directions = 8
    total_rings = 0

    for i in range(directions):
        angle = 2 * np.pi * i / directions
        rings = 0
        prev = 0

        for r in range(10, min(width, height) // 2, 2):
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))

            if x < 0 or x >= width or y < 0 or y >= height:
                break

            val = binary[y, x] // 255
            if val != prev:
                rings += 1
                prev = val

        total_rings += rings // 2

    ring_count = round(total_rings / directions)
    if ring_count < 0:
        ring_count = 0

    return {
        "success": True,
        "rings": ring_count,
        "message": f"检测到约 {ring_count} 个暗环",
        "width": width,
        "height": height
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