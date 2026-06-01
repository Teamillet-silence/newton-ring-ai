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
    """检测牛顿环暗环数"""
    h, w = gray.shape

    # ======== 先转成清晰的黑白图 ========
    # 1. CLAHE 增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. 高斯模糊去噪
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 3. DoG 提取环的边缘
    blur2 = cv2.GaussianBlur(enhanced, (31, 31), 0)
    detail = blur - blur2

    # 4. 转成黑白
    detail = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(detail, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 确保背景白色(255)、暗环黑色(0)
    cx, cy = w // 2, h // 2
    if binary[cy, cx] == 0:
        binary = 255 - binary

    # ======== 原方法：单行扫描 + 均值阈值 ========
    scan_line = binary[cy, cx:]

    # 均值阈值
    threshold = np.mean(scan_line)

    transitions = 0
    prev = 1

    for pixel in scan_line:
        val = 1 if pixel < threshold else 0
        if val != prev:
            transitions += 1
            prev = val

    ring_count = transitions // 2
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