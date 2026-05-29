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

    # 读取图片
    contents = await file.read()

    # numpy转换
    np_array = np.frombuffer(contents, np.uint8)

    # OpenCV解码
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # 判断读取
    if image is None:

        return {
            "success": False,
            "message": "图片读取失败"
        }

    # 灰度化
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # 获取尺寸
    height, width = gray.shape

    # 中心点
    center_x = width // 2
    center_y = height // 2

    # 高斯模糊
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # 从中心向右扫描
    scan_line = blur[center_y, center_x:]

    # 自动阈值
    threshold = np.mean(scan_line)

    # 二值化
    binary = []

    for pixel in scan_line:

        if pixel < threshold:

            binary.append(1)

        else:

            binary.append(0)

    # 统计变化次数
    transitions = 0

    for i in range(1, len(binary)):

        if binary[i] != binary[i - 1]:

            transitions += 1

    # 一个暗环大约对应两次变化
    ring_count = transitions // 2

    # 防止误判
    if ring_count < 0:
        ring_count = 0

    # =========================
    # 返回结果
    # =========================
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