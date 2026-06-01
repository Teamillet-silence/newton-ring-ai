"use client";

import { useState, useRef, useEffect } from "react";

import axios from "axios";

import ReactMarkdown from "react-markdown";

import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";

import "katex/dist/katex.min.css";

import DigitalHuman from "@/components/DigitalHuman";
import { useVoice } from "@/hooks/useVoice";
import { cleanLatexForSpeech } from "@/lib/latexToSpeech";

export default function Home() {

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  const [message, setMessage] = useState("");

  const [messages, setMessages] = useState<any[]>([
    {
      role: "assistant",
      content: `
你好！我是牛顿环实验 AI 助手。

你可以：

- 提问牛顿环公式
- 询问实验原理
- 上传牛顿环图片
- 自动检测圆环

例如：

$$
r_m^2 = m\\lambda R
$$
      `,
    },
  ]);

  const [loading, setLoading] = useState(false);

  const [uploadResult, setUploadResult] = useState("");
  const [binaryPreview, setBinaryPreview] = useState("");

  const chatBoxRef = useRef<HTMLDivElement>(null);

  const { isSpeaking, speak, mouthShape, mouthOpen, voices, switchVoice } = useVoice();

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages]);

  // 发送聊天
  const sendMessage = async () => {

    if (!message.trim()) return;

    const userMessage = {
      role: "user",
      content: message,
    };

    setMessages((prev) => [...prev, userMessage]);

    setLoading(true);

    try {

      const response = await axios.post(
        `${API_BASE}/chat`,
        {
          message: message,
        }
      );

      const reply = response.data.reply;

      const aiMessage = {
        role: "assistant",
        content: reply,
      };

      setMessages((prev) => [...prev, aiMessage]);

      speak(cleanLatexForSpeech(reply));

    } catch (error) {

      console.error(error);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "后端连接失败",
        },
      ]);
    }

    setMessage("");

    setLoading(false);
  };

  // 上传图片
  const handleUpload = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {

    const file = e.target.files?.[0];

    if (!file) return;

    const formData = new FormData();

    formData.append("file", file);

    try {

      const response = await axios.post(
        `${API_BASE}/upload`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setUploadResult(
        `检测结果：${response.data.message}`
      );

      // 获取黑白预览图
      const previewResp = await axios.post(
        `${API_BASE}/preview-binary`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          responseType: "blob",
        }
      );
      setBinaryPreview(URL.createObjectURL(previewResp.data));

    } catch (error) {

      console.error(error);

      setUploadResult("上传失败");

    }
  };

  return (

    <div className="container">

      <h1>牛顿环 AI 助手</h1>

      <div className="voice-row">
        <select
          className="voice-select"
          defaultValue=""
          onChange={(e) => {
            const v = voices.find((v) => v.name === e.target.value);
            if (v) switchVoice(v);
          }}
        >
          <option value="">默认语音</option>
          {voices.map((v) => (
            <option key={v.name} value={v.name}>{v.name}</option>
          ))}
        </select>
        <button className="voice-test-btn" onClick={() => speak("你好，我是牛顿环AI助手。")}>
          测试语音
        </button>
      </div>

      <div className="main-layout">

        {/* 数字人面板 */}
        <div className="digital-human-panel">
          <DigitalHuman isSpeaking={isSpeaking} mouthShape={mouthShape} mouthOpen={mouthOpen} />
        </div>

        {/* 聊天面板 */}
        <div className="chat-panel">

          {/* 上传 */}
          <div className="upload-box">

            <input
              type="file"
              onChange={handleUpload}
            />

          </div>

          {/* 上传结果 */}
          {
            uploadResult && (

              <div className="upload-result">

                <p>{uploadResult}</p>

                {binaryPreview && (
                  <img
                    src={binaryPreview}
                    alt="黑白预览"
                    style={{ width: 200, marginTop: 8, borderRadius: 6 }}
                  />
                )}

              </div>
            )
          }

          {/* 聊天区域 */}
          <div className="chat-box" ref={chatBoxRef}>

            {
              messages.map((msg, index) => (

                <div
                  key={index}
                  className={
                    msg.role === "user"
                      ? "user-message"
                      : "ai-message"
                  }
                >

                  <ReactMarkdown
                    remarkPlugins={[
                      remarkMath,
                      remarkGfm
                    ]}
                    rehypePlugins={[
                      rehypeKatex
                    ]}
                  >
                    {msg.content}
                  </ReactMarkdown>

                </div>
              ))
            }

            {
              loading && (
                <p>AI 思考中...</p>
              )
            }

          </div>

          {/* 输入区域 */}
          <div className="input-box">

            <input
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="请输入问题..."
              onKeyDown={(e) => {
                if (e.key === "Enter") sendMessage();
              }}
            />

            <button onClick={sendMessage}>
              发送
            </button>

          </div>

        </div>

      </div>

    </div>
  );
}