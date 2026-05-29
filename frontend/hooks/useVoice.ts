"use client";

import { useState, useRef, useCallback, useEffect } from "react";

const MOUTH_SHAPES = ["aa", "ih", "ou", "ee", "oh"] as const;

export function useVoice() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [mouthShape, setMouthShape] = useState<string>("");
  const [mouthOpen, setMouthOpen] = useState(0);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);

  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);
  const shapeTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const openTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const load = () => {
      const all = window.speechSynthesis.getVoices();
      setVoices(all.filter((v) => v.lang.startsWith("zh")));
    };
    load();
    window.speechSynthesis.addEventListener("voiceschanged", load);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", load);
  }, []);

  const stopShapeAnimation = useCallback(() => {
    if (shapeTimerRef.current) {
      clearInterval(shapeTimerRef.current);
      shapeTimerRef.current = null;
    }
    if (openTimerRef.current) {
      clearInterval(openTimerRef.current);
      openTimerRef.current = null;
    }
    setMouthShape("");
    setMouthOpen(0);
  }, []);

  const startMouthAnimation = useCallback(() => {
    let shapeIndex = 0;
    shapeTimerRef.current = setInterval(() => {
      setMouthShape(MOUTH_SHAPES[shapeIndex % MOUTH_SHAPES.length]);
      shapeIndex++;
    }, 160);

    let openDir = 1;
    let openVal = 0;
    openTimerRef.current = setInterval(() => {
      openVal += openDir * 0.15;
      if (openVal >= 1) { openVal = 1; openDir = -1; }
      if (openVal <= 0.3) { openVal = 0.3; openDir = 1; }
      setMouthOpen(openVal);
    }, 40);
  }, []);

  const speak = useCallback((text: string) => {
    if (!text.trim()) return;

    window.speechSynthesis.cancel();
    stopShapeAnimation();

    setTimeout(() => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "zh-CN";
      utterance.rate = 1.05;
      utterance.pitch = 1.5;
      if (voiceRef.current) utterance.voice = voiceRef.current;

      utterance.onstart = () => {
        setIsSpeaking(true);
        startMouthAnimation();
      };

      utterance.onend = () => {
        setIsSpeaking(false);
        stopShapeAnimation();
      };

      utterance.onerror = () => {
        setIsSpeaking(false);
        stopShapeAnimation();
      };

      window.speechSynthesis.speak(utterance);
    }, 50);
  }, [startMouthAnimation, stopShapeAnimation]);

  const switchVoice = useCallback((voice: SpeechSynthesisVoice) => {
    voiceRef.current = voice;
  }, []);

  const stop = useCallback(() => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
    stopShapeAnimation();
  }, [stopShapeAnimation]);

  useEffect(() => {
    return () => {
      window.speechSynthesis.cancel();
      stopShapeAnimation();
    };
  }, [stopShapeAnimation]);

  return { isSpeaking, speak, stop, mouthShape, mouthOpen, voices, switchVoice };
}
