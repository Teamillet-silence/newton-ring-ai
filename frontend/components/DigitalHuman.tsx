"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import type { VRM } from "@pixiv/three-vrm";

interface Props {
  isSpeaking?: boolean;
  mouthShape?: string;
  mouthOpen?: number;
}

const SHAPES = ["aa", "ih", "ou", "ee", "oh"];

export default function DigitalHuman({
  isSpeaking = false,
  mouthShape = "",
  mouthOpen = 0,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const vrmRef = useRef<VRM | null>(null);
  const speakingRef = useRef(false);
  const shapeRef = useRef("");
  const openRef = useRef(0);
  const timerRef = useRef(new THREE.Timer());

  useEffect(() => {
    speakingRef.current = isSpeaking;
    shapeRef.current = mouthShape;
    openRef.current = mouthOpen;
  }, [isSpeaking, mouthShape, mouthOpen]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const parent = canvas.parentElement;
    if (!parent) return;

    const w = parent.clientWidth;
    const h = parent.clientHeight;

    const scene = new THREE.Scene();

    const bgTex = new THREE.TextureLoader().load("/background.jpg");
    bgTex.colorSpace = THREE.SRGBColorSpace;
    scene.background = bgTex;

    const camera = new THREE.PerspectiveCamera(30, w / h, 0.1, 20);
    camera.position.set(0, 0.9, 3.5);
    camera.lookAt(0, 0.9, 0);

    const renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
    });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);

    const mainLight = new THREE.DirectionalLight(0xffffff, 1.2);
    mainLight.position.set(1, 2, 3);
    scene.add(mainLight);

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
    fillLight.position.set(-1, 1, 1);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xffffff, 0.3);
    rimLight.position.set(0, 2, -2);
    scene.add(rimLight);

    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    loader.load(
      "/avatar.vrm",
      (gltf) => {
        const vrm = gltf.userData.vrm as VRM;
        vrmRef.current = vrm;
        vrm.scene.scale.set(0.75, 0.75, 0.75);
        scene.add(vrm.scene);
        VRMUtils.rotateVRM0(vrm);
      },
      undefined,
      (error) => console.error("VRM load error:", error)
    );

    let blinkTimer = 3 + Math.random() * 4;

    const animate = () => {
      requestAnimationFrame(animate);

      const delta = timerRef.current.getDelta();
      const elapsed = timerRef.current.getElapsed();
      const vrm = vrmRef.current;

      if (vrm) {
        if (speakingRef.current && vrm.expressionManager) {
          const activeShape = shapeRef.current || "aa";
          for (const s of SHAPES) {
            vrm.expressionManager.setValue(s, s === activeShape ? openRef.current : 0);
          }
        } else if (vrm.expressionManager) {
          for (const s of SHAPES) {
            vrm.expressionManager.setValue(s, 0);
          }
        }

        blinkTimer -= delta;
        if (blinkTimer <= 0) {
          if (vrm.expressionManager) {
            vrm.expressionManager.setValue("blink", 1);
            setTimeout(() => {
              vrm.expressionManager?.setValue("blink", 0);
            }, 150);
          }
          blinkTimer = 3 + Math.random() * 4;
        }

        vrm.scene.position.y = Math.sin(elapsed * 1.5) * 0.003;

        const headNode = vrm.humanoid?.getRawBoneNode("head");
        if (headNode && speakingRef.current) {
          headNode.rotation.y = Math.sin(elapsed * 2) * 0.05;
          headNode.rotation.x = Math.sin(elapsed * 3) * 0.02;
        }

        const la = vrm.humanoid?.getNormalizedBoneNode("leftUpperArm");
        const ra = vrm.humanoid?.getNormalizedBoneNode("rightUpperArm");
        const lf = vrm.humanoid?.getNormalizedBoneNode("leftLowerArm");
        const rf = vrm.humanoid?.getNormalizedBoneNode("rightLowerArm");
        if (la) la.rotation.set(0, 0, -1.3);
        if (ra) ra.rotation.set(0, 0, 1.3);
        if (lf) lf.rotation.set(0, 0, -0.6);
        if (rf) rf.rotation.set(0, 0, 0.6);

        vrm.update(delta);
      }

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!parent) return;
      const nw = parent.clientWidth;
      const nh = parent.clientHeight;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      renderer.dispose();
      if (vrmRef.current) {
        VRMUtils.deepDispose(vrmRef.current.scene);
      }
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: "100%", display: "block" }}
    />
  );
}
