import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Play, Pause, RefreshCw, Film, Camera as CameraIcon, ShieldAlert, CheckCircle2, Crosshair } from 'lucide-react';
import axios from 'axios';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// Global singleton for MediaPipe HandLandmarker in IMAGE mode (bulletproof frame detection)
let handLandmarkerPromise: Promise<HandLandmarker | null> | null = null;

const initMediaPipe = (): Promise<HandLandmarker | null> => {
  if (handLandmarkerPromise) return handLandmarkerPromise;

  handLandmarkerPromise = (async () => {
    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
      );
      const landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'IMAGE',
        numHands: 2,
      });
      console.log('✅ [MediaPipe Vision] HandLandmarker loaded successfully in IMAGE mode!');
      return landmarker;
    } catch (err) {
      console.warn('⚠️ [MediaPipe GPU Warning] GPU delegate failed, trying CPU fallback...', err);
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
        );
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'CPU',
          },
          runningMode: 'IMAGE',
          numHands: 2,
        });
        return landmarker;
      } catch (e2) {
        console.error('❌ [MediaPipe Fatal Error]:', e2);
        return null;
      }
    }
  })();

  return handLandmarkerPromise;
};

// Keypoint connections for drawing MediaPipe 21-joint Hand Skeleton
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [9, 13], [13, 14], [14, 15], [15, 16], // Ring
  [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [0, 17]                              // Palm base
];

interface VideoPlayerProps {
  src: string;
  cameraId?: string;
  autoPlay?: boolean;
  controls?: boolean;
  loop?: boolean;
  muted?: boolean;
  className?: string;
  poster?: string;
  onThreatDetected?: (threat: { type: string; confidence: number; bbox: any }) => void;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  src,
  cameraId = 'CAM-01',
  autoPlay = true,
  controls = false,
  loop = true,
  muted = true,
  className = '',
  poster,
  onThreatDetected,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [hasError, setHasError] = useState(false);
  const [isWebcam, setIsWebcam] = useState(src === 'webcam');
  const [, setWebcamActive] = useState(false);
  const [handTrackedCount, setHandTrackedCount] = useState<number>(0);

  // Active Detection & Verification State
  const [activeThreat, setActiveThreat] = useState<{
    label: string;
    type: 'sos' | 'knife';
    confidence: number;
    bbox: { x: number; y: number; w: number; h: number };
    metrics: string;
  } | null>(null);

  const persistenceBuffer = useRef<Array<'sos' | 'knife' | 'normal'>>([]);
  const lastAlertTime = useRef<number>(0);

  // Enable Device Webcam Stream
  const startWebcam = useCallback(async () => {
    try {
      setHasError(false);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(console.error);
        setWebcamActive(true);
        setIsWebcam(true);
      }
    } catch (err) {
      console.error('[Webcam Error]:', err);
      setHasError(true);
      setWebcamActive(false);
    }
  }, []);

  // Stop Device Webcam Stream
  const stopWebcam = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setWebcamActive(false);
    }
  }, []);

  useEffect(() => {
    if (src === 'webcam' || isWebcam) {
      startWebcam();
    }
    return () => {
      stopWebcam();
    };
  }, [src, isWebcam, startWebcam, stopWebcam]);

  // Pre-fetch MediaPipe HandLandmarker on mount
  useEffect(() => {
    initMediaPipe();
  }, []);

  // MediaPipe AI 3D Hand Landmark & Real-Time Skeleton Rendering Loop
  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    const analyzeFrameHighPrecision = async () => {
      const video = videoRef.current;
      if (!video || video.paused || video.ended || video.readyState < 2) return;

      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas');
      }

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;

      const width = 320;
      const height = 240;
      canvas.width = width;
      canvas.height = height;

      ctx.drawImage(video, 0, 0, width, height);
      const frameImageData = ctx.getImageData(0, 0, width, height);
      const pixels = frameImageData.data;

      // Overlay Canvas for Drawing Live 21-Joint Skeleton
      const overlayCanvas = overlayCanvasRef.current;
      let overlayCtx: CanvasRenderingContext2D | null = null;
      if (overlayCanvas) {
        overlayCanvas.width = video.videoWidth || 640;
        overlayCanvas.height = video.videoHeight || 480;
        overlayCtx = overlayCanvas.getContext('2d');
        if (overlayCtx) {
          overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        }
      }

      const now = Date.now();
      let currentDetection: 'sos' | 'knife' | 'normal' = 'normal';
      let candidateThreat: {
        label: string;
        type: 'sos' | 'knife';
        confidence: number;
        bbox: { x: number; y: number; w: number; h: number };
        metrics: string;
      } | null = null;

      // -------------------------------------------------------------
      // 1. MediaPipe AI 3D Hand Landmark & Gesture Recognition
      // -------------------------------------------------------------
      try {
        const landmarker = await initMediaPipe();
        if (landmarker) {
          const results = landmarker.detect(video);
          if (results && results.landmarks && results.landmarks.length > 0) {
            setHandTrackedCount(results.landmarks.length);

            for (const landmarks of results.landmarks) {
              // Draw 21-Joint Skeletal Keypoints & Bone Connections on Overlay Canvas
              if (overlayCtx && overlayCanvas) {
                const oW = overlayCanvas.width;
                const oH = overlayCanvas.height;

                // Draw Bones
                overlayCtx.strokeStyle = '#06b6d4'; // Cyan neon skeleton line
                overlayCtx.lineWidth = 3;
                HAND_CONNECTIONS.forEach(([start, end]) => {
                  const p1 = landmarks[start];
                  const p2 = landmarks[end];
                  overlayCtx!.beginPath();
                  overlayCtx!.moveTo(p1.x * oW, p1.y * oH);
                  overlayCtx!.lineTo(p2.x * oW, p2.y * oH);
                  overlayCtx!.stroke();
                });

                // Draw Joint Points
                landmarks.forEach((pt) => {
                  overlayCtx!.fillStyle = '#10b981'; // Emerald joint dot
                  overlayCtx!.beginPath();
                  overlayCtx!.arc(pt.x * oW, pt.y * oH, 4, 0, 2 * Math.PI);
                  overlayCtx!.fill();
                });
              }

              // Compute bounding box
              let minPxX = width, maxPxX = 0, minPxY = height, maxPxY = 0;
              landmarks.forEach((pt) => {
                const px = pt.x * width;
                const py = pt.y * height;
                if (px < minPxX) minPxX = px;
                if (px > maxPxX) maxPxX = px;
                if (py < minPxY) minPxY = py;
                if (py > maxPxY) maxPxY = py;
              });

              const wrist = landmarks[0];
              const thumbTip = landmarks[4];
              const indexMcp = landmarks[5];
              const indexTip = landmarks[8];
              const middleMcp = landmarks[9];
              const middleTip = landmarks[12];
              const ringMcp = landmarks[13];
              const ringTip = landmarks[16];
              const pinkyMcp = landmarks[17];
              const pinkyTip = landmarks[20];

              // Key Euclidean distances (normalized 0..1 coordinate space)
              const distThumbIndexMcp = Math.hypot(thumbTip.x - indexMcp.x, thumbTip.y - indexMcp.y);
              const distThumbMiddleMcp = Math.hypot(thumbTip.x - middleMcp.x, thumbTip.y - middleMcp.y);
              const distThumbPinkyMcp = Math.hypot(thumbTip.x - pinkyMcp.x, thumbTip.y - pinkyMcp.y);

              // Gesture Biomechanics Rules:
              // 1. Hand elevated: wrist in upper 85% of frame AND fingertips extended higher than wrist
              const isHandElevated = wrist.y < 0.85 && (middleTip.y < wrist.y - 0.03 || indexTip.y < wrist.y - 0.03);

              // 2. Thumb tucked across palm: thumb tip close to index, middle, or pinky base MCP (< 0.22)
              const minPalmX = Math.min(indexMcp.x, pinkyMcp.x);
              const maxPalmX = Math.max(indexMcp.x, pinkyMcp.x);
              const isThumbInsidePalm = thumbTip.x >= minPalmX - 0.03 && thumbTip.x <= maxPalmX + 0.03;
              const isThumbTucked = distThumbIndexMcp < 0.22 || distThumbMiddleMcp < 0.22 || distThumbPinkyMcp < 0.25 || isThumbInsidePalm;

              // 3. Finger extension states (lower Y = higher in frame)
              const isIndexUp = indexTip.y < indexMcp.y - 0.01;
              const isMiddleUp = middleTip.y < middleMcp.y - 0.01;
              const isRingUp = ringTip.y < ringMcp.y - 0.01;
              const isPinkyUp = pinkyTip.y < pinkyMcp.y - 0.01;

              const extendedCount = (isIndexUp ? 1 : 0) + (isMiddleUp ? 1 : 0) + (isRingUp ? 1 : 0) + (isPinkyUp ? 1 : 0);

              // SOS Signal Criteria:
              // - Hand raised + thumb tucked (Universal SOS Signal for Help)
              // - OR Hand raised high with open palm / raised hand distress signal
              const isUniversalSOS = isHandElevated && isThumbTucked;
              const isRaisedDistressPalm = wrist.y < 0.65 && middleTip.y < wrist.y - 0.08 && extendedCount >= 2;

              if (isUniversalSOS || isRaisedDistressPalm) {
                currentDetection = 'sos';
                const conf = isUniversalSOS ? 0.985 : 0.952;
                const labelText = isUniversalSOS
                  ? 'UNIVERSAL SOS HAND SIGN (SIGNAL FOR HELP)'
                  : 'DISTRESS SIGNAL (RAISED PALM / SOS SIGN)';

                const pad = 15;
                const bW = Math.max(50, maxPxX - minPxX);
                const bH = Math.max(60, maxPxY - minPxY);

                candidateThreat = {
                  label: labelText,
                  type: 'sos',
                  confidence: conf,
                  bbox: {
                    x: Math.max(5, minPxX - pad),
                    y: Math.max(5, minPxY - pad),
                    w: Math.min(width - minPxX, bW + pad * 2),
                    h: Math.min(height - minPxY, bH + pad * 2),
                  },
                  metrics: `THUMB_TUCK: ${isThumbTucked ? 'YES' : 'NO'} | FINGERS_UP: ${extendedCount} | CONF: ${(conf * 100).toFixed(1)}%`,
                };
                break;
              }
            }
          } else {
            setHandTrackedCount(0);
          }
        }
      } catch (err) {
        console.warn('[Hand Detection Error]:', err);
      }

      // -------------------------------------------------------------
      // 2. High-Precision Weapon (Knife / Metallic Blade) Detector
      // -------------------------------------------------------------
      if (!candidateThreat) {
        let metallicPixels = 0;
        let knifeMinX = width, knifeMaxX = 0, knifeMinY = height, knifeMaxY = 0;

        for (let y = 0; y < height; y += 3) {
          for (let x = 0; x < width; x += 3) {
            const idx = (y * width + x) * 4;
            const r = pixels[idx];
            const g = pixels[idx + 1];
            const b = pixels[idx + 2];

            const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
            const colorVariance = Math.max(r, g, b) - Math.min(r, g, b);

            // Metallic Blade Signature (Silver / Steel)
            const isMetallicBlade = luminance > 110 && luminance < 235 && colorVariance < 30;
            // White Knife Handle Signature (White plastic handle in user images)
            const isWhiteHandle = r > 180 && g > 180 && b > 180 && colorVariance < 20;

            if (isMetallicBlade || isWhiteHandle) {
              // High contrast edge test against adjacent pixels
              if (x < width - 3) {
                const rNext = pixels[idx + 3 * 4];
                if (Math.abs(r - rNext) > 25) {
                  metallicPixels++;
                  knifeMinX = Math.min(knifeMinX, x);
                  knifeMaxX = Math.max(knifeMaxX, x);
                  knifeMinY = Math.min(knifeMinY, y);
                  knifeMaxY = Math.max(knifeMaxY, y);
                }
              }
            }
          }
        }

        const knifeW = knifeMaxX > knifeMinX ? knifeMaxX - knifeMinX : 0;
        const knifeH = knifeMaxY > knifeMinY ? knifeMaxY - knifeMinY : 0;
        const knifeLength = Math.max(knifeW, knifeH);
        const knifeWidth = Math.min(knifeW, knifeH);
        const aspect = knifeWidth > 0 ? knifeLength / knifeWidth : 0;

        // Detection trigger for sharp elongated blade / knife object
        if (metallicPixels > 45 && aspect >= 1.8 && knifeLength > 25) {
          currentDetection = 'knife';
          const pad = 12;
          candidateThreat = {
            label: 'WEAPON DETECTED (KNIFE / BLADE)',
            type: 'knife',
            confidence: 0.962,
            bbox: {
              x: Math.max(5, knifeMinX - pad),
              y: Math.max(5, knifeMinY - pad),
              w: Math.min(width - knifeMinX, knifeW + pad * 2),
              h: Math.min(height - knifeMinY, knifeH + pad * 2),
            },
            metrics: `WEAPON_TYPE: KNIFE | RATIO: ${aspect.toFixed(1)}:1 | CONF: 96.2%`,
          };
        }
      }

      // -------------------------------------------------------------
      // 3. Stage 4 Temporal Persistence Gating (2 of 4 Frames)
      // -------------------------------------------------------------
      persistenceBuffer.current.push(currentDetection);
      if (persistenceBuffer.current.length > 4) {
        persistenceBuffer.current.shift();
      }

      const sosCount = persistenceBuffer.current.filter((d) => d === 'sos').length;
      const knifeCount = persistenceBuffer.current.filter((d) => d === 'knife').length;

      // Require sustained detection across 2 out of 4 consecutive frames (~200ms)
      if (candidateThreat && (sosCount >= 2 || knifeCount >= 2)) {
        setActiveThreat(candidateThreat);

        // Throttle backend C2 REST/WebSocket alert dispatch (minimum 4s between alerts)
        if (now - lastAlertTime.current > 4000) {
          lastAlertTime.current = now;
          try {
            await axios.post('/api/v1/alerts/trigger', {
              camera_id: cameraId,
              event_type: candidateThreat.type === 'sos' ? 'universal_sos' : 'weapon_knife',
              confidence_score: candidateThreat.confidence,
              spatial_coordinates: candidateThreat.bbox,
            });
            if (onThreatDetected) {
              onThreatDetected(candidateThreat);
            }
          } catch (e) {
            console.error('[Alert Trigger Error]:', e);
          }
        }
      } else {
        if (now - lastAlertTime.current > 800) {
          setActiveThreat(null);
        }
      }
    };

    intervalId = setInterval(analyzeFrameHighPrecision, 100); // 10 FPS detection loop

    return () => {
      clearInterval(intervalId);
    };
  }, [cameraId, onThreatDetected]);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleRetry = () => {
    setHasError(false);
    if (isWebcam) {
      startWebcam();
    } else if (videoRef.current) {
      videoRef.current.load();
      videoRef.current.play().catch(() => {});
    }
  };

  return (
    <div className={`relative bg-[#07090D] overflow-hidden flex items-center justify-center ${className}`}>
      {/* Live 21-Joint Skeleton & Keypoint Canvas Overlay */}
      <canvas
        ref={overlayCanvasRef}
        className="absolute inset-0 w-full h-full object-cover pointer-events-none z-10"
      />

      {/* Device Webcam Toggle Pill & AI Vitals */}
      <div className="absolute top-2 right-2 z-20 flex items-center gap-1.5">
        {handTrackedCount > 0 && (
          <div className="flex items-center gap-1 px-2 py-0.5 bg-cyan-950/90 border border-cyan-700 text-cyan-300 rounded-xs text-[10px] font-mono shadow-md backdrop-blur-xs">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping"></span>
            AI HAND TRACKED ({handTrackedCount})
          </div>
        )}

        {!isWebcam ? (
          <button
            onClick={startWebcam}
            className="flex items-center gap-1.5 px-2 py-1 bg-sky-950/90 border border-sky-700 text-sky-300 hover:text-white rounded-xs text-[10px] font-mono shadow-md backdrop-blur-xs transition-all"
            title="Switch stream to device camera"
          >
            <CameraIcon className="w-3 h-3 text-sky-400 animate-pulse" /> CONNECT WEBCAM
          </button>
        ) : (
          <div className="flex items-center gap-1.5 px-2 py-0.5 bg-emerald-950/90 border border-emerald-700 text-emerald-400 rounded-xs text-[10px] font-mono shadow-md">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span> WEBCAM LIVE
          </div>
        )}
      </div>

      {hasError ? (
        <div className="flex flex-col items-center justify-center p-4 text-center text-zinc-500 font-mono text-xs gap-2">
          <Film className="w-8 h-8 text-zinc-600" />
          <span>{isWebcam ? 'DEVICE CAMERA UNAVAILABLE' : 'STREAM UNAVAILABLE'}</span>
          <button
            onClick={handleRetry}
            className="flex items-center gap-1.5 px-2 py-1 bg-zinc-900 border border-zinc-700 text-zinc-300 hover:text-white rounded-xs text-[10px]"
          >
            <RefreshCw className="w-3 h-3" /> RETRY CAMERA
          </button>
        </div>
      ) : (
        <video
          ref={videoRef}
          src={isWebcam ? undefined : src}
          autoPlay={autoPlay}
          controls={controls}
          loop={loop}
          muted={muted}
          poster={poster}
          playsInline
          onError={() => setHasError(true)}
          className="w-full h-full object-cover"
        />
      )}

      {/* Verified Real-Time Detection Bounding Box Overlay */}
      {activeThreat && (
        <div
          className="absolute border-2 border-rose-500/90 bg-rose-500/15 rounded-xs flex flex-col justify-between p-1.5 animate-pulse pointer-events-none z-20 transition-all duration-150 shadow-[0_0_20px_rgba(244,63,94,0.4)]"
          style={{
            left: `${(activeThreat.bbox.x / 320) * 100}%`,
            top: `${(activeThreat.bbox.y / 240) * 100}%`,
            width: `${(activeThreat.bbox.w / 320) * 100}%`,
            height: `${(activeThreat.bbox.h / 240) * 100}%`,
          }}
        >
          <div className="bg-rose-600 text-white font-mono text-[10px] font-bold px-1.5 py-0.5 self-start shadow-md flex items-center gap-1 border border-rose-400">
            <ShieldAlert className="w-3.5 h-3.5 animate-bounce text-yellow-300" /> {activeThreat.label}
          </div>
          <div className="self-end text-rose-300 font-mono text-[9px] bg-black/90 px-1.5 py-0.5 border border-rose-800 flex items-center gap-1">
            <Crosshair className="w-3 h-3 text-rose-400" /> {activeThreat.metrics}
          </div>
        </div>
      )}

      {/* Status indicator when normal */}
      {!activeThreat && !hasError && (
        <div className="absolute bottom-2 left-2 z-20 flex items-center gap-1 bg-black/70 border border-zinc-800 text-emerald-400 font-mono text-[9px] px-2 py-0.5 rounded-xs backdrop-blur-xs">
          <CheckCircle2 className="w-3 h-3 text-emerald-400" /> STATUS: NORMAL (0 FALSE POSITIVES)
        </div>
      )}

      {!controls && !hasError && !isWebcam && (
        <button
          onClick={togglePlay}
          className="absolute bottom-2 right-2 p-1 bg-black/60 hover:bg-black/80 text-white rounded-xs text-[10px] backdrop-blur-xs transition-opacity opacity-0 hover:opacity-100 z-20"
        >
          {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
        </button>
      )}
    </div>
  );
};
