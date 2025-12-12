import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality } from "@google/genai";
import type { LiveServerMessage } from "@google/genai";
import { GameState, FallingNa, GameMode, ScoreEntry } from '../types';
import { MODE_DURATIONS, BEAT_INTERVAL_MS, COUNTDOWN_SECONDS } from '../constants';
import { encode } from '../utils/audioUtils';
import FallingNaComponent from './FallingNa';

const HIT_WINDOW_MS = 100; // Tightened window, but we check 8th notes now
const CLEAR_THRESHOLDS = {
  '10s': 70,
  '30s': 210,
  'FULL': 365
};

// SVG Component for the Crowd
const Crowd: React.FC<{ progress: number }> = ({ progress }) => {
  // progress 0 to 1
  const showLevel1 = progress > 0.3;
  const showLevel2 = progress > 0.6;
  const showLevel3 = progress >= 1.0;

  return (
    <div className="absolute bottom-0 left-0 w-full h-1/2 pointer-events-none z-0 overflow-hidden">
      {/* Level 1: Back Row - Light Gray */}
      <div className={`absolute bottom-[-50px] left-0 w-full h-full flex justify-around items-end transition-transform duration-700 ${showLevel1 ? 'translate-y-0 opacity-100' : 'translate-y-full opacity-0'}`}>
         {Array.from({ length: 8 }).map((_, i) => (
             <div key={`l1-${i}`} className="w-24 h-32 bg-gray-200 rounded-t-full mx-1 animate-[crowd-jump_0.6s_infinite] origin-bottom" style={{ animationDelay: `${i * 0.1}s` }}></div>
         ))}
      </div>

      {/* Level 2: Mid Row - Medium Gray */}
      <div className={`absolute bottom-[-30px] left-[-5%] w-[110%] h-full flex justify-around items-end transition-transform duration-700 delay-100 ${showLevel2 ? 'translate-y-0 opacity-100' : 'translate-y-full opacity-0'}`}>
         {Array.from({ length: 7 }).map((_, i) => (
             <div key={`l2-${i}`} className="w-32 h-40 bg-gray-300 rounded-t-full mx-1 animate-[crowd-jump_0.5s_infinite] origin-bottom" style={{ animationDelay: `${i * 0.15}s` }}></div>
         ))}
      </div>

      {/* Level 3: Front Row - Darker Gray */}
      <div className={`absolute bottom-[-10px] left-[-10%] w-[120%] h-full flex justify-around items-end transition-transform duration-700 delay-200 ${showLevel3 ? 'translate-y-0 opacity-100' : 'translate-y-full opacity-0'}`}>
         {Array.from({ length: 6 }).map((_, i) => (
             <div key={`l3-${i}`} className="w-40 h-48 bg-gray-400 rounded-t-full mx-1 animate-[crowd-jump_0.4s_infinite] origin-bottom" style={{ animationDelay: `${i * 0.05}s` }}>
                {/* Hands */}
                <div className="absolute -top-10 -left-4 w-12 h-24 bg-gray-400 rounded-full rotate-[-20deg]"></div>
                <div className="absolute -top-10 -right-4 w-12 h-24 bg-gray-400 rounded-full rotate-[20deg]"></div>
             </div>
         ))}
      </div>
    </div>
  );
};

const Game: React.FC = () => {
  const [gameState, setGameState] = useState<GameState>('READY');
  const [gameMode, setGameMode] = useState<GameMode>('30s');
  const [score, setScore] = useState(0); // This is "Hits" now
  const [subscribers, setSubscribers] = useState(0); // This is the new "Score"
  const [timeLeft, setTimeLeft] = useState(MODE_DURATIONS['30s']);
  const [countdown, setCountdown] = useState(COUNTDOWN_SECONDS);
  const [fallingNas, setFallingNas] = useState<FallingNa[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [highScores, setHighScores] = useState<ScoreEntry[]>([]);
  
  // Animation State
  const [isHitAnimating, setIsHitAnimating] = useState(false);

  // Asset URLs - using jsDelivr for reliable CDN hosting from GitHub
  const defaultBgmUrl = 'https://cdn.jsdelivr.net/gh/mshinyukari/-@main/bgm/bgm.mp3';
  const menuBgmUrl = 'https://cdn.jsdelivr.net/gh/mshinyukari/-@main/bgm/top.mp3';

  // Settings State
  const [volume, setVolume] = useState(0.5);
  const [bgmSource, setBgmSource] = useState<string>(defaultBgmUrl);
  const [customUrl, setCustomUrl] = useState(''); // State for manual URL input
  const [micThreshold, setMicThreshold] = useState(30); // 0-100 visual scale
  const [inputLevel, setInputLevel] = useState(0);
  const [isMicTesting, setIsMicTesting] = useState(false);
  
  // Device Selection State
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');

  const sessionPromise = useRef<Promise<any> | null>(null);
  const lastTranscription = useRef('');
  const audioContextRef = useRef<AudioContext | null>(null);
  const metronomeInterval = useRef<number | null>(null);
  const gameTimer = useRef<number | null>(null);
  const bgmRef = useRef<HTMLAudioElement>(null);
  const menuBgmRef = useRef<HTMLAudioElement>(null);
  
  // Refs for Mic Test & Game Audio Analysis
  const streamRef = useRef<MediaStream | null>(null);
  const micAnalyserRef = useRef<AnalyserNode | null>(null);
  const micCheckInterval = useRef<number | null>(null);
  const micAudioCtxRef = useRef<AudioContext | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const isGameActiveRef = useRef(false); // Strict control for scoring loop

  // Logic Refs
  const lastAudioVolume = useRef(0);
  const isArmedRef = useRef(true); // For "Valley" detection
  const currentNotePeakRef = useRef(0); // Track peak of current sound to determine drop-off
  
  // Rhythm Refs
  const lastBeatTimeRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0); // To calculate elapsed time for inflation
  const pauseStartTimeRef = useRef<number>(0); // To adjust startTimeRef after pause

  // Sync BGM Volume
  useEffect(() => {
      // Only sync volume if NOT in countdown (countdown uses silent volume 0)
      if (gameState !== 'COUNTDOWN') {
          if (bgmRef.current) {
              bgmRef.current.volume = volume;
          }
      }
      if (menuBgmRef.current) {
          menuBgmRef.current.volume = volume;
      }
  }, [volume, gameState]);

  // Handle BGM File Selection
  const handleBgmSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) {
          const file = e.target.files[0];
          const url = URL.createObjectURL(file);
          setBgmSource(url);
          setCustomUrl(''); // Clear custom URL if file selected
          // Revoke old URL if it was a blob URL
          if (bgmSource.startsWith('blob:')) {
              URL.revokeObjectURL(bgmSource);
          }
      }
  };

  // Handle URL Input
  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const url = e.target.value;
      setCustomUrl(url);
      if (url) {
          setBgmSource(url);
      } else {
          setBgmSource(defaultBgmUrl);
      }
  };

  // --- Device Management ---
  const fetchAudioDevices = useCallback(async () => {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const inputs = devices.filter(d => d.kind === 'audioinput');
        setAudioDevices(inputs);
        
        // Default to first device if none selected, or if selected one is gone
        if (inputs.length > 0) {
             const exists = inputs.find(d => d.deviceId === selectedDeviceId);
             if (!selectedDeviceId || !exists) {
                 setSelectedDeviceId(inputs[0].deviceId);
             }
        }
    } catch (e) {
        console.warn("Could not enumerate devices");
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    fetchAudioDevices();
    // Listen for device changes (plug/unplug)
    navigator.mediaDevices.addEventListener('devicechange', fetchAudioDevices);
    return () => navigator.mediaDevices.removeEventListener('devicechange', fetchAudioDevices);
  }, [fetchAudioDevices]);

  // Helper to get stream with selected device - IMPROVED to reuse stream
  const getAudioStream = async () => {
      // Reuse active stream if possible
      if (streamRef.current) {
          const tracks = streamRef.current.getAudioTracks();
          if (tracks.length > 0 && tracks[0].readyState === 'live') {
              // Check if it matches selected device
              const settings = tracks[0].getSettings();
              if (!selectedDeviceId || settings.deviceId === selectedDeviceId) {
                  return streamRef.current;
              }
          }
      }
      
      // If we need to get a new one, stop the old one first
      if (streamRef.current) {
          streamRef.current.getTracks().forEach(t => t.stop());
      }
      
      const constraints = {
          audio: selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      return stream;
  };

  // --- Score Saving ---
  const saveScore = (finalSubscribers: number, finalHitCount: number) => {
    const key = `rhythm_na_subs_${gameMode}`;
    const existing = localStorage.getItem(key);
    let scores: ScoreEntry[] = existing ? JSON.parse(existing) : [];
    
    // Safety check to ensure we only save primitives to avoid circular JSON errors
    const safeSubs = typeof finalSubscribers === 'number' ? finalSubscribers : 0;
    const safeHits = typeof finalHitCount === 'number' ? finalHitCount : 0;
    
    // Check if scoreEntry supports hitCount (migration)
    scores.push({ score: safeSubs, hitCount: safeHits, date: Date.now() });
    
    // Sort descending and keep top 5
    scores.sort((a, b) => b.score - a.score);
    scores = scores.slice(0, 5);
    
    try {
        localStorage.setItem(key, JSON.stringify(scores));
    } catch (e) {
        console.error("Failed to save score");
    }
    setHighScores(scores);
  };

  const loadScores = () => {
    const key = `rhythm_na_subs_${gameMode}`;
    const existing = localStorage.getItem(key);
    if (existing) {
        try {
            setHighScores(JSON.parse(existing));
        } catch (e) {
            setHighScores([]);
        }
    } else {
        setHighScores([]);
    }
  };

  // --- Audio Cleanup ---
  const cleanupAudio = useCallback(() => {
    if (metronomeInterval.current) {
      clearInterval(metronomeInterval.current);
      metronomeInterval.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
  }, []);

  const stopMicAnalysis = useCallback((keepStream = false) => {
    if (micCheckInterval.current) {
        clearInterval(micCheckInterval.current);
        micCheckInterval.current = null;
    }
    if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
    }
    if (micAudioCtxRef.current && micAudioCtxRef.current.state !== 'closed') {
        micAudioCtxRef.current.close();
        micAudioCtxRef.current = null;
    }
    if (!keepStream && streamRef.current) {
         streamRef.current.getTracks().forEach(t => t.stop());
         streamRef.current = null;
    }
    setInputLevel(0);
    setIsMicTesting(false);
  }, []);

  // --- Audio Volume Helper ---
  const calculateVolume = (analyser: AnalyserNode, dataArray: Uint8Array) => {
    analyser.getByteFrequencyData(dataArray);
    let sum = 0;
    const binsToCheck = dataArray.length / 2;
    for(let i = 0; i < binsToCheck; i++) {
        sum += dataArray[i];
    }
    const average = sum / binsToCheck;
    return Math.min(100, (average / 128) * 100 * 2.0); 
  };

  // --- Mic Test Logic ---
  const toggleMicTest = async () => {
    if (isMicTesting) {
        stopMicAnalysis();
        return;
    }

    try {
        const stream = await getAudioStream();
        
        // If this was the first time granting permission, labels might now be available
        fetchAudioDevices();

        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        micAudioCtxRef.current = audioCtx;
        
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.5;
        source.connect(analyser);
        micAnalyserRef.current = analyser;
        
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        
        micCheckInterval.current = window.setInterval(() => {
            const normalized = calculateVolume(analyser, dataArray);
            setInputLevel(normalized);
        }, 50);
        
        setIsMicTesting(true);
        setError(null);
    } catch (err) {
        console.error("Mic access denied"); // Don't log event object
        setError("マイクにアクセスできませんでした。");
    }
  };

  // --- Game Audio Logic (Stricter Pulse Detection) ---
  const triggerHit = () => {
    if (!isGameActiveRef.current) return; // Strict check

    // Trigger Reactive Animation
    setIsHitAnimating(true);
    setTimeout(() => setIsHitAnimating(false), 100);

    // Increment Hits
    setScore(prev => prev + 1); 

    // Calculate Subscriber Gain based on Inflation Logic
    const elapsedTime = (Date.now() - startTimeRef.current) / 1000;
    let gain = 100;

    if (elapsedTime > 85) {
        gain = 1000000;
    } else if (elapsedTime > 60) {
        gain = 500000;
    } else if (elapsedTime > 30) {
        gain = 50000;
    } else if (elapsedTime > 10) {
        gain = 500;
    } else {
        gain = 100;
    }
    
    setSubscribers(prev => prev + gain);

    // Falling Na Effect
    const newNa: FallingNa = {
        id: Date.now(),
        left: `${10 + Math.random() * 80}%`,
        size: `${3 + Math.random() * 5}rem`,
        duration: `${2 + Math.random() * 2}s`,
    };
    setFallingNas(prev => [...prev, newNa]);
  };

  const startGameAudioAnalysis = async () => {
    try {
        const stream = await getAudioStream();

        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        micAudioCtxRef.current = audioCtx;
        
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        // Low smoothing to detect sharp attacks immediately
        analyser.smoothingTimeConstant = 0.2; 
        source.connect(analyser);
        
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        
        // Reset logic state
        isArmedRef.current = true;
        currentNotePeakRef.current = 0;
        lastAudioVolume.current = 0;
        
        // Only set start time if it's 0 (fresh start), otherwise we are resuming
        if (startTimeRef.current === 0) startTimeRef.current = Date.now(); 
        
        const loop = () => {
            if (!isGameActiveRef.current) return; // Stop loop if game ended

            const vol = calculateVolume(analyser, dataArray);
            
            // --- Stricter Scoring Algorithm ---
            // 1. Attack Detection: Significant volume jump + above threshold
            const SENSITIVITY = 15; // Requires a sharp increase
            const isSharpRise = (vol - lastAudioVolume.current) > SENSITIVITY;
            
            if (vol > micThreshold) {
                // We are in active volume range
                
                // Only trigger if we are "Armed" (meaning we previously went quiet/dropped)
                if (isArmedRef.current && isSharpRise) {
                    // --- Rhythm Check (Updated for 8th Notes) ---
                    const now = Date.now();
                    const timeSinceLastBeat = now - lastBeatTimeRef.current; // 0 to Interval
                    
                    // Normalize position in the beat (0.0 to 1.0)
                    const phase = (timeSinceLastBeat % BEAT_INTERVAL_MS) / BEAT_INTERVAL_MS;
                    
                    // Check distance to Quarter Note (0.0 or 1.0)
                    const distToQuarter = Math.min(phase, 1.0 - phase) * BEAT_INTERVAL_MS;
                    
                    // Check distance to Eighth Note (0.5)
                    const distToEighth = Math.abs(phase - 0.5) * BEAT_INTERVAL_MS;
                    
                    // Hit is valid if close to Quarter OR Eighth
                    if (distToQuarter < HIT_WINDOW_MS || distToEighth < HIT_WINDOW_MS) {
                        triggerHit();
                        isArmedRef.current = false; // Disarm until volume drops
                        currentNotePeakRef.current = vol; // Start tracking peak of this new note
                    }
                }

                // If not armed (currently sustaining a note), update peak tracking
                if (!isArmedRef.current) {
                    if (vol > currentNotePeakRef.current) {
                        currentNotePeakRef.current = vol;
                    }
                    
                    // RE-ARM Condition:
                    // Volume must drop significantly from the *peak* of the current note.
                    if (vol < currentNotePeakRef.current * 0.7) {
                        isArmedRef.current = true;
                        currentNotePeakRef.current = 0;
                    }
                }
            } else {
                // If we drop below absolute threshold, always re-arm
                isArmedRef.current = true;
                currentNotePeakRef.current = 0;
            }
            
            lastAudioVolume.current = vol;
            animationFrameRef.current = requestAnimationFrame(loop);
        };
        loop();

    } catch (err) {
        console.error("Game Audio Init Error"); // Don't log full error
        setError("マイクの開始に失敗しました。");
    }
  };


  // --- Game State Management ---
  const resetGame = useCallback(() => {
    setScore(0);
    setSubscribers(0);
    // Use the selected mode duration
    setTimeLeft(MODE_DURATIONS[gameMode]);
    setCountdown(COUNTDOWN_SECONDS);
    setFallingNas([]);
    setError(null);
    lastTranscription.current = '';
    isGameActiveRef.current = false; // Ensure inactive
    lastBeatTimeRef.current = 0;
    startTimeRef.current = 0;
    pauseStartTimeRef.current = 0;
    
    if (gameTimer.current) clearInterval(gameTimer.current);
    cleanupAudio();
    stopMicAnalysis(false); // Full stop including stream
    if(sessionPromise.current) {
        sessionPromise.current.then(session => session.close());
        sessionPromise.current = null;
    }
    
    // Stop BGM if playing
    if (bgmRef.current) {
        bgmRef.current.pause();
        bgmRef.current.currentTime = 0;
    }
  }, [cleanupAudio, stopMicAnalysis, gameMode]);
  
  const handleTranscription = useCallback((message: LiveServerMessage) => {
    if (message.serverContent?.turnComplete) {
      lastTranscription.current = '';
    }
  }, []);

  const connectToGemini = useCallback(async () => {
    try {
        const stream = await getAudioStream();
        
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

        sessionPromise.current = ai.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-09-2025',
            callbacks: {
                onopen: () => {
                    const inputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
                    const source = inputAudioContext.createMediaStreamSource(stream);
                    const scriptProcessor = inputAudioContext.createScriptProcessor(4096, 1, 1);
                    
                    scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
                        const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                        const l = inputData.length;
                        const int16 = new Int16Array(l);
                        for (let i = 0; i < l; i++) {
                            int16[i] = inputData[i] * 32768;
                        }
                        const pcmBlob = {
                            data: encode(new Uint8Array(int16.buffer)),
                            mimeType: 'audio/pcm;rate=16000',
                        };
                        
                        if (sessionPromise.current) {
                            sessionPromise.current.then((session) => {
                                session.sendRealtimeInput({ media: pcmBlob });
                            });
                        }
                    };
                    source.connect(scriptProcessor);
                    scriptProcessor.connect(inputAudioContext.destination);
                },
                onmessage: handleTranscription,
                onerror: (e: ErrorEvent) => {
                    console.error('Gemini API Error'); // Prevent circular log
                },
                onclose: () => {
                    console.log('Gemini session closed.');
                },
            },
            config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: {
                    voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }
                },
                inputAudioTranscription: {},
                systemInstruction: "You are a referee for a rhythm game. Just listen. Do not speak.",
            },
        });
        await sessionPromise.current;

    } catch (err) {
        console.error("Failed to initialize Gemini session"); // Prevent circular log
        setError("AIサーバーへの接続に失敗しました。ネット環境を確認するか、時間をおいて試してください。");
    }
  }, [handleTranscription]);
  
  const startAudio = () => {
    // Only create context if not exists or closed
    if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    } else if (audioContextRef.current.state === 'suspended') {
        audioContextRef.current.resume();
    }
    
    const playBeat = () => {
        if (!audioContextRef.current) return;
        const oscillator = audioContextRef.current.createOscillator();
        const gainNode = audioContextRef.current.createGain();
        oscillator.connect(gainNode);
        gainNode.connect(audioContextRef.current.destination);
        
        oscillator.type = 'sine'; // Soft sine wave for cuter sound
        oscillator.frequency.setValueAtTime(880, audioContextRef.current.currentTime);
        
        const vol = 0.1 * volume; 
        gainNode.gain.setValueAtTime(vol, audioContextRef.current.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.0001, audioContextRef.current.currentTime + 0.1);
        
        oscillator.start(audioContextRef.current.currentTime);
        oscillator.stop(audioContextRef.current.currentTime + 0.1);
        
        // --- Sync Record ---
        lastBeatTimeRef.current = Date.now();
    };
    
    // Play first beat immediately
    playBeat();
    if (metronomeInterval.current) clearInterval(metronomeInterval.current);
    metronomeInterval.current = window.setInterval(playBeat, BEAT_INTERVAL_MS);
  };
  
  useEffect(() => {
    if (gameState === 'COUNTDOWN') {
      if (countdown > 0) {
        const timer = setTimeout(() => setCountdown(c => c - 1), 1000);
        return () => clearTimeout(timer);
      } else {
        setGameState('PLAYING');
      }
    }
  }, [gameState, countdown]);

  useEffect(() => {
    // Menu BGM Logic
    if (gameState === 'READY') {
        const playMusic = () => {
            if (menuBgmRef.current && menuBgmRef.current.paused) {
                menuBgmRef.current.play().catch(() => {});
            }
        };

        if (menuBgmRef.current) {
            menuBgmRef.current.play().catch(() => {
                window.addEventListener('click', playMusic, { once: true });
                window.addEventListener('touchstart', playMusic, { once: true });
                window.addEventListener('keydown', playMusic, { once: true });
            });
        }
        
        return () => {
            window.removeEventListener('click', playMusic);
            window.removeEventListener('touchstart', playMusic);
            window.removeEventListener('keydown', playMusic);
        };
    } else {
        // Stop Menu BGM for all other states
        if (menuBgmRef.current) {
            menuBgmRef.current.pause();
            menuBgmRef.current.currentTime = 0;
        }
    }

    if (gameState === 'PLAYING') {
      isGameActiveRef.current = true;
      
      // If we are resuming from pause, adjust startTime to ignore the paused duration
      if (pauseStartTimeRef.current > 0) {
          const pausedDuration = Date.now() - pauseStartTimeRef.current;
          startTimeRef.current += pausedDuration;
          pauseStartTimeRef.current = 0;
      }
      
      startAudio();
      
      // RESTART BGM (Warmup Complete)
      // Reset time to 0 and restore volume
      if (bgmRef.current) {
          bgmRef.current.currentTime = 0;
          bgmRef.current.volume = volume; // Restore audible volume
          bgmRef.current.play().catch(e => console.warn("BGM play failed", e));
      }

      startGameAudioAnalysis(); 
      
      if (gameTimer.current) clearInterval(gameTimer.current);
      
      gameTimer.current = window.setInterval(() => {
        setTimeLeft(prev => {
          if (prev <= 1) {
            clearInterval(gameTimer.current!);
            isGameActiveRef.current = false; // KILL SWITCH for scoring
            setGameState('FINISHED');
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
      
    } else if (gameState === 'PAUSED') {
       isGameActiveRef.current = false;
       pauseStartTimeRef.current = Date.now();
       
       if (bgmRef.current) {
           bgmRef.current.pause();
       }
       if (gameTimer.current) {
           clearInterval(gameTimer.current);
           gameTimer.current = null;
       }
       if (metronomeInterval.current) {
           clearInterval(metronomeInterval.current);
           metronomeInterval.current = null;
       }
       stopMicAnalysis(true); // keepStream=true
       
    } else if (gameState === 'COUNTDOWN') {
        // KEEP ALIVE: Ensure BGM is playing silently to maintain "User Gesture" token
        if (bgmRef.current) {
            bgmRef.current.volume = 0; // Silence
            if (bgmRef.current.paused) {
                 bgmRef.current.play().catch(e => console.warn("Silent warmup failed", e));
            }
        }
    } else {
      // READY / FINISHED
      isGameActiveRef.current = false;
      // Pause Game BGM and reset
      if (bgmRef.current) {
          bgmRef.current.pause();
          bgmRef.current.currentTime = 0;
      }
      cleanupAudio();
      stopMicAnalysis(false); // stop stream
    }
    
    return () => {
        // Cleanup
    };
  }, [gameState]);

  // Save score when game finishes
  useEffect(() => {
    if (gameState === 'FINISHED') {
        saveScore(subscribers, score);
    }
  }, [gameState, subscribers, score]);

  // Improved Cleanup Logic
  useEffect(() => {
    const interval = setInterval(() => {
        setFallingNas(prev => {
            const now = Date.now();
            const filtered = prev.filter(na => now - na.id < 6000);
            return filtered.length !== prev.length ? filtered : prev;
        });
    }, 500);
    return () => clearInterval(interval);
  }, []);
  
  useEffect(() => {
    return () => {
      cleanupAudio();
      stopMicAnalysis(false);
      if (gameTimer.current) clearInterval(gameTimer.current);
      if(sessionPromise.current) {
          sessionPromise.current.then(session => session.close());
      }
      // Revoke BGM blob if exists
      if (bgmSource.startsWith('blob:')) {
          URL.revokeObjectURL(bgmSource);
      }
    };
  }, [cleanupAudio, stopMicAnalysis, bgmSource]);

  const handleStart = async () => {
    // 1. Reset Game State first (which might clean up old contexts)
    resetGame();

    // 2. UNLOCK BGM: Start playing SILENTLY immediately on click
    if (bgmRef.current) {
        try {
            bgmRef.current.volume = 0;
            bgmRef.current.currentTime = 0;
            await bgmRef.current.play();
            // Do NOT pause here. Keep it playing silently during countdown.
            // We will restart/unmute it when gameState becomes 'PLAYING'.
        } catch(e) { 
            console.warn("BGM Access Error", e);
            setError("BGMの再生に失敗しました。URLを確認してください。");
        }
    }

    // 3. UNLOCK AUDIO CONTEXT
    if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
    }

    // 4. Connect to API and Start Countdown
    connectToGemini(); 
    setGameState('COUNTDOWN');
  };

  const handlePlayAgain = () => {
    resetGame();
    loadScores(); 
    setGameState('READY');
  };

  const handleQuit = () => {
      resetGame();
      setGameState('READY');
  };
  
  const handlePause = () => {
      setGameState('PAUSED');
  };

  const handleResume = () => {
      setGameState('PLAYING');
  };

  const renderContent = () => {
    const threshold = CLEAR_THRESHOLDS[gameMode];
    // Avoid divide by zero if threshold is 0 (shouldn't happen)
    const isCleared = score >= threshold;

    // --- HUD Components to share between PLAYING and PAUSED ---
    const HUD = (
        <div className="absolute top-0 w-full flex flex-col p-4 pointer-events-none z-20 pl-20"> 
            {/* Top Bar: Time and Quota */}
            <div className="flex justify-between items-start w-full gap-4">
                 {/* Time Bar (Left) */}
                <div className="flex-1 max-w-sm">
                     <div className="bg-white/90 backdrop-blur rounded-full p-2 border-4 border-pink-400 shadow-md flex items-center gap-3">
                        <div className="text-pink-500 font-black text-lg px-2">のこり</div>
                        <div className="flex-1 h-6 bg-pink-100 rounded-full overflow-hidden border-2 border-pink-200">
                            <div 
                                className="h-full bg-pink-400 transition-all duration-1000 linear"
                                style={{ width: `${(timeLeft / MODE_DURATIONS[gameMode]) * 100}%` }}
                            ></div>
                        </div>
                        <div className="text-2xl font-black text-slate-700 w-16 text-right">{timeLeft}</div>
                     </div>
                </div>

                 {/* Quota Bar (Right) */}
                <div className="flex-1 max-w-sm">
                    <div className={`bg-white/90 backdrop-blur rounded-full p-2 border-4 shadow-md flex items-center gap-3 ${isCleared ? 'border-yellow-400 animate-pulse ring-4 ring-yellow-200' : 'border-cyan-400'}`}>
                        <div className={`${isCleared ? 'text-yellow-500' : 'text-cyan-500'} font-black text-lg px-2`}>
                            {isCleared ? 'クリア！' : 'ノルマ'}
                        </div>
                        <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden border-2 border-gray-200 relative">
                            <div 
                                className={`h-full transition-all duration-200 ease-out ${isCleared ? 'bg-gradient-to-r from-yellow-300 via-orange-300 to-yellow-300 animate-[scroll-bg_1s_linear_infinite] bg-[length:200%_100%]' : 'bg-cyan-400'}`}
                                style={{ width: `${Math.min(100, (score / threshold) * 100)}%` }}
                            ></div>
                        </div>
                        <div className="text-2xl font-black text-slate-700 w-24 text-right">
                            {score}<span className="text-sm text-gray-400">/{threshold}</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Subscribers Score (Center) */}
            <div className="flex justify-center mt-4">
                <div className="bg-white/90 backdrop-blur-sm rounded-3xl px-8 py-2 border-b-8 border-slate-200 shadow-xl text-center transform hover:scale-105 transition-transform">
                    <div className="text-xs font-bold text-slate-400 tracking-wider mb-1">チャンネル登録者数</div>
                    <div className="text-6xl md:text-7xl font-black text-slate-800 tracking-tighter leading-none flex items-baseline gap-2">
                         {subscribers.toLocaleString()}
                         <span className="text-2xl text-slate-500 font-bold">人</span>
                    </div>
                </div>
            </div>
        </div>
    );

    const BeatIndicator = (
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-10">
            <div 
              className={`w-64 h-64 rounded-full border-8 border-white flex items-center justify-center bg-pink-200/30 backdrop-blur-sm transition-transform duration-75 ${isHitAnimating ? 'scale-110' : 'scale-100'}`}
            >
                <div className={`w-48 h-48 bg-gradient-to-b from-pink-400 to-pink-500 rounded-full flex items-center justify-center shadow-[0_8px_0_rgba(219,39,119,0.2)] border-4 border-white transition-all duration-75 ${isHitAnimating ? 'brightness-110' : ''}`}>
                     <span className="text-6xl font-black text-white drop-shadow-md select-none">な！</span>
                </div>
            </div>
        </div>
    );

    if (gameState === 'PAUSED') {
        return (
             <>
                {/* Background Game UI (Dimmed & Blurred) */}
                <div className="absolute inset-0 pointer-events-none opacity-50 filter blur-sm">
                   {HUD}
                   {BeatIndicator}
                </div>
                
                {/* Pause Menu Overlay */}
                <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
                    <div className="bg-white/90 p-8 rounded-[3rem] border-8 border-yellow-300 shadow-2xl flex flex-col items-center gap-6 animate-[pop_0.3s_cubic-bezier(0.175,0.885,0.32,1.275)] min-w-[320px]">
                        <h2 className="text-4xl font-black text-slate-700 mb-2">一 時 停 止</h2>
                        
                        <button 
                            onClick={handleResume}
                            className="w-64 py-4 text-xl font-bold text-white bg-pink-500 rounded-full border-4 border-white shadow-lg hover:scale-105 active:scale-95 transition-all"
                        >
                            引き続きプレイ
                        </button>
                        
                        <button 
                            onClick={handleStart}
                            className="w-64 py-4 text-xl font-bold text-slate-600 bg-yellow-300 rounded-full border-4 border-white shadow-lg hover:scale-105 active:scale-95 transition-all"
                        >
                            初めから
                        </button>
                        
                        <button 
                            onClick={handleQuit}
                            className="w-64 py-4 text-xl font-bold text-slate-500 bg-gray-200 rounded-full border-4 border-white shadow-lg hover:scale-105 active:scale-95 transition-all"
                        >
                            終了
                        </button>
                    </div>
                </div>
             </>
        );
    }

    switch(gameState) {
      case 'READY':
        return (
          // Scroll container with custom scrollbar
          <div className="w-full h-full overflow-y-auto overflow-x-hidden relative z-10 scrollbar-thin scrollbar-thumb-pink-200 scrollbar-track-transparent">
            {/* Flex container that grows with content but centers if space allows */}
            <div className="min-h-full flex flex-col items-center justify-center p-4 py-8 md:py-4">
            
            {/* Title Section - Reduced margins */}
            <div className="relative mb-6 md:mb-8 flex flex-col items-center text-center z-20">
                {/* Subtitle Top */}
                <div className="mb-2 bg-yellow-300 text-slate-800 px-4 py-1 rounded-full font-black text-xs md:text-sm border-2 md:border-4 border-white shadow-md transform -rotate-2">
                    底辺チューバーからの脱却
                </div>
                
                {/* Main Title Logo Group - Scaled down */}
                <div className="flex flex-col items-center transform -rotate-1 hover:scale-105 transition-transform duration-300 cursor-default py-2">
                    {/* Line 1 - Compacted */}
                    <div className="flex items-end justify-center flex-wrap gap-x-2">
                        <span className="text-6xl sm:text-8xl md:text-[8rem] font-black text-pink-500 leading-none drop-shadow-xl" 
                              style={{ WebkitTextStroke: '3px white', paintOrder: 'stroke fill' }}>
                            名
                        </span>
                        <span className="text-2xl sm:text-4xl md:text-4xl font-black text-slate-800 leading-none"
                              style={{ WebkitTextStroke: '2px white', paintOrder: 'stroke fill' }}>
                            を
                        </span>
                        <span className="text-4xl sm:text-6xl md:text-6xl font-black text-slate-800 leading-none"
                              style={{ WebkitTextStroke: '2.5px white', paintOrder: 'stroke fill' }}>
                            たくさん
                        </span>
                        <span className="text-3xl sm:text-5xl md:text-5xl font-black text-slate-800 leading-none"
                          style={{ WebkitTextStroke: '2px white', paintOrder: 'stroke fill' }}>
                            伝えれる
                        </span>
                    </div>
                    
                    {/* Line 2 */}
                    <div className="flex items-baseline justify-center mt-1">
                        <span className="text-3xl sm:text-5xl md:text-7xl font-black text-slate-800"
                              style={{ WebkitTextStroke: '3px white', paintOrder: 'stroke fill' }}>
                            画期的ツール
                        </span>
                    </div>
                </div>

                {/* Subtitle Bottom */}
                <div className="text-slate-400 text-[10px] md:text-xs font-bold mt-2 tracking-widest bg-white/50 px-3 py-1 rounded-full">
                    的なもの
                </div>
            </div>

            <div className="flex flex-col lg:flex-row gap-4 w-full max-w-4xl items-stretch justify-center">
                {/* LEFT: Start & Mode Selection */}
                <div className="flex-1 flex flex-col gap-4 order-1">
                     {/* Mode Selectors - Compact */}
                     <div className="bg-white/80 backdrop-blur rounded-2xl p-3 shadow-[0_4px_0_rgba(0,0,0,0.1)] flex flex-col gap-2 border-4 border-white">
                         <div className="text-cyan-500 font-bold text-center text-sm mb-1">モードをえらんでね</div>
                         <div className="flex gap-2">
                            {(Object.keys(MODE_DURATIONS) as GameMode[]).map((mode) => (
                                <button
                                    key={mode}
                                    onClick={() => setGameMode(mode)}
                                    className={`flex-1 py-2 font-bold text-xs md:text-base rounded-xl transition-all transform ${
                                        gameMode === mode 
                                        ? 'bg-yellow-300 text-slate-800 shadow-[0_3px_0_#d97706] translate-y-[-1px]' 
                                        : 'bg-gray-100 text-gray-400 hover:bg-gray-200'
                                    }`}
                                >
                                    {mode === 'FULL' ? 'フル (73秒)' : mode === '10s' ? '10秒' : '30秒'}
                                </button>
                            ))}
                         </div>
                     </div>

                     <button 
                        onClick={handleStart} 
                        className="group relative flex-1 w-full focus:outline-none transform transition-all hover:scale-[1.02] active:scale-95 duration-200 min-h-[100px] md:min-h-[160px]"
                    >
                        <div className="absolute inset-0 bg-pink-300 rounded-[2rem] transform translate-y-2"></div>
                        <div className="absolute inset-0 bg-gradient-to-b from-pink-400 to-pink-500 rounded-[2rem] border-4 border-white shadow-inner flex flex-col items-center justify-center overflow-hidden">
                            {/* Star decoration */}
                            <div className="absolute top-[-20%] left-[-10%] w-32 h-32 bg-white/20 rotate-45 transform"></div>
                            <div className="absolute bottom-[-20%] right-[-10%] w-32 h-32 bg-white/20 rotate-12 transform"></div>
                            
                            <span className="text-4xl md:text-5xl font-black text-white drop-shadow-md z-10">始める！</span>
                            <div className="bg-white/30 rounded-full px-6 py-2 mt-4 md:mt-6 z-10">
                                <span className="text-lg md:text-2xl font-bold text-white">
                                    {gameMode === 'FULL' ? '73秒名を伝える' : gameMode === '10s' ? '10秒名を伝える' : '30秒名を伝える'}
                                </span>
                            </div>
                        </div>
                    </button>
                </div>

                {/* RIGHT: Settings Panel */}
                <div className="flex-1 order-2">
                     <div className="h-full bg-sky-200/90 backdrop-blur rounded-[2rem] border-4 md:border-8 border-white p-4 text-slate-700 flex flex-col justify-center gap-4 shadow-lg relative overflow-hidden">
                         <div className="absolute top-[-20px] left-[-20px] w-20 h-20 bg-sky-300 rounded-full opacity-50"></div>
                         
                         <div className="bg-white text-sky-500 font-black px-4 py-1 rounded-full inline-block self-center mb-2 shadow-sm text-lg border-2 border-sky-100">
                             ⚙️ せってい
                         </div>

                         {/* Volume Setting */}
                         <div className="bg-white/50 p-3 rounded-xl">
                             <div className="flex justify-between mb-1 font-bold text-sky-600 text-sm">
                                 <span>🔊 おんりょう</span>
                                 <span>{Math.round(volume * 100)}%</span>
                             </div>
                             <input 
                                type="range" 
                                min="0" max="1" step="0.1" 
                                value={volume}
                                onChange={(e) => setVolume(parseFloat(e.target.value))}
                                className="w-full accent-yellow-400"
                             />
                         </div>

                          {/* BGM Selection */}
                         <div className="bg-white/50 p-3 rounded-xl">
                             <div className="font-bold text-sky-600 mb-1 text-sm">📁 BGMをえらぶ</div>
                             <input 
                                 type="file" 
                                 accept="audio/*"
                                 onChange={handleBgmSelect}
                                 className="w-full text-xs text-slate-500 file:mr-2 file:py-1 file:px-2 file:rounded-full file:border-0 file:text-xs file:font-semibold file:bg-sky-50 file:text-sky-700 hover:file:bg-sky-100 cursor-pointer mb-2"
                             />
                             <input 
                                 type="text"
                                 placeholder="またはURLを入力 (GitHub Rawなど)"
                                 value={customUrl}
                                 onChange={handleUrlChange}
                                 className="w-full text-xs p-2 rounded-lg border-2 border-sky-200 text-slate-600 bg-white focus:outline-none focus:border-sky-400"
                             />
                             <div className="text-[10px] text-slate-400 mt-1 leading-tight">※Github等のURLを直接指定できます</div>
                         </div>

                         {/* Mic Test Section */}
                         <div className="bg-white/60 p-3 rounded-xl flex-1 flex flex-col justify-center border-2 border-white gap-2">
                             <div className="flex justify-between items-center">
                                 <span className="font-bold text-sky-600 text-sm">🎤 マイクテスト</span>
                                 <button 
                                    onClick={toggleMicTest}
                                    className={`px-3 py-1 text-xs font-bold rounded-full border-2 transition-colors ${isMicTesting ? 'bg-green-400 text-white border-green-200' : 'bg-white text-gray-400 border-gray-200'}`}
                                 >
                                     {isMicTesting ? 'ON' : 'OFF'}
                                 </button>
                             </div>

                             {/* Mic Selector */}
                             <div>
                                 <select 
                                     className="w-full text-xs p-2 rounded-lg border-2 border-sky-200 text-slate-600 bg-white focus:outline-none focus:border-sky-400 max-w-full truncate"
                                     value={selectedDeviceId}
                                     onChange={(e) => setSelectedDeviceId(e.target.value)}
                                 >
                                     {audioDevices.length === 0 && <option value="">マイクをさがしています...</option>}
                                     {audioDevices.map(device => (
                                         <option key={device.deviceId} value={device.deviceId}>
                                             {device.label || `マイク ${device.deviceId.slice(0, 5)}...`}
                                         </option>
                                     ))}
                                 </select>
                             </div>

                             {/* Visualizer & Threshold */}
                             <div className="relative h-8 bg-gray-200 rounded-full border-4 border-white overflow-hidden shadow-inner">
                                 <div 
                                    className="absolute top-0 bottom-0 w-1 bg-yellow-400 z-20"
                                    style={{ left: `${micThreshold}%` }}
                                 ></div>
                                 <div 
                                    className={`absolute top-0 left-0 bottom-0 transition-all duration-100 ease-out rounded-l-full ${inputLevel > micThreshold ? 'bg-green-400' : 'bg-sky-400'}`}
                                    style={{ width: `${inputLevel}%` }}
                                 ></div>
                             </div>

                             <div className="flex items-center gap-2">
                                <span className="text-[10px] text-slate-500 font-bold whitespace-nowrap">はんのうライン</span>
                                <input 
                                    type="range" 
                                    min="0" max="100" 
                                    value={micThreshold}
                                    onChange={(e) => setMicThreshold(parseInt(e.target.value))}
                                    className="flex-1 h-2 opacity-80"
                                />
                             </div>
                         </div>
                     </div>
                </div>
            </div>
            
            {error && <div className="mt-4 bg-red-400 text-white font-bold p-3 rounded-xl border-4 border-white animate-bounce shadow-lg max-w-lg text-center text-sm break-all">{error}</div>}
            
            </div>
          </div>
        );
      case 'COUNTDOWN':
        return (
          <div className="flex items-center justify-center h-full">
            <div className="text-[12rem] md:text-[16rem] font-black text-white drop-shadow-[0_8px_0_rgba(236,72,153,1)] animate-impact" style={{WebkitTextStroke: '4px #ec4899'}}>
              {countdown > 0 ? countdown : 'スタート!'}
            </div>
          </div>
        );
      case 'PLAYING':
        return (
          <>
            {/* Pause Button (Replaces Abort) */}
            <button 
                onClick={handlePause}
                className="absolute top-4 left-4 z-50 bg-white/90 text-slate-500 hover:bg-yellow-300 hover:text-white border-4 border-slate-200 hover:border-white rounded-full p-3 font-bold shadow-md transition-all active:scale-95"
                aria-label="一時停止"
            >
                <span className="block w-8 h-8 flex items-center justify-center">
                    {/* Pause Icon */}
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                        <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
                    </svg>
                </span>
            </button>

            {HUD}
            {BeatIndicator}
          </>
        );
      case 'FINISHED':
        return (
          <div className="text-center z-20 flex flex-col items-center w-full max-w-4xl px-4 relative">
            {/* Clear Message with Explosion Effect */}
            {isCleared && (
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-full z-0 pointer-events-none">
                     {/* Explosion Shape */}
                     <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-to-r from-red-500 to-orange-500 rounded-full opacity-20 animate-pulse blur-3xl"></div>
                </div>
            )}
            
            <h2 className="text-7xl md:text-9xl font-black text-pink-500 drop-shadow-[4px_4px_0_#fff] mb-8 animate-impact tracking-tight z-10">
                おしまい！
            </h2>
            
            {isCleared && (
                 <div className="mb-8 z-10 animate-bounce">
                     <p className="text-xl md:text-2xl font-black text-red-500 bg-yellow-200 px-6 py-2 rounded-full border-4 border-white shadow-lg">
                        （あなたの名を伝えすぎて、世界が爆発しました）
                     </p>
                 </div>
            )}
            
            <div className="flex flex-col md:flex-row gap-6 items-center w-full justify-center z-10">
                {/* Score Card */}
                <div className="bg-white/80 backdrop-blur-md p-8 rounded-[3rem] border-8 border-yellow-300 shadow-xl flex flex-col items-center transform rotate-[-2deg] min-w-[300px]">
                    <p className="text-xl text-yellow-500 font-bold mb-2">チャンネル登録者数</p>
                    <p className="text-6xl font-black text-slate-700">{subscribers.toLocaleString()} <span className="text-2xl">人</span></p>
                    <div className="mt-4 bg-gray-100 rounded-xl px-4 py-2 w-full flex justify-between">
                         <span className="font-bold text-gray-500">伝えた回数</span>
                         <span className="font-black text-slate-700">{score}回</span>
                    </div>
                    {isCleared && <p className="text-red-500 font-black mt-4 text-3xl animate-pulse">CLEAR!</p>}
                </div>

                {/* Ranking Board */}
                <div className="bg-sky-100/90 p-6 rounded-[2rem] border-4 border-white shadow-lg w-full max-w-sm transform rotate-[2deg]">
                    <div className="bg-sky-400 text-white text-xl font-black py-2 px-6 rounded-full inline-block mb-4 shadow-sm">
                        ランキング (人数)
                    </div>
                    <ul className="space-y-2">
                        {highScores.map((entry, i) => (
                            <li key={i} className={`flex justify-between items-center px-4 py-2 rounded-xl ${i === 0 ? 'bg-yellow-100 text-yellow-600 font-black border-2 border-yellow-200' : 'bg-white/50 text-slate-600 font-bold'}`}>
                                <div className="flex flex-col">
                                    <div className="flex items-center gap-2">
                                        <span>#{i + 1}</span>
                                        <span className="text-lg">{entry.score.toLocaleString()}人</span>
                                    </div>
                                    <span className="text-xs text-slate-400 ml-6">({entry.hitCount || 0}回)</span>
                                </div>
                            </li>
                        ))}
                        {highScores.length === 0 && <li className="text-gray-400 text-center py-4">まだきろくがないよ</li>}
                    </ul>
                </div>
            </div>
            
            <button 
              onClick={handlePlayAgain}
              className="mt-12 group relative inline-block focus:outline-none transform transition-transform hover:scale-105 active:scale-95 z-10"
            >
              <span className="absolute inset-0 translate-y-2 bg-pink-300 rounded-full"></span>
              <span className="relative inline-block border-4 border-white px-12 py-4 text-2xl font-black text-white bg-pink-500 rounded-full shadow-lg group-hover:-translate-y-1 transition-transform">
                もういちどあそぶ
              </span>
            </button>
          </div>
        );
    }
  };

  return (
    <div className="w-full h-screen flex items-center justify-center relative overflow-hidden bg-polka font-sans selection:bg-pink-200 selection:text-pink-900">
        <audio 
            ref={bgmRef} 
            loop 
            preload="auto"
            // ADDED playsInline for better mobile compatibility
            playsInline
            src={bgmSource}
            onCanPlay={() => {
                setError(null); 
            }}
            onError={(e) => {
                console.warn("BGM Load Error:", bgmSource);
                setError("BGMの読み込みに失敗しました: " + bgmSource);
            }}
        />
        {/* Menu BGM */}
        <audio 
            ref={menuBgmRef} 
            loop 
            preload="auto"
            // ADDED playsInline
            playsInline
            src={menuBgmUrl}
            onError={() => console.warn("Menu BGM not found")}
        />
        <Crowd progress={Math.min(score / CLEAR_THRESHOLDS[gameMode], 1)} />
        {fallingNas.map(na => <FallingNaComponent key={na.id} na={na} />)}
        {renderContent()}
    </div>
  );
};

export default Game;