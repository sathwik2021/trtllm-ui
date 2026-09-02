import React, { useEffect, useRef, useState } from "react";
import {
  Github,
  Mail,
  Phone,
  Linkedin,
  ArrowUpRight,
  Menu,
  X,
  GraduationCap,
  Award,
  Trophy,
  BadgeCheck,
  Code,
  Sparkles,
  Cpu,
  Layers,
  ChevronRight,
  ExternalLink,
  Download,
  FileText,
  Terminal,
} from "lucide-react";

/* ---------------------------------------------------------------------- */
/*  DATA — pulled from resume + github.com/sathwik2021                    */
/* ---------------------------------------------------------------------- */

const PROFILE = {
  name: "Peechara Sathwik",
  title: "Python Developer — AI & Data Analytics",
  summary:
    "I build AI systems that talk to each other. Chatbots that fail over between model providers, pipelines that turn a prompt into a manga panel, agents that argue about your code until it's right. NLP, computer vision, and the backend plumbing that keeps it all running.",
  email: "peecharasathwik@gmail.com",
  phone: "+91 8019002245",
  github: "https://github.com/sathwik2021",
  linkedin: "https://www.linkedin.com/in/peechara-sathwik-90414432b/",
  // Connects directly to raw GitHub URL — updating the file on GitHub auto-updates the download
  resume: "https://raw.githubusercontent.com/sathwik2021/Peechara_Sathwik_Resume/main/Peechara_Sathwik_Resume.pdf",
  localResume: "/Peechara_Sathwik_Resume.pdf",
};

const CHARACTER_PROFILES = [
  {
    id: "sathwik",
    name: "Peechara Sathwik",
    title: "Python Developer — AI & Data Analytics",
    summary: "I build AI systems that talk to each other. Chatbots that fail over between model providers, pipelines that turn a prompt into a manga panel, agents that argue about your code until it's right. NLP, computer vision, and the backend plumbing that keeps it all running.",
    image: "/sathwik_cutout.png",
    stats: [
      { label: "PRIMARY STACK", value: "Python, Flask, PyMuPDF, SQLAlchemy" },
      { label: "AI PIPELINE", value: "Groq + Gemini Failover, SSE Streaming" },
      { label: "FINE-TUNED LORA", value: "15,500 Steps on Kaggle Dual-T4 GPUs" },
      { label: "STATUS", value: "Available for Internships & Full-Time" },
    ],
    bgText: ["PEECHARA", "SATHWIK"],
  },
  {
    id: "nexus",
    name: "NexusAI Bot",
    title: "Multimodal AI Chatbot & Provider Failover",
    summary: "A resilient full-stack chatbot featuring automatic fallback between Groq, Gemini, and Ollama APIs. Streams response tokens via Server-Sent Events (SSE) and ingests PDF documents using PyMuPDF.",
    image: "/logo.png",
    stats: [
      { label: "FAILOVER ENGINE", value: "Groq <-> Gemini <-> Ollama" },
      { label: "STREAMING", value: "Server-Sent Events (SSE)" },
      { label: "DATABASE", value: "Normalized MySQL with SQLAlchemy" },
      { label: "ADVERSARIAL", value: "Stress-tested against model downtime" },
    ],
    bgText: ["NEXUS", "AI BOT"],
  },
  {
    id: "manga",
    name: "Manga Pipeline",
    title: "3-Stage GPU Text-to-Manga Generator",
    summary: "A multi-stage GPU system: Gemini extracts narrative beats from story text, a 20-rule layout engine composes page panels, and a LoRA fine-tuned Stable Diffusion 1.5 renders each panel.",
    image: "/logo.png",
    stats: [
      { label: "DATASET", value: "Manga109 (15,500 Training Steps)" },
      { label: "HARDWARE", value: "Kaggle Dual-T4 GPUs" },
      { label: "LAYOUT ENGINE", value: "20-Rule Composition Algorithm" },
      { label: "FRAMEWORK", value: "PyTorch, Diffusers, OpenCV" },
    ],
    bgText: ["MANGA", "LORA SD"],
  },
  {
    id: "codex",
    name: "Codex Council",
    title: "Multi-Agent Code Review & Consensus Engine",
    summary: "An autonomous multi-agent system where specialized AI agents debate, critique, and refactor code until reaching consensus on correctness, security, and performance.",
    image: "/logo.png",
    stats: [
      { label: "ENGINE", value: "Multi-Agent Consensus Loop" },
      { label: "CRITIQUE", value: "Automated Code & Security Review" },
      { label: "ORCHESTRATION", value: "Python & Agent Workflows" },
      { label: "STATUS", value: "Active Autonomous Project" },
    ],
    bgText: ["CODEX", "COUNCIL"],
  },
  {
    id: "trtllm",
    name: "TensorRT-LLM Control Panel",
    title: "Localhost GPU Inference Orchestrator",
    summary: "A FastAPI control panel for running TensorRT-LLM locally: model discovery, Docker deployment lifecycle, GPU polling, and command generation — no cloud, no auth surface exposed.",
    image: "/logo.png",
    stats: [
      { label: "BACKEND", value: "FastAPI, Uvicorn" },
      { label: "DEPLOYMENT", value: "Docker Lifecycle Management" },
      { label: "SCOPE", value: "Diagnostics, GPU Polling, Chat/Test UI" },
      { label: "SECURITY", value: "Localhost-Only, No Auth Exposed" },
    ],
    bgText: ["TRT", "LLM"],
  },
];

const ENGINEERING_PILLARS = [
  {
    title: "LLM Routing & Failover",
    desc: "Multi-provider fallback chains (Groq + Gemini + Ollama) with SSE response streaming.",
    icon: Cpu,
    tag: "NLP & Multi-Agent",
  },
  {
    title: "Fine-Tuned Diffusion",
    desc: "15,500 step LoRA model training on Kaggle dual-T4 GPUs for text-to-manga pipelines.",
    icon: Sparkles,
    tag: "Generative AI",
  },
  {
    title: "Multi-Video Tracking",
    desc: "Target tracking across continuous video feeds using YOLOv8 & OpenCV CSRT trackers.",
    icon: Terminal,
    tag: "Computer Vision",
  },
  {
    title: "Backend Data Plumbing",
    desc: "Normalized MySQL schemas with SQLAlchemy & stress-tested API failover mechanisms.",
    icon: Layers,
    tag: "Database & API",
  },
];

const NAV = ["Work", "Systems", "Education", "Contact"];

const PROJECTS = [
  {
    id: "nexus",
    name: "NexusAI",
    category: "AI & NLP",
    tagline: "AI chatbot with NLP & multimodal input",
    years: "2025–2026",
    featured: true,
    stages: ["Multimodal Input", "Provider Failover", "NLP Logging"],
    description:
      "A full-stack chatbot that pairs Groq and Gemini with automatic provider failover, streams responses over SSE, and accepts voice, image, and file uploads through PyMuPDF.",
    detail:
      "Every conversation and its NLP metadata lands in a normalized MySQL schema via SQLAlchemy. The pipeline was stress-tested against adversarial and malformed input on purpose — it needed to stay up when a provider didn't.",
    stack: ["Python", "Flask", "Groq API", "Gemini API", "MySQL", "SQLAlchemy", "PyMuPDF"],
    link: null,
  },
  {
    id: "manga",
    name: "AI Manga Generation Pipeline",
    category: "AI & NLP",
    tagline: "Text-to-manga system",
    years: "2025–2026",
    featured: true,
    stages: ["Narrative Beats", "Layout Engine", "LoRA Diffusion"],
    description:
      "A 3-stage GPU pipeline: Gemini extracts narrative beats from a story, a 20-rule layout engine composes the page, and a LoRA fine-tuned Stable Diffusion 1.5 model renders each panel.",
    detail:
      "The LoRA was trained on Manga109 across 15,500 steps on a Kaggle dual-T4 setup — including tracking down a symlink cache corruption bug that only showed up mid-run.",
    stack: ["Python", "Flask", "Stable Diffusion 1.5", "LoRA", "Gemini API", "OpenCV", "Pillow", "Hugging Face"],
    link: null,
  },
  {
    id: "codex",
    name: "Codex Council",
    category: "AI & NLP",
    tagline: "Multi-agent LLM code review pipeline",
    years: "2026",
    featured: true,
    stages: ["Generator", "Critic", "Verifier"],
    description:
      "Routes codebase queries to specialist teams — security, architecture, bugs, optimization — through a Generator → Critic → Verifier debate loop, so no single model gets the final word.",
    detail:
      "Backed by a 6-provider LLM fallback chain with automatic key rotation. Along the way: a JSON-escaping failure and a provider-naming collision that was silently disabling an entire team without erroring.",
    stack: ["Python", "Ollama", "Groq", "Gemini", "Mistral", "OpenRouter"],
    link: null,
  },
  {
    id: "cctv",
    name: "Smart CCTV Tracker",
    category: "Computer Vision",
    tagline: "Multi-video object tracker",
    years: "2026",
    featured: false,
    stages: ["YOLOv8 Detect", "CSRT Track", "Recovery"],
    description:
      "A continuous object tracker that follows a target across multiple video files, not just one clip — built for surveillance, wildlife footage, and sports analysis.",
    detail:
      "Combines YOLOv8 detection with OpenCV's CSRT/KCF/MOSSE trackers, and falls back to YOLO plus histogram matching when a target is lost. Tracking state is saved and resumed automatically between runs.",
    stack: ["Python", "OpenCV", "YOLOv8", "Ultralytics", "NumPy"],
    link: null,
  },
  {
    id: "trtllm-ui",
    name: "TensorRT-LLM Control Panel",
    category: "Infra & Tooling",
    tagline: "Localhost orchestrator for TensorRT-LLM",
    years: "2026",
    featured: false,
    stages: ["Model Discovery", "Docker Deploy", "GPU Monitor"],
    description:
      "A FastAPI-based localhost control panel for running TensorRT-LLM: model discovery, command generation, Docker deployment lifecycle, GPU polling, and a basic chat/test UI.",
    detail:
      "Deliberately scoped to v1 — no auth, no HF downloads, no engine building or quantization. Docker publishes ports to 127.0.0.1 only while TensorRT-LLM binds 0.0.0.0 inside the container, kept that way on purpose for the container network.",
    stack: ["Python", "FastAPI", "Docker", "TensorRT-LLM", "Uvicorn"],
    link: "https://github.com/sathwik2021/trtllm-ui",
  },
];

const SKILL_GROUPS = [
  { label: "Languages", items: ["Python", "Java"] },
  {
    label: "AI / ML",
    items: [
      "NLP",
      "Transformers",
      "Computer Vision",
      "Machine Learning",
      "Stable Diffusion",
      "LoRA Fine-Tuning",
      "Prompt Engineering",
    ],
  },
  { label: "Libraries", items: ["NumPy", "Pandas", "OpenCV", "Pillow", "Diffusers", "HF Transformers"] },
  { label: "Data & Cloud", items: ["SQL (MySQL)", "Google Cloud Fundamentals"] },
  { label: "Tools", items: ["Git", "GitHub", "Kaggle GPU Training"] },
];

const MARQUEE_TECH = [
  "Python", "Flask", "PyTorch", "Stable Diffusion", "YOLOv8", "OpenCV", 
  "Groq API", "Gemini API", "MySQL", "SQLAlchemy", "LoRA", "Hugging Face", "Ollama", "Git"
];

const EDUCATION = [
  {
    school: "Malla Reddy Engineering College, Hyderabad",
    degree: "B.Tech, Computer Science (Data Science)",
    period: "2023–2027",
    metric: "CGPA 7.47 / 10",
    description: "Specializing in Data Science, focusing on Machine Learning, AI system architecture, NLP, and database management.",
  },
  {
    school: "Alphores Junior College, Karimnagar",
    degree: "Senior Secondary (XII) — TSBIE, Science",
    period: "2021–2023",
    metric: "70.4%",
    description: "Coursework focused on Mathematics, Physics, and Computer Fundamentals.",
  },
  {
    school: "Alphores High School, Karimnagar",
    degree: "Secondary (X) — TSBSE",
    period: "2021",
    metric: "CGPA 10.00 / 10",
    description: "Graduated with perfect GPA, building strong mathematical and algorithmic foundations.",
  },
];

const CERTIFICATIONS = [
  { name: "Azure AI Fundamentals (AI-900)", org: "Microsoft", type: "Badge" },
  { name: "Data Analytics Job Simulation", org: "Deloitte (Forage)", type: "Certificate" },
  { name: "Machine Learning Certification", org: "Tutedude, Nov 2025", type: "Certificate" },
  { name: "Internship Certificate", org: "AimR Edu, 2024", type: "Certificate" },
  { name: "Data Structures", org: "TechAugusta, Oct 2024", type: "Certificate" },
  { name: "Google Cloud Computing Foundations", org: "Google, Apr 2026", type: "Badge" },
];

/* ---------------------------------------------------------------------- */
/*  UTIL — scroll reveal                                                  */
/* ---------------------------------------------------------------------- */

function useReveal() {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          io.unobserve(el);
        }
      },
      { threshold: 0.15 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return [ref, visible];
}

function Reveal({ children, className = "", delay = 0 }) {
  const [ref, visible] = useReveal();
  return (
    <div
      ref={ref}
      className={`pf-reveal ${visible ? "pf-reveal-in" : ""} ${className}`}
      style={{ transitionDelay: visible ? `${delay}ms` : "0ms" }}
    >
      {children}
    </div>
  );
}

/* ---------------------------------------------------------------------- */
/*  PIPELINE — the signature element                                      */
/* ---------------------------------------------------------------------- */

function Pipeline({ stages, size = "md" }) {
  return (
    <div className={`pf-pipeline pf-pipeline-${size}`}>
      {stages.map((stage, i) => (
        <React.Fragment key={stage}>
          <div className="pf-node" style={{ animationDelay: `${i * 0.5}s` }}>
            <span className="pf-node-dot" style={{ animationDelay: `${i * 0.5}s` }} />
            <span className="pf-node-label">{stage}</span>
          </div>
          {i < stages.length - 1 && (
            <div className="pf-edge" style={{ animationDelay: `${i * 0.5 + 0.25}s` }} />
          )}
        </React.Fragment>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------------- */
/*  MAIN                                                                   */
/* ---------------------------------------------------------------------- */

export default function Portfolio() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeCategory, setActiveCategory] = useState("All");
  const [activeSkillCategory, setActiveSkillCategory] = useState("All");
  const [activeCharaId, setActiveCharaId] = useState("sathwik");
  const [pageLoaded, setPageLoaded] = useState(false);
  const [loaderDone, setLoaderDone] = useState(false);

  const currentChara = CHARACTER_PROFILES.find((c) => c.id === activeCharaId) || CHARACTER_PROFILES[0];
  const [charaImg, setCharaImg] = useState(currentChara.image);

  useEffect(() => {
    const c = CHARACTER_PROFILES.find((item) => item.id === activeCharaId);
    if (c) setCharaImg(c.image);
  }, [activeCharaId]);

  // Fast stylish Logo Preloader sequence
  useEffect(() => {
    const t1 = setTimeout(() => setPageLoaded(true), 500); // Preloader starts fade out at 0.5s
    const t2 = setTimeout(() => setLoaderDone(true), 800);  // Preloader finishes & unmounts at 0.8s
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, []);

  const scrollTo = (id) => {
    setMenuOpen(false);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  const filteredProjects = PROJECTS.filter((p) => {
    if (activeCategory === "All") return true;
    return p.category === activeCategory;
  });

  const filteredSkillGroups = SKILL_GROUPS.filter((g) => {
    if (activeSkillCategory === "All") return true;
    return g.label === activeSkillCategory;
  });

  return (
    <div className="pf-root">

      {/* CLEAN LOGO PRELOADER */}
      {!loaderDone && (
        <div className={`pf-logo-loader ${pageLoaded ? "pf-logo-loader--done" : ""}`} aria-hidden="true">
          <img src="/logo.png" alt="S Logo" className="pf-logo-loader-img" />
        </div>
      )}

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body {
          overflow-x: hidden;
          width: 100%;
          max-width: 100vw;
          margin: 0;
          padding: 0;
        }

        .pf-root {
          --bg: #0A0D10;
          --bg-elev: #12161B;
          --bg-elev-2: #171C22;
          --border: #1F262E;
          --border-soft: #171D24;
          --text: #EAEDF0;
          --text-dim: #8792A0;
          --text-faint: #4E5966;
          --accent: #FF8A3D;
          --accent-soft: rgba(255, 138, 61, 0.14);
          --teal: #4FD1C5;
          --teal-soft: rgba(79, 209, 197, 0.14);

          background: var(--bg);
          color: var(--text);
          font-family: 'Inter', sans-serif;
          min-height: 100vh;
          width: 100%;
          max-width: 100vw;
          overflow-x: hidden;
          line-height: 1.5;
          -webkit-font-smoothing: antialiased;
        }
        .pf-root * { box-sizing: border-box; }
        .pf-root ::selection { background: var(--accent-soft); color: var(--accent); }

        .pf-mono { font-family: 'JetBrains Mono', monospace; }
        .pf-display { font-family: 'Space Grotesk', sans-serif; }

        .pf-wrap {
          max-width: 1040px;
          margin: 0 auto;
          padding: 0 24px;
          padding-left: max(24px, env(safe-area-inset-left));
          padding-right: max(24px, env(safe-area-inset-right));
          width: 100%;
          box-sizing: border-box;
        }
        @media (max-width: 768px) {
          .pf-wrap {
            padding: 0 20px;
            padding-left: max(20px, env(safe-area-inset-left));
            padding-right: max(20px, env(safe-area-inset-right));
          }
        }
        @media (max-width: 480px) {
          .pf-wrap {
            padding: 0 16px;
            padding-left: max(16px, env(safe-area-inset-left));
            padding-right: max(16px, env(safe-area-inset-right));
          }
        }

        /* ---------- reveal ---------- */
        .pf-reveal { opacity: 0; transform: translateY(18px); transition: opacity 0.7s ease, transform 0.7s ease; }
        .pf-reveal-in { opacity: 1; transform: translateY(0); }
        @media (prefers-reduced-motion: reduce) {
          .pf-reveal { opacity: 1; transform: none; transition: none; }
        }

        /* ---------- nav ---------- */
        .pf-nav {
          position: sticky; top: 0; z-index: 40;
          background: rgba(10, 13, 16, 0.85);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          border-bottom: 1px solid var(--border-soft);
        }
        .pf-nav-inner {
          display: flex; align-items: center; justify-content: space-between;
          height: 64px;
        }
        .pf-logo {
          display: flex; align-items: center; gap: 10px;
          font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 16px;
          color: var(--text); letter-spacing: 0.01em; cursor: pointer;
        }
        .pf-logo-mark {
          width: 30px; height: 30px; border-radius: 6px;
          transition: transform 0.2s ease, opacity 0.2s ease;
        }
        .pf-logo:hover .pf-logo-mark {
          transform: scale(1.05);
          opacity: 0.95;
        }
        .pf-footer-brand {
          display: flex; align-items: center; gap: 10px; cursor: pointer;
          color: var(--text); font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 14.5px;
          transition: opacity 0.2s ease;
        }
        .pf-footer-brand:hover { opacity: 0.9; }
        .pf-logo-mark-sm {
          width: 26px; height: 26px; border-radius: 6px;
          transition: transform 0.2s ease;
        }
        .pf-footer-brand:hover .pf-logo-mark-sm {
          transform: scale(1.06);
        }
        .pf-nav-links { display: flex; align-items: center; gap: 32px; }
        .pf-nav-link {
          background: none; border: none; cursor: pointer;
          color: var(--text-dim); font-size: 13.5px; font-family: 'JetBrains Mono', monospace;
          letter-spacing: 0.02em; padding: 6px 2px; position: relative;
          transition: color 0.2s ease;
        }
        .pf-nav-link:hover, .pf-nav-link:focus-visible { color: var(--text); }
        .pf-nav-link::after {
          content: ''; position: absolute; left: 0; bottom: 0; height: 1px; width: 0;
          background: var(--accent); transition: width 0.25s ease;
        }
        .pf-nav-link:hover::after, .pf-nav-link:focus-visible::after { width: 100%; }
        .pf-nav-cta {
          border: 1px solid var(--border); background: var(--bg-elev);
          color: var(--text); font-size: 13px; padding: 8px 14px; border-radius: 6px;
          cursor: pointer; font-family: 'Inter', sans-serif; font-weight: 500;
          transition: border-color 0.2s ease, background 0.2s ease;
        }
        .pf-nav-cta:hover, .pf-nav-cta:focus-visible { border-color: var(--accent); background: var(--accent-soft); }
        .pf-burger { display: none; background: none; border: none; color: var(--text); cursor: pointer; }
        .pf-mobile-menu {
          display: none; flex-direction: column; gap: 4px; padding: 8px 28px 20px;
          border-bottom: 1px solid var(--border-soft);
        }
        .pf-mobile-menu.pf-open { display: flex; }
        .pf-mobile-link {
          text-align: left; background: none; border: none; color: var(--text-dim);
          font-family: 'JetBrains Mono', monospace; font-size: 14px; padding: 10px 0; cursor: pointer;
          border-bottom: 1px solid var(--border-soft);
        }
        .pf-mobile-link:hover { color: var(--text); }

        @media (max-width: 720px) {
          .pf-nav-links { display: none; }
          .pf-burger { display: block; }
        }

        @media (max-width: 480px) {
          .pf-hero { padding: 56px 0 48px; }
          .pf-hero-ctas { flex-direction: column; align-items: flex-start; }
          .pf-hero-ctas a { width: 100%; justify-content: center; }
          .pf-contact { padding: 64px 0 40px; }
          .pf-section { padding: 64px 0; }
        }

        /* ---------- CLEAN LOGO PRELOADER ---------- */
        .pf-logo-loader {
          position: fixed; inset: 0; z-index: 9999; background: #0A0D10;
          display: flex; align-items: center; justify-content: center;
          transition: opacity 0.35s ease, transform 0.35s ease;
          pointer-events: all;
        }
        .pf-logo-loader--done {
          opacity: 0; transform: scale(1.04); pointer-events: none;
        }
        .pf-logo-loader-img {
          width: 52px; height: 52px; object-fit: contain;
          animation: pf-logo-intro 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        @keyframes pf-logo-intro {
          0%   { opacity: 0; transform: scale(0.7); }
          100% { opacity: 1; transform: scale(1); }
        }

        /* ---------- SEQUENTIAL HERO ENTRANCE ---------- */
        @keyframes pf-figure-enter {
          0%   { opacity: 0; transform: translate3d(60px, 0, 0) scale(0.96); }
          100% { opacity: 1; transform: translate3d(0, 0, 0) scale(1); }
        }
        @keyframes pf-hero-text-enter {
          0%   { opacity: 0; transform: translate3d(-28px, 0, 0) scale(0.985); }
          60%  { opacity: 1; transform: translate3d(6px, 0, 0) scale(1.002); }
          100% { opacity: 1; transform: translate3d(0, 0, 0) scale(1); }
        }
        @keyframes pf-pillars-up-enter {
          0%   { opacity: 0; transform: translate3d(0, 45px, 0); }
          100% { opacity: 1; transform: translate3d(0, 0, 0); }
        }
        @keyframes pf-watermark-enter {
          0%   { opacity: 0; }
          100% { opacity: 1; }
        }

        /* Initial hidden state before loader completes — ALL text, tabs, pillars, & photo hidden */
        .pf-figure-entrance,
        .pf-hero-entrance,
        .pf-mt-selector,
        .pf-pillars-grid,
        .pf-watermark-entrance {
          opacity: 0;
          will-change: transform, opacity;
        }

        /* Once loader is finished (loaderDone = true): */
        /* 1. PHOTO APPEARS FIRST (0.15s delay — smooth 0.75s slide RIGHT to LEFT) */
        .pf-animate-start .pf-figure-entrance {
          animation: pf-figure-enter 0.75s cubic-bezier(0.16, 1, 0.3, 1) 0.15s forwards;
        }
        /* 2. HERO TEXT & TABS APPEAR SECOND (0.5s delay — smooth 0.75s slide LEFT to RIGHT) */
        .pf-animate-start .pf-hero-entrance,
        .pf-animate-start .pf-mt-selector {
          animation: pf-hero-text-enter 0.9s cubic-bezier(0.22, 1, 0.36, 1) 0.5s forwards;
        }
        /* 3. CORE ENGINEERING PILLARS CARDS APPEAR THIRD (0.8s delay — smooth 0.75s slide BOTTOM to TOP) */
        .pf-animate-start .pf-pillars-grid {
          animation: pf-pillars-up-enter 0.75s cubic-bezier(0.16, 1, 0.3, 1) 0.8s forwards;
        }
        /* 4. WATERMARK FADES IN AT 0.2s */
        .pf-animate-start .pf-watermark-entrance {
          animation: pf-watermark-enter 0.9s ease 0.2s forwards;
        }

        /* ---------- INTERACTIVE TAB SWITCHING FADE IN ---------- */
        .pf-mt-data-anim {
          animation: pf-tab-slide 0.25s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        @keyframes pf-tab-slide {
          0%   { opacity: 0; transform: translate3d(-12px, 0, 0); }
          100% { opacity: 1; transform: translate3d(0, 0, 0); }
        }

        /* ---------- FIGURE WRAP ---------- */
        .pf-mt-figure-wrap {
          position: relative; width: 100%; max-width: 320px;
        }

        /* ---------- MARRIAGE TOXIN ANIME STYLE HERO & CHARACTER SHOWCASE ---------- */
        .pf-mt-hero {
          position: relative; padding: 50px 0 64px; overflow: hidden;
          opacity: 0; visibility: hidden; transition: opacity 0.25s ease, visibility 0.25s ease;
        }
        .pf-animate-start.pf-mt-hero {
          opacity: 1; visibility: visible;
        }
        .pf-mt-bg-watermark {
          position: absolute; top: 10px; right: -10px; z-index: 0; pointer-events: none;
          font-family: 'Space Grotesk', sans-serif; font-weight: 900; font-size: clamp(70px, 13vw, 160px);
          line-height: 0.85; text-transform: uppercase; color: rgba(255, 255, 255, 0.03);
          letter-spacing: -0.03em; user-select: none; text-align: right;
          -webkit-text-stroke: 1.5px rgba(255, 255, 255, 0.05);
        }
        .pf-mt-layout {
          position: relative; z-index: 2; display: grid; grid-template-columns: 1fr 340px; gap: 44px; align-items: start;
        }
        @media (max-width: 960px) {
          .pf-mt-layout { grid-template-columns: 1fr; }
        }

        .pf-mt-selector {
          position: relative; z-index: 3;
          display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 32px;
        }
        .pf-mt-tab {
          background: rgba(18, 22, 27, 0.8); border: 1px solid var(--border);
          color: var(--text-dim); font-family: 'Space Grotesk', sans-serif; font-size: 13px;
          padding: 8px 18px; border-radius: 6px; cursor: pointer;
          transition: all 0.25s ease; display: flex; align-items: center; gap: 6px;
          position: relative;
        }
        .pf-mt-tab::before {
          content: "//";
          color: var(--text-faint);
          margin-right: 4px;
          font-size: 12px;
          letter-spacing: 0.08em;
        }
        .pf-mt-tab:hover { border-color: var(--accent-soft); color: var(--text); }
        .pf-mt-tab.active {
          background: var(--accent); color: #1A0E04; border-color: var(--accent); font-weight: 700;
          box-shadow: 0 0 16px rgba(255, 138, 61, 0.35);
        }

        .pf-mt-kanji {
          font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--teal);
          letter-spacing: 0.16em; font-weight: 600; text-transform: uppercase; margin-bottom: 8px;
          display: flex; align-items: center; gap: 8px;
        }
        .pf-mt-kanji-slash { color: var(--accent); }
        .pf-mt-name {
          font-family: 'Space Grotesk', sans-serif; font-weight: 800;
          font-size: clamp(38px, 5vw, 62px); line-height: 1.02; margin: 0 0 8px; letter-spacing: -0.02em;
        }
        .pf-mt-role {
          font-size: clamp(16px, 2vw, 20px); font-weight: 600; color: var(--accent);
          margin-bottom: 18px; font-family: 'Inter', sans-serif;
        }
        .pf-mt-divider {
          height: 2px; width: 100%; margin: 18px 0 24px;
          background: linear-gradient(90deg, var(--accent), var(--teal), transparent);
          position: relative;
        }
        .pf-mt-divider::after {
          content: ''; position: absolute; left: 0; top: -3px; width: 8px; height: 8px;
          background: var(--accent); transform: rotate(45deg);
        }

        .pf-mt-stats-grid {
          display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-top: 24px; margin-bottom: 12px;
        }
        @media (max-width: 540px) {
          .pf-mt-stats-grid { grid-template-columns: 1fr; }
        }
        .pf-mt-stat-card {
          background: rgba(18, 22, 27, 0.85); border: 1px solid var(--border);
          border-left: 3px solid var(--teal); padding: 12px 16px; border-radius: 6px;
        }
        .pf-mt-stat-label {
          font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--text-faint);
          letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px;
        }
        .pf-mt-stat-value {
          font-size: 13px; font-weight: 600; color: var(--text); font-family: 'Inter', sans-serif;
        }

        .pf-mt-figure-col {
          display: flex; justify-content: center; align-items: center; position: relative; z-index: 2;
        }
        .pf-mt-figure-card {
          position: relative; width: 100%; max-width: 320px; display: flex; justify-content: center; align-items: center;
          background: transparent; border: none; outline: none; box-shadow: none; border-radius: 0; overflow: visible;
        }
        .pf-mt-figure-img {
          width: 100%; max-height: 440px; object-fit: contain; display: block;
          border: none !important; outline: none !important; box-shadow: none !important; border-radius: 0 !important;
          mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 82%, rgba(0,0,0,0) 100%);
          -webkit-mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 82%, rgba(0,0,0,0) 100%);
          filter: drop-shadow(0 14px 28px rgba(0, 0, 0, 0.65));
          transition: transform 0.3s ease;
        }
        .pf-mt-figure-img:hover {
          transform: scale(1.02);
        }
        .pf-eyebrow {
          display: inline-flex; align-items: center; gap: 8px;
          font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--teal);
          text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 22px;
          padding: 6px 14px; border-radius: 20px; background: var(--teal-soft);
          border: 1px solid rgba(79, 209, 197, 0.2);
        }
        .pf-eyebrow-pulse {
          width: 7px; height: 7px; border-radius: 50%; background: var(--teal);
          box-shadow: 0 0 0 3px var(--teal-soft);
          animation: pf-pulse 2s ease-in-out infinite;
        }
        @keyframes pf-pulse {
          0%, 100% { opacity: 1; } 50% { opacity: 0.35; }
        }
        .pf-h1 {
          font-family: 'Space Grotesk', sans-serif; font-weight: 700;
          font-size: clamp(38px, 6vw, 64px); line-height: 1.04; letter-spacing: -0.01em;
          margin: 0 0 18px; max-width: 780px;
        }
        .pf-h1-title {
          display: block; font-size: clamp(17px, 2.2vw, 21px); font-weight: 500;
          color: var(--text-dim); font-family: 'Inter', sans-serif; margin-top: 14px; letter-spacing: 0;
        }
        .pf-summary {
          max-width: 620px; color: var(--text-dim); font-size: 16.5px; line-height: 1.7;
          margin: 0 0 36px;
        }
        .pf-hero-ctas { display: flex; flex-wrap: wrap; gap: 14px; }
        .pf-btn-primary, .pf-btn-secondary {
          display: inline-flex; align-items: center; gap: 8px;
          font-family: 'Inter', sans-serif; font-weight: 600; font-size: 14.5px;
          padding: 12px 20px; border-radius: 7px; cursor: pointer; text-decoration: none;
          transition: transform 0.15s ease, background 0.2s ease, border-color 0.2s ease;
        }
        .pf-btn-primary {
          background: var(--accent); color: #1A0E04; border: 1px solid var(--accent);
        }
        .pf-btn-primary:hover { transform: translateY(-1px); background: #ff9a54; }
        .pf-btn-secondary {
          background: transparent; color: var(--text); border: 1px solid var(--border);
        }
        .pf-btn-secondary:hover { transform: translateY(-1px); border-color: var(--text-faint); }

        /* ---------- engineering pillars grid ---------- */
        .pf-pillars-grid {
          display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 54px;
        }
        @media (max-width: 900px) {
          .pf-pillars-grid { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 540px) {
          .pf-pillars-grid { grid-template-columns: 1fr; }
        }
        .pf-pillar-card {
          border: 1px solid var(--border); border-radius: 12px; padding: 20px;
          background: var(--bg-elev); transition: border-color 0.25s ease, transform 0.2s ease, background 0.2s ease;
          display: flex; flex-direction: column; justify-content: space-between;
        }
        .pf-pillar-card:hover { border-color: var(--accent-soft); transform: translateY(-2px); background: var(--bg-elev-2); }
        .pf-pillar-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
        .pf-pillar-tag {
          font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--accent);
          background: var(--accent-soft); padding: 3px 8px; border-radius: 12px; font-weight: 600;
        }
        .pf-pillar-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 15px; margin: 0 0 6px; color: var(--text); }
        .pf-pillar-desc { font-size: 13px; color: var(--text-dim); line-height: 1.55; margin: 0; }

        /* ---------- marquee tech bar ---------- */
        .pf-marquee-wrap {
          position: relative; overflow: hidden; margin: 32px 0 0; padding: 14px 0;
          border-top: 1px solid var(--border-soft); border-bottom: 1px solid var(--border-soft);
          background: rgba(18, 22, 27, 0.4);
        }
        .pf-marquee-wrap::before, .pf-marquee-wrap::after {
          content: ''; position: absolute; top: 0; bottom: 0; width: 60px; z-index: 2; pointer-events: none;
        }
        .pf-marquee-wrap::before { left: 0; background: linear-gradient(90deg, var(--bg), transparent); }
        .pf-marquee-wrap::after { right: 0; background: linear-gradient(-90deg, var(--bg), transparent); }
        .pf-marquee-track {
          display: flex; gap: 24px; width: max-content;
          animation: pf-scroll 35s linear infinite;
        }
        .pf-marquee-item {
          font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--text-dim);
          display: flex; align-items: center; gap: 8px; white-space: nowrap;
        }
        .pf-marquee-item::before { content: '•'; color: var(--accent); opacity: 0.7; }
        @keyframes pf-scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }

        /* ---------- pipeline (signature) ---------- */
        .pf-pipeline {
          display: flex; align-items: center; flex-wrap: wrap; gap: 8px 0;
          padding: 20px 18px; border: 1px solid var(--border); border-radius: 12px;
          background: linear-gradient(180deg, var(--bg-elev), var(--bg-elev-2));
          max-width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch;
        }
        .pf-node { display: flex; align-items: center; gap: 9px; opacity: 0.4;
          animation: pf-node-light 3s ease-in-out infinite; }
        .pf-node-dot {
          width: 9px; height: 9px; border-radius: 50%; background: var(--text-faint);
          animation: pf-dot-light 3s ease-in-out infinite; flex-shrink: 0;
        }
        .pf-node-label {
          font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--text-dim);
          white-space: nowrap;
        }
        .pf-edge {
          width: 34px; height: 1px; background: var(--border); margin: 0 10px;
          position: relative; overflow: hidden; flex-shrink: 0;
        }
        .pf-edge::after {
          content: ''; position: absolute; inset: 0; background: var(--accent);
          transform: translateX(-100%); animation: pf-edge-flow 3s ease-in-out infinite;
        }
        @keyframes pf-node-light {
          0%, 8% { opacity: 0.4; } 16%, 100% { opacity: 1; }
        }
        @keyframes pf-dot-light {
          0%, 8% { background: var(--text-faint); box-shadow: none; }
          16%, 100% { background: var(--accent); box-shadow: 0 0 0 4px var(--accent-soft); }
        }
        @keyframes pf-edge-flow {
          0%, 8% { transform: translateX(-100%); } 20%, 100% { transform: translateX(0); }
        }
        @media (prefers-reduced-motion: reduce) {
          .pf-node, .pf-node-dot, .pf-edge::after { animation: none; opacity: 1; }
          .pf-node-dot { background: var(--accent); box-shadow: 0 0 0 4px var(--accent-soft); }
          .pf-edge::after { transform: translateX(0); }
        }
        .pf-pipeline-sm { padding: 14px 16px; border-radius: 9px; gap: 0; }
        .pf-pipeline-sm .pf-node-label { font-size: 11px; }
        .pf-pipeline-sm .pf-edge { width: 20px; margin: 0 6px; }
        .pf-pipeline-sm .pf-node-dot { width: 7px; height: 7px; }

        /* ---------- section shell ---------- */
        .pf-section { padding: 72px 0; border-top: 1px solid var(--border-soft); }
        .pf-section-head { display: flex; align-items: baseline; gap: 10px 14px; margin-bottom: 24px; flex-wrap: wrap; }
        .pf-section-tag {
          font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--accent);
          letter-spacing: 0.08em; text-transform: uppercase;
        }
        .pf-section-title {
          font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 30px;
          margin: 0; letter-spacing: -0.01em;
        }
        .pf-section-sub { color: var(--text-dim); font-size: 15px; margin: 0 0 32px; max-width: 560px; }

        /* ---------- category filter tabs ---------- */
        .pf-filter-tabs { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 28px; }
        .pf-tab-btn {
          font-family: 'JetBrains Mono', monospace; font-size: 12.5px; padding: 7px 16px;
          border-radius: 20px; border: 1px solid var(--border); background: var(--bg-elev);
          color: var(--text-dim); cursor: pointer; transition: all 0.2s ease;
        }
        .pf-tab-btn:hover { border-color: var(--accent); color: var(--text); }
        .pf-tab-btn.pf-tab-active {
          background: var(--accent); color: #1A0E04; border-color: var(--accent); font-weight: 600;
        }

        /* ---------- projects ---------- */
        .pf-projects { display: flex; flex-direction: column; gap: 20px; }
        .pf-project {
          border: 1px solid var(--border); border-radius: 14px; padding: 30px;
          background: var(--bg-elev); transition: border-color 0.25s ease, transform 0.25s ease;
        }
        .pf-project:hover { border-color: var(--text-faint); }
        .pf-project-top {
          display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;
          margin-bottom: 6px; flex-wrap: wrap;
        }
        .pf-project-name {
          font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 21px; margin: 0;
          display: flex; align-items: center; gap: 10px;
        }
        .pf-featured-badge {
          font-family: 'JetBrains Mono', monospace; font-size: 10.5px; padding: 2px 8px;
          border-radius: 12px; background: var(--accent-soft); color: var(--accent); font-weight: 600;
        }
        .pf-project-tagline { color: var(--text-dim); font-size: 14.5px; margin: 4px 0 20px; }
        .pf-project-years {
          font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-faint);
          white-space: nowrap; padding-top: 4px;
        }
        .pf-project-desc { font-size: 15px; color: var(--text); line-height: 1.65; margin: 22px 0 10px; }
        .pf-project-detail { font-size: 14px; color: var(--text-dim); line-height: 1.65; margin: 0 0 22px; }
        .pf-stack { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
        .pf-chip {
          font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: var(--text-dim);
          border: 1px solid var(--border); padding: 5px 10px; border-radius: 5px; background: var(--bg);
        }
        .pf-project-link {
          display: inline-flex; align-items: center; gap: 6px; color: var(--teal);
          font-size: 13.5px; font-weight: 600; text-decoration: none; font-family: 'Inter', sans-serif;
        }
        .pf-project-link:hover { text-decoration: underline; }

        /* ---------- skills ---------- */
        .pf-skills-grid {
          display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 18px;
        }
        .pf-skill-card { border: 1px solid var(--border); border-radius: 12px; padding: 22px; background: var(--bg-elev); }
        .pf-skill-label {
          font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--accent);
          text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 16px;
        }
        .pf-skill-items { display: flex; flex-wrap: wrap; gap: 8px; }
        .pf-skill-pill {
          font-size: 13px; padding: 6px 12px; border-radius: 20px; background: var(--bg);
          border: 1px solid var(--border-soft); color: var(--text); transition: border-color 0.2s ease;
        }
        .pf-skill-pill:hover { border-color: var(--accent-soft); }

        /* ---------- education / timeline ---------- */
        .pf-timeline { position: relative; padding-left: 28px; }
        .pf-timeline::before {
          content: ''; position: absolute; left: 8px; top: 12px; bottom: 12px; width: 2px;
          background: linear-gradient(180deg, var(--accent), var(--teal), var(--border));
        }
        .pf-timeline-item { position: relative; margin-bottom: 32px; }
        .pf-timeline-item:last-child { margin-bottom: 0; }
        .pf-timeline-marker {
          position: absolute; left: -28px; top: 4px; width: 18px; height: 18px; border-radius: 50%;
          background: var(--bg-elev); border: 2px solid var(--accent); display: flex;
          align-items: center; justify-content: center; box-shadow: 0 0 0 4px var(--bg);
        }
        .pf-timeline-content {
          border: 1px solid var(--border); border-radius: 12px; padding: 22px; background: var(--bg-elev);
        }
        .pf-edu-degree { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 17px; margin: 0 0 4px; word-break: break-word; }
        .pf-edu-school { color: var(--text-dim); font-size: 14.5px; margin: 0 0 12px; word-break: break-word; }
        .pf-edu-desc { color: var(--text-dim); font-size: 13.5px; line-height: 1.6; margin: 0 0 14px; }
        .pf-edu-meta { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
        .pf-edu-period, .pf-edu-metric {
          font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-faint);
        }
        .pf-edu-metric {
          color: var(--teal); padding: 3px 8px; border-radius: 4px; background: var(--teal-soft); font-weight: 600;
        }

        @media (max-width: 600px) {
          .pf-timeline { padding-left: 18px; }
          .pf-timeline::before { left: 4px; }
          .pf-timeline-marker { left: -18px; width: 12px; height: 12px; top: 6px; box-shadow: 0 0 0 3px var(--bg); }
          .pf-timeline-content { padding: 16px; }
          .pf-project { padding: 20px 16px; }
          .pf-oss-card { padding: 20px 16px; }
        }

        /* ---------- certs + achievements ---------- */
        .pf-two-col { display: grid; grid-template-columns: 1.4fr 1fr; gap: 48px; }
        @media (max-width: 800px) { .pf-two-col { grid-template-columns: 1fr; } }
        .pf-cert-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }
        .pf-cert-card {
          border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px;
          background: var(--bg-elev); display: flex; gap: 12px; align-items: flex-start;
          transition: border-color 0.2s ease;
        }
        .pf-cert-card:hover { border-color: var(--text-faint); }
        .pf-cert-icon { color: var(--accent); flex-shrink: 0; margin-top: 2px; }
        .pf-cert-name { font-size: 13.5px; font-weight: 600; margin: 0 0 3px; line-height: 1.4; }
        .pf-cert-org { font-size: 12.5px; color: var(--text-dim); margin: 0; }
        .pf-achieve-card {
          border: 1px solid var(--border); border-radius: 12px; padding: 24px;
          background: var(--bg-elev); display: flex; gap: 16px;
        }
        .pf-achieve-icon { color: var(--teal); flex-shrink: 0; }
        .pf-achieve-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 16px; margin: 0 0 6px; }
        .pf-achieve-desc { font-size: 14px; color: var(--text-dim); line-height: 1.65; margin: 0; }

        /* ---------- open source spotlight ---------- */
        .pf-oss-card {
          border: 1px solid var(--border); border-radius: 14px; padding: 28px;
          background: linear-gradient(135deg, var(--bg-elev), var(--bg-elev-2));
          display: flex; align-items: center; justify-content: space-between; gap: 20px; flex-wrap: wrap;
          margin-top: 28px;
        }
        .pf-oss-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 19px; margin: 0 0 6px; }
        .pf-oss-desc { color: var(--text-dim); font-size: 14px; margin: 0; max-width: 540px; }

        /* ---------- contact / footer ---------- */
        .pf-contact { text-align: center; padding: 88px 0 56px; }
        .pf-contact-title {
          font-family: 'Space Grotesk', sans-serif; font-weight: 700;
          font-size: clamp(30px, 5vw, 46px); margin: 0 0 18px; letter-spacing: -0.01em;
        }
        .pf-contact-sub { color: var(--text-dim); font-size: 16px; max-width: 460px; margin: 0 auto 36px; }
        .pf-contact-links { display: flex; justify-content: center; flex-wrap: wrap; gap: 14px; margin-bottom: 56px; }
        .pf-contact-link {
          display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--border);
          border-radius: 8px; padding: 11px 18px; color: var(--text); text-decoration: none;
          font-size: 14px; font-weight: 500; background: var(--bg-elev);
          transition: border-color 0.2s ease, transform 0.15s ease;
        }
        .pf-contact-link:hover { border-color: var(--accent); transform: translateY(-1px); }
        .pf-footer {
          display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
          padding: 26px 0; border-top: 1px solid var(--border-soft);
          font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-faint);
        }

        a:focus-visible, button:focus-visible {
          outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px;
        }
      `}</style>

      {/* NAV */}
      <nav className="pf-nav">
        <div className="pf-wrap pf-nav-inner">
          <div className="pf-logo" onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>
            <img src="/logo.png" alt="S Logo" className="pf-logo-mark" />
            <span>sathwik</span>
          </div>
          <div className="pf-nav-links">
            {NAV.map((item) => (
              <button key={item} className="pf-nav-link" onClick={() => scrollTo(item.toLowerCase())}>
                {item}
              </button>
            ))}
            <button className="pf-nav-cta" onClick={() => scrollTo("contact")}>
              Get in touch
            </button>
          </div>
          <button
            className="pf-burger"
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            onClick={() => setMenuOpen((v) => !v)}
          >
            {menuOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
        <div className={`pf-mobile-menu ${menuOpen ? "pf-open" : ""}`}>
          {NAV.map((item) => (
            <button key={item} className="pf-mobile-link" onClick={() => scrollTo(item.toLowerCase())}>
              {item}
            </button>
          ))}
          <button className="pf-mobile-link" onClick={() => scrollTo("contact")}>
            Get in touch
          </button>
        </div>
      </nav>

      {/* MARRIAGE TOXIN STYLED HERO & CHARACTER SHOWCASE */}
      <header className={`pf-hero pf-wrap pf-mt-hero ${loaderDone ? "pf-animate-start" : ""}`}>
        {/* Giant Watermark Background Typography */}
        <div className="pf-mt-bg-watermark pf-watermark-entrance">
          {currentChara.bgText.map((t, idx) => (
            <div key={idx}>{t}</div>
          ))}
        </div>

        {/* Character Selector Tabs */}
        <div className="pf-mt-selector">
          {CHARACTER_PROFILES.map((chara) => (
            <button
              key={chara.id}
              className={`pf-mt-tab ${activeCharaId === chara.id ? "active" : ""}`}
              onClick={() => setActiveCharaId(chara.id)}
            >
              {chara.name}
            </button>
          ))}
        </div>

        <div className="pf-mt-layout">
          {/* Left Column: Character Data — key forces fast 0.25s re-mount animation on tab change */}
          <div key={activeCharaId} className="pf-mt-data pf-hero-entrance pf-mt-data-anim">
            <h1 className="pf-mt-name">{currentChara.name}</h1>
            <div className="pf-mt-role">{currentChara.title}</div>

            <div className="pf-mt-divider" />

            <p className="pf-summary">{currentChara.summary}</p>

            <div className="pf-hero-ctas">
              <a className="pf-btn-primary" href={`mailto:${PROFILE.email}`}>
                <Mail size={16} /> Get in Touch
              </a>
              <a className="pf-btn-secondary" href={PROFILE.resume} target="_blank" rel="noreferrer" download="Peechara_Sathwik_Resume.pdf">
                <Download size={16} /> Download Resume
              </a>
              <a className="pf-btn-secondary" href={PROFILE.github} target="_blank" rel="noreferrer">
                <Github size={16} /> GitHub Profile
              </a>
            </div>

            {/* Character Specs Grid */}
            <div className="pf-mt-stats-grid">
              {currentChara.stats.map((st) => (
                <div key={st.label} className="pf-mt-stat-card">
                  <div className="pf-mt-stat-label">{st.label}</div>
                  <div className="pf-mt-stat-value">{st.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Right Column: Person Cutout */}
          <div className="pf-mt-figure-col pf-figure-entrance">
            <div className="pf-mt-figure-wrap">
              <div className="pf-mt-figure-card">
                <img src="/sathwik_cutout.png" alt={PROFILE.name} className="pf-mt-figure-img" />
              </div>
            </div>
          </div>
        </div>

        {/* CORE ENGINEERING PILLARS GRID */}
        <div className="pf-pillars-grid pf-hero-entrance">
          {ENGINEERING_PILLARS.map((p) => {
            const Icon = p.icon;
            return (
              <div key={p.title} className="pf-pillar-card">
                <div>
                  <div className="pf-pillar-head">
                    <span className="pf-pillar-tag">{p.tag}</span>
                    <Icon size={18} color="var(--accent)" />
                  </div>
                  <h4 className="pf-pillar-title">{p.title}</h4>
                  <p className="pf-pillar-desc">{p.desc}</p>
                </div>
              </div>
            );
          })}
        </div>

        <Reveal>
          <Pipeline stages={["Input Prompt", "Model Router", "NLP / CV", "Verified Output"]} />
        </Reveal>
      </header>

      {/* MARQUEE TECH BAR */}
      <div className="pf-marquee-wrap">
        <div className="pf-marquee-track">
          {[...MARQUEE_TECH, ...MARQUEE_TECH].map((tech, i) => (
            <span key={i} className="pf-marquee-item">
              {tech}
            </span>
          ))}
        </div>
      </div>

      {/* PROJECTS */}
      <section id="work" className="pf-section pf-wrap">
        <Reveal>
          <div className="pf-section-head">
            <span className="pf-section-tag">01</span>
            <h2 className="pf-section-title">Featured Systems</h2>
          </div>
        </Reveal>
        <Reveal>
          <p className="pf-section-sub">
            Four projects, four execution pipelines. Click filter categories or view the architecture stages for each system.
          </p>
        </Reveal>

        {/* PROJECT FILTER TABS */}
        <div className="pf-filter-tabs">
          {["All", "AI & NLP", "Computer Vision", "Infra & Tooling"].map((cat) => (
            <button
              key={cat}
              className={`pf-tab-btn ${activeCategory === cat ? "pf-tab-active" : ""}`}
              onClick={() => setActiveCategory(cat)}
            >
              {cat}
            </button>
          ))}
        </div>

        <div className="pf-projects">
          {filteredProjects.map((p, i) => (
            <Reveal key={p.name} delay={i * 60}>
              <article className="pf-project">
                <div className="pf-project-top">
                  <h3 className="pf-project-name">
                    {p.name}
                    {p.featured && <span className="pf-featured-badge">FEATURED</span>}
                  </h3>
                  <span className="pf-project-years pf-mono">{p.years}</span>
                </div>
                <p className="pf-project-tagline">{p.tagline}</p>
                <Pipeline stages={p.stages} size="sm" />
                <p className="pf-project-desc">{p.description}</p>
                <p className="pf-project-detail">{p.detail}</p>
                <div className="pf-stack">
                  {p.stack.map((s) => (
                    <span key={s} className="pf-chip">
                      {s}
                    </span>
                  ))}
                </div>
                {p.link && (
                  <a className="pf-project-link" href={p.link} target="_blank" rel="noreferrer">
                    View repository <ArrowUpRight size={14} />
                  </a>
                )}
              </article>
            </Reveal>
          ))}
        </div>

        {/* OPEN SOURCE SPOTLIGHT */}
        <Reveal delay={100}>
          <div className="pf-oss-card">
            <div>
              <h4 className="pf-oss-title">Explore Open Source Code</h4>
              <p className="pf-oss-desc">
                Check out active repositories, computer vision trackers, and Python experiments on my GitHub profile.
              </p>
            </div>
            <a className="pf-btn-secondary" href={PROFILE.github} target="_blank" rel="noreferrer" style={{ whitespace: "nowrap" }}>
              <Github size={16} /> @sathwik2021
            </a>
          </div>
        </Reveal>
      </section>

      {/* SKILLS */}
      <section id="systems" className="pf-section pf-wrap">
        <Reveal>
          <div className="pf-section-head">
            <span className="pf-section-tag">02</span>
            <h2 className="pf-section-title">Technical Skills</h2>
          </div>
        </Reveal>
        <Reveal>
          <p className="pf-section-sub">Comprehensive toolkit used across machine learning, backend engineering, and pipeline automation.</p>
        </Reveal>

        {/* SKILLS TAB FILTER */}
        <div className="pf-filter-tabs">
          {["All", "Languages", "AI / ML", "Libraries", "Data & Cloud", "Tools"].map((cat) => (
            <button
              key={cat}
              className={`pf-tab-btn ${activeSkillCategory === cat ? "pf-tab-active" : ""}`}
              onClick={() => setActiveSkillCategory(cat)}
            >
              {cat}
            </button>
          ))}
        </div>

        <div className="pf-skills-grid">
          {filteredSkillGroups.map((group, i) => (
            <Reveal key={group.label} delay={i * 50}>
              <div className="pf-skill-card">
                <p className="pf-skill-label">{group.label}</p>
                <div className="pf-skill-items">
                  {group.items.map((item) => (
                    <span key={item} className="pf-skill-pill">
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* EDUCATION & TIMELINE */}
      <section id="education" className="pf-section pf-wrap">
        <Reveal>
          <div className="pf-section-head">
            <span className="pf-section-tag">03</span>
            <h2 className="pf-section-title">Education &amp; Journey</h2>
          </div>
        </Reveal>
        <div className="pf-two-col">
          <Reveal>
            <div className="pf-timeline">
              {EDUCATION.map((e) => (
                <div className="pf-timeline-item" key={e.school}>
                  <div className="pf-timeline-marker" />
                  <div className="pf-timeline-content">
                    <h3 className="pf-edu-degree">{e.degree}</h3>
                    <p className="pf-edu-school">{e.school}</p>
                    <p className="pf-edu-desc">{e.description}</p>
                    <div className="pf-edu-meta">
                      <span className="pf-edu-period">{e.period}</span>
                      <span className="pf-edu-metric">{e.metric}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Reveal>

          <Reveal delay={100}>
            <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
              <div className="pf-achieve-card">
                <div className="pf-achieve-icon">
                  <Trophy size={22} />
                </div>
                <div>
                  <h4 className="pf-achieve-title">Vishesh Hackathon</h4>
                  <p className="pf-achieve-desc">
                    Competed under high pressure, designing, building, and shipping functional code with a team.
                  </p>
                </div>
              </div>

              <div className="pf-achieve-card">
                <div className="pf-achieve-icon">
                  <Sparkles size={22} />
                </div>
                <div>
                  <h4 className="pf-achieve-title">B.Tech Specialization</h4>
                  <p className="pf-achieve-desc">
                    Focused on Data Science, artificial intelligence model integration, computer vision, and modern backend architectures.
                  </p>
                </div>
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      {/* CERTIFICATIONS & BADGES */}
      <section className="pf-section pf-wrap">
        <Reveal>
          <div className="pf-section-head">
            <span className="pf-section-tag">04</span>
            <h2 className="pf-section-title">Certifications &amp; Badges</h2>
          </div>
        </Reveal>
        <div className="pf-cert-grid">
          {CERTIFICATIONS.map((c, i) => (
            <Reveal key={c.name} delay={i * 40}>
              <div className="pf-cert-card">
                <div className="pf-cert-icon">
                  {c.type === "Badge" ? <BadgeCheck size={18} color="var(--teal)" /> : <Award size={18} />}
                </div>
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                    <p className="pf-cert-name" style={{ margin: 0 }}>{c.name}</p>
                    {c.type === "Badge" && (
                      <span className="pf-mono" style={{ fontSize: "10px", padding: "2px 6px", borderRadius: "4px", background: "var(--teal-soft)", color: "var(--teal)", fontWeight: "600" }}>
                        BADGE
                      </span>
                    )}
                  </div>
                  <p className="pf-cert-org pf-mono">{c.org}</p>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* CONTACT */}
      <section id="contact" className="pf-contact pf-wrap">
        <Reveal>
          <h2 className="pf-contact-title">Let's build something.</h2>
        </Reveal>
        <Reveal delay={60}>
          <p className="pf-contact-sub">
            Open to internships and junior roles in AI, backend, or data.
          </p>
        </Reveal>
        <Reveal delay={120}>
          <div className="pf-contact-links">
            <a className="pf-contact-link" href={`mailto:${PROFILE.email}`}>
              <Mail size={16} /> {PROFILE.email}
            </a>
            <a className="pf-contact-link" href={`tel:${PROFILE.phone.replace(/\s/g, "")}`}>
              <Phone size={16} /> {PROFILE.phone}
            </a>
            <a className="pf-contact-link" href={PROFILE.resume} target="_blank" rel="noreferrer" download="Peechara_Sathwik_Resume.pdf">
              <Download size={16} /> Resume PDF
            </a>
            <a className="pf-contact-link" href={PROFILE.github} target="_blank" rel="noreferrer">
              <Github size={16} /> GitHub
            </a>
            <a className="pf-contact-link" href={PROFILE.linkedin} target="_blank" rel="noreferrer">
              <Linkedin size={16} /> LinkedIn
            </a>
          </div>
        </Reveal>
      </section>

      <footer className="pf-footer pf-wrap">
        <div className="pf-footer-brand" onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>
          <img src="/logo.png" alt="S Logo" className="pf-logo-mark-sm" />
          <span>sathwik</span>
        </div>
        <span>© {new Date().getFullYear()} {PROFILE.name}. All rights reserved.</span>
      </footer>
    </div>
  );
}
