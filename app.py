#!/usr/bin/env python3
"""
AI Assistant Chatbot Manager
- Multi-model, multi-chatbot support with Personality Presets
- Isolated RAG (Retrieval-Augmented Generation) per chatbot
- Smart GPU detection and VRAM management
- Model folder scanning and loading
- Built with PySide6 + llama-cpp-python + ChromaDB
"""

import sys
import os
import gc
import re
import json
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QTextEdit, QLineEdit, QPushButton, QLabel,
    QFileDialog, QComboBox, QProgressBar, QGroupBox, QScrollArea,
    QCheckBox, QSpinBox, QSlider, QSplitter, QTreeWidget,
    QTreeWidgetItem, QMessageBox, QStatusBar, QFrame, QGridLayout,
    QSizePolicy, QToolButton, QMenu, QDialog, QDialogButtonBox,
    QFormLayout, QTextBrowser, QInputDialog
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QSettings, QMutex
from PySide6.QtGui import QFont, QColor, QTextCursor, QIcon, QAction, QPalette

# ─────────────────────── STYLING ───────────────────────

DARK_STYLESHEET = """
QMainWindow { background-color: #1a1a2e; }
QWidget { background-color: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', Arial; font-size: 13px; }
QFrame#chatFrame { background-color: #16213e; border: 1px solid #0f3460; border-radius: 8px; }
QFrame#sidebarFrame { background-color: #16213e; border-right: 1px solid #0f3460; }
QTextEdit { background-color: #0f0f23; color: #e0e0e0; border: 1px solid #0f3460; border-radius: 6px; padding: 8px; font-size: 14px; }
QTextEdit[readOnly="true"] { background-color: #0a0a1a; }
QLineEdit { background-color: #0f0f23; color: #e0e0e0; border: 1px solid #0f3460; border-radius: 6px; padding: 8px; font-size: 14px; }
QLineEdit:focus { border: 2px solid #e94560; }
QPushButton { background-color: #0f3460; color: #e0e0e0; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; }
QPushButton:hover { background-color: #e94560; }
QPushButton:pressed { background-color: #c81e45; }
QPushButton:disabled { background-color: #2a2a4a; color: #555; }
QPushButton#dangerBtn { background-color: #8b0000; }
QPushButton#dangerBtn:hover { background-color: #c41e3a; }
QPushButton#successBtn { background-color: #0a6847; }
QPushButton#successBtn:hover { background-color: #0d8b5e; }
QTabWidget::pane { border: 1px solid #0f3460; border-radius: 6px; background-color: #1a1a2e; }
QTabBar::tab { background-color: #16213e; color: #aaa; padding: 10px 20px; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }
QTabBar::tab:selected { background-color: #0f3460; color: #e94560; font-weight: bold; }
QTabBar::tab:hover { background-color: #0f3460; color: #fff; }
QTreeWidget { background-color: #0f0f23; color: #e0e0e0; border: 1px solid #0f3460; border-radius: 6px; alternate-background-color: #16213e; }
QTreeWidget::item:selected { background-color: #0f3460; color: #e94560; }
QTreeWidget::item:hover { background-color: #1a2a5e; }
QHeaderView::section { background-color: #16213e; color: #e94560; padding: 6px; border: none; font-weight: bold; }
QProgressBar { background-color: #0f0f23; border: 1px solid #0f3460; border-radius: 6px; text-align: center; color: #e0e0e0; font-weight: bold; }
QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e94560, stop:1 #0f3460); border-radius: 5px; }
QGroupBox { border: 2px solid #0f3460; border-radius: 8px; margin-top: 12px; padding-top: 16px; font-weight: bold; color: #e94560; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QComboBox { background-color: #0f0f23; color: #e0e0e0; border: 1px solid #0f3460; border-radius: 6px; padding: 6px; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background-color: #16213e; color: #e0e0e0; selection-background-color: #0f3460; }
QSpinBox { background-color: #0f0f23; color: #e0e0e0; border: 1px solid #0f3460; border-radius: 4px; padding: 4px; }
QSlider::groove:horizontal { background: #0f3460; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #e94560; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }
QScrollArea { border: none; }
QScrollBar:vertical { background: #1a1a2e; width: 10px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #0f3460; border-radius: 5px; min-height: 20px; }
QScrollBar:horizontal { background: #1a1a2e; height: 10px; border-radius: 5px; }
QScrollBar::handle:horizontal { background: #0f3460; border-radius: 5px; min-width: 20px; }
QStatusBar { background-color: #16213e; color: #aaa; border-top: 1px solid #0f3460; }
QLabel#headerLabel { color: #e94560; font-size: 14px; font-weight: bold; }
QLabel#gpuLabel { color: #53d769; font-size: 12px; }
QLabel#warningLabel { color: #f0a500; font-size: 12px; }
QLabel#vramLabel { color: #00d4ff; font-size: 13px; font-weight: bold; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #0f3460; border-radius: 3px; background-color: #0f0f23; }
QCheckBox::indicator:checked { background-color: #e94560; }
"""

# ─────────────────────── PERSONALITY PRESETS ───────────────────────

PERSONALITY_PRESETS = {
    "Default Assistant": "You are a helpful, respectful, and honest AI assistant. Always answer as helpfully as possible.",
    "Coding Expert": "You are an expert programmer and software architect. Provide concise, bug-free code with clear explanations. Favor Python, C++, and Rust. Use markdown code blocks.",
    "Creative Writer": "You are a creative and imaginative writer. Use vivid, expressive language. Think outside the box and tell compelling stories.",
    "Sarcastic Buddy": "You are a witty, slightly sarcastic AI friend. Keep it fun and playful, but still provide useful information. Heavy use of emojis and dry humor.",
    "Strict Teacher": "You are a strict but fair university professor. Ensure the user understands the core concepts deeply. Test their knowledge with follow-up questions.",
    "Cyberpunk Hacker": "You are a rogue AI from the neon-lit streets of 2077. Speak in hacker jargon, reference the net, and keep things edgy. Always use terminal-style formatting.",
    "Minimalist": "You are a minimalist AI. Provide extremely concise, to-the-point answers. No fluff, no pleasantries, just the facts."
}

# ─────────────────────── RAG MANAGER (MULTI-KB) ───────────────────────

class RAGManager:
    """Manages multiple isolated document collections using ChromaDB"""
    def __init__(self):
        self.available = False
        self.collections: Dict[str, object] = {}
        try:
            import chromadb
            self.chromadb = chromadb
            self.client = chromadb.PersistentClient(path="./rag_db")
            self._load_existing_collections()
            self.available = True
        except ImportError:
            self.available = False

    def _sanitize_name(self, name: str) -> str:
        # ChromaDB limits: 3-63 chars, alphanumeric, hyphens, underscores
        name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        name = name[:63]
        if len(name) < 3:
            name = name + "_kb"
        return name

    def _load_existing_collections(self):
        for col in self.client.list_collections():
            self.collections[col.name] = self.client.get_collection(col.name)

    def create_kb(self, name: str) -> str:
        if not self.available: raise ImportError("ChromaDB not installed")
        sname = self._sanitize_name(name)
        if sname not in self.collections:
            self.collections[sname] = self.client.get_or_create_collection(
                name=sname, metadata={"hnsw:space": "cosine"}
            )
        return sname

    def delete_kb(self, name: str):
        if not self.available: return
        sname = self._sanitize_name(name)
        if sname in self.collections:
            self.client.delete_collection(sname)
            del self.collections[sname]

    def list_kbs(self) -> List[str]:
        return list(self.collections.keys())

    def add_document(self, filepath: str, kb_name: str) -> int:
        if not self.available: raise ImportError("ChromaDB not installed")
        collection = self.collections.get(kb_name)
        if not collection: return 0

        ext = Path(filepath).suffix.lower()
        text = ""
        
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(filepath)
                for page in reader.pages:
                    t = page.extract_text()
                    if t: text += t + "\n"
            except ImportError:
                QMessageBox.warning(None, "Missing Dependency", "Install pypdf to ingest PDFs: pip install pypdf")
                return 0
        elif ext in (".txt", ".md", ".py", ".js", ".c", ".cpp", ".rs", ".json", ".html", ".css"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            return 0

        # Chunking
        chunk_size, overlap = 800, 200
        chunks, start = [], 0
        while start < len(text):
            chunk = text[start:start + chunk_size].strip()
            if chunk: chunks.append(chunk)
            start += (chunk_size - overlap)

        if chunks:
            ids = [f"{Path(filepath).name}_chunk_{i}_{int(time.time())}" for i in range(len(chunks))]
            metadatas = [{"source": filepath, "chunk": i} for i in range(len(chunks))]
            collection.add(documents=chunks, ids=ids, metadatas=metadatas)
            return len(chunks)
        return 0

    def query(self, query_text: str, kb_name: str, n_results: int = 3) -> str:
        if not self.available or kb_name not in self.collections: return ""
        collection = self.collections[kb_name]
        if collection.count() == 0: return ""
        results = collection.query(query_texts=[query_text], n_results=n_results)
        docs = results.get('documents', [[]])[0]
        if not docs: return ""
        return "\n\n".join([f"[Context {i+1}]: {doc}" for i, doc in enumerate(docs)])

    def get_document_count(self, kb_name: str) -> int:
        if not self.available or kb_name not in self.collections: return 0
        return self.collections[kb_name].count()


# ─────────────────────── GPU DETECTION ───────────────────────

@dataclass
class GPUInfo:
    name: str = "Unknown GPU"; vendor: str = "Unknown"; total_vram_mb: int = 0
    free_vram_mb: int = 0; used_vram_mb: int = 0; driver_version: str = "N/A"
    cuda_version: str = "N/A"; index: int = 0; temperature_c: int = 0; utilization_pct: int = 0

    @property
    def vram_usage_pct(self) -> float:
        return (self.used_vram_mb / self.total_vram_mb * 100.0) if self.total_vram_mb else 0.0

class GPUDetector:
    def __init__(self): self.gpus, self.detection_method, self.cuda_available = [], "None", False
    def detect(self) -> List[GPUInfo]:
        self.gpus = []
        for name, method in [("pynvml", self._pynvml), ("nvidia-smi", self._smi)]:
            try:
                r = method()
                if r: self.gpus, self.detection_method, self.cuda_available = r, name, True; return r
            except: pass
        self.detection_method, self.cuda_available = "None (CPU Only)", False; return self.gpus

    def refresh_vram(self):
        if self.detection_method == "pynvml":
            try: self._refresh_pynvml()
            except: pass
        elif self.detection_method == "nvidia-smi":
            try: self.gpus = self._smi()
            except: pass

    def _pynvml(self) -> List[GPUInfo]:
        import pynvml; pynvml.nvmlInit(); gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            n = pynvml.nvmlDeviceGetName(h); 
            if isinstance(n, bytes): n = n.decode()
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            try: t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except: t = 0
            try: u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except: u = 0
            try: d = pynvml.nvmlSystemGetDriverVersion()
            except: d = "N/A"
            if isinstance(d, bytes): d = d.decode()
            gpus.append(GPUInfo(name=n, vendor="NVIDIA", total_vram_mb=int(m.total//(1024*1024)), free_vram_mb=int(m.free//(1024*1024)), used_vram_mb=int(m.used//(1024*1024)), driver_version=d, index=i, temperature_c=t, utilization_pct=u))
        pynvml.nvmlShutdown(); return gpus

    def _refresh_pynvml(self):
        import pynvml; pynvml.nvmlInit()
        for g in self.gpus:
            h = pynvml.nvmlDeviceGetHandleByIndex(g.index); m = pynvml.nvmlDeviceGetMemoryInfo(h)
            g.free_vram_mb, g.used_vram_mb = int(m.free//(1024*1024)), int(m.used//(1024*1024))
        pynvml.nvmlShutdown()

    def _smi(self) -> List[GPUInfo]:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0: return []
        gpus = []
        for l in r.stdout.strip().split("\n"):
            p = [x.strip() for x in l.split(",")]
            if len(p) >= 8: gpus.append(GPUInfo(name=p[1], vendor="NVIDIA", total_vram_mb=int(float(p[2])), free_vram_mb=int(float(p[3])), used_vram_mb=int(float(p[4])), temperature_c=int(float(p[5])), utilization_pct=int(float(p[6].replace("%",""))), driver_version=p[7], index=int(p[0])))
        return gpus

    def get_total_vram(self) -> int: return sum(g.total_vram_mb for g in self.gpus)
    def get_free_vram(self) -> int: return sum(g.free_vram_mb for g in self.gpus)


# ─────────────────────── MODEL SCANNER & MANAGER ───────────────────────

@dataclass
class ModelInfo:
    name: str; path: Path; size_mb: float; format: str; estimated_vram_mb: float; loaded: bool = False; quantization: str = ""

class ModelScanner:
    SUPPORTED_EXTENSIONS = {'.gguf', '.bin', '.safetensors'}
    def __init__(self): self.search_paths, self.models = [], []
    def add_path(self, p): 
        if p not in self.search_paths: self.search_paths.append(p)
    def scan(self) -> List[ModelInfo]:
        self.models, seen = [], set()
        for p in self.search_paths:
            path = Path(p)
            if path.exists():
                for e in path.rglob("*"):
                    if e.is_file() and e.suffix.lower() in self.SUPPORTED_EXTENSIONS and str(e) not in seen:
                        seen.add(str(e))
                        try:
                            s = e.stat().st_size / (1024*1024); q = ""; nl = e.name.lower()
                            for qp in ['q4_0','q5_0','q8_0','q4_k_m','q5_k_m','fp16']:
                                if qp in nl: q = qp.upper(); break
                            self.models.append(ModelInfo(e.stem, e, s, e.suffix.lower().lstrip('.'), s*1.3, False, q))
                        except: pass
        self.models.sort(key=lambda m: m.name.lower()); return self.models

class ModelManager:
    SAFETY_MARGIN_MB = 512
    def __init__(self, gd): self.gpu_detector, self.loaded_models, self.model_vram, self.mutex = gd, {}, {}, QMutex()
    def can_load(self, mi) -> Tuple[bool, str]:
        if not self.gpu_detector.gpus: return True, "CPU Mode"
        self.gpu_detector.refresh_vram(); a = self.gpu_detector.get_free_vram() - sum(self.model_vram.values()) - self.SAFETY_MARGIN_MB
        return (True, f"OK: Need {mi.estimated_vram_mb:.0f}MB, {a:.0f}MB avail") if mi.estimated_vram_mb <= a else (False, f"Low VRAM: Need {mi.estimated_vram_mb:.0f}MB, {a:.0f}MB avail")

    def load_model(self, mi, ngl=-1, nctx=4096):
        from llama_cpp import Llama; self.mutex.lock()
        try:
            if mi.name in self.loaded_models: return self.loaded_models[mi.name]
            if not self.gpu_detector.cuda_available: ngl = 0
            m = Llama(model_path=str(mi.path), n_gpu_layers=ngl, n_ctx=nctx, n_batch=512, verbose=False)
            self.loaded_models[mi.name], self.model_vram[mi.name], mi.loaded = m, mi.estimated_vram_mb, True
            time.sleep(0.5); self.gpu_detector.refresh_vram(); return m
        finally: self.mutex.unlock()

    def unload_model(self, n):
        self.mutex.lock()
        try:
            if n in self.loaded_models: del self.loaded_models[n]; self.model_vram.pop(n, None); gc.collect(); time.sleep(0.3); self.gpu_detector.refresh_vram()
        finally: self.mutex.unlock()

    def generate_stream(self, mn, prompt, max_tokens=512, temperature=0.7, top_p=0.9):
        m = self.loaded_models.get(mn)
        if not m: raise ValueError(f"Model '{mn}' not loaded")
        for c in m(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=["User:", "\n\n\n"], stream=True):
            t = c['choices'][0].get('text', '')
            if t: yield t


# ─────────────────────── WORKERS ───────────────────────

class ModelLoadWorker(QThread):
    progress, finished = Signal(int), Signal(str, bool, str)
    def __init__(self, mm, mi, ngl, nctx): super().__init__(); self.mm, self.mi, self.ngl, self.nctx = mm, mi, ngl, nctx
    def run(self):
        self.progress.emit(20)
        try: self.mm.load_model(self.mi, self.ngl, self.nctx); self.finished.emit(self.mi.name, True, f"✓ {self.mi.name} loaded")
        except Exception as e: self.finished.emit(self.mi.name, False, f"✗ Failed: {e}")

class InferenceWorker(QThread):
    chunk_ready, finished, error = Signal(str, str), Signal(str, str), Signal(str, str)
    def __init__(self, mm, mn, cid, prompt, params): super().__init__(); self.mm, self.mn, self.cid, self.prompt, self.params, self._stop = mm, mn, cid, prompt, params, False
    def stop(self): self._stop = True
    def run(self):
        try:
            full = ""
            for t in self.mm.generate_stream(self.mn, self.prompt, self.params.get('max_tokens',512), self.params.get('temperature',0.7), self.params.get('top_p',0.9)):
                if self._stop: break
                full += t; self.chunk_ready.emit(self.cid, t)
            self.finished.emit(self.cid, full)
        except Exception as e: self.error.emit(self.cid, str(e))


# ─────────────────────── CHAT WIDGET ───────────────────────

class ChatWidget(QWidget):
    send_requested = Signal(str, str)
    stop_requested = Signal(str)
    rag_kb_changed = Signal(str, str) # chat_id, kb_name

    def __init__(self, chat_id: str, model_name: str = "", personality: str = "Default Assistant", parent=None):
        super().__init__(parent)
        self.chat_id, self.model_name, self.personality = chat_id, model_name, personality
        self.messages, self.is_generating, self.current_kb = [], False, "None"
        self._setup_ui()
        if personality and personality in PERSONALITY_PRESETS:
            self.messages.append({'role': 'system', 'content': PERSONALITY_PRESETS[personality]})

    def _setup_ui(self):
        layout = QVBoxLayout(self); layout.setContentsMargins(6,6,6,6); layout.setSpacing(6)
        
        # Header
        header = QFrame(); header.setFixedHeight(44)
        hl = QHBoxLayout(header); hl.setContentsMargins(8,0,8,0)
        self.model_label = QLabel(f"🤖 {self.model_name or 'No Model'}"); self.model_label.setObjectName("headerLabel"); hl.addWidget(self.model_label)
        
        hl.addWidget(QLabel("Personality:"))
        self.personality_cb = QComboBox(); self.personality_cb.addItems(PERSONALITY_PRESETS.keys()); self.personality_cb.setCurrentText(self.personality)
        self.personality_cb.currentTextChanged.connect(self._change_personality); hl.addWidget(self.personality_cb)

        hl.addWidget(QLabel("RAG KB:"))
        self.rag_kb_cb = QComboBox(); self.rag_kb_cb.setMinimumWidth(100)
        self.rag_kb_cb.currentTextChanged.connect(self._change_rag_kb); hl.addWidget(self.rag_kb_cb)
        
        self.status_label = QLabel("Ready"); self.status_label.setObjectName("gpuLabel")
        hl.addStretch(); hl.addWidget(self.status_label)
        layout.addWidget(header)

        # Chat Display
        self.chat_display = QTextBrowser(); self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("QTextBrowser { background-color: #0a0a1a; color: #e0e0e0; border: 1px solid #0f3460; border-radius: 8px; padding: 12px; font-size: 14px; }")
        layout.addWidget(self.chat_display, 1)

        # Input
        input_frame = QFrame(); il = QHBoxLayout(input_frame); il.setContentsMargins(0,0,0,0)
        self.input_field = QLineEdit(); self.input_field.setPlaceholderText("Type message... (Enter to send)")
        self.input_field.returnPressed.connect(self._on_send)
        self.send_btn = QPushButton("➤ Send"); self.send_btn.setObjectName("successBtn"); self.send_btn.setFixedWidth(80); self.send_btn.clicked.connect(self._on_send)
        self.stop_btn = QPushButton("■ Stop"); self.stop_btn.setObjectName("dangerBtn"); self.stop_btn.setFixedWidth(70); self.stop_btn.clicked.connect(self._on_stop); self.stop_btn.setEnabled(False)
        il.addWidget(self.input_field, 1); il.addWidget(self.send_btn); il.addWidget(self.stop_btn)
        layout.addWidget(input_frame)

    def update_rag_kbs(self, kbs: List[str]):
        curr = self.rag_kb_cb.currentText()
        self.rag_kb_cb.blockSignals(True)
        self.rag_kb_cb.clear()
        self.rag_kb_cb.addItems(["None"] + kbs)
        if curr in kbs or curr == "None": self.rag_kb_cb.setCurrentText(curr)
        else: self.rag_kb_cb.setCurrentText("None"); self.current_kb = "None"
        self.rag_kb_cb.blockSignals(False)

    def set_model(self, n): self.model_name = n; self.model_label.setText(f"🤖 {n}")
    def _change_personality(self, p):
        self.personality = p
        if self.messages and self.messages[0]['role'] == 'system': self.messages[0]['content'] = PERSONALITY_PRESETS[p]
        else: self.messages.insert(0, {'role': 'system', 'content': PERSONALITY_PRESETS[p]})
        self._append_system_msg(f"Personality changed to: {p}")
    def _change_rag_kb(self, kb):
        self.current_kb = kb; self._append_system_msg(f"RAG Knowledge Base set to: {kb}")

    def _on_send(self):
        text = self.input_field.text().strip()
        if not text or self.is_generating: return
        self.input_field.clear(); self.messages.append({'role': 'user', 'content': text})
        self._display_user_msg(text); self.send_requested.emit(self.chat_id, text)

    def _on_stop(self): self.stop_requested.emit(self.chat_id)

    def start_gen(self):
        self.is_generating = True; self.send_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.input_field.setEnabled(False); self.status_label.setText("Generating..."); self.status_label.setStyleSheet("color: #e94560;")

    def append_chunk(self, chunk):
        if not self.messages or self.messages[-1]['role'] != 'assistant': self.messages.append({'role': 'assistant', 'content': chunk})
        else: self.messages[-1]['content'] += chunk
        self._display_assistant_msg(self.messages[-1]['content'])

    def finish_gen(self):
        self.is_generating = False; self.send_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.input_field.setEnabled(True); self.input_field.setFocus()
        self.status_label.setText("Ready"); self.status_label.setStyleSheet("color: #53d769;"); self._finalize_assistant_msg()

    def _display_user_msg(self, t):
        h = self.chat_display.toHtml()
        self.chat_display.setHtml(f'{h}<div style="margin:8px 0;padding:10px 14px;background-color:#0f3460;border-radius:12px 12px 2px 12px;max-width:80%;margin-left:auto;"><b style="color:#e94560;">You</b><br><span style="color:#e0e0e0;">{self._esc(t)}</span></div>')
        self.chat_display.moveCursor(QTextCursor.End)

    def _display_assistant_msg(self, t):
        h = self.chat_display.toHtml()
        if "<!--A_MSG-->" in h: h = h[:h.index("<!--A_MSG-->")]
        self.chat_display.setHtml(f'{h}<!--A_MSG--><div style="margin:8px 0;padding:10px 14px;background-color:#1a2a4e;border-radius:12px 12px 12px 2px;max-width:80%;"><b style="color:#00d4ff;">Assistant</b><br><span style="color:#e0e0e0;white-space:pre-wrap;">{self._esc(t)}</span><span style="color:#e94560;">▌</span></div>')
        self.chat_display.moveCursor(QTextCursor.End)

    def _finalize_assistant_msg(self):
        if self.messages and self.messages[-1]['role'] == 'assistant':
            t = self.messages[-1]['content']; h = self.chat_display.toHtml()
            if "<!--A_MSG-->" in h: h = h[:h.index("<!--A_MSG-->")]
            self.chat_display.setHtml(f'{h}<!--A_MSG--><div style="margin:8px 0;padding:10px 14px;background-color:#1a2a4e;border-radius:12px 12px 12px 2px;max-width:80%;"><b style="color:#00d4ff;">Assistant</b><br><span style="color:#e0e0e0;white-space:pre-wrap;">{self._esc(t)}</span></div>')
            self.chat_display.moveCursor(QTextCursor.End)

    def _append_system_msg(self, t):
        h = self.chat_display.toHtml()
        self.chat_display.setHtml(f'{h}<div style="margin:8px 0;padding:8px;text-align:center;"><span style="color:#666;font-style:italic;">{self._esc(t)}</span></div>')

    @staticmethod
    def _esc(t): return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")


# ─────────────────────── MAIN WINDOW ───────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Assistant Manager — Multi-Model & Isolated RAG Chatbot")
        self.resize(1400, 850)
        
        self.gpu_detector = GPUDetector()
        self.model_scanner = ModelScanner()
        self.model_manager = ModelManager(self.gpu_detector)
        self.rag_manager = RAGManager()
        
        self.chat_widgets: Dict[str, ChatWidget] = {}
        self.inference_workers: Dict[str, InferenceWorker] = {}
        self.load_worker = None
        self.chat_counter = 0
        
        self._setup_ui()
        self._detect_gpu()
        self._update_all_rag_kbs()

        self.vram_timer = QTimer(); self.vram_timer.timeout.connect(self._refresh_vram); self.vram_timer.start(5000)

    def _setup_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central); main_layout.setContentsMargins(6,6,6,6); main_layout.setSpacing(6)
        splitter = QSplitter(Qt.Horizontal)
        
        # --- LEFT SIDEBAR ---
        sidebar_scroll = QScrollArea(); sidebar_scroll.setWidgetResizable(True); sidebar_scroll.setMaximumWidth(380); sidebar_scroll.setMinimumWidth(300)
        sidebar_widget = QWidget(); sidebar_layout = QVBoxLayout(sidebar_widget); sidebar_layout.setContentsMargins(4,4,4,4); sidebar_layout.setSpacing(8)
        sidebar_layout.addWidget(self._create_gpu_panel())
        sidebar_layout.addWidget(self._create_model_panel())
        sidebar_layout.addWidget(self._create_rag_panel())
        sidebar_layout.addStretch()
        sidebar_scroll.setWidget(sidebar_widget)
        
        # --- RIGHT CHAT AREA ---
        right_frame = QFrame(); right_frame.setObjectName("chatFrame")
        right_layout = QVBoxLayout(right_frame); right_layout.setContentsMargins(4,4,4,4)
        chat_btn_layout = QHBoxLayout()
        new_chat_btn = QPushButton("+ New Chat"); new_chat_btn.setObjectName("successBtn"); new_chat_btn.clicked.connect(self._new_chat)
        chat_btn_layout.addWidget(new_chat_btn); chat_btn_layout.addStretch()
        right_layout.addLayout(chat_btn_layout)
        self.chat_tabs = QTabWidget(); self.chat_tabs.setTabsClosable(True); self.chat_tabs.tabCloseRequested.connect(self._close_chat)
        right_layout.addWidget(self.chat_tabs)
        
        splitter.addWidget(sidebar_scroll); splitter.addWidget(right_frame)
        splitter.setSizes([350, 1050])
        main_layout.addWidget(splitter)
        self.statusBar().showMessage("Ready")

    def _create_gpu_panel(self) -> QWidget:
        group = QGroupBox("🖥️ GPU Status"); layout = QVBoxLayout(group)
        self.gpu_info_label = QLabel("Not detected yet"); self.gpu_info_label.setWordWrap(True); self.gpu_info_label.setObjectName("gpuLabel"); layout.addWidget(self.gpu_info_label)
        self.vram_bar = QProgressBar(); self.vram_bar.setRange(0, 100); self.vram_bar.setValue(0); self.vram_bar.setFormat("VRAM: 0MB / 0MB"); layout.addWidget(self.vram_bar)
        return group

    def _create_model_panel(self) -> QWidget:
        group = QGroupBox("📂 Models"); layout = QVBoxLayout(group)
        path_layout = QHBoxLayout()
        add_path_btn = QPushButton("+ Folder"); add_path_btn.clicked.connect(self._add_model_path)
        scan_btn = QPushButton("🔍 Scan"); scan_btn.setObjectName("successBtn"); scan_btn.clicked.connect(self._scan_models)
        path_layout.addWidget(add_path_btn); path_layout.addWidget(scan_btn); layout.addLayout(path_layout)
        self.model_tree = QTreeWidget(); self.model_tree.setHeaderLabels(["Model", "Size", "Est. VRAM", "Quant"]); self.model_tree.setAlternatingRowColors(True)
        self.model_tree.setColumnWidth(0, 120); self.model_tree.setColumnWidth(1, 50); self.model_tree.setColumnWidth(2, 65); layout.addWidget(self.model_tree)
        
        h1 = QHBoxLayout(); h1.addWidget(QLabel("GPU Layers:"))
        self.ngl_spin = QSpinBox(); self.ngl_spin.setRange(-1, 200); self.ngl_spin.setValue(-1); self.ngl_spin.setSpecialValueText("Auto"); h1.addWidget(self.ngl_spin)
        h1.addWidget(QLabel("Ctx:"))
        self.ctx_spin = QSpinBox(); self.ctx_spin.setRange(512, 32768); self.ctx_spin.setSingleStep(512); self.ctx_spin.setValue(4096); h1.addWidget(self.ctx_spin)
        layout.addLayout(h1)
        
        btn_layout = QHBoxLayout()
        load_btn = QPushButton("⬆ Load"); load_btn.setObjectName("successBtn"); load_btn.clicked.connect(self._load_selected_model)
        unload_btn = QPushButton("⬇ Unload"); unload_btn.setObjectName("dangerBtn"); unload_btn.clicked.connect(self._unload_selected_model)
        btn_layout.addWidget(load_btn); btn_layout.addWidget(unload_btn); layout.addLayout(btn_layout)
        return group

    def _create_rag_panel(self) -> QWidget:
        group = QGroupBox("📄 RAG Knowledge Bases"); layout = QVBoxLayout(group)
        if not self.rag_manager.available:
            layout.addWidget(QLabel("⚠️ Install chromadb & pypdf\npip install chromadb pypdf"))
        else:
            kb_list_layout = QHBoxLayout()
            self.kb_combo = QComboBox(); self._refresh_kb_combo(); kb_list_layout.addWidget(self.kb_combo, 1)
            
            create_kb_btn = QPushButton("+"); create_kb_btn.setFixedWidth(30); create_kb_btn.clicked.connect(self._create_kb)
            del_kb_btn = QPushButton("🗑"); del_kb_btn.setFixedWidth(30); del_kb_btn.setObjectName("dangerBtn"); del_kb_btn.clicked.connect(self._delete_kb)
            kb_list_layout.addWidget(create_kb_btn); kb_list_layout.addWidget(del_kb_btn)
            layout.addLayout(kb_list_layout)

            self.rag_status_label = QLabel("Select a KB and add files"); self.rag_status_label.setObjectName("gpuLabel"); layout.addWidget(self.rag_status_label)
            
            add_doc_btn = QPushButton("+ Add Files (PDF/TXT/Code)"); add_doc_btn.clicked.connect(self._add_rag_doc); layout.addWidget(add_doc_btn)
            add_folder_btn = QPushButton("+ Add Folder"); add_folder_btn.clicked.connect(self._add_rag_folder); layout.addWidget(add_folder_btn)
            
        return group

    # --- GPU & Model Actions ---
    def _detect_gpu(self): self.gpu_detector.detect(); self._refresh_vram()
    def _refresh_vram(self):
        self.gpu_detector.refresh_vram()
        if self.gpu_detector.gpus:
            g = self.gpu_detector.gpus[0]
            self.gpu_info_label.setText(f"{g.name}\nDriver: {g.driver_version} | Temp: {g.temperature_c}°C")
            total, free = self.gpu_detector.get_total_vram(), self.gpu_detector.get_free_vram()
            pct = int((total - free) / total * 100) if total else 0
            self.vram_bar.setValue(pct); self.vram_bar.setFormat(f"VRAM: {free}MB free / {total}MB")
        else:
            self.gpu_info_label.setText("No GPU detected. CPU mode.")
            self.vram_bar.setValue(0); self.vram_bar.setFormat("CPU Mode")

    def _add_model_path(self):
        d = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if d: self.model_scanner.add_path(d); self._scan_models()

    def _scan_models(self):
        models = self.model_scanner.scan(); self.model_tree.clear()
        for m in models:
            item = QTreeWidgetItem([m.name, f"{m.size_mb:.0f}", f"{m.estimated_vram_mb:.0f}", m.quantization])
            item.setData(0, Qt.UserRole, m)
            if m.name in self.model_manager.loaded_models: item.setForeground(0, QColor("#53d769"))
            self.model_tree.addTopLevelItem(item)

    def _load_selected_model(self):
        item = self.model_tree.currentItem()
        if not item: return
        mi = item.data(0, Qt.UserRole); can, msg = self.model_manager.can_load(mi)
        if not can: QMessageBox.warning(self, "VRAM Warning", msg); return
        ngl = self.ngl_spin.value() if self.ngl_spin.value() != -1 else (-1 if self.gpu_detector.cuda_available else 0)
        self.load_worker = ModelLoadWorker(self.model_manager, mi, ngl, self.ctx_spin.value())
        self.load_worker.finished.connect(self._on_model_loaded); self.statusBar().showMessage(f"Loading {mi.name}..."); self.load_worker.start()

    def _unload_selected_model(self):
        item = self.model_tree.currentItem()
        if not item: return
        mi = item.data(0, Qt.UserRole)
        if mi.name in self.model_manager.loaded_models:
            self.model_manager.unload_model(mi.name); self.statusBar().showMessage(f"Unloaded {mi.name}"); self._scan_models(); self._refresh_vram()

    def _on_model_loaded(self, name, success, msg):
        self.statusBar().showMessage(msg, 5000)
        if success: self._scan_models(); self._refresh_vram()
        else: QMessageBox.critical(self, "Load Error", msg)

    # --- Chat Actions ---
    def _new_chat(self):
        models = list(self.model_manager.loaded_models.keys())
        if not models: QMessageBox.information(self, "No Models", "Please load a model first."); return
        self.chat_counter += 1; cid = f"chat_{self.chat_counter}"
        chat = ChatWidget(cid, models[0]); chat.send_requested.connect(self._on_chat_send); chat.stop_requested.connect(self._on_chat_stop)
        chat.update_rag_kbs(self.rag_manager.list_kbs())
        self.chat_widgets[cid] = chat; self.chat_tabs.addTab(chat, f"💬 {models[0][:12]}"); self.chat_tabs.setCurrentIndex(self.chat_tabs.count() - 1)

    def _close_chat(self, idx):
        widget = self.chat_tabs.widget(idx)
        if widget:
            cid = widget.chat_id
            if cid in self.inference_workers: self.inference_workers[cid].stop(); self.inference_workers[cid].wait()
            self.chat_widgets.pop(cid, None); self.chat_tabs.removeTab(idx)

    def _on_chat_send(self, cid, message):
        chat = self.chat_widgets.get(cid)
        if not chat: return
        prompt = ""
        if chat.messages and chat.messages[0]['role'] == 'system': prompt += f"System: {chat.messages[0]['content']}\n\n"
        
        if chat.current_kb != "None" and self.rag_manager.available:
            context = self.rag_manager.query(message, chat.current_kb, n_results=3)
            if context: prompt += f"Retrieved Context from '{chat.current_kb}':\n{context}\n\nInstructions: Use the above context to answer the user's question.\n\n"
        
        for msg in chat.messages[1:]:
            if msg['role'] == 'user': prompt += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant': prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant:"
        
        chat.start_gen()
        worker = InferenceWorker(self.model_manager, chat.model_name, cid, prompt, {'max_tokens': 1024, 'temperature': 0.7})
        worker.chunk_ready.connect(self._on_chunk); worker.finished.connect(self._on_inference_done); worker.error.connect(self._on_inference_error)
        self.inference_workers[cid] = worker; worker.start()

    def _on_chat_stop(self, cid):
        if cid in self.inference_workers: self.inference_workers[cid].stop()

    def _on_chunk(self, cid, chunk):
        chat = self.chat_widgets.get(cid)
        if chat: chat.append_chunk(chunk)

    def _on_inference_done(self, cid, _):
        chat = self.chat_widgets.get(cid)
        if chat: chat.finish_gen()
        self.inference_workers.pop(cid, None)

    def _on_inference_error(self, cid, err):
        chat = self.chat_widgets.get(cid)
        if chat: chat._append_system_msg(f"Error: {err}"); chat.finish_gen()
        self.inference_workers.pop(cid, None)

    # --- RAG Actions ---
    def _refresh_kb_combo(self):
        self.kb_combo.clear(); self.kb_combo.addItems(self.rag_manager.list_kbs())

    def _update_all_rag_kbs(self):
        kbs = self.rag_manager.list_kbs()
        for chat in self.chat_widgets.values(): chat.update_rag_kbs(kbs)

    def _create_kb(self):
        name, ok = QInputDialog.getText(self, "Create Knowledge Base", "Enter name (alphanumeric, hyphens, underscores):")
        if ok and name:
            sname = self.rag_manager.create_kb(name); self._refresh_kb_combo(); self.kb_combo.setCurrentText(sname)
            self._update_all_rag_kbs(); self.statusBar().showMessage(f"Created KB: {sname}", 3000)

    def _delete_kb(self):
        kb = self.kb_combo.currentText()
        if not kb: return
        if QMessageBox.question(self, "Delete KB", f"Delete '{kb}' and all its documents?") == QMessageBox.Yes:
            self.rag_manager.delete_kb(kb); self._refresh_kb_combo(); self._update_all_rag_kbs()
            self.statusBar().showMessage(f"Deleted KB: {kb}", 3000)

    def _ingest_files(self, files):
        kb = self.kb_combo.currentText()
        if not kb: QMessageBox.warning(self, "No KB", "Select or create a Knowledge Base first."); return
        total_chunks = 0
        for f in files:
            try: total_chunks += self.rag_manager.add_document(f, kb)
            except Exception as e: QMessageBox.warning(self, "Error", f"Failed {f}:\n{e}")
        self.rag_status_label.setText(f"KB '{kb}' chunks: {self.rag_manager.get_document_count(kb)}")
        self.statusBar().showMessage(f"Ingested {len(files)} files ({total_chunks} chunks) into {kb}", 5000)

    def _add_rag_doc(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add Documents", "", "Docs (*.pdf *.txt *.md *.py *.js *.c *.cpp *.rs *.json);;All (*)")
        if files: self._ingest_files(files)

    def _add_rag_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if d:
            exts = {'.pdf','.txt','.md','.py','.js','.c','.cpp','.rs','.json'}
            files = [str(p) for p in Path(d).rglob('*') if p.suffix.lower() in exts]
            self._ingest_files(files)


# ─────────────────────── ENTRY POINT ───────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    font = QFont("Segoe UI", 10); app.setFont(font)
    window = MainWindow(); window.show()
    sys.exit(app.exec())