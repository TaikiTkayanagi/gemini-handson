import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types

EMBEDDING_MODEL = "gemini-embedding-2"
CACHE_FILE = Path(__file__).parent / "phase_embeddings.json"

PHASES: dict[str, str] = {
    "企画": (
        "プロジェクト企画フェーズ。ビジネス要件定義、プロジェクト目的・背景・課題の整理、"
        "スコープ定義、ステークホルダー分析、リスク評価、費用対効果（ROI）試算、"
        "プロジェクト計画書、WBS、スケジュール策定、予算計画、承認・意思決定プロセス。"
        "市場調査、ユーザーニーズ分析、提案書、RFP、プロジェクト憲章。"
    ),
    "設計": (
        "システム設計フェーズ。要件定義書、基本設計、詳細設計、アーキテクチャ設計、"
        "データベース設計、ER図、テーブル定義、API設計、インターフェース仕様書、"
        "画面設計、ワイヤーフレーム、UI/UXデザイン、シーケンス図、クラス図、"
        "インフラ構成図、非機能要件定義、セキュリティ設計、性能設計。"
    ),
    "開発": (
        "システム開発・実装フェーズ。ソースコード実装、プログラミング、コーディング規約、"
        "フロントエンド開発、バックエンド開発、API実装、データベース構築、"
        "単体テスト、コードレビュー、ブランチ管理、プルリクエスト、Git、"
        "CI/CD、ビルド、デプロイメント、環境構築、ライブラリ選定、技術スタック。"
    ),
    "テスト": (
        "テスト・品質保証フェーズ。テスト計画書、テスト仕様書、テストケース設計、"
        "単体テスト、結合テスト、システムテスト、受入テスト（UAT）、回帰テスト、"
        "性能テスト、負荷テスト、セキュリティテスト、バグ票、不具合管理、"
        "テスト報告書、品質指標、カバレッジ、QA、検証・確認。"
    ),
    "リリース・運用": (
        "リリース・運用保守フェーズ。リリース計画、本番環境デプロイ、移行手順書、"
        "カットオーバー、ロールバック手順、監視・アラート設定、運用手順書、"
        "障害対応、インシデント管理、変更管理、SLA、サービスレベル定義、"
        "ヘルプデスク、ユーザートレーニング、保守・改善計画、定期報告。"
    ),
}

phase_vectors: dict[str, list[float]] = {}

HTML_CONTENT = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PDF フェーズ検出</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #f8fafc;
      color: #1e293b;
      min-height: 100vh;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      padding: 40px 16px;
    }
    .container { width: 100%; max-width: 640px; }
    h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: 8px; }
    .subtitle { color: #64748b; margin-bottom: 32px; font-size: 0.95rem; }
    .upload-zone {
      border: 2px dashed #cbd5e1;
      border-radius: 12px;
      padding: 40px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
      background: white;
    }
    .upload-zone:hover, .upload-zone.drag-over {
      border-color: #6366f1;
      background: #f5f3ff;
    }
    .upload-icon { font-size: 2.5rem; margin-bottom: 12px; }
    .upload-text { color: #475569; font-size: 0.95rem; }
    .upload-text strong { color: #6366f1; }
    .file-name { margin-top: 12px; font-size: 0.85rem; color: #6366f1; font-weight: 500; }
    input[type="file"] { display: none; }
    .btn {
      display: block;
      width: 100%;
      margin-top: 16px;
      padding: 14px;
      background: #6366f1;
      color: white;
      border: none;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s, opacity 0.2s;
    }
    .btn:hover:not(:disabled) { background: #4f46e5; }
    .btn:disabled { opacity: 0.6; cursor: not-allowed; }
    .spinner {
      display: inline-block;
      width: 16px; height: 16px;
      border: 2px solid rgba(255,255,255,0.4);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: middle;
      margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .error-box {
      margin-top: 20px;
      padding: 12px 16px;
      background: #fef2f2;
      border: 1px solid #fca5a5;
      border-radius: 8px;
      color: #dc2626;
      font-size: 0.9rem;
      display: none;
    }
    .results { margin-top: 32px; display: none; }
    .result-header { margin-bottom: 20px; }
    .file-info { font-size: 0.8rem; color: #94a3b8; margin-bottom: 12px; }
    .top-phase {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px 20px;
      background: #f0fdf4;
      border: 1.5px solid #86efac;
      border-radius: 10px;
    }
    .top-phase-label { font-size: 0.8rem; color: #16a34a; font-weight: 600; }
    .top-phase-name { font-size: 1.3rem; font-weight: 700; color: #15803d; }
    .top-phase-score { margin-left: auto; font-size: 1.1rem; font-weight: 700; color: #15803d; }
    .divider { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }
    .phase-list-label { font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin: 20px 0 12px; }
    .phase-list { display: flex; flex-direction: column; gap: 12px; }
    .phase-item {
      background: white;
      border-radius: 10px;
      padding: 14px 16px;
      border: 1px solid #e2e8f0;
    }
    .phase-item-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .phase-name { font-weight: 600; font-size: 0.95rem; }
    .phase-score { font-size: 0.9rem; color: #64748b; }
    .bar-track { background: #f1f5f9; border-radius: 999px; height: 8px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; width: 0; transition: width 0.7s ease-in-out; }
    .bar-fill.top { background: #22c55e; }
    .bar-fill.other { background: #6366f1; }
  </style>
</head>
<body>
  <div class="container">
    <h1>PDF フェーズ検出</h1>
    <p class="subtitle">PDFをアップロードすると、プロジェクトのどのフェーズに近い文書かをAIが判定します。</p>
    <div class="upload-zone" id="dropZone">
      <div class="upload-icon">📄</div>
      <p class="upload-text"><strong>クリックしてPDFを選択</strong><br>またはここにドラッグ＆ドロップ</p>
      <p class="file-name" id="fileName"></p>
      <input type="file" id="fileInput" accept=".pdf,application/pdf">
    </div>
    <button class="btn" id="analyzeBtn" disabled>分析開始</button>
    <div class="error-box" id="errorBox"></div>
    <div class="results" id="results">
      <hr class="divider">
      <div class="result-header">
        <div class="file-info" id="fileInfo"></div>
        <div class="top-phase">
          <div>
            <div class="top-phase-label">最も近いフェーズ</div>
            <div class="top-phase-name" id="topPhaseName"></div>
          </div>
          <div class="top-phase-score" id="topPhaseScore"></div>
        </div>
      </div>
      <div class="phase-list-label">全フェーズスコア</div>
      <div class="phase-list" id="phaseList"></div>
    </div>
  </div>
  <script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const errorBox = document.getElementById('errorBox');
    const resultsEl = document.getElementById('results');
    let selectedFile = null;

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f) setFile(f);
    });
    fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

    function setFile(f) {
      selectedFile = f;
      fileName.textContent = f.name;
      analyzeBtn.disabled = false;
      errorBox.style.display = 'none';
      resultsEl.style.display = 'none';
    }

    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) return;
      analyzeBtn.disabled = true;
      analyzeBtn.innerHTML = '<span class="spinner"></span>分析中...';
      errorBox.style.display = 'none';
      resultsEl.style.display = 'none';
      try {
        const form = new FormData();
        form.append('file', selectedFile);
        const res = await fetch('/analyze', { method: 'POST', body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '分析に失敗しました');
        renderResults(data);
      } catch (err) {
        errorBox.textContent = err.message;
        errorBox.style.display = 'block';
      } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = '分析開始';
      }
    });

    function renderResults(data) {
      document.getElementById('fileInfo').textContent = 'ファイル: ' + data.filename;
      const top = data.results[0];
      document.getElementById('topPhaseName').textContent = top.phase;
      document.getElementById('topPhaseScore').textContent = (top.score * 100).toFixed(1) + '%';
      const scores = data.results.map(r => r.score);
      const minScore = Math.min(...scores);
      const maxScore = Math.max(...scores);
      const range = maxScore - minScore || 1;
      const list = document.getElementById('phaseList');
      list.innerHTML = '';
      data.results.forEach((r, i) => {
        const pct = (r.score * 100).toFixed(1);
        const barWidth = (((r.score - minScore) / range) * 100).toFixed(1);
        const isTop = i === 0;
        const item = document.createElement('div');
        item.className = 'phase-item';
        item.innerHTML =
          '<div class="phase-item-header">' +
            '<span class="phase-name">' + r.phase + '</span>' +
            '<span class="phase-score">' + pct + '%</span>' +
          '</div>' +
          '<div class="bar-track">' +
            '<div class="bar-fill ' + (isTop ? 'top' : 'other') + '" data-width="' + barWidth + '%"></div>' +
          '</div>';
        list.appendChild(item);
      });
      resultsEl.style.display = 'block';
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          list.querySelectorAll('.bar-fill').forEach(el => {
            el.style.width = el.dataset.width;
          });
        });
      });
      resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  </script>
</body>
</html>"""


def format_for_embedding(text: str) -> str:
    return f"task: clustering | query: {text}"


def load_phase_embeddings() -> dict[str, list[float]] | None:
    if CACHE_FILE.exists():
        with CACHE_FILE.open() as f:
            return json.load(f)
    return None


def save_phase_embeddings(embeddings: dict[str, list[float]]) -> None:
    with CACHE_FILE.open("w") as f:
        json.dump(embeddings, f)


def compute_phase_embeddings(client: genai.Client) -> dict[str, list[float]]:
    result = {}
    for name, desc in PHASES.items():
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[format_for_embedding(desc)],
        )
        result[name] = response.embeddings[0].values
    return result


def embed_pdf(client: genai.Client, pdf_bytes: bytes) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[
            types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf",
            )
        ],
    )
    return response.embeddings[0].values


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / norm) if norm else 0.0


def rank_phases(pdf_vector: list[float], phase_vecs: dict[str, list[float]]) -> list[dict]:
    scores = [
        {"phase": name, "score": cosine_similarity(pdf_vector, vec)}
        for name, vec in phase_vecs.items()
    ]
    return sorted(scores, key=lambda x: x["score"], reverse=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global phase_vectors
    client = genai.Client()
    cached = load_phase_embeddings()
    if cached and set(cached.keys()) == set(PHASES.keys()):
        phase_vectors = cached
    else:
        phase_vectors = compute_phase_embeddings(client)
        save_phase_embeddings(phase_vectors)
    app.state.genai_client = client
    yield


app = FastAPI(title="PDF Phase Detector", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_CONTENT


@app.post("/analyze")
async def analyze(file: Annotated[UploadFile, File(description="PDF file")]):
    if not (
        file.content_type in ("application/pdf", "application/octet-stream")
        or (file.filename or "").endswith(".pdf")
    ):
        raise HTTPException(400, "PDF ファイルをアップロードしてください")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 20 * 1024 * 1024:
        raise HTTPException(413, "ファイルサイズは20MB以下にしてください")

    client: genai.Client = app.state.genai_client
    pdf_vector = embed_pdf(client, pdf_bytes)
    results = rank_phases(pdf_vector, phase_vectors)

    return JSONResponse({
        "filename": file.filename,
        "results": results,
    })


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
