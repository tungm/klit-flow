"""klit-flow web portal.

Serves a single-page application (SPA) that visualises the dependency and
screen-flow graphs and exposes five REST endpoints consumed by the SPA:

- ``GET /``                     — SPA HTML (no external dependencies).
- ``GET /api/graph``            — all nodes and edges (includes condition field).
- ``GET /api/search?q=&k=``     — hybrid BM25 + semantic search.
- ``GET /api/flows?screen=``    — NAVIGATES_TO edges, optionally filtered.
- ``GET /api/node/{node_id}``   — one node with its inbound/outbound edges.

:func:`create_web_app` builds the FastAPI instance from pre-opened resources
(store, bm25, embedder).  The CLI opens ONE :class:`~klit_flow.graph.store.LadybugGraphStore`
and passes it to both :func:`create_web_app` and ``create_server`` so a single
DB connection is shared between the web thread and the MCP stdio thread.
Thread safety is provided by the ``threading.Lock`` inside ``LadybugGraphStore``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from klit_flow.graph.store import GraphStore, parse_conditions_json
from klit_flow.index.bm25 import BM25Index
from klit_flow.index.search import hybrid_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedded SPA (no CDN, fully offline)
# ---------------------------------------------------------------------------

_SPA_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>klit-flow portal</title>
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    :root{
      --bg:#0f172a;--surface:#1e293b;--border:#334155;
      --text:#e2e8f0;--muted:#64748b;--accent:#7c3aed;--accent-l:#a78bfa;
      --green:#34d399;--blue:#38bdf8;--orange:#fb923c;--pink:#f472b6;--yellow:#fbbf24
    }
    body{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif;
         height:100dvh;display:flex;flex-direction:column;overflow:hidden}

    /* Header */
    header{background:var(--surface);border-bottom:1px solid var(--border);
           padding:0 16px;height:48px;display:flex;align-items:center;gap:12px;flex-shrink:0}
    .logo{font-weight:700;font-size:.9rem;color:var(--accent-l);white-space:nowrap}
    .tabs{display:flex;gap:4px}
    .tab{background:none;border:none;color:var(--muted);padding:6px 12px;border-radius:6px;
         cursor:pointer;font-size:.85rem;transition:background .15s,color .15s}
    .tab:hover{background:#334155;color:var(--text)}
    .tab.active{background:var(--accent);color:#fff}
    #search-input{flex:1;max-width:360px;margin-left:auto;background:var(--bg);
                  border:1px solid var(--border);color:var(--text);padding:6px 12px;
                  border-radius:6px;font-size:.85rem;outline:none}
    #search-input:focus{border-color:var(--accent)}

    /* Layout */
    main{flex:1;display:flex;overflow:hidden;position:relative}
    .view{flex:1;display:none;overflow:hidden}
    .view.active{display:flex}

    /* Canvas views */
    .canvas-wrap{flex:1;position:relative;display:flex}
    canvas{width:100%;height:100%;display:block;cursor:grab}
    canvas:active{cursor:grabbing}
    .graph-controls{position:absolute;top:12px;right:12px;display:flex;flex-direction:column;gap:6px}
    .gc-btn{background:var(--surface);border:1px solid var(--border);color:var(--muted);
            width:32px;height:32px;border-radius:6px;cursor:pointer;font-size:1rem;
            display:flex;align-items:center;justify-content:center}
    .gc-btn:hover{color:var(--text);border-color:var(--accent)}
    .legend{position:absolute;bottom:12px;left:12px;background:#1e293bcc;
            border:1px solid var(--border);border-radius:8px;padding:10px 14px;
            font-size:.75rem;backdrop-filter:blur(4px)}
    .li{display:flex;align-items:center;gap:8px;margin-bottom:4px}
    .li:last-child{margin-bottom:0}
    .dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}

    /* Sidebar */
    #sidebar{width:0;background:var(--surface);border-left:1px solid var(--border);
             overflow:hidden;transition:width .2s ease;flex-shrink:0}
    #sidebar.open{width:280px}
    #sb-inner{width:280px;padding:16px;overflow-y:auto;height:100%}
    #sb-inner .close-btn{float:right;background:none;border:none;color:var(--muted);
                         cursor:pointer;font-size:1rem;padding:0}
    #sb-inner .close-btn:hover{color:var(--text)}
    .sb-badge{display:inline-flex;align-items:center;padding:2px 7px;border-radius:4px;
              font-size:.72rem;font-weight:600;margin-bottom:6px}
    #sb-inner h2{font-size:.9rem;font-weight:600;margin-bottom:2px}
    #sb-inner .meta{font-size:.75rem;color:var(--muted);margin-bottom:12px;word-break:break-all}
    .es{margin-top:14px}
    .es h3{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;
           color:var(--muted);margin-bottom:6px}
    .er{display:flex;justify-content:space-between;align-items:center;padding:5px 0;
        border-bottom:1px solid #33415522;cursor:pointer;font-size:.8rem}
    .er:hover{color:var(--accent-l)}
    .er .et{font-size:.7rem;color:var(--muted)}

    /* Table views */
    .tv{flex:1;overflow-y:auto;padding:20px;flex-direction:column}
    .tv h2{font-size:.95rem;font-weight:600;margin-bottom:14px}
    .fr{display:flex;gap:8px;margin-bottom:12px;align-items:center}
    .fr input{background:var(--bg);border:1px solid var(--border);color:var(--text);
              padding:5px 10px;border-radius:6px;font-size:.8rem;flex:1;max-width:300px;outline:none}
    .fr input:focus{border-color:var(--accent)}
    .fr button{background:var(--surface);border:1px solid var(--border);color:var(--muted);
               padding:5px 10px;border-radius:6px;font-size:.8rem;cursor:pointer}
    .fr button:hover{color:var(--text)}
    table{width:100%;border-collapse:collapse;font-size:.83rem}
    th{text-align:left;padding:8px 12px;background:var(--surface);color:var(--muted);
       font-weight:500;font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;
       border-bottom:1px solid var(--border);position:sticky;top:0;z-index:1}
    td{padding:8px 12px;border-bottom:1px solid #33415533;vertical-align:middle}
    tr:hover td{background:#1e293b66;cursor:pointer}
    .badge{display:inline-flex;align-items:center;padding:2px 7px;border-radius:4px;
           font-size:.72rem;font-weight:600}
    .none{color:var(--muted);font-size:.8rem}
    .empty-td{color:var(--muted);font-size:.85rem;padding:20px 12px}
    .cond-cell{font-size:.78rem;color:#94a3b8;font-style:italic;max-width:200px;
               white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

    /* Screen Flows sidebar */
    #scr-sidebar{width:0;background:var(--surface);border-left:1px solid var(--border);
                 overflow:hidden;transition:width .2s ease;flex-shrink:0}
    #scr-sidebar.open{width:300px}
    #scr-sb-inner{width:300px;padding:16px;overflow-y:auto;height:100%}
    #scr-sb-inner .close-btn{float:right;background:none;border:none;color:var(--muted);
                              cursor:pointer;font-size:1rem;padding:0}
    #scr-sb-inner .close-btn:hover{color:var(--text)}
    .cond-chain{margin-top:10px}
    .cond-chain h3{font-size:.7rem;font-weight:600;text-transform:uppercase;
                   letter-spacing:.06em;color:var(--muted);margin-bottom:6px}
    .cond-step{display:flex;align-items:flex-start;gap:6px;margin-bottom:6px;font-size:.8rem}
    .cond-step-num{min-width:18px;height:18px;border-radius:50%;background:var(--accent);
                   color:#fff;font-size:.65rem;font-weight:700;display:flex;
                   align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}
    .cond-step-text{color:var(--accent-l);line-height:1.4}
    .api-section{margin-top:14px;border-top:1px solid var(--border);padding-top:12px}
    .api-section h3{font-size:.7rem;font-weight:600;text-transform:uppercase;
                    letter-spacing:.06em;color:var(--muted);margin-bottom:6px}
    .api-item{display:flex;align-items:center;gap:8px;padding:5px 0;
              border-bottom:1px solid #33415522;font-size:.8rem}
    .api-item .api-kind{font-size:.68rem;color:var(--muted);min-width:52px}
    .api-item .api-name{color:var(--blue);font-weight:500}
    .api-item .api-file{font-size:.72rem;color:var(--muted)}

    /* Tooltip */
    #tip{position:fixed;background:var(--surface);border:1px solid var(--border);
         border-radius:6px;padding:6px 10px;font-size:.78rem;pointer-events:none;
         opacity:0;transition:opacity .1s;z-index:999;max-width:280px;line-height:1.5}
    #tip.on{opacity:1}

    /* Loading overlay (per-canvas) */
    .loading-overlay{position:absolute;inset:0;display:flex;flex-direction:column;
             align-items:center;justify-content:center;background:var(--bg);gap:12px;z-index:10}
    .loading-overlay p{color:var(--muted);font-size:.85rem}
    .spin{width:32px;height:32px;border:3px solid var(--border);
          border-top-color:var(--accent);border-radius:50%;animation:sp .8s linear infinite}
    @keyframes sp{to{transform:rotate(360deg)}}
  </style>
</head>
<body>
<header>
  <span class="logo">&#x2B21; klit-flow</span>
  <nav class="tabs">
    <button class="tab active" data-view="deps">Dependencies</button>
    <button class="tab" data-view="screens">Screen Flows</button>
    <button class="tab" data-view="flows">Flows</button>
    <button class="tab" data-view="search">Search</button>
  </nav>
  <input id="search-input" type="search" placeholder="Search symbols\u2026" autocomplete="off">
</header>
<main>

  <!-- ── Dependencies canvas ── -->
  <div id="deps-view" class="view active">
    <div class="canvas-wrap">
      <div class="loading-overlay" id="dep-loading"><div class="spin"></div><p>Loading graph\u2026</p></div>
      <canvas id="graph-canvas"></canvas>
      <div class="graph-controls">
        <button class="gc-btn" id="dep-zin" title="Zoom in">+</button>
        <button class="gc-btn" id="dep-zout" title="Zoom out">\u2212</button>
        <button class="gc-btn" id="dep-fit" title="Fit">\u229e</button>
      </div>
      <div class="legend">
        <div class="li"><div class="dot" style="background:var(--green)"></div>Class</div>
        <div class="li"><div class="dot" style="background:var(--blue)"></div>File</div>
        <div class="li"><div class="dot" style="background:var(--orange)"></div>Function</div>
        <div class="li"><div class="dot" style="background:var(--pink)"></div>Method</div>
        <div class="li"><div class="dot" style="background:#94a3b8"></div>Module / Other</div>
      </div>
    </div>
    <div id="sidebar">
      <div id="sb-inner">
        <button class="close-btn" id="sb-close">\u2715</button>
        <div id="sb-content"><p class="none">Click a node to inspect it.</p></div>
      </div>
    </div>
  </div>

  <!-- ── Screen Flows canvas ── -->
  <div id="screens-view" class="view">
    <div class="canvas-wrap">
      <div class="loading-overlay" id="scr-loading" style="display:none"><div class="spin"></div><p>Loading flows\u2026</p></div>
      <canvas id="scr-canvas"></canvas>
      <div class="graph-controls">
        <button class="gc-btn" id="scr-zin" title="Zoom in">+</button>
        <button class="gc-btn" id="scr-zout" title="Zoom out">\u2212</button>
        <button class="gc-btn" id="scr-fit" title="Fit">\u229e</button>
      </div>
      <div class="legend">
        <div class="li"><div class="dot" style="background:var(--accent-l)"></div>Screen</div>
        <div class="li" style="margin-top:6px;gap:6px">
          <div class="dot" style="background:var(--green)"></div><span style="font-size:.7rem;color:var(--muted)">button_tap</span>
        </div>
        <div class="li" style="gap:6px">
          <div class="dot" style="background:var(--blue)"></div><span style="font-size:.7rem;color:var(--muted)">api_response</span>
        </div>
        <div class="li" style="gap:6px">
          <div class="dot" style="background:#64748b"></div><span style="font-size:.7rem;color:var(--muted)">programmatic</span>
        </div>
        <div class="li" style="margin-top:6px;color:var(--muted);font-size:.7rem">Click screen to see API calls</div>
      </div>
    </div>
    <div id="scr-sidebar">
      <div id="scr-sb-inner">
        <button class="close-btn" id="scr-sb-close">\u2715</button>
        <div id="scr-sb-content"><p class="none">Click a screen to inspect it.</p></div>
      </div>
    </div>
  </div>

  <!-- ── Flows table ── -->
  <div id="flows-view" class="view tv" style="display:none">
    <h2>Navigation Flows</h2>
    <div class="fr">
      <input id="flows-q" type="search" placeholder="Filter by screen name\u2026" autocomplete="off">
      <button id="flows-reset">All</button>
    </div>
    <table>
      <thead><tr><th>From</th><th>To</th><th>Trigger</th><th>Condition</th><th>Conf</th></tr></thead>
      <tbody id="flows-body"></tbody>
    </table>
  </div>

  <!-- ── Search ── -->
  <div id="search-view" class="view tv" style="display:none">
    <h2>Search Results</h2>
    <table>
      <thead><tr><th>Kind</th><th>Name</th><th>File</th></tr></thead>
      <tbody id="search-body"><tr><td colspan="3" class="empty-td">Type a query above.</td></tr></tbody>
    </table>
  </div>

</main>
<div id="tip"></div>

<script>
'use strict';
// ── Palette ───────────────────────────────────────────────────────────────────
const KC={Screen:'#a78bfa',Class:'#34d399',File:'#38bdf8',
          Function:'#fb923c',Method:'#f472b6',Module:'#94a3b8',Interface:'#fbbf24'};
const TRIGGER_COLOR={button_tap:'#34d399',api_response:'#38bdf8',programmatic:'#64748b',deep_link:'#fbbf24'};
const NODE_R={Screen:16,File:9,Module:9};const DEF_R=6;
function kc(k){return KC[k]||'#94a3b8'}
function nr(n){return NODE_R[n.kind]||DEF_R}
function tc(t){return TRIGGER_COLOR[t]||'#64748b'}

// ── Helpers ───────────────────────────────────────────────────────────────────
async function api(url){const r=await fetch(url);if(!r.ok)throw new Error('HTTP '+r.status);return r.json()}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
function base(fp){return fp?fp.replace(/\\\\/g,'/').split('/').pop():''}
function badge(kind){const c=kc(kind);return`<span class="badge" style="background:${c}22;color:${c}">${esc(kind)}</span>`}
function trigBadge(t){const c=tc(t||'programmatic');return`<span class="badge" style="background:${c}22;color:${c}">${esc(t||'')}</span>`}

function roundRect(cx,x,y,w,h,r){
  cx.beginPath();cx.moveTo(x+r,y);cx.lineTo(x+w-r,y);
  cx.quadraticCurveTo(x+w,y,x+w,y+r);cx.lineTo(x+w,y+h-r);
  cx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);cx.lineTo(x+r,y+h);
  cx.quadraticCurveTo(x,y+h,x,y+h-r);cx.lineTo(x,y+r);
  cx.quadraticCurveTo(x,y,x+r,y);cx.closePath();
}

// ── Graph data (shared) ───────────────────────────────────────────────────────
let allNodes=[],allEdges=[],nodeById={};
let graphLoaded=false;

async function ensureGraph(){
  if(graphLoaded)return;
  const d=await api('/api/graph');
  allNodes=d.nodes;allEdges=d.edges;
  nodeById=Object.fromEntries(allNodes.map(n=>[n.id,n]));
  graphLoaded=true;
}

// ════════════════════════════════════════════════════════════════════════════════
// ── DEPENDENCIES GRAPH ────────────────────────────────────────────────────────
// ════════════════════════════════════════════════════════════════════════════════
const depCanvas=document.getElementById('graph-canvas');
const depCtx=depCanvas.getContext('2d');
let depNodes=[],depEdges=[];
let depPan={x:0,y:0},depZoom=1,depAlpha=0,depRunning=false;
let depDrag=null,depPanStart=null,depMd=null,depSel=null;

new ResizeObserver(()=>{
  const p=depCanvas.parentElement;
  depCanvas.width=p.clientWidth;depCanvas.height=p.clientHeight;depDraw();
}).observe(depCanvas.parentElement);

async function loadDeps(){
  document.getElementById('dep-loading').style.display='flex';
  try{
    await ensureGraph();
    depNodes=allNodes.filter(n=>n.kind!=='Screen');
    const ids=new Set(depNodes.map(n=>n.id));
    depEdges=allEdges
      .filter(e=>e.type!=='NAVIGATES_TO'&&ids.has(e.src)&&ids.has(e.dst))
      .map(e=>({...e,s:nodeById[e.src],t:nodeById[e.dst]}))
      .filter(e=>e.s&&e.t);
    const W=depCanvas.width||800,H=depCanvas.height||600;
    depNodes.forEach((n,i)=>{
      if(n.x===undefined){
        const a=2*Math.PI*i/depNodes.length;
        n.x=W/2+Math.min(W,H)*.3*Math.cos(a)+(Math.random()-.5)*20;
        n.y=H/2+Math.min(W,H)*.3*Math.sin(a)+(Math.random()-.5)*20;
        n.vx=0;n.vy=0;
      }
    });
    depSim();
    document.getElementById('dep-loading').style.display='none';
  }catch(e){
    document.getElementById('dep-loading').innerHTML=`<p style="color:#f87171">Failed: ${esc(e.message)}</p>`;
  }
}

function depSim(){depAlpha=1;if(!depRunning){depRunning=true;requestAnimationFrame(depStep)}}
function depStep(){
  if(depAlpha<0.008){depRunning=false;depDraw();return}
  depAlpha*=0.96;
  depNodes.forEach(n=>{n.fx=0;n.fy=0});
  for(let i=0;i<depNodes.length;i++){
    const a=depNodes[i];
    for(let j=i+1;j<depNodes.length;j++){
      const b=depNodes[j];
      const dx=b.x-a.x,dy=b.y-a.y,d2=dx*dx+dy*dy||1,d=Math.sqrt(d2);
      const f=3000/d2,fx=dx/d*f,fy=dy/d*f;
      a.fx-=fx;a.fy-=fy;b.fx+=fx;b.fy+=fy;
    }
  }
  for(const e of depEdges){
    const dx=e.t.x-e.s.x,dy=e.t.y-e.s.y,d=Math.sqrt(dx*dx+dy*dy)||1;
    const f=(d-120)*.05,fx=dx/d*f,fy=dy/d*f;
    e.s.fx+=fx;e.s.fy+=fy;e.t.fx-=fx;e.t.fy-=fy;
  }
  const W=depCanvas.width,H=depCanvas.height;
  depNodes.forEach(n=>{n.fx+=(W/2-n.x)*.003;n.fy+=(H/2-n.y)*.003});
  depNodes.forEach(n=>{
    if(n===depDrag)return;
    n.vx=(n.vx+n.fx)*.7;n.vy=(n.vy+n.fy)*.7;
    n.x+=n.vx*depAlpha;n.y+=n.vy*depAlpha;
  });
  depDraw();requestAnimationFrame(depStep);
}

function depDraw(){
  const W=depCanvas.width,H=depCanvas.height;
  depCtx.clearRect(0,0,W,H);
  depCtx.save();depCtx.translate(depPan.x,depPan.y);depCtx.scale(depZoom,depZoom);
  for(const e of depEdges){
    depCtx.beginPath();depCtx.moveTo(e.s.x,e.s.y);depCtx.lineTo(e.t.x,e.t.y);
    depCtx.strokeStyle='#47556955';depCtx.lineWidth=.8;depCtx.stroke();
  }
  for(const n of depNodes){
    const r=nr(n),c=kc(n.kind),sel=depSel&&depSel.id===n.id;
    if(sel){depCtx.beginPath();depCtx.arc(n.x,n.y,r+5,0,Math.PI*2);depCtx.fillStyle=c+'33';depCtx.fill()}
    depCtx.beginPath();depCtx.arc(n.x,n.y,r,0,Math.PI*2);
    depCtx.fillStyle=sel?c:c+'99';depCtx.fill();
    depCtx.strokeStyle=c;depCtx.lineWidth=sel?2:1;depCtx.stroke();
    if(depZoom>.6&&n.kind!=='Method'&&n.kind!=='Function'){
      const lbl=n.name.length>16?n.name.slice(0,14)+'\u2026':n.name;
      depCtx.font=`${Math.round(10/depZoom)}px system-ui`;
      depCtx.textAlign='center';depCtx.fillStyle='#cbd5e1';
      depCtx.fillText(lbl,n.x,n.y+r+11/depZoom);
    }
  }
  depCtx.restore();
}

function depWp(cx,cy){const rc=depCanvas.getBoundingClientRect();return{x:(cx-rc.left-depPan.x)/depZoom,y:(cy-rc.top-depPan.y)/depZoom}}
function depHit(pt){for(let i=depNodes.length-1;i>=0;i--){const n=depNodes[i],r=nr(n)+4;if((n.x-pt.x)**2+(n.y-pt.y)**2<=r*r)return n}return null}

depCanvas.addEventListener('mousedown',e=>{
  e.preventDefault();depMd={cx:e.clientX,cy:e.clientY};
  const h=depHit(depWp(e.clientX,e.clientY));
  if(h)depDrag=h;else depPanStart={cx:e.clientX,cy:e.clientY,px:depPan.x,py:depPan.y};
});
depCanvas.addEventListener('mousemove',e=>{
  if(depDrag){const p=depWp(e.clientX,e.clientY);depDrag.x=p.x;depDrag.y=p.y;depDrag.vx=0;depDrag.vy=0;if(!depRunning)depDraw();return}
  if(depPanStart){depPan.x=depPanStart.px+(e.clientX-depPanStart.cx);depPan.y=depPanStart.py+(e.clientY-depPanStart.cy);if(!depRunning)depDraw();return}
  const h=depHit(depWp(e.clientX,e.clientY));
  const tip=document.getElementById('tip');
  if(h){
    tip.innerHTML=`<strong>${esc(h.name)}</strong><br><span style="color:#64748b">${esc(h.kind)} \u00b7 ${esc(base(h.file))}</span>`;
    tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-10)+'px';tip.classList.add('on');
  }else tip.classList.remove('on');
});
depCanvas.addEventListener('mouseup',e=>{
  const dn=depDrag;depDrag=null;depPanStart=null;
  if(!depMd)return;const dx=e.clientX-depMd.cx,dy=e.clientY-depMd.cy;depMd=null;
  if(dx*dx+dy*dy>25)return;
  const h=depHit(depWp(e.clientX,e.clientY));
  if(h)depSelNode(h);else{depSel=null;if(!depRunning)depDraw()}
});
depCanvas.addEventListener('mouseleave',()=>{document.getElementById('tip').classList.remove('on');depDrag=null;depPanStart=null});
depCanvas.addEventListener('wheel',e=>{
  e.preventDefault();const f=e.deltaY>0?.88:1.14;
  const rc=depCanvas.getBoundingClientRect(),cx=e.clientX-rc.left,cy=e.clientY-rc.top;
  depPan.x=cx-(cx-depPan.x)*f;depPan.y=cy-(cy-depPan.y)*f;
  depZoom=Math.max(.15,Math.min(5,depZoom*f));if(!depRunning)depDraw();
},{passive:false});

document.getElementById('dep-zin').addEventListener('click',()=>{depZoom=Math.min(5,depZoom*1.25);if(!depRunning)depDraw()});
document.getElementById('dep-zout').addEventListener('click',()=>{depZoom=Math.max(.15,depZoom/1.25);if(!depRunning)depDraw()});
document.getElementById('dep-fit').addEventListener('click',depFit);
function depFit(){
  if(!depNodes.length)return;
  const xs=depNodes.map(n=>n.x),ys=depNodes.map(n=>n.y);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const W=depCanvas.width,H=depCanvas.height,pad=60;
  depZoom=Math.max(.15,Math.min(5,Math.min((W-pad*2)/(x1-x0||1),(H-pad*2)/(y1-y0||1))));
  depPan.x=W/2-(x0+x1)/2*depZoom;depPan.y=H/2-(y0+y1)/2*depZoom;
  if(!depRunning)depDraw();
}

async function depSelNode(n){
  depSel=n;if(!depRunning)depDraw();
  document.getElementById('sidebar').classList.add('open');
  document.getElementById('sb-content').innerHTML='<p class="none">Loading\u2026</p>';
  try{
    const d=await api(`/api/node/${encodeURIComponent(n.id)}`);
    renderDet(d);
  }catch{document.getElementById('sb-content').innerHTML='<p class="none">Error.</p>'}
}
document.getElementById('sb-close').addEventListener('click',()=>{
  document.getElementById('sidebar').classList.remove('open');depSel=null;if(!depRunning)depDraw();
});

function renderDet({node,outbound,inbound}){
  const c=kc(node.kind);
  document.getElementById('sb-content').innerHTML=`
    <span class="sb-badge" style="background:${c}22;color:${c}">${esc(node.kind)}</span>
    <h2>${esc(node.name)}</h2>
    <div class="meta">${esc(base(node.file))} \u00b7 L${node.start_line}\u2013${node.end_line} \u00b7 ${esc(node.language||'')}</div>
    <div class="es"><h3>Outbound (${outbound.length})</h3>
      ${outbound.length?outbound.map(e=>`<div class="er" onclick="depJump('${esc(e.id)}')"><span>${esc(e.name)}</span><span class="et">${esc(e.type)}</span></div>`).join(''):'<span class="none">None</span>'}
    </div>
    <div class="es"><h3>Inbound (${inbound.length})</h3>
      ${inbound.length?inbound.map(e=>`<div class="er" onclick="depJump('${esc(e.id)}')"><span>${esc(e.name)}</span><span class="et">${esc(e.type)}</span></div>`).join(''):'<span class="none">None</span>'}
    </div>`;
}
function depJump(id){
  const n=depNodes.find(n=>n.id===id);
  if(!n){alert('Node not in dependency view.');return}
  depPan.x=depCanvas.width/2-n.x*depZoom;depPan.y=depCanvas.height/2-n.y*depZoom;
  depSelNode(n);if(!depRunning)depDraw();
}

// ════════════════════════════════════════════════════════════════════════════════
// ── SCREEN FLOWS GRAPH ────────────────────────────────────────────────────────
// ════════════════════════════════════════════════════════════════════════════════
const scrCanvas=document.getElementById('scr-canvas');
const scrCtx=scrCanvas.getContext('2d');
let scrNodes=[],scrEdges=[];
let scrPan={x:0,y:0},scrZoom=1,scrAlpha=0,scrRunning=false;
let scrDrag=null,scrPanStart=null,scrMd=null,scrSel=null;
let scrLoaded=false;

new ResizeObserver(()=>{
  const p=scrCanvas.parentElement;
  scrCanvas.width=p.clientWidth;scrCanvas.height=p.clientHeight;scrDraw();
}).observe(scrCanvas.parentElement);

async function loadScreenFlows(){
  if(scrLoaded)return;
  document.getElementById('scr-loading').style.display='flex';
  try{
    await ensureGraph();
    scrNodes=allNodes.filter(n=>n.kind==='Screen');
    const ids=new Set(scrNodes.map(n=>n.id));
    scrEdges=allEdges
      .filter(e=>e.type==='NAVIGATES_TO'&&ids.has(e.src)&&ids.has(e.dst))
      .map(e=>({...e,s:nodeById[e.src],t:nodeById[e.dst]}))
      .filter(e=>e.s&&e.t);
    const W=scrCanvas.width||800,H=scrCanvas.height||600;
    scrNodes.forEach((n,i)=>{
      const a=2*Math.PI*i/scrNodes.length;
      n.sx=W/2+Math.min(W,H)*.32*Math.cos(a)+(Math.random()-.5)*20;
      n.sy=H/2+Math.min(W,H)*.32*Math.sin(a)+(Math.random()-.5)*20;
      n.svx=0;n.svy=0;
    });
    scrSim();
    document.getElementById('scr-loading').style.display='none';
    scrLoaded=true;
  }catch(e){
    document.getElementById('scr-loading').innerHTML=`<p style="color:#f87171">Failed: ${esc(e.message)}</p>`;
  }
}

function scrSim(){scrAlpha=1;if(!scrRunning){scrRunning=true;requestAnimationFrame(scrStep)}}
function scrStep(){
  if(scrAlpha<0.008){scrRunning=false;scrDraw();return}
  scrAlpha*=0.96;
  scrNodes.forEach(n=>{n.sfx=0;n.sfy=0});
  for(let i=0;i<scrNodes.length;i++){
    const a=scrNodes[i];
    for(let j=i+1;j<scrNodes.length;j++){
      const b=scrNodes[j];
      const dx=b.sx-a.sx,dy=b.sy-a.sy,d2=dx*dx+dy*dy||1,d=Math.sqrt(d2);
      const f=6000/d2,fx=dx/d*f,fy=dy/d*f;
      a.sfx-=fx;a.sfy-=fy;b.sfx+=fx;b.sfy+=fy;
    }
  }
  for(const e of scrEdges){
    const dx=e.t.sx-e.s.sx,dy=e.t.sy-e.s.sy,d=Math.sqrt(dx*dx+dy*dy)||1;
    const f=(d-180)*.04,fx=dx/d*f,fy=dy/d*f;
    e.s.sfx+=fx;e.s.sfy+=fy;e.t.sfx-=fx;e.t.sfy-=fy;
  }
  const W=scrCanvas.width,H=scrCanvas.height;
  scrNodes.forEach(n=>{n.sfx+=(W/2-n.sx)*.002;n.sfy+=(H/2-n.sy)*.002});
  scrNodes.forEach(n=>{
    if(n===scrDrag)return;
    n.svx=(n.svx+n.sfx)*.7;n.svy=(n.svy+n.sfy)*.7;
    n.sx+=n.svx*scrAlpha;n.sy+=n.svy*scrAlpha;
  });
  scrDraw();requestAnimationFrame(scrStep);
}

function scrBezierMid(s,t){
  const cpx=(s.sx+t.sx)/2-(t.sy-s.sy)*.2;
  const cpy=(s.sy+t.sy)/2+(t.sx-s.sx)*.2;
  return{
    lx:.25*s.sx+.5*cpx+.25*t.sx,
    ly:.25*s.sy+.5*cpy+.25*t.sy,
    cpx,cpy
  };
}

function scrDraw(){
  const W=scrCanvas.width,H=scrCanvas.height;
  scrCtx.clearRect(0,0,W,H);
  scrCtx.save();scrCtx.translate(scrPan.x,scrPan.y);scrCtx.scale(scrZoom,scrZoom);

  for(const e of scrEdges){
    const {cpx,cpy}=scrBezierMid(e.s,e.t);
    scrCtx.beginPath();scrCtx.moveTo(e.s.sx,e.s.sy);
    scrCtx.quadraticCurveTo(cpx,cpy,e.t.sx,e.t.sy);
    const col=tc(e.trigger);
    scrCtx.strokeStyle=col+'cc';scrCtx.lineWidth=1.8;scrCtx.stroke();
    scrDrawArrow(e.s,e.t,col);
    scrDrawEdgeLabel(e);
  }

  for(const n of scrNodes){
    const r=16,c='#a78bfa',sel=scrSel&&scrSel.id===n.id;
    if(sel){scrCtx.beginPath();scrCtx.arc(n.sx,n.sy,r+6,0,Math.PI*2);scrCtx.fillStyle=c+'33';scrCtx.fill()}
    scrCtx.beginPath();scrCtx.arc(n.sx,n.sy,r,0,Math.PI*2);
    scrCtx.fillStyle=sel?c:c+'99';scrCtx.fill();
    scrCtx.strokeStyle=c;scrCtx.lineWidth=sel?2.5:1.5;scrCtx.stroke();
    const lbl=n.name.length>18?n.name.slice(0,16)+'\u2026':n.name;
    scrCtx.font=`bold ${Math.round(11/scrZoom)}px system-ui`;
    scrCtx.textAlign='center';scrCtx.fillStyle='#e2e8f0';
    scrCtx.fillText(lbl,n.sx,n.sy+r+13/scrZoom);
  }
  scrCtx.restore();
}

function scrDrawArrow(s,t,col){
  const ang=Math.atan2(t.sy-s.sy,t.sx-s.sx);
  const r=18,ax=t.sx-Math.cos(ang)*r,ay=t.sy-Math.sin(ang)*r;
  scrCtx.beginPath();
  scrCtx.moveTo(ax-Math.cos(ang-.42)*9,ay-Math.sin(ang-.42)*9);
  scrCtx.lineTo(ax,ay);
  scrCtx.lineTo(ax-Math.cos(ang+.42)*9,ay-Math.sin(ang+.42)*9);
  scrCtx.strokeStyle=(col||'#a78bfa')+'dd';scrCtx.lineWidth=1.8;scrCtx.stroke();
}

function scrDrawEdgeLabel(e){
  const {lx,ly}=scrBezierMid(e.s,e.t);
  const conds=e.conditions||[];
  if(!conds.length&&!e.trigger)return;
  const first=conds.length?conds[0].expression:e.trigger||'';
  const extra=conds.length>1?` +${conds.length-1}`:'';
  const short=(first.length>22?first.slice(0,20)+'\u2026':first)+extra;
  scrCtx.save();
  scrCtx.font=`${Math.round(9/scrZoom)}px system-ui`;
  const tw=scrCtx.measureText(short).width;
  const pad=5/scrZoom,h=14/scrZoom;
  const bx=lx-tw/2-pad,by=ly-h/2-2/scrZoom,bw=tw+pad*2,bh=h+4/scrZoom;
  const col=tc(e.trigger);
  scrCtx.fillStyle='#0f172add';
  roundRect(scrCtx,bx,by,bw,bh,4/scrZoom);scrCtx.fill();
  scrCtx.strokeStyle=col+'77';scrCtx.lineWidth=.8/scrZoom;scrCtx.stroke();
  // If nested, draw a small stacked indicator dot
  if(conds.length>1){
    scrCtx.fillStyle=col+'aa';
    roundRect(scrCtx,bx+bw+2/scrZoom,by+bh/4,6/scrZoom,bh/2,2/scrZoom);
    scrCtx.fill();
  }
  scrCtx.fillStyle=col;scrCtx.textAlign='center';scrCtx.textBaseline='middle';
  scrCtx.fillText(short,lx,ly);
  scrCtx.restore();
}

function scrWp(cx,cy){const rc=scrCanvas.getBoundingClientRect();return{x:(cx-rc.left-scrPan.x)/scrZoom,y:(cy-rc.top-scrPan.y)/scrZoom}}
function scrHitNode(pt){for(let i=scrNodes.length-1;i>=0;i--){const n=scrNodes[i],r=20;if((n.sx-pt.x)**2+(n.sy-pt.y)**2<=r*r)return n}return null}
function scrHitEdge(pt){
  for(const e of scrEdges){
    const {lx,ly}=scrBezierMid(e.s,e.t);
    if((pt.x-lx)**2+(pt.y-ly)**2<=20*20)return e;
  }
  return null;
}

scrCanvas.addEventListener('mousedown',e=>{
  e.preventDefault();scrMd={cx:e.clientX,cy:e.clientY};
  const h=scrHitNode(scrWp(e.clientX,e.clientY));
  if(h)scrDrag=h;else scrPanStart={cx:e.clientX,cy:e.clientY,px:scrPan.x,py:scrPan.y};
});
scrCanvas.addEventListener('mousemove',e=>{
  if(scrDrag){const p=scrWp(e.clientX,e.clientY);scrDrag.sx=p.x;scrDrag.sy=p.y;scrDrag.svx=0;scrDrag.svy=0;if(!scrRunning)scrDraw();return}
  if(scrPanStart){scrPan.x=scrPanStart.px+(e.clientX-scrPanStart.cx);scrPan.y=scrPanStart.py+(e.clientY-scrPanStart.cy);if(!scrRunning)scrDraw();return}
  const pt=scrWp(e.clientX,e.clientY);
  const hn=scrHitNode(pt);
  const he=hn?null:scrHitEdge(pt);
  const tip=document.getElementById('tip');
  if(hn){
    tip.innerHTML=`<strong>${esc(hn.name)}</strong><br><span style="color:#64748b">Screen \u00b7 click to inspect</span>`;
    tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-10)+'px';tip.classList.add('on');
  }else if(he){
    // Build condition chain display from structured conditions
    const conds=he.conditions||[];
    const chainHtml=conds.length>1
      ?`<div style="margin-top:5px;border-top:1px solid #334155;padding-top:5px">`
       +conds.map((c,i)=>`<div style="display:flex;gap:5px;margin-bottom:3px">
          <span style="background:#7c3aed;color:#fff;border-radius:50%;width:14px;height:14px;font-size:.6rem;
                       display:inline-flex;align-items:center;justify-content:center;flex-shrink:0">${i+1}</span>
          <span style="color:${c.kind==='else'?'#fb923c':'#a78bfa'};font-size:.78rem">[${esc(c.kind)}] ${esc(c.expression)}</span></div>`).join('')+`</div>`
      :conds.length===1?`<br><em style="color:#a78bfa;font-size:.78rem">[${esc(conds[0].kind)}] ${esc(conds[0].expression)}</em>`:'';
    tip.innerHTML=`<strong>${esc(he.s.name)} \u2192 ${esc(he.t.name)}</strong><br>
      <span style="color:${tc(he.trigger)}">${esc(he.trigger||'')}</span>
      <span style="color:#64748b"> \u00b7 ${(he.confidence*100).toFixed(0)}% conf</span>
      ${chainHtml}`;
    tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-10)+'px';tip.classList.add('on');
  }else tip.classList.remove('on');
});
scrCanvas.addEventListener('mouseup',e=>{
  const dn=scrDrag;scrDrag=null;scrPanStart=null;
  if(!scrMd)return;const dx=e.clientX-scrMd.cx,dy=e.clientY-scrMd.cy;scrMd=null;
  if(dx*dx+dy*dy>25)return;
  const h=scrHitNode(scrWp(e.clientX,e.clientY));
  scrSel=h||null;if(!scrRunning)scrDraw();
  if(h)scrClickScreen(h);
  else{document.getElementById('scr-sidebar').classList.remove('open')}
});

async function scrClickScreen(n){
  document.getElementById('scr-sidebar').classList.add('open');
  document.getElementById('scr-sb-content').innerHTML='<p class="none">Loading\u2026</p>';
  try{
    // Outbound navigation edges from this screen
    const nd=await api(`/api/node/${encodeURIComponent(n.id)}`);
    const navOut=nd.outbound.filter(e=>e.type==='NAVIGATES_TO');
    // API dependencies (transitive graph traversal)
    const ap=await api(`/api/screen-apis/${encodeURIComponent(n.id)}`);
    const apiDeps=ap.api_deps||[];

    // Build outbound edges section
    const edgesHtml=navOut.length
      ?navOut.map(e=>{
          const targetEdge=scrEdges.find(se=>se.dst===e.id&&se.src===n.id);
          const conds=targetEdge?.conditions||e.conditions||[];
          const chainHtml=conds.length
            ?`<div class="cond-chain"><h3>Condition chain (${conds.length} level${conds.length>1?'s':''})</h3>`
              +conds.map((c,i)=>`<div class="cond-step">
                  <span class="cond-step-num">${i+1}</span>
                  <span class="cond-step-text"><span style="color:${c.kind==='else'?'#fb923c':'#64748b'};font-size:.7rem">[${esc(c.kind)}]</span> ${esc(c.expression)}${c.source_line?' <span style="color:#64748b;font-size:.68rem">L'+c.source_line+'</span>':''}</span></div>`).join('')
              +`</div>`:'';
          return`<div style="padding:8px 0;border-bottom:1px solid #33415533">
            <div style="display:flex;justify-content:space-between;align-items:center">
              <span style="font-size:.82rem;font-weight:500">${esc(e.name)}</span>
              ${targetEdge?trigBadge(targetEdge.trigger):''}
            </div>${chainHtml}</div>`;
        }).join('')
      :'<p class="none">No outbound navigation.</p>';

    // Build API deps section
    const apiHtml=apiDeps.length
      ?apiDeps.map(a=>`<div class="api-item">
          <span class="api-kind">${esc(a.kind)}</span>
          <div style="flex:1;min-width:0">
            <div class="api-name">${esc(a.name)}</div>
            <div class="api-file">${esc(base(a.file))}</div>
          </div></div>`).join('')
      :'<p class="none" style="margin-top:4px">No API dependencies found within 5 hops.</p>';

    const c='#a78bfa';
    document.getElementById('scr-sb-content').innerHTML=`
      <span class="sb-badge" style="background:${c}22;color:${c}">Screen</span>
      <h2 style="font-size:.9rem;font-weight:600;margin-bottom:12px">${esc(n.name)}</h2>
      <div class="es"><h3>Navigates to (${navOut.length})</h3>${edgesHtml}</div>
      <div class="api-section">
        <h3>API calls (${apiDeps.length} found, \u22645 hops)</h3>${apiHtml}
      </div>`;
  }catch(err){
    document.getElementById('scr-sb-content').innerHTML=`<p class="none">Error: ${esc(err.message)}</p>`;
  }
}
document.getElementById('scr-sb-close').addEventListener('click',()=>{
  document.getElementById('scr-sidebar').classList.remove('open');
  scrSel=null;if(!scrRunning)scrDraw();
});
scrCanvas.addEventListener('mouseleave',()=>{document.getElementById('tip').classList.remove('on');scrDrag=null;scrPanStart=null});
scrCanvas.addEventListener('wheel',e=>{
  e.preventDefault();const f=e.deltaY>0?.88:1.14;
  const rc=scrCanvas.getBoundingClientRect(),cx=e.clientX-rc.left,cy=e.clientY-rc.top;
  scrPan.x=cx-(cx-scrPan.x)*f;scrPan.y=cy-(cy-scrPan.y)*f;
  scrZoom=Math.max(.1,Math.min(5,scrZoom*f));if(!scrRunning)scrDraw();
},{passive:false});

document.getElementById('scr-zin').addEventListener('click',()=>{scrZoom=Math.min(5,scrZoom*1.25);if(!scrRunning)scrDraw()});
document.getElementById('scr-zout').addEventListener('click',()=>{scrZoom=Math.max(.1,scrZoom/1.25);if(!scrRunning)scrDraw()});
document.getElementById('scr-fit').addEventListener('click',scrFit);
function scrFit(){
  if(!scrNodes.length)return;
  const xs=scrNodes.map(n=>n.sx),ys=scrNodes.map(n=>n.sy);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const W=scrCanvas.width,H=scrCanvas.height,pad=80;
  scrZoom=Math.max(.1,Math.min(5,Math.min((W-pad*2)/(x1-x0||1),(H-pad*2)/(y1-y0||1))));
  scrPan.x=W/2-(x0+x1)/2*scrZoom;scrPan.y=H/2-(y0+y1)/2*scrZoom;
  if(!scrRunning)scrDraw();
}

// ════════════════════════════════════════════════════════════════════════════════
// ── VIEWS ────────────────────────────────────────────────────────────────────
// ════════════════════════════════════════════════════════════════════════════════
const VIEWS=['deps','screens','flows','search'];
function showView(name){
  VIEWS.forEach(v=>{
    const el=document.getElementById(v+'-view');
    const isCanvas=v==='deps'||v==='screens';
    const active=v===name;
    el.classList.toggle('active',active);
    el.style.display=active?(isCanvas?'flex':'flex'):'none';
  });
  document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b.dataset.view===name));
  if(name==='screens')loadScreenFlows();
  if(name==='flows')loadFlows();
}
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>showView(b.dataset.view)));

// ════════════════════════════════════════════════════════════════════════════════
// ── FLOWS TABLE ───────────────────────────────────────────────────────────────
// ════════════════════════════════════════════════════════════════════════════════
async function loadFlows(screen=''){
  const url=screen?`/api/flows?screen=${encodeURIComponent(screen)}`:'/api/flows';
  try{
    const d=await api(url);
    const tb=document.getElementById('flows-body');
    if(!d.flows.length){
      tb.innerHTML='<tr><td colspan="5" class="empty-td">No navigation edges found.</td></tr>';return;
    }
    tb.innerHTML=d.flows.map(f=>{
      const conds=f.conditions||[];
      const condText=conds.length?conds.map(c=>`[${c.kind}] ${c.expression}`).join(' \u2192 '):'\u2013';
      const condTitle=conds.length?conds.map((c,i)=>`${i+1}. [${c.kind}] ${c.expression}`).join('\\n'):'';
      return`<tr onclick="showFlowFor('${esc(f.from)}')">
      <td>${esc(f.from)}</td><td>${esc(f.to)}</td>
      <td>${trigBadge(f.trigger)}</td>
      <td class="cond-cell" title="${esc(condTitle)}">${esc(condText)}</td>
      <td>${(f.confidence*100).toFixed(0)}%</td></tr>`;
    }).join('');
  }catch(e){
    document.getElementById('flows-body').innerHTML=`<tr><td colspan="5" class="empty-td">Error: ${esc(e.message)}</td></tr>`;
  }
}
function showFlowFor(s){document.getElementById('flows-q').value=s;loadFlows(s)}
document.getElementById('flows-q').addEventListener('input',e=>{const v=e.target.value.trim();loadFlows(v||'')});
document.getElementById('flows-reset').addEventListener('click',()=>{document.getElementById('flows-q').value='';loadFlows()});

// ════════════════════════════════════════════════════════════════════════════════
// ── SEARCH ────────────────────────────────────────────────────────────────────
// ════════════════════════════════════════════════════════════════════════════════
let stimer;
document.getElementById('search-input').addEventListener('input',e=>{
  clearTimeout(stimer);const q=e.target.value.trim();
  if(!q)return;showView('search');
  stimer=setTimeout(()=>doSearch(q),280);
});
async function doSearch(q){
  document.getElementById('search-body').innerHTML='<tr><td colspan="3" class="empty-td">Searching\u2026</td></tr>';
  try{
    const d=await api(`/api/search?q=${encodeURIComponent(q)}`);
    if(!d.results.length){document.getElementById('search-body').innerHTML='<tr><td colspan="3" class="empty-td">No results.</td></tr>';return}
    document.getElementById('search-body').innerHTML=d.results.map(r=>`
      <tr onclick="jumpFromSearch('${esc(r.id)}')">
        <td>${badge(r.kind)}</td><td>${esc(r.name)}</td>
        <td style="color:#64748b;font-size:.78rem">${esc(base(r.file))}</td>
      </tr>`).join('');
  }catch(e){
    document.getElementById('search-body').innerHTML=`<tr><td colspan="3" class="empty-td">Error: ${esc(e.message)}</td></tr>`;
  }
}
function jumpFromSearch(id){showView('deps');depJump(id)}

// ── Boot ──────────────────────────────────────────────────────────────────────
loadDeps();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def create_web_app(store: GraphStore, bm25: BM25Index, embedder: Any) -> FastAPI:
    """Return a FastAPI application serving the SPA and REST API."""
    app = FastAPI(title="klit-flow portal", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def spa() -> str:
        return _SPA_HTML

    @app.get("/api/graph")
    async def get_graph() -> dict[str, Any]:
        nodes = store.query(
            "MATCH (n:KlitNode) RETURN n.id, n.kind, n.name, n.file_path, n.start_line"
        )
        edges = store.query(
            "MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode) "
            "RETURN a.id, b.id, e.type, e.confidence, e.trigger, e.condition"
        )
        return {
            "nodes": [
                {"id": r[0], "kind": r[1], "name": r[2], "file": r[3], "start_line": r[4]}
                for r in nodes
            ],
            "edges": [
                {
                    "src": r[0],
                    "dst": r[1],
                    "type": r[2],
                    "confidence": r[3],
                    "trigger": r[4],
                    "conditions": parse_conditions_json(r[5]),
                }
                for r in edges
            ],
        }

    @app.get("/api/search")
    async def search(q: str, k: int = 10) -> dict[str, Any]:
        node_ids = hybrid_search(q, bm25, store, embedder, k=k)
        results = []
        for nid in node_ids:
            rows = store.query(
                f"MATCH (n:KlitNode {{id: '{_esc(nid)}'}}) RETURN n.id, n.kind, n.name, n.file_path"
            )
            if rows:
                r = rows[0]
                results.append({"id": r[0], "kind": r[1], "name": r[2], "file": r[3]})
        return {"results": results}

    @app.get("/api/flows")
    async def get_flows(screen: str = "") -> dict[str, Any]:
        if screen:
            s = _esc(screen)
            rows = store.query(
                f"MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode) "
                f"WHERE e.type = 'NAVIGATES_TO' AND (a.name = '{s}' OR b.name = '{s}') "
                f"RETURN a.name, b.name, e.trigger, e.condition, e.confidence "
                f"ORDER BY a.name, b.name"
            )
        else:
            rows = store.query(
                "MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode) "
                "WHERE e.type = 'NAVIGATES_TO' "
                "RETURN a.name, b.name, e.trigger, e.condition, e.confidence "
                "ORDER BY a.name, b.name"
            )
        return {
            "flows": [
                {
                    "from": r[0],
                    "to": r[1],
                    "trigger": r[2],
                    "conditions": parse_conditions_json(r[3]),
                    "confidence": r[4],
                }
                for r in rows
            ]
        }

    @app.get("/api/screen-apis/{screen_id}")
    async def get_screen_apis(screen_id: str) -> dict[str, Any]:
        """Return API/Service/Repository nodes reachable from *screen_id* within 5 hops."""
        nid = _esc(screen_id)
        screen_rows = store.query(f"MATCH (n:KlitNode {{id: '{nid}'}}) RETURN n.name, n.file_path")
        if not screen_rows:
            raise HTTPException(status_code=404, detail="Screen not found")
        screen_name, screen_file = screen_rows[0][0], screen_rows[0][1]

        # All non-navigation edges as adjacency list
        edge_rows = store.query(
            "MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode) "
            "WHERE e.type <> 'NAVIGATES_TO' RETURN a.id, b.id"
        )
        adj: dict[str, list[str]] = {}
        for row in edge_rows:
            adj.setdefault(row[0], []).append(row[1])

        # All non-Screen nodes for lookup
        node_rows = store.query(
            "MATCH (n:KlitNode) WHERE n.kind <> 'Screen' RETURN n.id, n.name, n.kind, n.file_path"
        )
        nodes_by_id = {
            r[0]: {"id": r[0], "name": r[1], "kind": r[2], "file": r[3]} for r in node_rows
        }

        # BFS seeds: the Class and File nodes corresponding to this screen
        seed_rows = store.query(
            f"MATCH (n:KlitNode) "
            f"WHERE (n.name = '{_esc(screen_name)}' AND n.kind = 'Class') "
            f"OR (n.file_path = '{_esc(screen_file)}' AND n.kind = 'File') "
            f"RETURN n.id"
        )
        visited: set[str] = {screen_id} | {r[0] for r in seed_rows}
        queue: list[str] = [
            nid2 for sid in visited for nid2 in adj.get(sid, []) if nid2 not in visited
        ]

        _API_KW = {"api", "service", "repository", "client", "http", "retrofit", "volley"}
        api_nodes: list[dict] = []
        seen_names: set[str] = set()

        for _ in range(5):
            next_q: list[str] = []
            for nid2 in queue:
                if nid2 in visited:
                    continue
                visited.add(nid2)
                node = nodes_by_id.get(nid2)
                if node and any(kw in node["name"].lower() for kw in _API_KW):
                    if node["name"] not in seen_names:
                        api_nodes.append(node)
                        seen_names.add(node["name"])
                next_q.extend(n2 for n2 in adj.get(nid2, []) if n2 not in visited)
            queue = next_q
            if not queue:
                break

        return {"screen_id": screen_id, "screen_name": screen_name, "api_deps": api_nodes}

    @app.get("/api/node/{node_id}")
    async def get_node(node_id: str) -> dict[str, Any]:
        nid = _esc(node_id)
        rows = store.query(
            f"MATCH (n:KlitNode {{id: '{nid}'}}) "
            f"RETURN n.id, n.kind, n.name, n.file_path, n.start_line, n.end_line, n.language"
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Node not found")
        r = rows[0]
        node = {
            "id": r[0],
            "kind": r[1],
            "name": r[2],
            "file": r[3],
            "start_line": r[4],
            "end_line": r[5],
            "language": r[6],
        }
        out = store.query(
            f"MATCH (a:KlitNode {{id: '{nid}'}}) -[e:KlitEdge]->(b:KlitNode) "
            f"RETURN e.type, b.id, b.name, b.kind"
        )
        inb = store.query(
            f"MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode {{id: '{nid}'}}) "
            f"RETURN e.type, a.id, a.name, a.kind"
        )
        return {
            "node": node,
            "outbound": [{"type": r[0], "id": r[1], "name": r[2], "kind": r[3]} for r in out],
            "inbound": [{"type": r[0], "id": r[1], "name": r[2], "kind": r[3]} for r in inb],
        }

    return app


# ---------------------------------------------------------------------------
# Legacy entry point (kept for direct use; CLI now uses create_web_app directly)
# ---------------------------------------------------------------------------

_KLIT_DIR = ".klit-flow"
_DB_NAME = "graph.db"
_BM25_NAME = "bm25.pkl"


def run_web_server(target: Path, port: int) -> None:
    """Open the index for *target* and serve the web portal on *port*."""
    import uvicorn

    from klit_flow.graph.store import LadybugGraphStore
    from klit_flow.index.bm25 import BM25Index
    from klit_flow.index.embeddings import Embedder

    klit_dir = target / _KLIT_DIR
    db_path = klit_dir / _DB_NAME
    bm25_path = klit_dir / _BM25_NAME

    store = LadybugGraphStore(db_path)
    bm25 = BM25Index.load(bm25_path) if bm25_path.exists() else _empty_bm25()
    embedder = Embedder()

    web_app = create_web_app(store, bm25, embedder)
    try:
        uvicorn.run(web_app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        store.close()


def _empty_bm25() -> BM25Index:
    idx = BM25Index()
    idx.build()
    return idx
