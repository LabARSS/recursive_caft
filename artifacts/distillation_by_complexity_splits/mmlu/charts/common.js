// Shared data and chart helpers for the complexity-bin charts.
// Generated from the analysis notebook export; edit data here, not in the HTML files.
const D={
 q2048:[[.6035,.6010,.5960,.5761,.6209],[.3491,.3965,.4264,.3815,.3990],[.3342,.3691,.3566,.4015,.3641],[.3017,.3292,.2843,.3267,.2968],[.2818,.2893,.2818,.2943,.3067],[.2195,.2369,.2294,.2070,.2045]],
 q4096:[[.5985,.5985,.5985,.5835,.6284],[.3441,.4015,.4364,.3815,.4065],[.3392,.3865,.3691,.3940,.3865],[.2993,.3342,.2868,.3192,.2893],[.2668,.3042,.3017,.3217,.3117],[.2070,.2269,.2768,.2195,.2020]],
 p2048:[[.0499,.6085,.6135,.6284,.6334],[.0299,.2793,.3940,.3616,.4090],[.0025,.2668,.3092,.3541,.3516],[.0025,.2020,.2668,.2618,.2793],[.0025,.2643,.3242,.3242,.3367],[.0050,.2170,.2618,.3042,.2843]],
 p4096:[[.0599,.6135,.6060,.6234,.6409],[.0499,.2519,.3591,.3741,.4190],[.0150,.2419,.2968,.3641,.3466],[.0150,.1621,.2294,.2469,.2768],[.0125,.2369,.2893,.2918,.3117],[.0175,.1496,.2145,.2594,.2494]],
 l2048:[[.5486,.5636,.5786,.5536,.5810],[.4165,.4389,.3741,.3890,.3716],[.3217,.3192,.2943,.3392,.2818],[.3092,.3042,.2843,.3117,.2843],[.3017,.2668,.2818,.3042,.2843],[.2294,.2893,.2718,.2768,.2793]]};
const T={
 p2048:[[155,16,2,0,0],[99,32,6,2,5],[67,8,2,2,1],[46,16,3,0,7],[36,3,0,0,2],[64,13,3,1,1]],
 p4096:[[202,33,14,3,0],[277,116,115,7,5],[179,121,32,19,14],[247,232,51,35,34],[245,184,151,114,115],[286,269,162,139,86]]};
const EP=[5,10,20,35,50];
const SER=[{k:'q2048',lab:'Qwen-3B · cap2048',c:'#2a6fdb',d:''},{k:'q4096',lab:'Qwen-3B · cap4096',c:'#2a6fdb',d:'5 4'},{k:'p2048',lab:'Phi-4-mini · cap2048',c:'#c2563a',d:''},{k:'p4096',lab:'Phi-4-mini · cap4096',c:'#c2563a',d:'5 4'},{k:'l2048',lab:'Llama-3B · cap2048',c:'#0d9488',d:''}];
const best=(k,b)=>{const a=D[k]&&D[k][b];return a?Math.max(...a.slice(k[0]==='p'?1:0)):null};
const S='http://www.w3.org/2000/svg';
function el(n,a){const e=document.createElementNS(S,n);for(const k in a)e.setAttribute(k,a[k]);return e}
function svg(w,h){const s=el('svg',{viewBox:`0 0 ${w} ${h}`,width:'100%'});s.style.display='block';return s}
function txt(s,x,y,t,o={}){const e=el('text',{x,y,'font-size':o.fs||11,fill:o.f||'#6b7385','text-anchor':o.a||'middle','font-family':'IBM Plex Mono,monospace'});e.textContent=t;if(o.w)e.setAttribute('font-weight',o.w);s.appendChild(e);return e}
function axes(s,L,R,T0,B,ymin,ymax,ticks,xs,xlabs){
  s.appendChild(el('line',{x1:L,y1:B,x2:R,y2:B,stroke:'#8b93a1','stroke-width':1}));
  s.appendChild(el('line',{x1:L,y1:T0,x2:L,y2:B,stroke:'#8b93a1','stroke-width':1}));
  const Y=v=>B-(v-ymin)/(ymax-ymin)*(B-T0);
  ticks.forEach(v=>{s.appendChild(el('line',{x1:L,y1:Y(v),x2:R,y2:Y(v),stroke:'rgba(26,42,68,.08)'}));txt(s,L-7,Y(v)+4,(v*100).toFixed(0)+'%',{a:'end'})});
  xs.forEach((x,i)=>txt(s,x,B+16,xlabs[i]));
  return Y;
}
function legendSeries(id,items){document.getElementById(id).innerHTML=items.map(x=>`<span><i style="border-top-style:${x.d?'dashed':'solid'};border-top-color:${x.c}"></i>${x.lab}</span>`).join('')}
const GAIN={},ACC={};['q','p','l'].forEach(m=>{GAIN[m]=[];ACC[m]=[];
  for(let b=0;b<6;b++){const ks=[m+'2048',m+'4096'].filter(k=>D[k]&&D[k][b]);
    if(!ks.length){GAIN[m].push(null);ACC[m].push(null);continue}
    const bs=Math.max(...ks.map(k=>best(k,b)));
    const e10=Math.max(...ks.map(k=>D[k][b][1]));
    GAIN[m].push((bs-e10)*100);ACC[m].push(bs*100);}});
const GC=['#2a6fdb','#6b4edd','#c2563a','#d4a017','#0d9488','#8b6f47'];
const RB_P=null; // Phi random seed42: pending recompute
const RB_Q={rand:[.36,.3533,.3383,.3333,.3267],bal:[.3633,.385,.33,.3633,.3517]};
const RB_L={rand:[.3117,.3567,.3267,.3333,.3167],bal:[.3733,.3567,.355,.2983,.3417]};
const CR_L=[[.3233,.3217,.3117,.3317,.3217],[.3667,.3383,.3217,.3217,.3317],[.3133,.355,.32,.305,.3083],[.3417,.37,.34,.31,.3367],[.3233,.355,.3467,.3233,.3283],[.32,.375,.3667,.3167,.3517]];
const CR_Q=[[.37,.365,.3633,.35,.3667],[.2867,.34,.3117,.345,.3217],[.3267,.3783,.3617,.335,.3833],[.3317,.3583,.3617,.3333,.3517],[.33,.3867,.315,.3517,.3367],[.3467,.36,.3533,.3233,.3183]];
const CB_L=[[.3267,.3317,.33,.31,.3533],[.3433,.3817,.3383,.35,.3417],[.36,.3283,.3367,.3583,.3483],[.3217,.3917,.3367,.3233,.3283],[.345,.34,.3433,.3317,.3467],[.3733,.3817,.3367,.3433,.3333]];
const CB_Q=[[.395,.4167,.3817,.3717,.375],[.3283,.3633,.3633,.3383,.3867],[.3467,.3867,.3633,.365,.36],[.365,.3783,.3633,.3433,.3617],[.3617,.3917,.335,.35,.34],[.3517,.38,.3717,.3633,.3417]];
const CR_P=[[.0217,.2533,.2867,.305,.3483],[.025,.2433,.3383,.3317,.3567],[.005,.2783,.36,.3867,.4133],[.0083,.2467,.3533,.3633,.355],[.0033,.2967,.3633,.3817,.3417],[.0033,.2267,.3267,.3517,.385]];
const CB_P=[[.0133,.2483,.3317,.3083,.2917],[.01,.2717,.3333,.3533,.3367],[.0017,.3,.315,.39,.3733],[.005,.2667,.3367,.385,.3467],[0,.3,.36,.3867,.3783],[.0117,.2483,.3033,.36,.3583]];
const TM_L=[[.62,.4,.29,.35,.3,.23],[.61,.4,.36,.4,.34,.22],[.57,.42,.38,.36,.3,.3],[.61,.4,.35,.41,.34,.28],[.64,.34,.35,.37,.36,.28],[.63,.41,.39,.37,.38,.3]];
const TM_Q=[[.66,.52,.42,.35,.38,.27],[.63,.55,.39,.3,.37,.29],[.6,.49,.36,.33,.36,.33],[.64,.47,.35,.34,.42,.28],[.61,.47,.41,.29,.37,.29],[.65,.46,.37,.36,.36,.22]];
const TM_P=[[.67,.43,.31,.21,.3,.18],[.59,.42,.4,.27,.29,.24],[.67,.46,.37,.33,.32,.26],[.68,.4,.31,.29,.32,.32],[.6,.48,.34,.34,.33,.36],[.61,.36,.28,.3,.37,.31]];
const TIP=(()=>{const d=document.createElement('div');d.style.cssText='position:fixed;z-index:99;pointer-events:none;opacity:0;transition:opacity .12s;background:#1a2a44;color:#fffdf8;font:11.5px/1.45 IBM Plex Mono,monospace;padding:6px 9px;border-radius:5px;white-space:nowrap;box-shadow:0 4px 14px rgba(0,0,0,.22)';document.body.appendChild(d);return d})();
function tipOn(node,html){
  node.style.cursor='pointer';
  node.addEventListener('mouseenter',e=>{TIP.innerHTML=html;TIP.style.opacity='1'});
  node.addEventListener('mousemove',e=>{TIP.style.left=(e.clientX+14)+'px';TIP.style.top=(e.clientY-10)+'px'});
  node.addEventListener('mouseleave',()=>{TIP.style.opacity='0'});
}
const HID=new Set();
const REG=[];
function applyVis(){REG.forEach(r=>{const off=HID.has(r.bin);r.g.style.opacity=off?.08:1;r.g.style.pointerEvents=off?'none':'auto'});
  document.querySelectorAll('[data-legend-bin]').forEach(b=>{b.style.opacity=HID.has(+b.dataset.legendBin)?.35:1});}
function binLegend(mount){
  if(document.querySelector('#'+mount+' .lgd'))return;
  const d=document.createElement('div');d.className='lgd';
  d.style.cssText='display:flex;gap:12px;flex-wrap:wrap;width:100%;margin:0 0 6px;font:11.5px IBM Plex Mono,monospace;color:#6b7385';
  d.innerHTML='<span style="color:#8b93a1">click to isolate:</span>'+[0,1,2,3,4,5].map(b=>
    '<span data-legend-bin="'+b+'" style="display:flex;align-items:center;gap:5px;cursor:pointer"><span style="width:10px;height:10px;border-radius:2px;background:'+GC[b]+'"></span>bin '+b+'</span>').join('')+
    '<span data-legend-reset style="cursor:pointer;color:#2a6fdb">reset</span>';
  d.querySelectorAll('[data-legend-bin]').forEach(el2=>el2.onclick=()=>{const b=+el2.dataset.legendBin;
    if(HID.size===5&&!HID.has(b)){HID.clear()}else{HID.clear();[0,1,2,3,4,5].forEach(x=>{if(x!==b)HID.add(x)})}applyVis()});
  d.querySelector('[data-legend-reset]').onclick=()=>{HID.clear();applyVis()};
  const mt=document.getElementById(mount);mt.insertBefore(d,mt.firstChild);
}
function epochPanel(data,mount,title,lo,hi,tks,skip5,rb){
  const w=1000,h=330,L=68,R=858,T0=34,B=266,EPl=[5,10,20,35,50];
  const s=svg(w,h);const X=i=>L+i/4*(R-L),Y=v=>B-(v-lo)/(hi-lo)*(B-T0);
  txt(s,L-4,20,title,{a:'start',w:600,fs:13.5,f:'#1a2a44'});
  s.appendChild(el('line',{x1:L,y1:B,x2:R,y2:B,stroke:'#8b93a1'}));
  s.appendChild(el('line',{x1:L,y1:T0,x2:L,y2:B,stroke:'#8b93a1'}));
  tks.forEach(v=>{s.appendChild(el('line',{x1:L,y1:Y(v),x2:R,y2:Y(v),stroke:'rgba(26,42,68,.07)'}));txt(s,L-8,Y(v)+4,Math.round(v*100)+'%',{a:'end'})});
  EPl.forEach((e,i)=>txt(s,X(i),B+17,e));
  txt(s,(L+R)/2,B+36,'epoch');
  const ends=data.map((arr,g)=>({g,y:Y(arr[4])+4,lab:'bin '+g,c:null})).concat(rb?[{g:-1,y:Y(rb[4])+4,lab:'random',c:'#6b7385'}]:[]).sort((a,b)=>a.y-b.y);
  for(let i=1;i<ends.length;i++)if(ends[i].y-ends[i-1].y<15)ends[i].y=ends[i-1].y+15;
  const i0=0;const clipB=v=>Math.min(Y(v),B-4);
  if(rb){const pts=rb.map((v,i)=>[X(i),clipB(v)]);
    const g=el('g',{});
    g.appendChild(el('path',{d:pts.map((p,i)=>(i?'L':'M')+p[0]+' '+p[1]).join(' '),fill:'none',stroke:'#6b7385','stroke-width':2.6,'stroke-dasharray':'8 5','stroke-linejoin':'round'}));
    pts.forEach((p,k)=>{g.appendChild(el('circle',{cx:p[0],cy:p[1],r:3.6,fill:'#fffdf8',stroke:'#6b7385','stroke-width':2}));
      const hit=el('circle',{cx:p[0],cy:p[1],r:10,fill:'transparent'});g.appendChild(hit);
      tipOn(hit,'<b>random seed42 · epoch '+EPl[k]+'</b><br>'+(rb[k]*100).toFixed(1)+'% · fixed random sample of the same size'+(Y(rb[k])>B-4?'<br>point below the axis range (format warm-up), clipped to the edge':''));});

    s.appendChild(g);}
  data.forEach((arr,g)=>{
    const gg=el('g',{});s.appendChild(gg);REG.push({bin:g,g:gg});
    const pts=arr.map((v,i)=>[X(i),clipB(v)]);
    gg.appendChild(el('path',{d:pts.map((p,i)=>(i?'L':'M')+p[0]+' '+p[1]).join(' '),fill:'none',stroke:GC[g],'stroke-width':2.4,'stroke-linejoin':'round'}));
    const lab=ends.find(o=>o.g===g);
    tipOn(txt(gg,R+10,lab.y,'trained on bin '+g,{a:'start',f:GC[g],w:600,fs:11.5}),'<b>trained on bin '+g+'</b><br>'+title+'<br>best '+(Math.max(...arr)*100).toFixed(1)+'% @ ep'+EPl[arr.indexOf(Math.max(...arr))]);
    pts.forEach((p,k)=>{const i=k+i0;const hit=el('circle',{cx:p[0],cy:p[1],r:9,fill:'transparent'});
      gg.appendChild(el('circle',{cx:p[0],cy:p[1],r:4,fill:'#fffdf8',stroke:GC[g],'stroke-width':2.2}));
      gg.appendChild(hit);
      tipOn(hit,'<b>bin '+g+' · epoch '+EPl[i]+'</b><br>'+(arr[i]*100).toFixed(1)+'%'+(i?' · Δep10 '+((arr[i]-arr[1])*100>=0?'+':'')+((arr[i]-arr[1])*100).toFixed(1)+' pp':''));});
  });
  if(rb){const lr=ends.find(o=>o.g===-1);txt(s,R+10,lr.y,'random',{a:'start',f:'#6b7385',w:600,fs:11.5});}
  document.getElementById(mount).appendChild(s);
  binLegend(mount);
}
const TK_LQ=[.28,.31,.34,.37,.40,.43],TK_P0=[0,.1,.2,.3,.4];
function bestPanel(mount,title,dL,dQ,dP,lo,hi,tks,rbs){
  const EPl=[5,10,20,35,50],w=1000,h=380,L=68,R=890,T0=34,B=266,PAD=26;
  const s=svg(w,h);const X=b=>L+PAD+b*(R-L-PAD)/5,Y=v=>B-(v-lo)/(hi-lo)*(B-T0);
  txt(s,L-6,20,title,{a:'start',w:600,fs:13.5,f:'#1a2a44'});
  s.appendChild(el('line',{x1:L,y1:B,x2:R,y2:B,stroke:'#8b93a1'}));
  s.appendChild(el('line',{x1:L,y1:T0,x2:L,y2:B,stroke:'#8b93a1'}));
  tks.forEach(v=>{s.appendChild(el('line',{x1:L,y1:Y(v),x2:R,y2:Y(v),stroke:'rgba(26,42,68,.07)'}));txt(s,L-8,Y(v)+4,Math.round(v*100)+'%',{a:'end'})});
  for(let b=0;b<6;b++)txt(s,X(b),B+17,'bin '+b,{w:600});
  txt(s,(L+R)/2,B+36,'training bin');
  const obst=[];
  (rbs||[]).forEach(r=>{const a=r.skip5?r.arr.slice(1):r.arr,v=Math.max(...a),e=EPl[r.arr.indexOf(v)],y=Y(v),g=el('g',{});
    g.appendChild(el('line',{x1:L,y1:y,x2:R,y2:y,stroke:r.c,'stroke-width':1.8,'stroke-dasharray':'8 5',opacity:.8}));
    const hit=el('rect',{x:L,y:y-7,width:R-L,height:14,fill:'transparent'});g.appendChild(hit);
    tipOn(hit,'<b>'+r.nm+' · random seed42</b><br>best epoch '+e+' · '+(v*100).toFixed(1)+'%<br>fixed random sample of the same size, all bins mixed');
    s.appendChild(g);
    const str='random · '+r.nm.split('-')[0].replace('2.5','')+' '+(v*100).toFixed(1)+'%',ww=str.length*6.2;
    let ly=y-4,bx={x1:R-8-ww,x2:R-8,y1:ly-10,y2:ly+2};
    if(obst.some(o=>o.x1<bx.x2&&bx.x1<o.x2&&o.y1<bx.y2&&bx.y1<o.y2)){ly=y+13;bx={x1:R-8-ww,x2:R-8,y1:ly-10,y2:ly+2};}
    obst.push(bx);txt(s,R-8,ly,str,{a:'end',f:r.c,w:600,fs:10});});
  const series=[['Llama-3B',dL,'#0d9488',1,0],['Qwen2.5-3B',dQ,'#2a6fdb',-1,0],['Phi-4-mini',dP,'#c2563a',1,1]].map(([nm,data,c,dir,skip5])=>({
    nm,c,dir,pts:data.map((arr,b)=>{const a=skip5?arr.map((v,i)=>i?v:-1):arr;const i=a.indexOf(Math.max(...a));return{b,v:arr[i],e:EPl[i]}})}));
  series.forEach(se=>s.appendChild(el('path',{d:se.pts.map((p,i)=>(i?'L':'M')+X(p.b)+' '+Y(p.v)).join(' '),fill:'none',stroke:se.c,'stroke-width':2.2,'stroke-linejoin':'round'})));
  const dots=obst.slice(),labels=[];
  for(let bi=0;bi<6;bi++)series.forEach(se=>{
    const p=se.pts[bi],x=X(bi),y=Y(p.v),g=el('g',{});
    g.appendChild(el('circle',{cx:x,cy:y,r:5,fill:'#fffdf8',stroke:se.c,'stroke-width':2.4}));
    const hit=el('circle',{cx:x,cy:y,r:11,fill:'transparent'});g.appendChild(hit);
    tipOn(hit,'<b>'+se.nm+' · trained on bin '+bi+'</b><br>best epoch '+p.e+' · '+(p.v*100).toFixed(1)+'%');
    s.appendChild(g);
    dots.push({x1:x-6,x2:x+6,y1:y-6,y2:y+6});
    labels.push({bi,x,y,se,p,g});});
  const RINGS=series.map(se=>{const vs=se.pts.map(p=>p.v);return{se,vs,bi:vs.indexOf(Math.max(...vs)),wi:vs.indexOf(Math.min(...vs))}});
  RINGS.forEach(r=>[r.bi,r.wi].forEach(b=>dots.push({x1:X(b)-12,x2:X(b)+12,y1:Y(r.vs[b])-12,y2:Y(r.vs[b])+12})));
  const boxOf=(x,y,str,fs,anchor)=>{const ww=str.length*fs*.6,x1=anchor==='start'?x:anchor==='end'?x-ww:x-ww/2;
    return{x1,x2:x1+ww,y1:y-fs*.78,y2:y+fs*.26}};
  const over=(u,v)=>u.x1<v.x2&&v.x1<u.x2&&u.y1<v.y2&&v.y1<u.y2;
  const taken=dots.slice();
  labels.sort((u,v)=>u.y-v.y).forEach(it=>{
    const cand=it.bi===5?[[-13,4],[-13,-13],[-13,18],[0,-14],[0,18]]:[[0,-14],[0,18],[15,4],[-15,4],[0,-26],[0,30]];
    for(const [dx,dy] of cand){
      const cy=it.y+dy;if(cy<T0+10||cy>B-4)continue;
      const anchor=dx>0?'start':dx<0?'end':'middle';
      const bx=boxOf(it.x+dx,cy,'ep'+it.p.e,10,anchor);
      if(taken.some(t=>over(bx,t)))continue;
      taken.push(bx);txt(it.g,it.x+dx,cy,'ep'+it.p.e,{fs:10,f:it.se.c,w:600,a:anchor});return;}
  });
  RINGS.forEach(({se,vs,bi,wi})=>{
    const rb=el('circle',{cx:X(bi),cy:Y(vs[bi]),r:10,fill:'none',stroke:'#d4a017','stroke-width':2.6});
    tipOn(rb,'<b>'+se.nm+'</b><br>best bin: '+bi+' · '+(vs[bi]*100).toFixed(1)+'%');s.appendChild(rb);
    const rw=el('circle',{cx:X(wi),cy:Y(vs[wi]),r:10,fill:'none',stroke:'#c2563a','stroke-width':2,'stroke-dasharray':'3 3',opacity:.75});
    tipOn(rw,'<b>'+se.nm+'</b><br>worst bin: '+wi+' · '+(vs[wi]*100).toFixed(1)+'%');s.appendChild(rw);});

  s.appendChild(el('circle',{cx:L+232,cy:16,r:7,fill:'none',stroke:'#d4a017','stroke-width':2.6}));
  txt(s,L+244,20,'best bin per model',{a:'start',fs:10.5});
  s.appendChild(el('circle',{cx:L+372,cy:16,r:7,fill:'none',stroke:'#c2563a','stroke-width':2,'stroke-dasharray':'3 3'}));
  txt(s,L+384,20,'worst',{a:'start',fs:10.5});
  const ends=series.map(se=>({nm:se.nm,c:se.c,y:Y(se.pts[5].v)+4})).sort((a,b)=>a.y-b.y);
  for(let i=1;i<ends.length;i++)if(ends[i].y-ends[i-1].y<15)ends[i].y=ends[i-1].y+15;
  ends.forEach(o=>txt(s,R+10,o.y,o.nm.split('-')[0].replace('2.5','').replace('Phi','Phi-4'),{a:'start',f:o.c,w:600,fs:11.5}));
  document.getElementById(mount).appendChild(s);
}
