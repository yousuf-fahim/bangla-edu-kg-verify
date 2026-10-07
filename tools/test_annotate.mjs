// Headless test for tools/annotate.html. Stubs just enough DOM to drive the
// tool, then checks both modes against the real CSVs in this repo.
//
//   node tools/test_annotate.mjs
//
// The tool is a single HTML file with no build step, so this extracts the inline
// <script> and runs it in a Function scope rather than importing a module.
import fs from 'node:fs';
import path from 'node:path';
const REPO = path.resolve(import.meta.dirname, '..');

// --- minimal DOM stub ----------------------------------------------------
class El {
  constructor(id){ this.id=id; this._html=''; this.hidden=false; this.textContent='';
                   this.style={}; this.dataset={}; this._kids=[]; this._lis={}; }
  set innerHTML(v){ this._html=v;
    this._kids=[...String(v).matchAll(/data-v="([^"]+)"/g)].map(m=>{
      const e=new El(null); e.dataset.v=m[1]; return e; }); }
  get innerHTML(){ return this._html; }
  addEventListener(t,f){ (this._lis[t]=this._lis[t]||[]).push(f); }
  querySelectorAll(){ return this._kids; }
  querySelector(){ return this._kids[0]||new El(null); }
  click(){ (this._lis.click||[]).forEach(f=>f()); }
}
const els={}; const get=id=>els[id]||(els[id]=new El(id));
['setup','app','file','resume','done','dn','summary','export2','reset','keys','hint','card','stats','fill']
  .forEach(get);
let captured=null;
globalThis.document={ getElementById:get, addEventListener(){}, querySelectorAll:()=>[],
  createElement:()=>({ set href(v){}, get href(){return 'x';}, click(){}, set download(v){this._d=v;} }) };
globalThis.localStorage={ _d:{}, getItem(k){return this._d[k]??null;},
  setItem(k,v){this._d[k]=v;}, removeItem(k){delete this._d[k];} };
globalThis.Blob=class{ constructor(parts){ captured=parts.join(''); } };
globalThis.URL={ createObjectURL:()=>'blob:x', revokeObjectURL(){} };
globalThis.FileReader=class{};

// --- load the tool's script ---------------------------------------------
const html=fs.readFileSync(`${REPO}/tools/annotate.html`,'utf8');
const m=html.match(/<script>([\s\S]*?)<\/script>/);
if(!m) throw new Error('no inline <script> found in annotate.html');
let src=m[1];
src+=`\nglobalThis.__T={start,exportCSV,mark,detectMode,parseCSV,MODES,tally,
  get mode(){return mode}, get idx(){return idx}, get verdicts(){return verdicts}};`;
const run=new Function(src); run();
const T=globalThis.__T;

const ok=(c,m)=>{ console.log((c?'  PASS  ':'  FAIL  ')+m); if(!c) process.exitCode=1; };

// --- verdict mode --------------------------------------------------------
const vg=T.parseCSV(fs.readFileSync(`${REPO}/eval/run_G/verdict_gold_sample.csv`,'utf8'));
ok(vg.length===200, `parsed 200 rows from verdict_gold_sample.csv (got ${vg.length})`);
ok(T.detectMode(vg[0])==='verdict', 'detects verdict mode from a `claim` column');
T.start(vg,'verdict_gold_sample.csv',null);
ok(T.mode==='verdict','start() entered verdict mode');
ok(get('keys').innerHTML.includes('not_in_curriculum'),'rendered the not_in_curriculum button');
ok(!get('card').innerHTML.includes('NLI entail'),'scores hidden by default (no anchoring)');
ok(!/verdict__/.test(get('card').innerHTML),'system verdict never shown to the annotator');
const q=vg[0].question;
ok(!q || get('card').innerHTML.includes('Question asked'),'shows the question that was asked');

T.mark('supported'); T.mark('contradicted'); T.mark('not_in_curriculum'); T.mark('unsure');
ok(T.idx===4,'four marks advanced the cursor to 4');
const t=T.tally();
ok(t.judged===4 && t.supported===1 && t.unsure===1, `tally correct (${JSON.stringify(t)})`);

T.exportCSV();
const head=captured.split('\n')[0];
ok(head.endsWith('gold') || head.includes(',gold'),'export has a `gold` column');
ok(!/,verdict(,|$)/.test(head),'export does NOT add a stray `verdict` column');
const back=T.parseCSV(captured);
ok(back.length===200,`round-trips 200 rows (got ${back.length})`);
ok(back[0].gold==='supported' && back[3].gold==='unsure','labels landed on the right rows');
ok(back[4].gold==='','unlabelled rows stay blank');
const first=vg[0];
ok(back[0].claim===first.claim,'Bangla claim text survives the CSV round-trip intact');

// resuming a part-labelled file should pick the labels back up
const T2=globalThis.__T;
T2.start(T2.parseCSV(captured),'verdict_gold_sample_reviewed.csv',null);
ok(T2.tally().judged===4,'reopening a part-labelled export restores 4 labels');

// --- triple mode (backward compatibility with review_200) ----------------
const tr=T.parseCSV(fs.readFileSync(`${REPO}/kg/triples/review_200.csv`,'utf8'));
ok(T.detectMode(tr[0])==='triple','still detects triple mode for review_200.csv');
T.start(tr,'review_200.csv',null);
ok(T.mode==='triple','start() entered triple mode');
T.mark('correct'); T.mark('wrong_relation');
T.exportCSV();
ok(/,verdict$/.test(captured.split('\n')[0]),'triple mode still exports `verdict`');
ok(T.parseCSV(captured)[1].verdict==='wrong_relation','triple labels land correctly');

console.log(process.exitCode ? String.fromCharCode(10) + 'FAILED'
                             : String.fromCharCode(10) + 'all checks passed');
