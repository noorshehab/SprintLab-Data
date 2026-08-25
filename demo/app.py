"""Minimal interactive demo: answer science questions, watch priors and
behavioral deltas move.

Run:
    ./.venv/Scripts/python.exe -m uvicorn demo.app:app --port 8500
Then open http://localhost:8500
"""
from __future__ import annotations
import math
import random
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from demo.runtime import (CALIBRATION_WINDOW, CHECKPOINT_EVERY, DEMO_STUDENT,
                          PROFILE, build_runtime)
from server.dto import question_dto
from services.log_setup import get_logger

log = get_logger('demo_api')

runtime: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.clear()
    runtime.update(build_runtime())
    yield


app = FastAPI(title='SprintLab prior-update demo', lifespan=lifespan)


def _rt() -> dict:
    if not runtime:
        raise RuntimeError('demo runtime not initialised')
    return runtime


#--- helpers --------------------------------------------------------------

def _student_state(rt) -> dict:
    data = rt['data']
    s = data.get_student(DEMO_STUDENT)
    stored = s.get_treatment_plan().get('treatment_plan')
    constraints = []
    if stored:
        constraints = rt['selection']._unwrap_treatment_plan(stored[1])

    windows = sorted(s.get_deltas().keys())
    delta_windows = [{'timestamp': ts,
                      'deltas': {k: round(v, 3) for k, v in
                                 s.get_deltas()[ts].items()
                                 if isinstance(v, (int, float))
                                 and math.isfinite(v)}}
                     for ts in windows]

    prior_windows = [{'timestamp': str(ts),
                      'priors': s.get_priors_history()[ts]}
                     for ts in sorted(s.get_priors_history().keys(),
                                      key=lambda k: str(k))[-6:]]

    return {
        'studentId': DEMO_STUDENT,
        'name': PROFILE['name'],
        'diagnoses': s.get_diagnoses(),
        'calibrationWindow': CALIBRATION_WINDOW,
        'checkpointEvery': CHECKPOINT_EVERY,
        'answersSoFar': rt['answer_count'],
        'priors': {k: round(v, 4) for k, v in s.get_priors().items()},
        'contentGaps': s.get_content_gaps(),
        'constraints': constraints,
        'deltaWindows': delta_windows,
        'priorWindows': prior_windows,
    }


def _question_payload(rt) -> dict | None:
    """Serve via Match_Service; fall back to bandit ranking over the whole
    pool when treatment constraints leave fewer than 2 fresh candidates."""
    data = rt['data']
    answered = set(rt['answered'])
    served = rt['match'].set_match(DEMO_STUDENT)
    fresh = [item for item in served if item['q'].id not in answered]
    via = 'treatment-constrained match'

    if len(fresh) < 2:
        # bandit over the full unanswered pool (permissive constraint set)
        candidates = [q for q in data.list_questions() if q.id not in answered]
        if not candidates:
            return None
        student = data.get_student(DEMO_STUDENT)
        import pandas as pd
        from server.reprocessing import _skill_ids  # noqa: F401  (context only)
        contexts = pd.DataFrame([
            {'q_id': q.id, 'skill_ids': q.skill_ids or [],
             'difficulty': q.difficulty_level, 'learning': q.p_g or 0.1}
            for q in candidates])
        scores = rt['selection'].Bandit.select(
            {'id': DEMO_STUDENT, 'priors': student.get_priors()}, contexts)
        order = scores['q_id'].tolist()
        by_id = {q.id: q for q in candidates}
        fresh = [{'q': by_id[qid]} for qid in order if qid in by_id]
        via = 'bandit fallback (constraints exhausted)'

    chosen = fresh[0]['q']
    rt['answered'].append(chosen.id)

    options = [chosen.correct_answer_content,
               chosen.distractor_1_content,
               chosen.distractor_2_content,
               chosen.distractor_3_content]
    options = [o for o in options if o not in (None, '')]
    shuffled = options[:]
    random.shuffle(shuffled)

    payload = question_dto(chosen)
    payload.update({
        'options': shuffled,
        'servedVia': via,
        'skills': chosen.skill_ids or [],
        'bloom': chosen.bloom_taxonomy_level,
        'timeAllowed': chosen.time_allowed,
    })
    return payload


#--- routes ---------------------------------------------------------------

class AnswerIn(BaseModel):
    question_id: str
    answer: str
    response_time_seconds: float = 20.0


@app.post('/api/answer')
def submit_answer(body: AnswerIn):
    rt = _rt()
    data = rt['data']
    q = data.get_question(body.question_id)
    q = q[0] if q else None
    if q is None:
        from fastapi import HTTPException
        raise HTTPException(404, f'question {body.question_id} not found')

    correct = (q.correct_answer_content or '').strip().lower() == \
        body.answer.strip().lower()

    priors_before = dict(data.get_student(DEMO_STUDENT).get_priors())
    rt['diagnosis'].add_student_response(
        DEMO_STUDENT, [q.id], [1 if correct else 0],
        [body.response_time_seconds], [None])
    priors_after = data.get_student(DEMO_STUDENT).get_priors()

    prior_changes = {
        skill: {'before': round(priors_before.get(skill, 0.0), 4),
                'after': round(priors_after.get(skill, 0.0), 4)}
        for skill in priors_after
        if abs(priors_after[skill] - priors_before.get(skill, 0.0)) > 1e-9}

    checkpoint = None
    rt['answer_count'] += 1
    if rt['answer_count'] % CHECKPOINT_EVERY == 0:
        bd_result = rt['bd_engine'].diagnose_student(DEMO_STUDENT)
        rt['treatment'].update_treatment_plan(DEMO_STUDENT)
        log.info('demo checkpoint at %d answers', rt['answer_count'])

    state = _student_state(rt)
    latest = state['deltaWindows'][-1] if state['deltaWindows'] else None
    if latest and rt['answer_count'] % CHECKPOINT_EVERY == 0:
        checkpoint = latest

    return {'correct': correct,
            'correctAnswer': q.correct_answer_content,
            'priorChanges': prior_changes,
            'checkpoint': checkpoint,
            'state': state}


@app.get('/api/state')
def get_state():
    return _student_state(_rt())


@app.get('/api/next')
def next_question():
    payload = _question_payload(_rt())
    if payload is None:
        return {'exhausted': True}
    return payload


@app.get('/api/reset')
def reset():
    """Rebuild the whole runtime - fresh student, fresh session."""
    runtime.clear()
    runtime.update(build_runtime())
    return {'reset': True}


@app.get('/', response_class=HTMLResponse)
def index():
    return _PAGE


#--- frontend -------------------------------------------------------------

_PAGE = """<!doctype html>
<html><head><meta charset='utf-8'>
<title>SprintLab Demo</title>
<style>
 :root{--bg:#10141c;--card:#1a2130;--line:#2a3448;--fg:#dce4f2;--dim:#8fa0bd;
        --ok:#4cc38a;--bad:#e5484d;--accent:#6ea8fe;}
 body{background:var(--bg);color:var(--fg);font-family:'Segoe UI',system-ui,sans-serif;
      margin:0;padding:24px;display:flex;gap:24px;flex-wrap:wrap}
 h1{font-size:20px;margin:0 0 4px} h2{font-size:14px;color:var(--dim);
      text-transform:uppercase;letter-spacing:.08em;margin:0 0 10px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
       padding:18px;min-width:340px;flex:1}
 .col{display:flex;flex-direction:column;gap:16px;max-width:560px;flex:1.2}
 .chip{display:inline-block;background:#233047;border:1px solid var(--line);
       border-radius:999px;padding:2px 10px;margin:2px;font-size:12px;color:var(--accent)}
 .opt{display:block;width:100%;text-align:left;background:#202a3d;color:var(--fg);
      border:1px solid var(--line);border-radius:8px;padding:12px;margin:8px 0;
      cursor:pointer;font-size:15px}
 .opt:hover{border-color:var(--accent)}
 .opt.ok{border-color:var(--ok);background:#17301f}
 .opt.bad{border-color:var(--bad);background:#321618}
 .bar{height:14px;background:#233047;border-radius:7px;overflow:hidden;margin:4px 0 10px}
 .fill{height:100%;background:var(--accent);transition:width .6s ease;border-radius:7px}
 .row{display:flex;justify-content:space-between;font-size:13px;color:var(--dim)}
 .delta{font-size:12px;padding:6px 8px;border-left:3px solid var(--line);
        margin:6px 0;background:#151c29;border-radius:0 6px 6px 0}
 .up{color:var(--ok)} .down{color:var(--bad)} .muted{color:var(--dim);font-size:12px}
 button.primary{background:var(--accent);border:none;color:#0b1220;font-weight:600;
      padding:10px 18px;border-radius:8px;cursor:pointer;font-size:14px}
 .flash{padding:10px;border-radius:8px;margin-bottom:12px;font-weight:600;display:none}
 .flash.ok{background:#17301f;color:var(--ok);display:block}
 .flash.bad{background:#321618;color:var(--bad);display:block}
 table{width:100%;border-collapse:collapse;font-size:12px}
 td,th{padding:4px 6px;text-align:left;border-bottom:1px solid var(--line);color:var(--dim)}
</style></head><body>

<div class='col'>
 <div class='card'>
   <h1 id='stuname'></h1>
   <div id='diagchips' class='muted'></div>
   <div class='muted' id='progress' style='margin-top:8px'></div>
 </div>
 <div class='card' id='qcard'>
   <h2>Question</h2>
   <div class='muted' id='meta'></div>
   <p id='qtext' style='font-size:16px;line-height:1.5'></p>
   <div id='opts'></div>
   <div class='flash' id='flash'></div>
   <button class='primary' onclick='loadNext()' id='nextbtn' style='visibility:hidden'>Next question &rarr;</button>
 </div>
</div>

<div class='col'>
 <div class='card'><h2>Skill priors</h2><div id='priors'></div>
   <div class='muted'>updated live with every answer (BKT)</div></div>
 <div class='card'><h2>Behavioral diagnosis deltas</h2><div id='deltas'></div>
   <div class='muted'>new window appended every <span id='ckpt'></span> answers</div></div>
 <div class='card'><h2>Active serving constraints</h2><div id='constraints'></div></div>
</div>

<script>
let current=null;
const $=id=>document.getElementById(id);

async function j(url,opt){const r=await fetch(url,opt);return r.json();}

async function refreshState(){
  const st=await j('/api/state');
  $('stuname').textContent=st.studentId+' — '+st.name;
  $('diagchips').innerHTML=st.diagnoses.map(d=>`<span class='chip'>${d}</span>`).join('')
    +st.contentGaps && Object.entries(st.contentGaps||{}).map(([k,v])=>`<span class='chip'>${k}: ${v}</span>`).join('');
  $('progress').textContent=`answers: ${st.answersSoFar} · calibration window: ${st.calibrationWindow}`;
  $('ckpt').textContent=st.checkpointEvery;
  const max=Math.max(...Object.values(st.priors),1);
  $('priors').innerHTML=Object.entries(st.priors).map(([k,v])=>
    `<div class='row'><span>${k}</span><span>${v}</span></div>
     <div class='bar'><div class='fill' style='width:${v/max*100}%'></div></div>`).join('');
  $('deltas').innerHTML=(st.deltaWindows||[]).slice().reverse().map(w=>
    `<div class='delta'><b>${w.timestamp.slice(0,10)}</b>`+
    Object.entries(w.deltas).map(([k,v])=>{
      const cls=v>=0?'up':'down';const arrow=v>=0?'▲':'▼';
      return `<div class='${cls}'>${arrow} ${k}: ${v}</div>`}).join('')+`</div>`).join('')
    ||'<span class=muted>no windows yet</span>';
  $('constraints').innerHTML=(st.constraints||[]).map(c=>
    `<table><tr><td>${c.Topic}</td><td>${c.Attribute}</td><td>${c.Operator}</td><td>${JSON.stringify(c.Threshold)}</td></tr></table>`).join('')
    ||'<span class=muted>none</span>';
}

async function loadNext(){
  $('flash').className='flash';$('nextbtn').style.visibility='hidden';
  const q=await j('/api/next');
  if(q.exhausted){$('qtext').textContent='All questions served — hit reset.';$('opts').innerHTML='';return;}
  current=q;
  $('meta').textContent=`${q.servedVia} · skills: ${(q.skills||[]).join(', ')} · bloom: ${q.bloom}`;
  $('qtext').textContent=q.question_text;
  $('opts').innerHTML='';
  (q.options||[]).forEach(opt=>{
    const b=document.createElement('button');b.className='opt';b.textContent=opt;
    b.onclick=()=>submitAnswer(opt,b);$('opts').appendChild(b);});
}

async function submitAnswer(choice,btn){
  document.querySelectorAll('.opt').forEach(b=>b.disabled=true);
  const res=await j('/api/answer',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question_id:current.id,answer:choice,
                         response_time_seconds:20})});
  btn.classList.add(res.correct?'ok':'bad');
  if(!res.correct){
    document.querySelectorAll('.opt').forEach(b=>{
      if(b.textContent===res.correctAnswer)b.classList.add('ok');});}
  const f=$('flash');
  f.className='flash '+(res.correct?'ok':'bad');
  const changes=Object.entries(res.priorChanges||{}).map(([k,c])=>
     `${k}: ${c.before} → ${c.after}`).join(' · ');
  f.textContent=(res.correct?'Correct! ':'Not quite. ')+
     (changes?`Priors updated → ${changes}`:'Priors unchanged (pre-calibration)');
  if(res.checkpoint)f.innerHTML+=`<div class='delta up'><b>checkpoint:</b> new behavioral window written</div>`;
  $('nextbtn').style.visibility='visible';
  refreshState();
}

refreshState();loadNext();
</script></body></html>"""
