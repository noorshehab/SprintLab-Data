"""API lifecycle tests against the full in-process graph.

Covers the seven spec routes end to end: batch insert (with corpus
reprocessing), questions query, diagnostics job, match creation,
answer submission and completion.
"""
import pytest
from fastapi.testclient import TestClient

from server.main import _job_store, _match_store, app


@pytest.fixture
def client():
    _match_store._matches.clear()
    _job_store._jobs.clear()
    with TestClient(app) as c:
        yield c


SEED_QUESTIONS = [
    {'q_id': f'q{i}', 'skill_cluster_id': 'KC-BIO-01' if i % 2 == 0 else 'KC-PHYS-01',
     'unit_id': 'U1', 'difficulty_level': 0.2 + 0.1 * i, 'p_g': 0.1,
     'question_text': ('What is the powerhouse of the cell and how does it '
                       'produce energy for the organism during respiration?'
                       if i % 2 == 0 else
                       'A ball rolls down an inclined plane; calculate its '
                       'final velocity after three seconds of acceleration.'),
     'correct_answer_content': 'x = y + z * 2',
     }
    for i in range(8)
]


def test_health(client):
    body = client.get('/health').json()
    assert body['status'] == 'ok'
    assert body['runtimeReady'] is True


def test_batch_insert_reprocesses_corpus(client):
    resp = client.post('/questions/batch', json={'Questions': SEED_QUESTIONS})
    assert resp.status_code == 201
    assert resp.json()['NumInserted'] == len(SEED_QUESTIONS)

    #reprocessing derived quartiles + BKT params onto every question
    all_qs = client.request('GET', '/questions', json={'allQs': 'True'}).json()['Questions']
    assert len(all_qs) == len(SEED_QUESTIONS)
    for q in all_qs:
        assert q['language_level'] in ('Q1', 'Q2', 'Q3', 'Q4')
        assert q['reasoning_level'] in ('Q1', 'Q2', 'Q3', 'Q4')
        assert q['p_t'] >= 0

    #duplicate insert does not double count
    dup = client.post('/questions/batch',
                      json={'Questions': SEED_QUESTIONS[:1]}).json()
    assert dup['NumInserted'] == 0


def test_get_questions_by_ids(client):
    client.post('/questions/batch', json={'Questions': SEED_QUESTIONS})
    body = client.request('GET', '/questions', json={'allQs': 'False',
                                          'ids': ['q0', 'q2']}).json()
    assert {q['id'] for q in body['Questions']} == {'q0', 'q2'}

    #no filter -> error
    assert client.request('GET', '/questions', json={'allQs': 'False'}).status_code == 400


def test_full_match_lifecycle(client):
    client.post('/questions/batch', json={'Questions': SEED_QUESTIONS})
    client.post('/students', json={'studentId': 'S-API'})
    client.post('/students', json={'studentId': 'S-GHOST'})  # unused

    #diagnostics job completes (background tasks run inline under TestClient)
    job = client.post('/diagnostics/run', json={
        'studentId': 'S-API',
        'Answers': [{'questionId': 'q0', 'response': 1,
                     'responseTimeSeconds': 12.5},
                    {'questionId': 'q1', 'response': 0,
                     'responseTimeSeconds': 40.0,
                     'stressTrigger': 1}]}).json()
    assert job['status'] == 'processing'

    status = client.get(f"/diagnostics/{job['jobId']}").json()
    assert status['status'] == 'completed'
    assert status['result']['treatmentPlanWritten'] is True
    assert status['result']['priorsWritten'] >= 0

    unknown = client.get('/diagnostics/missing')
    assert unknown.status_code == 404

    #create a match - bandit serves from the seeded pool
    created = client.post('/matches', json={'studentId': 'S-API'})
    assert created.status_code == 201
    match = created.json()
    match_id = match['matchId']
    assert match_id
    served_ids = {q['question']['id'] for q in match['questions']}
    assert served_ids and served_ids <= {f'q{i}' for i in range(8)}

    #append answers including stress + impulsive-atag variants so those
    #behavioral metrics have signal
    for i, qid in enumerate(sorted(served_ids)):
        answer = {
            'questionId': qid,
            'response': 1 if i % 3 else 0,
            'responseTimeSeconds': 10.0 + i,
        }
        if i % 2:
            answer['stressTrigger'] = 1
        if i % 3 == 0:
            answer['answerTag'] = 'Distactor_Impulsive'
        assert client.post(f'/matches/{match_id}/answers',
                           json=answer).json()['accepted'] is True

    assert client.post(f'/matches/nope/answers',
                       json={'questionId': 'q0', 'response': 1}).status_code == 404
    assert client.post('/matches', json={'studentId': 'NOPE'}).status_code == 404

    #complete -> progress tracking over accumulated answers
    done = client.post(f'/matches/{match_id}/complete')
    assert done.status_code == 200
    result = done.json()

    assert set(result.keys()) == {'deltaMastery', 'updatedPriors',
                                  'behavioralScores', 'updatedTreatmentPlan'}
    assert isinstance(result['deltaMastery'], dict)
    assert isinstance(result['updatedPriors'], dict)
    assert isinstance(result['updatedTreatmentPlan'], list)
    for constraint in result['updatedTreatmentPlan']:
        assert {'Topic', 'Attribute', 'Operator', 'Threshold'} <= set(constraint)

    scores = result['behavioralScores']
    assert isinstance(scores, list)
    for score in scores:
        assert {'metric', 'value', 'diagnosed'} <= set(score)
    metrics = {s['metric'] for s in scores}
    assert {'stress_ratio', 'impulse_error_rate', 'attention'} <= metrics

    #student profile reflects the loop
    profile = client.get('/students/S-API').json()
    assert profile['priors']
    assert profile['responsesCount'] >= len(served_ids) + 2
