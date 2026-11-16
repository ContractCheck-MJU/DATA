"""Exercise real checkpoint inference and the boundary with reviewed inference."""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import create_app
from api.legacy import RUN

DEV_ROUTE = '/dev/v1/legacy/clauses/classify'
CHECKPOINT = RUN / 'fold_1/checkpoint'


class LegacyAPIConfigurationTests(unittest.TestCase):
    def test_disabled_defaults(self):
        with patch.dict('os.environ', {}, clear=True), TestClient(create_app()) as client:
            self.assertEqual(client.get('/health/live').status_code, 200)
            self.assertEqual(client.get('/health/ready').status_code, 503)
            self.assertEqual(client.get('/dev/v1/legacy/health/ready').status_code, 503)
            response = client.post(DEV_ROUTE, json={'clause_id': 'demo', 'text': '요약문'})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()['detail']['code'], 'LEGACY_MODEL_DISABLED')

    def test_invalid_fold_fails_startup(self):
        with patch.dict('os.environ', {'CONTRACTCHECK_LEGACY_FOLD': '6'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'CONTRACTCHECK_LEGACY_FOLD'):
                with TestClient(create_app()):
                    self.fail('Invalid fold unexpectedly started')

    def test_legacy_checkpoint_rejected_by_reviewed_loader(self):
        with patch.dict('os.environ', {'CONTRACTCHECK_MODEL_DIR': str(CHECKPOINT)}, clear=True):
            with self.assertRaisesRegex(FileNotFoundError, 'inference_config.json'):
                with TestClient(create_app()):
                    self.fail('Legacy artifact unexpectedly accepted as reviewed')


@unittest.skipUnless((CHECKPOINT/'model.safetensors').is_file(), 'Local fold 1 artifact required')
class RealLegacyAPITests(unittest.TestCase):
    def test_real_checkpoint_predictions_and_request_errors(self):
        # Public API documentation example, not a released dataset row.
        fixtures = [dict(sample_id='demo-1',text='사업자가 계약 내용을 일방적으로 변경할 수 있도록 규정함.',
                         predicted=1,model_score=.9959288239479065)]
        env = {'CONTRACTCHECK_LEGACY_FOLD': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
        with patch.dict('os.environ', env, clear=True), TestClient(create_app()) as client:
            ready = client.get('/dev/v1/legacy/health/ready')
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json()['fold'], 1)
            self.assertEqual(ready.json()['max_tokens'], 192)
            self.assertEqual(client.get('/health/ready').status_code, 503)
            for row in fixtures:
                with self.subTest(sample=row['sample_id']):
                    body = {'clause_id': row['sample_id'], 'text': row['text']}
                    response = client.post(DEV_ROUTE, json=body)
                    self.assertEqual(response.status_code, 200, response.text)
                    data = response.json()
                    self.assertEqual(data['clause_id'], row['sample_id'])
                    self.assertEqual(data['model_version'], 'cv_20261116_seed42/fold_1')
                    self.assertEqual(data['weak_label'], int(row['predicted']))
                    self.assertAlmostEqual(data['before_summary_model_score'], float(row['model_score']), places=5)
                    self.assertGreater(data['input_tokens'], 0)
                    self.assertTrue(data['research_only'])
                    self.assertNotIn('is_risky_candidate', data)

            for invalid in ({'clause_id':'demo'}, {'clause_id':'demo','text':'  '},
                            {'clause_id':42,'text':'요약문'}, {'clause_id':'demo','text':'가'*10001},
                            {'clause_id':'demo','text':'요약문','domain':'residential_rental'}):
                with self.subTest(keys=list(invalid)):
                    self.assertEqual(client.post(DEV_ROUTE, json=invalid).status_code, 422)
            long = client.post(DEV_ROUTE, json={'clause_id':'demo','text':'계약 조항 ' * 300})
            self.assertEqual(long.status_code, 422)
            self.assertEqual(long.json()['detail'], {'code':'TOKEN_LIMIT_EXCEEDED','max_tokens':192})
            reviewed = client.post('/v1/clauses/classify', json={
                'clause_id':'demo','text':'요약문','domain':'residential_rental',
            })
            self.assertEqual(reviewed.status_code, 503)
            self.assertEqual(reviewed.json()['detail'], 'No reviewed-data model configured')
