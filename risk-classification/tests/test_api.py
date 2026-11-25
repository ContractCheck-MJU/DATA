import secrets
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.app import create_app


TEST_KEY = secrets.token_hex(24)
HEADERS = {"X-Internal-Api-Key": TEST_KEY}

class FakePredictor:
    def predict(self, text, domain):
        if text == 'too long':
            raise ValueError('token limit')
        return True, 'test-fixture-not-a-model'


class APITests(unittest.TestCase):
    def test_no_model_returns_503(self):
        with patch.dict('os.environ',{'CONTRACTCHECK_INTERNAL_API_KEY': TEST_KEY},clear=True), TestClient(create_app(), headers=HEADERS) as c:
            self.assertEqual(c.get('/health/live').status_code,200)
            self.assertEqual(c.get('/health/ready').status_code,503)
            self.assertEqual(c.post('/v1/clauses/classify',json={'clause_id':'a','text':'조항','domain':'residential_rental'}).status_code,503)

    def test_contract_and_input_guards(self):
        with patch.dict('os.environ', {'CONTRACTCHECK_INTERNAL_API_KEY': TEST_KEY}), TestClient(create_app(FakePredictor()), headers=HEADERS) as c:
            body={'clause_id':'a','text':'조항','domain':'residential_rental'}
            response=c.post('/v1/clauses/classify',json=body)
            self.assertEqual(response.status_code,200)
            data=response.json()
            self.assertTrue(data['needs_human_review'])
            self.assertIsNone(data['risk_level'])
            self.assertIsNone(data['probability_model_score'])
            for text in ('   ','too long','x'*10001):
                self.assertEqual(c.post('/v1/clauses/classify',json=dict(body,text=text)).status_code,422)
            self.assertEqual(c.post('/v1/clauses/classify',json=dict(body,domain='commercial')).status_code,422)
