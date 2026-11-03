import unittest
try:
 from fastapi.testclient import TestClient
 from qualiaform_part2.service import create_app
 HAVE_API=True
except ImportError:HAVE_API=False
class FakeService:
 name='test-double'
 def predict(self,items):return [{'label':'BN','yi_score_uncalibrated':.1,'decision_threshold':.5,'model_id':self.name,'input_truncated':False,'nearest_training_examples':[]} for _ in items]
 def benchmark_summary(self):return {'selected':self.name,'test_n':913,'models':{}}
@unittest.skipUnless(HAVE_API,'optional FastAPI dependencies unavailable')
class ServiceTests(unittest.TestCase):
 def setUp(self):
  self.client=TestClient(create_app(FakeService()));self.client.__enter__()
  self.payload={'masked_context':'PRIVATE_SENTINEL[TARGET_NP]。','noun_lemma':'书','verb_lemma':'买','relation':'obj'}
 def tearDown(self):self.client.__exit__(None,None,None)
 def test_health(self):self.assertEqual(self.client.get('/healthz').json()['status'],'ready')
 def test_prediction(self):
  r=self.client.post('/predict',json=self.payload);self.assertEqual(r.status_code,200);self.assertEqual(r.json()['label'],'BN');self.assertIn('X-Request-ID',r.headers)
 def test_gold_field_rejected(self):self.assertEqual(self.client.post('/predict',json={**self.payload,'observed_form':'BN'}).status_code,422)
 def test_no_mask_rejected(self):self.assertEqual(self.client.post('/predict',json={**self.payload,'masked_context':'没有占位符'}).status_code,422)
 def test_two_masks_rejected(self):self.assertEqual(self.client.post('/predict',json={**self.payload,'masked_context':'[TARGET_NP][TARGET_NP]'}).status_code,422)
 def test_blank_lemma_rejected(self):self.assertEqual(self.client.post('/predict',json={**self.payload,'noun_lemma':'  '}).status_code,422)
 def test_long_input_rejected(self):self.assertEqual(self.client.post('/predict',json={**self.payload,'masked_context':'字'*1300+'[TARGET_NP]'}).status_code,422)
 def test_batch(self):self.assertEqual(len(self.client.post('/predict_batch',json={'items':[self.payload]*3}).json()),3)
 def test_oversize_batch_rejected(self):self.assertEqual(self.client.post('/predict_batch',json={'items':[self.payload]*17}).status_code,422)
 def test_invalid_relation_rejected(self):self.assertEqual(self.client.post('/predict',json={**self.payload,'relation':'root'}).status_code,422)
 def test_openapi(self):self.assertIn('/predict',self.client.get('/openapi.json').json()['paths'])
 def test_metrics(self):self.assertIn('qualiaform_requests_total',self.client.get('/metrics').text)
 def test_logs_exclude_text(self):
  with self.assertLogs('qualiaform.requests',level='INFO') as logs:self.client.post('/predict',json=self.payload)
  self.assertNotIn('PRIVATE_SENTINEL',' '.join(logs.output))
 def test_demo_page(self):self.assertIn('QualiaForm',self.client.get('/').text)
