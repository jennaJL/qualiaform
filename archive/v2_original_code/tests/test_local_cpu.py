import json,pathlib,unittest
from qualiaform_part2.local_cpu import response_candidates,constrained_decoder,requests_for,parse_response
from qualiaform_part2.extract import read_rows
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TinyTokenizer:
    eos_token_id=0
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]
class Ids(list):
    def tolist(self):return list(self)
class LocalProtocol(unittest.TestCase):
    def test_all_json_paths_accepted(self):
        tok=TinyTokenizer()
        for condition in ('direct','structured'):
            decoder=constrained_decoder(tok,condition,2)
            for obj in response_candidates(condition):
                text=json.dumps(obj,ensure_ascii=False,separators=(',',':'));prefix=Ids([99,99])
                for token in tok.encode(text)+[0]:
                    self.assertIn(token,decoder(0,prefix));prefix.append(token)
                self.assertEqual(parse_response(text,condition),obj)
    def test_enum_counts(self):
        self.assertEqual(len(response_candidates('direct')),2)
        self.assertEqual(len(response_candidates('structured')),24)
    def test_no_fabricated_evidence(self):
        self.assertTrue(all(x['evidence']=='' for x in response_candidates('structured')))
    def test_invalid_unconstrained_pilot_stays_invalid(self):
        with self.assertRaises(ValueError):parse_response('{"typicality":"ATypical","qualia":"NONE","form_prediction":"BN","evidence":""}','structured')
    def test_train_dev_sources(self):
        train=read_rows(ROOT/'data/local_llm_v1/train_rows.jsonl');dev=read_rows(ROOT/'data/local_llm_v1/dev_rows.jsonl')
        self.assertEqual(len(train),256);self.assertEqual(len(dev),40)
        self.assertTrue(all(r['source_split']=='train' for r in train));self.assertTrue(all(r['source_split']=='dev' for r in dev))
        self.assertFalse({r['sentence_hash'] for r in train}&{r['sentence_hash'] for r in dev})
    def test_corpus_demo_overlap_rejected(self):
        rows=read_rows(ROOT/'data/processed/ud_candidates_v1/candidates.jsonl')
        with self.assertRaises(ValueError):requests_for(rows,'direct','corpus_few')
    def test_safe_inputs_and_direct_demos(self):
        rows=read_rows(ROOT/'data/local_llm_v1/pilot_rows.jsonl')
        requests=requests_for(rows,'direct','corpus_few')
        for r in requests:
            self.assertEqual(set(r['request']['input']),{'masked_context','noun_lemma','verb_lemma','relation'})
            self.assertTrue(all(set(d['output'])=={'form_prediction'} for d in r['request']['demonstrations']))
    def test_reference_label_does_not_change_request(self):
        rows=read_rows(ROOT/'data/local_llm_v1/pilot_rows.jsonl')[:1]
        before=requests_for(rows,'structured','zero');rows[0]['observed_form']='OTHER'
        self.assertEqual(before,requests_for(rows,'structured','zero'))
if __name__=='__main__':unittest.main()
