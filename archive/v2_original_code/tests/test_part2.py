import copy, json, pathlib, tempfile, unittest
from qualiaform_part2.extract import conllu, extract, read_rows, digest
from qualiaform_part2.experiment import model_input, validate_response, demonstrations
from qualiaform_part2.evaluate import metrics, evaluate
ROOT=pathlib.Path(__file__).resolve().parents[1]
def token(i,form,pos,head,rel,xpos='_'):
    return '\t'.join(map(str,[i,form,form,pos,xpos,'_',head,rel,'_','_']))
def example(yi=False,number='一',determiner=False,clf_rel='clf',noun_head=1):
    lines=['# sent_id = fixture','# text = synthetic',token(1,'买','VERB',0,'root')]
    if yi:
        lines += [token(2,number,'NUM',4,'nummod'),token(3,'本','NOUN',2,clf_rel,'NNB'),token(4,'书','NOUN',noun_head,'obj')]
    elif determiner:
        lines += [token(2,'这','DET',3,'det'),token(3,'书','NOUN',noun_head,'obj')]
    else: lines += [token(2,'书','NOUN',noun_head,'obj')]
    return '\n'.join(lines)+'\n'
class Extraction(unittest.TestCase):
    def parse(self,text):
        with tempfile.TemporaryDirectory() as tmp:
            p=pathlib.Path(tmp)/'x.conllu';p.write_text(text);return list(extract(p,'dev'))
    def test_bare(self): self.assertEqual(self.parse(example())[0]['observed_form'],'BN')
    def test_yi(self): self.assertEqual(self.parse(example(True))[0]['observed_form'],'YI_CL_N')
    def test_two_excluded(self): self.assertEqual(self.parse(example(True,number='两')),[])
    def test_det_excluded(self): self.assertEqual(self.parse(example(determiner=True)),[])
    def test_missing_clf_excluded(self): self.assertEqual(self.parse(example(True,clf_rel='dep')),[])
    def test_mask_identical_between_forms(self):
        self.assertEqual(model_input(self.parse(example())[0]),model_input(self.parse(example(True))[0]))
    def test_noun_not_verb(self): self.assertEqual(self.parse(example().replace('VERB','NOUN')),[])
    def test_no_eof_blank_required(self): self.assertEqual(len(self.parse(example().strip())),1)
    def test_invalid_head(self):
        with self.assertRaises(ValueError): self.parse(example(noun_head=99))
    def test_cycle(self):
        with self.assertRaises(ValueError): self.parse(example().replace('VERB\t_\t_\t0','VERB\t_\t_\t2'))
    def test_nested_determination(self):
        text=example()+token(3,'词汇','NOUN',2,'appos')+'\n'+token(4,'这个','DET',3,'det')
        self.assertEqual(self.parse(text),[])
    def test_skip_empty_and_multiword(self):
        text=example()+'1-2\t买书\t_\t_\t_\t_\t_\t_\t_\t_\n2.1\t它\t它\tPRON\t_\t_\t1\tobj\t_\t_'
        self.assertEqual(len(self.parse(text)),1)
class Evaluation(unittest.TestCase):
    def test_known_matrix(self):
        m=metrics([('BN','BN'),('BN','YI_CL_N'),('YI_CL_N','YI_CL_N'),('YI_CL_N','INVALID')])
        self.assertEqual(m['accuracy'],.5);self.assertEqual(m['balanced_accuracy'],.5)
        self.assertAlmostEqual(m['macro_f1'],7/12);self.assertEqual(m['invalid'],1)
    def test_absent_class(self): self.assertIsNone(metrics([('BN','BN')])['balanced_accuracy'])
    def test_empty(self): self.assertIsNone(metrics([])['accuracy'])
    def test_unknown_id(self):
        with self.assertRaises(ValueError): evaluate([], [{'candidate_id':'x'}])
    def test_duplicate_id(self):
        with self.assertRaises(ValueError): evaluate([], [{'candidate_id':'x'},{'candidate_id':'x'}])
    def test_invalid_response(self):
        with self.assertRaises(ValueError): validate_response({'form_prediction':'OTHER'},'direct')
    def test_structured_schema(self):
        with self.assertRaises(ValueError): validate_response({'form_prediction':'BN'},'structured')
class Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.rows=read_rows(ROOT/'data/processed/ud_candidates_v1/candidates.jsonl')
    def test_manifest_hashes(self):
        folder=ROOT/'data/processed/ud_candidates_v1';manifest=json.loads((folder/'manifest.json').read_text())
        for name,h in manifest['output_sha256'].items(): self.assertEqual(digest((folder/name).read_bytes()),h)
    def test_trace_every_occurrence(self):
        for r in self.rows:
            ts={t['id']:t for t in r['dependency_info']}
            self.assertEqual(ts[r['noun_token_id']]['form'],r['noun']);self.assertEqual(ts[r['verb_token_id']]['upos'],'VERB')
            self.assertEqual(r['masked_context'].count('[TARGET_NP]'),1)
            self.assertEqual(set(model_input(r)),{'masked_context','noun_lemma','verb_lemma','relation'})
    def test_no_sentence_split_overlap(self):
        sets=[{r['sentence_hash'] for r in self.rows if r['source_split']==s} for s in ('train','dev','test')]
        for i in range(3):
            for j in range(i): self.assertFalse(sets[i]&sets[j])
    def test_synthetic_forbidden_outside_mock(self):
        with self.assertRaises(ValueError): demonstrations(ROOT/'tests/synthetic_demos.jsonl',self.rows,'direct')
    def test_demo_leakage_rejected(self):
        demo=json.loads((ROOT/'tests/synthetic_demos.jsonl').read_text().splitlines()[0]);demo['input']=model_input(self.rows[0])
        with tempfile.TemporaryDirectory() as tmp:
            p=pathlib.Path(tmp)/'d.jsonl';p.write_text(json.dumps(demo))
            with self.assertRaises(ValueError): demonstrations(p,self.rows,'direct',True)
if __name__=='__main__':unittest.main()

class Fairness(unittest.TestCase):
    def test_direct_demos_omit_semantic_fields(self):
        demos=demonstrations(ROOT/'tests/synthetic_demos.jsonl',[],'direct',True)
        self.assertEqual(set(demos[0]['output']),{'form_prediction'})
    def test_structured_demos_keep_semantics(self):
        demos=demonstrations(ROOT/'tests/synthetic_demos.jsonl',[],'structured',True)
        self.assertIn('qualia',demos[0]['output'])
