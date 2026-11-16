"""Legacy experiments must stay separate from human-approved training."""
import unittest
from pathlib import Path
from experiments.legacy_roberta.run import load_protocol, metrics, prediction_rows, BASE, ROOT
from ml.train_klue_roberta import preflight

LEGACY_DATA = (ROOT/'data/raw/clause_classification_original.csv').is_file() and (ROOT/'data/raw/clause_pairs_original.csv').is_file()


class LegacyExperimentTests(unittest.TestCase):
    @unittest.skipUnless(LEGACY_DATA, 'Authorized legacy CSVs required')
    def test_exact_baseline_folds_and_complete_oof(self):
        rows,y,folds,assignments=load_protocol()
        self.assertEqual(len(rows),143)
        self.assertEqual(len(folds),5)
        self.assertEqual(len(assignments),143)
        heldout=[]
        for tr,va in folds:
            self.assertFalse({rows[i]['doc_id'] for i in tr}&{rows[i]['doc_id'] for i in va})
            self.assertEqual(set(y[tr]),{0,1})
            self.assertEqual(set(y[va]),{0,1})
            heldout.extend(va.tolist())
        self.assertEqual(sorted(heldout),list(range(143)))

    def test_metrics_matrix_orientation(self):
        m=metrics([0,0,1,1],[0,1,1,1])
        self.assertEqual(m['confusion_matrix_true_rows_pred_cols_0_1'],[[1,1],[0,2]])
        self.assertAlmostEqual(m['classification_report']['1']['recall'],1.)
        self.assertAlmostEqual(m['macro_f1'],(2/3+.8)/2)

    def test_prediction_records_preserve_weak_labels(self):
        rows=[dict(sample_id='example',clause_id='example',doc_id='sample',text='테스트 문장',label_weak='0')]
        pred=prediction_rows(rows,[0],1,[1],[.7])[0]
        self.assertEqual(pred['sample_id'],'example')
        self.assertIn('label_true_weak',pred)
        self.assertNotIn('approved_label',pred)

    def test_final_training_still_blocked(self):
        _,_,issues=preflight()
        self.assertTrue(issues)

    def test_output_tree_separate(self):
        self.assertTrue(BASE.is_relative_to(ROOT/'experiments'))
        self.assertFalse(BASE.is_relative_to(ROOT/'artifacts'))


if __name__=='__main__':unittest.main()
