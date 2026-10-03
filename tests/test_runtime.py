import json, tempfile,unittest
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from tracer_runtime.vocab import vocab,TokenTransform,load_vocab,save_vocab
from tracer_runtime.forest import NumericForest
from tracer_runtime.batchnorm import CompatibleBatchNorm1d
from Model.GCN.network import MolecularGCN
ROOT=Path(__file__).resolve().parents[1]

class RuntimeTests(unittest.TestCase):
    def test_vocab_order_specials_unknown_and_padding(self):
        v=vocab(Counter({'O':6,'C':10,'rare':1}),5,['<unk>','<pad>','<bos>','<eos>']);v.set_default_index(0)
        self.assertEqual(v.get_itos(),['<unk>','<pad>','<bos>','<eos>','O','C'])
        self.assertEqual(TokenTransform(v,5,True)([['C','?'],['O']]).tolist(),[[2,5,0,3,1],[2,4,3,1,1]])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'vocab.json';save_vocab(v,p);self.assertEqual(load_vocab(p).get_stoi(),v.get_stoi())
        with self.assertRaises(IndexError):v.lookup_tokens([-1])
    def test_numeric_forest_saved_predictions(self):
        with np.load(ROOT/'tests/runtime_validation/reference.npz',allow_pickle=False) as reference:
            x=reference['fingerprints']
            for name in ['AKT1','DRD2','CXCR4']:
                f=NumericForest(ROOT/'Model/QSAR'/('qsar_'+name+'_optimized.npz'))
                np.testing.assert_allclose(f.predict_proba(x),reference['qsar_'+name],rtol=0,atol=1e-12)
                with self.assertRaises(ValueError):f.predict_proba(np.full((1,2048),np.nan))
    def test_object_npz_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'unsafe.npz';np.savez(p,offsets=np.array([object()],dtype=object))
            with self.assertRaises(ValueError):NumericForest(p)
    def test_gcn_inference_reference_and_top10(self):
        torch.set_num_threads(1);m=MolecularGCN(256,1,3,.1);m.load_state_dict(torch.load(ROOT/'ckpts/GCN/GCN.pth',map_location='cpu',weights_only=True));m.eval()
        with np.load(ROOT/'tests/runtime_validation/reference.npz',allow_pickle=False) as r:
            with torch.no_grad():out=m(*(torch.from_numpy(r['graph_'+key]) for key in ['x','edge_index','batch'])).numpy()
            np.testing.assert_allclose(out,r['gcn_logits'],rtol=0,atol=1e-5)
            np.testing.assert_array_equal(np.argsort(-out,axis=1)[:,:10],r['gcn_top10'])
    def test_batchnorm_eval_formula_and_supported_training(self):
        torch.manual_seed(73);m=CompatibleBatchNorm1d(3);x=torch.randn(10,3,requires_grad=True);m.train();out=m(x);out.square().sum().backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(int(m.num_batches_tracked),1)
        m.eval();alpha=m.weight/torch.sqrt(m.running_var+m.eps);expected=x*alpha+(m.bias-m.running_mean*alpha);torch.testing.assert_close(m(x),expected,rtol=0,atol=0)
        with self.assertRaises(ValueError):m(x.double())
    def test_model_reload_and_mode_override(self):
        m=MolecularGCN(256,1,3,.1);n=MolecularGCN(256,1,3,.1,old_compatible=False);n.load_state_dict(m.state_dict());self.assertIsInstance(m.bn1,CompatibleBatchNorm1d);self.assertIsInstance(n.bn1,torch.nn.BatchNorm1d)
        for k,v in m.state_dict().items():torch.testing.assert_close(v,n.state_dict()[k],rtol=0,atol=0)
    def test_runtime_metadata_checkpoint_safe_roundtrip(self):
        from tracer_runtime.runtime import metadata
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'runtime.pth';torch.save({'runtime':metadata(),'state_dict':{'w':torch.zeros(1)}},p);loaded=torch.load(p,map_location='cpu',weights_only=True);self.assertEqual(loaded['runtime']['torch'],str(torch.__version__))

    def test_checkpoint_unsupported_pickle_rejected(self):
        import builtins
        class Bad:
            def __reduce__(self):return builtins.eval,('1 + 1',)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'unsafe.pth';torch.save({'model_state_dict':Bad()},p)
            with self.assertRaises(Exception):torch.load(p,map_location='cpu',weights_only=True)

    def test_layernorm_old_statistics_forward_and_gradients(self):
        from tracer_runtime.normalization import CompatibleLayerNorm
        from tracer_runtime import _legacy_math
        with np.load(ROOT/'tests/runtime_validation/normalization-reference.npz',allow_pickle=False) as r:
            m=CompatibleLayerNorm(512);m.weight.data.copy_(torch.from_numpy(r['weight']));m.bias.data.copy_(torch.from_numpy(r['bias']));x=torch.from_numpy(r['x'].copy()).requires_grad_();out=m(x)
            np.testing.assert_array_equal(out.detach().numpy(),r['output'])
            mean=np.empty((len(x),1),np.float32);inv=np.empty_like(mean);_legacy_math.moments(r['x'],mean,inv,512,1e-5)
            np.testing.assert_array_equal(mean,r['mean']);np.testing.assert_array_equal(inv,r['invstd'])
            out.square().sum().backward();self.assertTrue(torch.isfinite(x.grad).all())
            with self.assertRaises(ValueError):_legacy_math.moments(r['x'].astype(np.float64),mean,inv,512,1e-5)
            with self.assertRaises(ValueError):_legacy_math.moments(r['x'],mean,inv,0,1e-5)
    def test_attention_old_masked_self_attention(self):
        from tracer_runtime.attention import CompatibleMultiheadAttention
        torch.set_num_threads(1)
        with np.load(ROOT/'tests/runtime_validation/attention-reference.npz',allow_pickle=False) as r:
            m=CompatibleMultiheadAttention(512,8,dropout=0);m.in_proj_weight.data.copy_(torch.from_numpy(r['weight']));m.in_proj_bias.data.copy_(torch.from_numpy(r['bias']));m.out_proj.weight.data.copy_(torch.from_numpy(r['out_weight']));m.out_proj.bias.data.copy_(torch.from_numpy(r['out_bias']));m.eval();x=torch.from_numpy(r['x'])
            with torch.no_grad():out,_=m(x,x,x,key_padding_mask=torch.from_numpy(r['padding']),need_weights=False)
            np.testing.assert_array_equal(out.numpy(),r['output'])
            with self.assertRaises(ValueError):m(x.double(),x.double(),x.double(),need_weights=False)
    def test_logsoftmax_buffer_rejection_and_derivative(self):
        from tracer_runtime.softmax import log_softmax
        from tracer_runtime import _legacy_math
        x=torch.tensor([[1.,2.,3.,-20.]],requires_grad=True);out=log_softmax(x,True);torch.testing.assert_close(out,torch.log_softmax(x,dim=-1),rtol=0,atol=1e-6);out.sum().backward();torch.testing.assert_close(x.grad,1-4*out.detach().exp(),rtol=0,atol=1e-6)
        with self.assertRaises(ValueError):_legacy_math.logsoftmax(np.zeros(4,np.float32),np.empty(3,np.float32),4)
        with self.assertRaises((ValueError,BufferError)):a=np.empty(4,np.float32);a.setflags(write=False);_legacy_math.logsoftmax(np.zeros(4,np.float32),a,4)

if __name__=='__main__':unittest.main()
