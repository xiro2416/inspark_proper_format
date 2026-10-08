"""ZipVoice resource, profile and isolation checks without initializing CUDA."""
import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from inspark_infer.build import zipvoice
from inspark_infer.runtime.zipvoice import cli

class ZipVoiceBundleTest(unittest.TestCase):
 def test_plugin_package_is_bound_to_actual_batch(self):
  self.assertEqual(zipvoice.plugin_package(1),'inspark_infer.ops.tensorrt.zipvoice.a1007.b1')
  self.assertEqual(zipvoice.plugin_package(1,'inspark_infer.ops.tensorrt.zipvoice.a1007.b1_geo1'),'inspark_infer.ops.tensorrt.zipvoice.a1007.b1_geo1')
  for value in ('','inspark_infer.ops.tensorrt.zipvoice.a1007.b2_geo1','foreign.b1','inspark_infer.ops.tensorrt.zipvoice.a1007.b1../b2'):
   with self.assertRaisesRegex(ValueError,'target batch'):zipvoice.plugin_package(1,value)
 def test_unsupported_batch_never_downloads(self):
  with self.assertRaisesRegex(ValueError,'Supported INT8'):zipvoice.ensure(128)
 def test_path_traversal_is_rejected(self):
  with tempfile.TemporaryDirectory(dir='/workspace') as d:
   for path in ['../outside','/absolute']:
    with self.assertRaises(ValueError):zipvoice.safe_path(Path(d),path)
 def test_profile_checks_all_components(self):
  m={'batch':32,'engines':{'fm':{'shape_profile':{'x':[[32,600,100],[32,760,100],[32,920,100]]}},'text':{'shape_profile':{'token_ids':[[32,52],[32,78],[32,141]]}},'vocos':{'shape_profile':{'mel':[[32,100,225],[32,100,385],[32,100,545]]}}}}
  w={'batch':32,'prompt_frames':375,'target_frames':385,'total_frames':760,'joint_tokens':77,'padded_tokens':78,'steps':8,'t_shift':.5,'guidance':1.,'feat_scale':.1}
  zipvoice.check_workload(m,w)
  for change in [{'target_frames':546,'total_frames':921},{'target_frames':224,'total_frames':599},{'joint_tokens':141,'padded_tokens':142},{'joint_tokens':50,'padded_tokens':51},{'prompt_frames':374},{'batch':16}]:
   with self.assertRaises(ValueError):zipvoice.check_workload(m,{**w,**change})
 def test_manifest_binds_engine_to_inventory(self):
  with tempfile.TemporaryDirectory(dir='/workspace') as d:
   p=Path(d);engine=p/'engine.plan';engine.write_bytes(b'wrong')
   m={'schema':1,'model':'zipvoice','precision':'int8','batch':32,'certified_for_production':False,'engines':{k:{'path':'engine.plan','sha256':'bad','shape_profile':{'x':[]}} for k in zipvoice.COMPONENTS},'files':{'engine.plan':{'sha256':'bad','bytes':5}}}
   (p/'manifest.json').write_text(json.dumps(m))
   with self.assertRaisesRegex(ValueError,'integrity'):zipvoice.validate_bundle(p)
 def test_worker_does_not_inherit_index_triton(self):
  with patch.dict(os.environ,{'INSPARK_ZIPVOICE_PYTHON':os.sys.executable,'PYTHONPATH':'/workspace/foreign-triton','ACC_TRT113_SITE':'/workspace/foreign-trt'}),patch('subprocess.call',return_value=0) as call:
   self.assertEqual(cli.main(['infer','--help']),0)
   env=call.call_args.kwargs['env'];self.assertEqual(env['PYTHONPATH'],str(zipvoice.root()/'src'));self.assertNotIn('ACC_TRT113_SITE',env)
 def test_command_routes_explicit_model_only(self):
  from inspark_infer.command import main
  with patch('inspark_infer.runtime.zipvoice.cli.main',return_value=0) as worker:
   self.assertEqual(main(['trt','ensure','--model=zipvoice','--batches','32']),0)
   self.assertEqual(worker.call_args.args[0][0],'ensure')
  with patch('inspark_infer.runtime.zipvoice.cli.main') as worker:
   with self.assertRaises(SystemExit):main(['trt','ensure','--model','indextts2','--ref-audio','zipvoice'])
   worker.assert_not_called()
 def test_registry_targets_a1007_only(self):
  r=zipvoice.registry();self.assertEqual(zipvoice.BATCHES,(1,2,4,8,16,32,64));self.assertEqual(r['precision'],'int8')
  self.assertEqual(r['frame_profile'],[600,760,920]);self.assertEqual(r['token_profile'],[52,78,141])
  if r.get('status')=='a1007_build_and_validation_in_progress':self.assertEqual(r['bundles'],{})
  else:self.assertEqual(set(r['bundles']),{'1','2','4','8','16','32','64'})

if __name__=='__main__':unittest.main()
