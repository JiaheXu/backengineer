from pathlib import Path
import json,numpy as np,torch,onnx,onnxruntime as ort
from onnx import helper
from backengineer.depth_recovery.onnx_reference import OnnxDepthReference
from backengineer.depth_recovery.torch_network import ZedDepthNetwork
from backengineer.depth_recovery.compare_images import load_stereo_images

class TraceReference(OnnxDepthReference):
 def _add_segment(self,nodes,remaining):
  produced={name for node in nodes for name in node.output};consumed={name for node in nodes for name in node.input if name}
  inputs=sorted(consumed-produced-self.initializers.keys());outputs=sorted(produced)
  assert all(name in self.metadata for name in outputs)
  graph=helper.make_graph(nodes,'trace_segment',[self.metadata[x] for x in inputs],[self.metadata[x] for x in outputs],[v for v in self.model.graph.initializer if v.name in consumed])
  model=helper.make_model(graph,opset_imports=list(self.model.opset_import),ir_version=self.model.ir_version)
  self.options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  session=ort.InferenceSession(model.SerializeToString(),self.options,providers=['CPUExecutionProvider'])
  self.steps.append(('ort',session,inputs,outputs))

root=Path('backengineer/depth_recovery');torch.set_num_threads(2)
for name in ['neural_depth_5.4','neural_depth_5.6']:
 path=root/(name+'.onnx');dtype=np.float16 if name.endswith('5.6') else np.float32
 left,right,_=load_stereo_images(root/'input/im0.png',root/'input/im1.png',dtype,height=256,width=512)
 model=ZedDepthNetwork(path).eval();model.load_state_dict(torch.load(path.with_suffix('.pth'),weights_only=True))
 with torch.no_grad():tout,trace=model(torch.from_numpy(left),torch.from_numpy(right),return_intermediates=True)
 ref=TraceReference(path);values=dict(ref.initializers,left=left,right=right)
 for step in ref.steps:
  inputs=ref._inputs(step)
  if step[0]=='ort':
   _,session,names,outputs=step;results=session.run(outputs,{n:values[n] for n in names})
  else:
   _,node,attrs=step;outputs=node.output;args=[values[n] if n else None for n in node.input]
   import backengineer.depth_recovery.onnx_reference as r
   results=[r.quantize_linear(*args,axis=attrs.get('axis',1)) if node.op_type=='QuantizeLinear' else getattr(r,'census_xnor_cost_volume' if node.op_type=='CensusXnorCostVolume' else node.op_type)(*args,**attrs)]
  values.update(zip(outputs,results))
 records=[]
 for index,(nodename,op,inputs,outputs,attrs) in enumerate(model.nodes):
  key=outputs[0];actual=trace[key].numpy();expected=values[key]
  diff=np.abs(actual.astype(np.float64)-expected.astype(np.float64));different=actual!=expected
  record={'index':index,'node':nodename,'op':op,'output':key,'dtype':str(actual.dtype),'shape':list(actual.shape),'max_abs':float(diff.max()) if diff.size else 0,'mean_abs':float(diff.mean()) if diff.size else 0,'changed':int(different.sum()),'elements':int(actual.size)}
  if op=='QuantizeLinear':
   inp=inputs[0];a=trace[inp].detach().numpy().astype(np.float64);b=values[inp].astype(np.float64);record['input_max_abs']=float(np.abs(a-b).max());record['input_changed']=int((a!=b).sum())
  if op=='QuantizeLinear' and record['changed'] and not any(x['op']=='QuantizeLinear' and x['changed'] for x in records):
   import backengineer.depth_recovery.onnx_reference as r
   args=[trace[k].detach().numpy() if k else None for k in inputs]
   isolated=r.quantize_linear(*args,axis=attrs.get('axis',1))
   record['same_input_numpy_matches_torch_exactly']=bool(np.array_equal(isolated,actual))
   loc=tuple(np.argwhere(different)[0]);scale=args[1]
   record['first_changed_index']=list(map(int,loc));record['torch_input']=float(trace[inputs[0]].detach().numpy()[loc]);record['onnx_input']=float(values[inputs[0]][loc]);record['scale']=float(scale.reshape(-1)[0]);record['torch_quantized']=int(actual[loc]);record['onnx_quantized']=int(expected[loc])
  if op=='CensusXnorCostVolume':
   import backengineer.depth_recovery.onnx_reference as r
   args=[trace[k].detach().numpy() for k in inputs]
   isolated=r.census_xnor_cost_volume(*args,**attrs)
   record['same_input_numpy_matches_torch_exactly']=bool(np.array_equal(isolated,actual))
   record['left_input_sign_changes']=int(((args[0]>0)!=(values[inputs[0]]>0)).sum());record['right_input_sign_changes']=int(((args[1]>0)!=(values[inputs[1]]>0)).sum())
  records.append(record)
 dest=root/'pth_onnx_comparison/diagnostics'/(name+'.trace.json');dest.write_text(json.dumps(records,indent=2)+'\n')
 print(name,'first nonexact',next((x for x in records if x['changed']),None),flush=True)
 print(name,'first quant diff',next((x for x in records if x['op']=='QuantizeLinear' and x['changed']),None),flush=True)
 print(name,'first census',next(x for x in records if x['op']=='CensusXnorCostVolume'),flush=True)
 print(name,'outputs',[(k,float(np.abs(tout[k].numpy().astype(np.float64)-values[k].astype(np.float64)).max())) for k in model.output_names],flush=True)
 del model,ref,values,trace,tout,records
