"""Static inventory of schedule/device hard-coding outside Planner V2."""
import ast,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
KEYS={'bm','bn','bk','warps','stages','num_warps','num_stages','split_k','schedule','sm'}
class Visitor(ast.NodeVisitor):
 def __init__(self,path):self.path=path;self.rows=[]
 def record(self,node,key,value):self.rows.append(dict(path=str(self.path.relative_to(ROOT)),line=node.lineno,key=key,value=value))
 def visit_Call(self,node):
  for item in node.keywords:
   if item.arg in KEYS:
    try:value=ast.literal_eval(item.value)
    except Exception:value=ast.unparse(item.value)
    self.record(item,item.arg,value)
  self.generic_visit(node)
 def visit_Dict(self,node):
  for key,value in zip(node.keys,node.values):
   try:name=ast.literal_eval(key)
   except Exception:name=None
   if name in KEYS:
    try:raw=ast.literal_eval(value)
    except Exception:raw=ast.unparse(value)
    self.record(key,name,raw)
  self.generic_visit(node)
 def visit_Compare(self,node):
  text=ast.unparse(node)
  if '.sm' in text or 'sm ' in text:self.record(node,'sm_condition',text)
  self.generic_visit(node)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();rows=[]
 for path in sorted((ROOT/'src/acc_infer_clear').rglob('*.py')):
  if 'planner_v2' in path.parts:continue
  try:tree=ast.parse(path.read_text())
  except SyntaxError:continue
  visitor=Visitor(path);visitor.visit(tree);rows.extend(visitor.rows)
 grouped={}
 for row in rows:grouped[row['path']]=grouped.get(row['path'],0)+1
 result=dict(total=len(rows),files=len(grouped),by_file=dict(sorted(grouped.items(),key=lambda item:(-item[1],item[0]))),rows=rows,
  policy='Delete only after the corresponding V2 role passes component, service, and quality gates.')
 out=ROOT/'reports'/a.output;out.mkdir(parents=True,exist_ok=False);(out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(total=result['total'],files=result['files'],top=list(result['by_file'].items())[:15]),indent=2))
if __name__=='__main__':main()
