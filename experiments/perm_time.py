"""GPU symmetric-permutation time with sporder.reorder (CSR + features), per matrix."""
import sys, time, numpy as np, pandas as pd, torch, spmatkit, sporder
rows=[]
for mat in open(sys.argv[1]).read().split():
    A=spmatkit.load_matrix(mat,load_mode='pattern').tocsr(); A.sort_indices(); n=A.shape[0]
    p=torch.tensor(A.indptr,dtype=torch.int32,device='cuda'); i=torch.tensor(A.indices,dtype=torch.int32,device='cuda')
    perm=torch.tensor(spmatkit._load_row_perm(mat,'sporder-csg','a609629886c6'),dtype=torch.int32,device='cuda')
    X=torch.randn(n,128,device='cuda')
    for _ in range(3): sporder.reorder(p,i,perm,symmetric=True); X[perm.long()]
    ts=[]
    for _ in range(5):
        torch.cuda.synchronize(); t0=time.perf_counter()
        sporder.reorder(p,i,perm,symmetric=True); X[perm.long()]
        torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1e3)
    rows.append(dict(matrix=mat,perm_ms_sporder=float(np.median(ts)))); print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(sys.argv[2] if len(sys.argv) > 2 else 'perm_time_sporder.csv',index=False)
