#!/usr/bin/env python3
"""Exact unweighted edge centralities via grounded Laplacians; compare CSR scores."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def exact_scores(path):
    edges=set()
    with Path(path).open() as source:
        for line in source:
            if not line.strip() or line.lstrip().startswith(('#','%')): continue
            fields=line.split()
            if len(fields)<2: raise ValueError('invalid edge line')
            u,v=map(int,fields[:2])
            if min(u,v)<0: raise ValueError('negative vertex')
            if u!=v: edges.add((min(u,v),max(u,v)))
    if not edges: raise ValueError('empty graph')
    n=max(max(e) for e in edges)+1
    # This is exactly the source-major sorted, deduplicated adjacency order.
    directed=sorted(list(edges)+[(v,u) for u,v in edges])
    row=np.array([u for u,v in directed]); col=np.array([v for u,v in directed])
    adjacency=coo_matrix((np.ones(len(row)),(row,col)),shape=(n,n)).tocsr()
    if np.any(np.diff(adjacency.indptr)==0): raise ValueError('noncontiguous IDs or isolated vertex')
    components,labels=connected_components(adjacency,directed=False)
    result=np.zeros(len(row))
    residuals=[]
    for component in range(components):
        nodes=np.flatnonzero(labels==component)
        a=adjacency[nodes][:,nodes].toarray()
        laplacian=np.diag(a.sum(axis=1))-a
        grounded=laplacian[:-1,:-1]
        factor=cho_factor(grounded,lower=True)
        inverse=cho_solve(factor,np.eye(len(nodes)-1))
        # Certify the solve on deterministic probe vectors without another cubic product.
        probe=np.random.default_rng(42).normal(size=(len(nodes)-1,4))
        solved=cho_solve(factor,probe)
        residuals.append(float(np.max(np.abs(grounded@solved-probe))))
        green=np.zeros_like(laplacian);green[:-1,:-1]=inverse
        local=np.full(n,-1,dtype=np.int64);local[nodes]=np.arange(len(nodes))
        positions=np.flatnonzero(labels[row]==component)
        u,v=local[row[positions]],local[col[positions]]
        result[positions]=green[u,u]+green[v,v]-2*green[u,v]
    # Foster's theorem: sum of undirected effective resistances = n-components.
    foster_error=abs(result.sum()/2-(n-components))
    if not np.all(np.isfinite(result)) or max(residuals)>1e-6 or foster_error>1e-5*max(1,n):
        raise RuntimeError('exact reference failed residual/Foster checks')
    return result,{'vertices':n,'directed_edges':len(row),'components':components,
                   'max_solve_residual':max(residuals),'foster_error':foster_error}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('graph',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--scores',action='append',required=True,help='epsilon=score-file, repeatable')
    args=p.parse_args()
    exact,validation=exact_scores(args.graph)
    np.save(args.output.with_suffix('.exact.npy'),exact)
    rows=[]
    for spec in args.scores:
        epsilon,path=spec.split('=',1);epsilon=float(epsilon)
        estimated=np.fromfile(path,dtype=np.float32)
        if estimated.shape!=exact.shape or not np.all(np.isfinite(estimated)): raise ValueError('invalid CSR score output: '+path)
        errors=np.abs(estimated-exact)
        rows.append({'epsilon':epsilon,'max_absolute_error':float(errors.max()),
                     'mean_absolute_error':float(errors.mean()),'rmse':float(np.sqrt(np.mean(errors**2))),
                     'directed_edges_over_epsilon':int(np.count_nonzero(errors>epsilon)),
                     'within_epsilon':bool(errors.max()<=epsilon)})
    with args.output.open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    args.output.with_suffix('.reference.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps({'reference':validation,'accuracy':rows},indent=2))

if __name__=='__main__':main()
