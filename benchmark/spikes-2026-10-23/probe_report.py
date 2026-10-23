import sys,statistics
rows=[l.rstrip('\n').split('\t') for l in open(sys.argv[1])][1:]
dm=[float(r[2]) for r in rows]; vm=[float(r[4]) for r in rows]
mm=[i for i,r in enumerate(rows) if r[1]!=r[3]]
print("positions",len(rows),"distinct decode tokens",len(set(r[1] for r in rows)),"mismatches",len(mm))
print("decode margin median %.3f; verify margin median %.3f"%(statistics.median(dm),statistics.median(vm)))
print("decode margins < 0.25:",sum(m<0.25 for m in dm),"< 0.5:",sum(m<0.5 for m in dm))
for i in mm[:12]: print(" mismatch pos",i,rows[i][1:],)
# margins at agreeing positions vs mismatching
if mm: print("decode margin at mismatches: max %.3f"%max(dm[i] for i in mm))
