"""Stage the current Overleaf article with reproducible benchmark additions."""
from pathlib import Path
import json,shutil,hashlib,difflib
ROOT=Path(__file__).resolve().parents[2]
ARTICLE=Path('C:/Users/serge/Downloads/OhMyU1_public/Article')
stage=ROOT/'tmp/protes_article';stage.mkdir(parents=True,exist_ok=True)
for p in ARTICLE.iterdir():
    if p.is_file() and p.suffix in ['.tex','.bib','.sty','.bst']:
        shutil.copy2(p,stage/p.name)
shutil.copytree(ARTICLE/'pictures',stage/'pictures',dirs_exist_ok=True)
names=['benchmark_protes_memory_julia','benchmark_protes_sampling_julia','benchmark_protes_u1_5s_julia','benchmark_protes_u1_50s_julia']
for name in names:shutil.copy2(ROOT/'examples/pictures'/f'{name}.pdf',stage/'pictures'/f'{name}.pdf')
original=(ARTICLE/'main_v1.tex').read_text(encoding='utf-8')
assert '% BEGIN PROTES BENCHMARK ADDITION' not in original
(stage/'source_sha256.txt').write_text(hashlib.sha256((ARTICLE/'main_v1.tex').read_bytes()).hexdigest())
(stage/'main_v1.before_protes.tex').write_text(original,encoding='utf-8')
full=json.loads((ROOT/'examples/protes_full_comparison_results.json').read_text())
old=json.loads((ROOT/'examples/protes_scaling_results.json').read_text())['results']
sens=json.loads((ROOT/'examples/protes_sensitivity_results.json').read_text())['results']
tables=[r'''\begin{table}[t]
\centering\small
\caption{Recorded 50-second PROTES runs at $n=8$. Triples are new distinct feasible points in seed order $j=0,1,2$; ranges cover the three runs. Generated counts include infeasible draws.}
\label{tab:protes_long_counts}
\begin{tabular}{llrrr}\toprule
Configuration & Objective & New points & Generated draws (range) & Actual time (s)\\\midrule''']
for name in ['default','rank20','slow10','batch400','fit20']:
    for obj in ['linear','quadratic']:
        phase='long' if name=='default' else f'sensitivity_{name}_long'
        g=sorted([r for r in old+sens if r['config']['phase']==phase and r['config']['n']==8 and r['config']['objective']==obj],key=lambda r:r['config']['repetition'])
        triple=', '.join(str(r['novel_unique']) for r in g)
        generated=f"{min(r['generated'] for r in g)}--{max(r['generated'] for r in g)}"
        times=f"{min(r['elapsed_seconds'] for r in g):.2f}--{max(r['elapsed_seconds'] for r in g):.2f}"
        tables.append(f'{name} & {obj} & {triple} & {generated} & {times}'+r'\\')
tables.append(r'''\bottomrule\end{tabular}\end{table}
\begin{table}[t]
\centering\small
\caption{Recorded EELS workloads. Iterations are completed/started outer iterations; the last started iteration may be interrupted. All generated draws counted here are feasible. Novelty excludes the initial observations.}
\label{tab:u1_timed_counts}
\begin{tabular}{rlrrrrrr}\toprule
$n$ & Objective & $j$ & Budget (s) & Actual (s) & Iterations & Draws & New points\\\midrule''')
for r in full['data_u1']:
    tables.append(f"{r['n']} & {r['objective']} & {r['repetition']} & {float(r['budget']):.0f} & {float(r['elapsed_seconds']):.3f} & {r['completed_iterations']}/{r['iterations']} & {r['generated']} & {r['novel_unique']}"+r'\\')
tables.append(r'\bottomrule\end{tabular}\end{table}')
# Finite-data PROTES workload ranges document the short screen without a 70-row table.
tables.append(r'''\begin{table}[t]\centering\small
\caption{PROTES 5-second screen: workload ranges over the 14 size/objective pairs per configuration. Each pair uses one seed. Exact per-instance counts and times are stored in the results snapshot.}
\label{tab:protes_short_workload}
\begin{tabular}{lrr}\toprule Configuration & Generated draws (range) & Actual time (s)\\\midrule''')
for name in ['default','rank20','slow10','batch400','fit20']:
    phase='pilot' if name=='default' else f'sensitivity_{name}_short'
    g=[r for r in old+sens if r['config']['phase']==phase and r['config']['repetition']==0]
    assert len(g)==14
    tables.append(f"{name} & {min(r['generated'] for r in g)}--{max(r['generated'] for r in g)} & {min(r['elapsed_seconds'] for r in g):.2f}--{max(r['elapsed_seconds'] for r in g):.2f}"+r'\\')
tables.append(r'\bottomrule\end{tabular}\end{table}')
main=(ROOT/'scripts/protes/article_benchmark_main.tex').read_text()
appendix=(ROOT/'scripts/protes/article_benchmark_appendix.tex').read_text().replace('% GENERATED BENCHMARK TABLES','\n'.join(tables))
updated=original.replace(r'\section{Acknowledgements}',main+'\n\n'+r'\section{Acknowledgements}',1)
updated=updated.replace(r'\end{document}',appendix+'\n\n'+r'\end{document}',1)
(stage/'main_v1.tex').write_text(updated,encoding='utf-8')
patch=''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),fromfile='main_v1.before_protes.tex',tofile='main_v1.tex'))
(ROOT/'examples/main_v1.protes_benchmarks.patch').write_text(patch,encoding='utf-8')
print(stage/'main_v1.tex')
