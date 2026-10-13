"""Stage the current Overleaf article with reproducible benchmark additions."""
from pathlib import Path
import json,shutil,hashlib,difflib,re,statistics,math
from backend_article import update_backend_article
ROOT=Path(__file__).resolve().parents[2]
ARTICLE=Path('C:/Users/serge/Downloads/OhMyU1_public/Article')
stage=ROOT/'tmp/protes_article';stage.mkdir(parents=True,exist_ok=True)
for p in ARTICLE.iterdir():
    if p.is_file() and p.suffix in ['.tex','.bib','.sty','.bst']:
        shutil.copy2(p,stage/p.name)
shutil.copytree(ARTICLE/'pictures',stage/'pictures',dirs_exist_ok=True)
names=['benchmark_protes_memory_julia','benchmark_protes_sampling_julia','benchmark_protes_u1_5s_julia','benchmark_protes_u1_50s_julia','LibsComparison_fair']
for name in names:shutil.copy2(ROOT/'examples/pictures'/f'{name}.pdf',stage/'pictures'/f'{name}.pdf')
original=(ARTICLE/'main_v1.tex').read_text(encoding='utf-8')
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
\caption{Recorded EELS workloads with the legacy sampler. Iterations are completed/started outer iterations; the last started iteration may be interrupted. All generated draws counted here are feasible. Novelty excludes the initial observations.}
\label{tab:u1_timed_counts}
\begin{tabular}{rlrrrrrr}\toprule
$n$ & Objective & $j$ & Budget (s) & Actual (s) & Iterations & Draws & New points\\\midrule''')
for r in full['data_u1']:
    tables.append(f"{r['n']} & {r['objective']} & {r['repetition']} & {float(r['budget']):.0f} & {float(r['elapsed_seconds']):.3f} & {r['completed_iterations']}/{r['iterations']} & {r['generated']} & {r['novel_unique']}"+r'\\')
tables.append(r'\bottomrule\end{tabular}\end{table}')
if full.get('data_u1_table'):
    tables.append(r'''\begin{table}[t]\centering\small
\caption{EELS workloads with the table sampler under the same inputs and budgets as the legacy runs. Iterations are completed/started; novelty excludes initial observations. All counted draws are feasible.}
\label{tab:u1_timed_counts_table}
\begin{tabular}{rlrrrrrr}\toprule
$n$ & Objective & $j$ & Budget (s) & Actual (s) & Iterations & Draws & New points\\\midrule''')
    for r in full['data_u1_table']:
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

def sci(value):
    exponent=math.floor(math.log10(value))
    return f'{value/10**exponent:.1f}'+r'\cdot10^'+f'{{{exponent}}}'

def update_statistics(text):
    rates={n:statistics.median(float(r['samples_per_second']) for r in full['full_u1'] if int(r['n'])==n) for n in [8,10]}
    dense={v['n']:statistics.median(float(r['samples_per_second']) for r in v['trials']) for v in full['full_protes'] if v['variant']=='general'}
    memory=next(r for r in full['full_u1'] if int(r['n'])==16)
    text=re.sub(r'object occupies \$[^$]+\$ GiB, compared with \$[^$]+\$ MiB',
        lambda _:f"object occupies ${int(memory['object_bytes'])/2**30:.2f}$ GiB, compared with ${int(memory['numeric_bytes'])/2**20:.1f}$ MiB",text)
    start=text.index('approximately $',text.index('At $n=8$ and $n=10$'))
    end=text.index(' These measurements',start)
    text=text[:start]+('approximately $'+sci(rates[8])+'$ and $'+sci(rates[10])+r'''$ samples per second,
respectively, versus $'''+sci(dense[8])+'$ and $'+sci(dense[10])+r'''$ for the more
memory-efficient \texttt{protes\_general}. The resulting speedups are
approximately $'''+f'{rates[8]/dense[8]:.1f}'+r'\times$ and $'+f'{rates[10]/dense[10]:.1f}'+r'''\times$.
Small instances do not exhibit universal superiority.''')+text[end:]
    bounds=[]
    for obj in ['linear','quadratic']:
        values=[int(r['novel_unique']) for r in full['data_u1'] if float(r['budget'])==50 and r['objective']==obj]
        bounds.append(f'{min(values)}--{max(values)}')
    text=re.sub(r'while EELS (?:with the legacy sampler )?finds [0-9]+--[0-9]+ and [0-9]+--[0-9]+, respectively\.',
                'while EELS finds '+bounds[0]+' and '+bounds[1]+', respectively.',text)
    return text

main=update_statistics(main)
refresh=r'''% BEGIN U1 REFRESH PROVENANCE
OhMyU1 measurements were repeated on 12 October 2026 using revision
\texttt{'''+full['u1_refresh']['commit'][:7]+r'''}, with unchanged inputs, seeds, CPU affinity,
batch sizes, and time budgets. The PROTES measurements were retained.
The source hashes and separate memory-audit and timing runs are archived.
OhMyU1 uses the serial \texttt{sample\_nondeg!} routine, with both
\texttt{backend=:legacy} and \texttt{backend=:table} reported separately.
Full-indicator table timings are the archived repeat after the memory audit;
data-initialized table runs were measured separately with identical inputs.
% END U1 REFRESH PROVENANCE
'''
appendix=appendix.replace(r'\label{app:protes_protocol}',r'\label{app:protes_protocol}'+'\n\n'+refresh)
if '% BEGIN PROTES BENCHMARK ADDITION' in original:
    a=original.index('% BEGIN PROTES BENCHMARK ADDITION')
    b=original.index('% END PROTES BENCHMARK ADDITION',a)+len('% END PROTES BENCHMARK ADDITION')
    updated=original[:a]+update_statistics(original[a:b])+original[b:]
    # Update only the U1 table and provenance, preserving edits to the article.
    label=updated.index(r'\label{tab:u1_timed_counts}')
    a=updated.rfind(r'\begin{table}',0,label)
    b=updated.index(r'\end{table}',label)+len(r'\end{table}')
    label_new=appendix.index(r'\label{tab:u1_timed_counts}')
    a_new=appendix.rfind(r'\begin{table}',0,label_new)
    b_new=appendix.index(r'\end{table}',label_new)+len(r'\end{table}')
    updated=updated[:a]+appendix[a_new:b_new]+updated[b:]
    if full.get('data_u1_table'):
        label_new=appendix.index(r'\label{tab:u1_timed_counts_table}')
        a_new=appendix.rfind(r'\begin{table}',0,label_new)
        b_new=appendix.index(r'\end{table}',label_new)+len(r'\end{table}')
        table_text=appendix[a_new:b_new]
        if r'\label{tab:u1_timed_counts_table}' in updated:
            pos=updated.index(r'\label{tab:u1_timed_counts_table}')
            a=updated.rfind(r'\begin{table}',0,pos);b=updated.index(r'\end{table}',pos)+len(r'\end{table}')
            updated=updated[:a]+table_text+updated[b:]
        else:
            pos=updated.index(r'\label{tab:u1_timed_counts}')
            b=updated.index(r'\end{table}',pos)+len(r'\end{table}')
            updated=updated[:b]+'\n\n'+table_text+updated[b:]
    updated=re.sub(r'% BEGIN U1 REFRESH PROVENANCE.*?% END U1 REFRESH PROVENANCE\n?', '',updated,flags=re.S)
    updated=updated.replace(r'\label{app:protes_protocol}',r'\label{app:protes_protocol}'+'\n\n'+refresh)
else:
    updated=original.replace(r'\section{Acknowledgements}',main+'\n\n'+r'\section{Acknowledgements}',1)
    updated=updated.replace(r'\end{document}',appendix+'\n\n'+r'\end{document}',1)
# Keep descriptions and tables synchronized with both sampler variants.
updated=update_backend_article(updated,full,ROOT)
main=update_backend_article(main,full,ROOT,protes_only=True)
# Keep the insertion templates synchronized with the current recorded results.
(ROOT/'scripts/protes/article_benchmark_main.tex').write_text(main,encoding='utf-8')
(stage/'main_v1.tex').write_text(updated,encoding='utf-8')
patch=''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),fromfile='main_v1.before_protes.tex',tofile='main_v1.tex'))
(ROOT/'examples/main_v1.protes_benchmarks.patch').write_text(patch,encoding='utf-8')
print(stage/'main_v1.tex')
