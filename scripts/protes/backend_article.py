"""Compact manuscript updates for the two measured OhMyU1 samplers."""
import csv
import re
import statistics

def replace_marked(text, tag, content, anchor):
    block=f'% BEGIN {tag}\n{content}\n% END {tag}\n'
    pattern=rf'% BEGIN {tag}.*?% END {tag}'
    if re.search(pattern,text,re.S):
        return re.sub(pattern,lambda _:block,text,flags=re.S)
    return text.replace(anchor,anchor+'\n'+block,1)

def update_backend_article(text, full, root, protes_only=False):
    if not full.get('full_u1_table'):
        return text
    med=lambda data,n:statistics.median(float(r['samples_per_second']) for r in data if int(r['n'])==n)
    table_rates=[med(full['full_u1_table'],n) for n in (8,10)]
    text=text.replace('At $n=8$ and $n=10$, OhMyU1 generates','At $n=8$ and $n=10$, OhMyU1 with the legacy sampler generates')
    text=replace_marked(text,'PROTES TABLE RATES',
        r'The table sampler yields approximately '+f'{table_rates[0]:.0f} and {table_rates[1]:.0f}'+
        r' samples per second at these sizes; its per-call preparation is included. Both variants are shown.',
        'Small instances do not exhibit universal superiority.')
    text=text.replace('while EELS finds ','while EELS with the legacy sampler finds ')
    ranges=[]
    for objective in ['linear','quadratic']:
        vals=[int(r['novel_unique']) for r in full['data_u1_table'] if float(r['budget'])==50 and r['objective']==objective]
        ranges.append(f'{min(vals)}--{max(vals)}')
    text=replace_marked(text,'PROTES TABLE NOVELTY',
        'With the table sampler the corresponding EELS ranges are '+ranges[0]+' and '+ranges[1]+'.',
        'initialization and update rule. New points need not improve the objective.')
    text=text.replace('The distributions and binary variable order are identical. Both OhMyU1 sampling backends are shown.','The distributions and binary variable order are identical.')
    text=text.replace('The distributions and binary variable order are identical.',
        'The distributions and binary variable order are identical. Both OhMyU1 sampling backends are shown.')
    text=text.replace(r'$U(1)$-EELS uses the fixed settings above.',
        r'$U(1)$-EELS uses the fixed settings above, with both sampling backends shown.')
    text=text.replace('configurations are reported, not only the best observed configuration.',
        'configurations and both EELS sampling backends are reported, not only the best observed configuration.')
    if protes_only:
        return text
    note=r'''OhMyU1 is measured with two sampling backends. The \texttt{legacy}
backend follows sampled charge paths using the block dictionaries;
\texttt{table} first constructs transition tables for all charge states on
each sampling call, then amortizes that preparation over the batch.
Both implement the same conditional probabilities. Training is unchanged;
the recorded training-time differences between runs reflect timing variation,
not a different training algorithm.'''
    text=text.replace(r'OhMyU1 sampling in this comparison uses \texttt{backend=:table}.','')
    text=text.replace(r'The benchmark uses OhMyU1 revision \texttt{f69c0d0}',
        r'The refreshed benchmark uses OhMyU1 revision \texttt{'+full['u1_refresh']['commit'][:7]+'}')
    text=replace_marked(text,'SAMPLING BACKEND DESCRIPTION',note,
        'both libraries were trained on exactly the same data.')
    with (root/'examples/assignment_backend_comparison.csv').open() as f:
        rows=list(csv.DictReader(f))
    by={(r['sampling_backend'],int(r['n'])):r for r in rows}
    speedups={b:[float(by['upstream',n]['balanced_serial_s'])/float(by[b,n]['balanced_serial_s']) for n in range(4,11)] for b in ('legacy','table')}
    sentence='OhMyU1 is between\n'+f'${min(speedups["legacy"]):.1f}'+r'\times$ and '+f'${max(speedups["legacy"]):.1f}'+r'\times$ faster with legacy, and between '+f'${min(speedups["table"]):.1f}'+r'\times$ and '+f'${max(speedups["table"]):.1f}'+r'\times$ with table, on the measured grid.'
    text=re.sub(r'OhMyU1 is between\s+.*?on the measured grid\.',lambda _:sentence,text,count=1,flags=re.S)
    text=text.replace('Initial-state construction is excluded. Both OhMyU1 samplers are shown. The degeneracy','Initial-state construction is excluded. The degeneracy')
    text=text.replace('Initial-state construction is excluded. The degeneracy',
        'Initial-state construction is excluded. Both OhMyU1 samplers are shown. The degeneracy')
    table=[r'''\begin{table}[t]\centering\small
\caption{Median times (seconds) for one training sweep and 400 serial samples.
L and T denote legacy and table. Total is the sum of the two kernel medians.
ConstrainTNet measurements are retained; both OhMyU1 variants use the same inputs.
Construction, warm-up, and copying are excluded.}
\label{tab:lib_comparison_fair}
\begin{tabular}{rrrrrrrr}\toprule
$n$ & Train L & Sample L & Total L & Train T & Sample T & Total T & ConstrainTNet\\\midrule''']
    for n in range(4,11):
        vals=[float(by[b,n][k]) for b in ('legacy','table') for k in ('train_s','sample_serial_s','balanced_serial_s')]
        vals.append(float(by['upstream',n]['balanced_serial_s']))
        table.append(str(n)+' & '+' & '.join(f'{v:.4f}' for v in vals)+r'\\')
    table.append(r'\bottomrule\end{tabular}\end{table}')
    pos=text.index(r'\label{tab:lib_comparison_fair}')
    a=text.rfind(r'\begin{table}',0,pos);b=text.index(r'\end{table}',pos)+len(r'\end{table}')
    text=text[:a]+'\n'.join(table)+text[b:]
    return text
