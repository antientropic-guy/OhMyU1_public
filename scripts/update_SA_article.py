"""Stage the generic algorithm, frozen partition protocol, and audited SA results.

Run article_statistics.jl and analyze.jl first. No optimization is performed.
The real manuscript is changed only when the caller copies the reviewed staging output.
"""
from pathlib import Path
import argparse
import csv
import shutil

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ROOT / 'scripts/article_sections'
DATA = ROOT / 'examples/sa_cluster_analysis'

def read_csv(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

def section(name):
    return (SECTIONS / (name + '.tex')).read_text(encoding='utf-8')

def f1(x):
    x = float(x)
    return f'{0 if abs(x) < .05 else x:.1f}'

def pformat(x):
    x = float(x)
    if x >= .001:
        return f'{x:.3f}'
    mantissa, exponent = f'{x:.1e}'.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'

def rank_table():
    rows = read_csv(ROOT / 'examples/partition_locality_grid/rank_pooled.csv')
    lines = [r'\begin{table}[ht]', r'\centering\small',
             r'\caption{Exact-indicator ranks in the frozen locality grid: mean $\pm$ sample SD over all 135 instances at each span.}',
             r'\label{tab:partition_ranks}', r'\begin{tabular}{rrr}\toprule',
             r'$w$ & Maximum internal rank & Mean internal rank\\\midrule']
    for r in rows:
        lines.append(r['w'] + ' & ' + f1(r['mean_max']) + r' $\pm$ ' + f1(r['sd_max']) +
                     ' & ' + f1(r['mean_mean']) + r' $\pm$ ' + f1(r['sd_mean']) + r'\\')
    return '\n'.join(lines + [r'\bottomrule\end{tabular}', r'\end{table}'])

def endpoint_table():
    rows = read_csv(DATA / 'group_summary.csv')
    lookup = {(r['family'], r['objective'], int(r['n']), r['method']): r for r in rows if r['budget'] == '200'}
    lines = [r'\begin{table}[ht]', r'\centering\small',
             r'\caption{200-second endpoints. Each cell is mean feasible-objective gain (\%) / mean objective calls (millions), over 40 tasks. Means include unsuccessful runs.}',
             r'\label{tab:sa_endpoints}', r'\setlength{\tabcolsep}{5pt}',
             r'\begin{tabular}{llrrrrr}\toprule',
             r'Family & Objective & $n$ & Informed SA & Penalty SA & EELS & Full TT\\\midrule']
    for f in ('assignment', 'partition'):
        for o in ('quadratic', 'entropic'):
            for n in (10, 12, 14):
                cells = [('Assignment' if f == 'assignment' else 'Set Partitioning'), o.title(), str(n)]
                for m in ('SA', 'SA_penalty', 'U1_data_EELS', 'U1_full'):
                    r = lookup[f,o,n,m]
                    cells.append(f1(r['mean_gain_pct']) + ' / ' + f"{float(r['mean_calls'])/1e6:.2f}")
                lines.append(' & '.join(cells) + r'\\')
    return '\n'.join(lines + [r'\bottomrule\end{tabular}', r'\end{table}'])

def penalty_table():
    rows = [r for r in read_csv(DATA / 'paired_vs_penalty.csv') if r['budget'] == '200']
    assert len(rows) == 24 and all(float(r['p_holm72']) < .05 for r in rows)
    lookup = {(r['family'],r['objective'],int(r['n']),r['method']):r for r in rows}
    lines = [r'\begin{table}[ht]', r'\centering\footnotesize',
             r'\caption{Paired improvements over penalty SA at 200 seconds: mean $100(C_{\rm penalty}-C_{\rm TT})/|C_{\rm penalty}|$ (\%), wins/ties/losses, and one-sided Wilcoxon $p$ after Holm correction over all 72 TT--penalty comparisons (including 10 and 50 seconds).}',
             r'\label{tab:sa_penalty_tests}', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{llrrrrrrr}\toprule',
             r'Family & Objective & $n$ & EELS (\%) & W/T/L & Holm $p$ & Full TT (\%) & W/T/L & Holm $p$\\\midrule']
    for f in ('assignment','partition'):
        for o in ('quadratic','entropic'):
            for n in (10,12,14):
                cells=[('Assignment' if f=='assignment' else 'Set Partitioning'),o.title(),str(n)]
                for m in ('U1_data_EELS','U1_full'):
                    r=lookup[f,o,n,m]
                    cells += [f1(r['mean_pct']),f"{r['wins']}/{r['ties']}/{r['losses']}",pformat(r['p_holm72'])]
                lines.append(' & '.join(cells)+r'\\')
    return '\n'.join(lines+[r'\bottomrule\end{tabular}',r'\end{table}'])

def appendix_figures():
    pieces=[]
    for n in (10,12):
        for axis in ('time','calls'):
            caption=(f'Additional size $n={n}$ ($N={n*n}$): 200-second runs, 40 tasks per panel. '
                     + ('Gain versus optimization time; conventions match \\cref{fig:sa_time}.' if axis=='time'
                        else 'Gain versus objective calls; conventions and common-support truncation match \\cref{fig:sa_calls}.')
                     + ' Top: assignment; bottom: strip Set Partitioning. Left: quadratic; right: entropic. '
                     + 'All four methods use matched initial data; TT amplitudes are reset/rebuilt each iteration. '
                     + 'Shading denotes pointwise 95\\% bootstrap intervals for mean per-task gain.')
            pieces.append('\n'.join([r'\clearpage',r'\begin{figure}[p]',r'\centering',
                rf'\includegraphics[width=0.98\textwidth]{{pictures/SA_n{n}_{axis}.pdf}}',
                r'\caption{'+caption+'}',rf'\label{{fig:sa_n{n}_{axis}}}',r'\end{figure}']))
    return '\n'.join(pieces)+'\n\\clearpage\n'

def main(source, output):
    text=source.read_text(encoding='utf-8')
    # Accept either the original draft or a previously generated version.
    start=text.index('Algorithm~\\ref{alg:tt_generative} describes')
    end=text.index('For the constrained problem',start)
    text=text[:start]+section('generative_overview')+text[end:]
    label=text.index(r'\label{alg:tt_generative}')
    start=text.rfind(r'\begin{algorithm}',0,label)
    end=text.index(r'\end{algorithm}',label)+len(r'\end{algorithm}')
    text=text[:start]+section('generative_algorithm')+text[end:]
    text=text.replace('(the current $\\mathcal{T}_{\\mathrm{new}}$ in \\cref{alg:tt_generative}, not a cumulative history)',
                      '(the selected $\\mathcal{T}_{\\mathrm{new}}$ in \\cref{alg:tt_generative}, with retention determined by the experiment)')
    text=text.replace('For ADD, unlike the fixed-topology loop in \\cref{alg:tt_generative},',
                      'For the ADD selection-and-rebuild variant of \\cref{alg:tt_generative},')
    text=text.replace('Unlike the fixed-topology loop in \\cref{alg:tt_generative},\nADD reconstructs',
                      'In the ADD variant of \\cref{alg:tt_generative},\nwe reconstruct')
    main_block=section('partition_main')+'\n'+section('sa_main')
    appendix=section('partition_appendix').replace('% PARTITION_RANK_TABLE',rank_table())+'\n'+section('sa_appendix')
    appendix=appendix.replace('% SA_ENDPOINT_TABLE',endpoint_table()).replace('% SA_PENALTY_TEST_TABLE',penalty_table()).replace('% SA_APPENDIX_FIGURES',appendix_figures())
    for begin,end,anchor,content in [('% BEGIN SA AND PARTITION RESULTS','% END SA AND PARTITION RESULTS',
        r'\subsection{Benchmarking library performance}',main_block),
        ('% BEGIN SA AND PARTITION APPENDIX','% END SA AND PARTITION APPENDIX',
         r'\section{Detailed library performance benchmarking}', '\\clearpage\n'+appendix)]:
        if begin in text:
            i=text.index(begin);j=text.index(end,i)+len(end)
            text=text[:i]+text[j:]
        text=text.replace(anchor,begin+'\n'+content+'\n'+end+'\n\n'+anchor,1)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(text,encoding='utf-8')
    pictures=output.parent/'pictures';pictures.mkdir(exist_ok=True)
    for n in (10,12,14):
        for axis in ('time','calls'):
            shutil.copy2(DATA/f'article_n{n}_{axis}.pdf',pictures/f'SA_n{n}_{axis}.pdf')
    print(output)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.source,args.output)
