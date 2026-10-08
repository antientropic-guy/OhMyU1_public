"""Append reproducible exact-model and matched-data comparisons to the notebook."""
from pathlib import Path
import nbformat as nb
root=Path(__file__).resolve().parents[2]
path=root/'examples/PROTES_assignment_scaling.ipynb'
book=nb.read(path,as_version=4)
book.cells=[c for c in book.cells if not c.metadata.get('full_comparison_extension')]
for c in book.cells:
    if c.cell_type=='markdown' and 'реальный размер U(1)-объекта' in c.source:
        c.source=c.source.replace(r'n\le12',r'n\le16')
def md(s):
    c=nb.v4.new_markdown_cell(s);c.metadata['full_comparison_extension']=True;book.cells.append(c)
def code(s):
    c=nb.v4.new_code_cell(s);c.metadata['full_comparison_extension']=True;book.cells.append(c)
md(r'''## 9. Действительно построенные полные модели и скорость генерации

**Уточнение предыдущего эксперимента:** прежние большие числа памяти PROTES были аналитическими оценками плотных ядер, а не фактическими выделениями памяти. U(1)-объекты тогда строились до $n=12$. Теперь полные U(1)-MPS построены до $n=16$, а обе плотные реализации PROTES — на $n=4,\ldots,10$. Большие PROTES-модели на этом ноутбуке не создавались: при $n=16$ даже профиль переменных рангов требует сотен GiB. Эти точки не заменяются выдуманными измерениями скорости.

Обе модели описывают **всё** множество перестановок, в одинаковом построчном бинарном порядке. Динамическое программирование сохраняет множество уже занятых столбцов на каждом разрезе. Допустимым переходам соответствуют единицы, остальным — нули. Проверены точные ранги и число путей $n!$.

В U(1) обратный проход считает число продолжений $C_k(q)$. Скалярный блок заменяется на $\sqrt{C_{k+1}(q')/C_k(q)}$. Произведение вдоль любого пути равно $1/\sqrt{n!}$; это точная правоканоническая форма равномерного распределения. Используется неизменённый `sample_nondeg!` из проекта. Подготовка такой формы не включена во время генерации и записана отдельно.

В PROTES реально выделены плотные ядра индикатора: с дополнением до общего ранга для быстрого `protes` и с минимальными переменными рангами для `protes_general`. Используются их неизменённые `_sample`, `_interface_matrices` и JAX-векторизация. Это равномерное распределение на тех же $n!$ точках. Обучение выключено: проверяется именно сэмплирование из точно заданной модели.

**Протокол скорости:** один и тот же логический CPU 0; float64; батчи по 100; три seed; для каждого seed не менее 1 секунды суммарного времени вызовов генератора. Построение, JIT-прогрев, проверка допустимости и подсчёт уникальных точек исключены из этого времени у обоих методов. У PROTES учитываются получение результата на CPU и пересчёт правых окружений для каждого батча, как в штатном цикле. Сохраняется фактическое число батчей и длительность; крупный батч может превысить 1 секунду. Поэтому основной показатель — сэмплы/секунду, а количество уникальных точек сопровождается фактической длительностью.

«Новые» точки здесь отсутствуют в одной и той же контрольной выборке из 400 точек (`linear`, seed №0); она не используется для построения полного индикатора. При $n=4$ эта выборка уже содержит всё пространство, поэтому новых точек быть не может. При $n=5$ число ещё не виденных решений тоже мало. **Скорость всех сэмплов и скорость новых уникальных точек — разные показатели.**

Размер объектов измерен отдельным полным обходом `Base.summarysize`. Первичные прогоны на $n=15,16$ показали давление на память и подкачку на машине с 8 GB RAM. Поэтому скоростные прогоны повторены без глубокого обхода памяти, с `GC.gc()` перед прогревом; первичные результаты сохранены в снимке как `full_u1_initial_memory_audit`. На графике показаны повторные измерения и их диапазон. Это локальные показатели данной реализации и машины, не независимые от оборудования оценки сложности.
''')
code('''full_snapshot=json.loads((ROOT/'examples/protes_full_comparison_results.json').read_text())
u1_full=full_snapshot['full_u1']
dense_full=full_snapshot['full_protes']
full_trials=[dict(r,method='U1') for r in u1_full]
for result in dense_full:
    full_trials.extend(dict(r,method='PROTES '+result['variant']) for r in result['trials'])
methods=['U1','PROTES fast','PROTES general']
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for j,method in enumerate(methods):
    sizes=sorted({int(r['n']) for r in full_trials if r['method']==method})
    med=[];lo=[];hi=[];new=[]
    for n in sizes:
        g=[r for r in full_trials if r['method']==method and int(r['n'])==n]
        rates=[float(r['samples_per_second']) for r in g]
        med.append(np.median(rates));lo.append(min(rates));hi.append(max(rates))
        new.append(np.median([int(r['novel_unique'])/float(r['seconds']) for r in g]))
    axes[0].semilogy(sizes,med,'o-',label=method)
    axes[0].fill_between(sizes,lo,hi,alpha=.15)
    axes[1].plot(sizes,new,'o-',label=method)
axes[0].set(ylabel='Samples / second',title='Full uniform indicator: median and trial range')
axes[1].set(ylabel='New distinct feasible points / second',yscale='symlog',title='Novelty relative to the same 400 reference points')
for ax in axes:
    ax.set(xlabel='Assignment size n',xticks=range(4,17,2));ax.grid(alpha=.25);ax.legend()
fig.tight_layout();savefig(fig,'assignment_full_sampling_speed');plt.show()
table(['n','Sampler','Median samples/s','Feasible / generated','Unique per trial','New per trial','Measured seconds per trial'],
      [[n,method,f'{np.median([float(r["samples_per_second"]) for r in g]):.1f}',
        f'{sum(int(r["feasible"]) for r in g)} / {sum(int(r["samples"]) for r in g)}',
        str([int(r['unique']) for r in g]),str([int(r['novel_unique']) for r in g]),
        str([round(float(r['seconds']),3) for r in g])]
       for n in range(4,17) for method in methods
       if (g:=[r for r in full_trials if int(r['n'])==n and r['method']==method])])
table(['n','U1 / PROTES fast throughput','U1 / PROTES general throughput'],
      [[n,*[f'{np.median([float(r["samples_per_second"]) for r in full_trials if int(r["n"])==n and r["method"]=="U1"])/np.median([float(r["samples_per_second"]) for r in full_trials if int(r["n"])==n and r["method"]==method]):.2f}'
             for method in ['PROTES fast','PROTES general']]] for n in range(4,11)])
''')
md(r'''## 10. U(1)-EELS и PROTES при одинаковых исходных данных и времени

Теперь U(1)-EELS строится **только по тем же 400 наблюдениям**, а не по полному множеству. Наблюдения и целевые функции экспортированы из NumPy без повторной генерации Julia RNG; SHA-256 исходных байтов проверяется в обоих языках. Остаётся принципиальное различие доступа к структуре: U(1) использует $A,b$, PROTES получает только данные и внешний оценщик. Поэтому это сравнение рассматриваемых постановок, а не изоляция эффекта одной архитектуры при одинаковой информации.

Параметры EELS: $\gamma=1$, вырожденность сектора 1, один двусторонний обучающий проход, шаг 0.05, до 400 лучших точек на следующей итерации, 10000 генераций на итерацию. Внешние допустимые точки после инициализации не добавляются. Обучение использует те же Boltzmann-веса и обновления, что `run_EELS_optimization.jl`. Чтобы сравнивать **время**, число внешних итераций ограничено дедлайном, а не прежней константой 20; фактическое число итераций записано. Для проверки дедлайна генерация разделена на батчи по 100. Батч, завершившийся после дедлайна, не принимается, как у PROTES. Построение, обучение, вычисление цели и учёт новых точек входят в бюджет. Дедлайн кооперативный: текущая операция построения/обучения может его превысить; фактические времена приведены ниже.

Перед измерением прогревается одна полная итерация на том же размере; модель и данные прогрева не переносятся в измеряемый прогон. Все запуски последовательные, один логический CPU, без одновременного запуска конкурирующих бенчмарков. Обзор: $n=4,\ldots,10$, один seed, 5 с. Длинные прогоны: $n=8$, три seed, 50 с, обе цели. Это те же диагностические экземпляры, а не новые файлы held-out test из статьи.
''')
code('''u1_data=full_snapshot['data_u1']
for r in u1_data:
    ref=base[int(r['n']),r['objective'],int(r['repetition'])]
    assert r['initial_sha256']==ref['initial_sha256'] and r['objective_sha256']==ref['objective_sha256']
    assert int(r['generated'])==int(r['feasible'])
    assert int(r['novel_unique'])<=ref['unseen_feasible']
u1_long=[r for r in u1_data if float(r['budget'])==50]
comparison_names=names+['U1 EELS']
fig,axes=plt.subplots(1,2,figsize=(13,4.7),sharex=True,sharey=True)
for ax,obj in zip(axes,['linear','quadratic']):
    for j,name in enumerate(comparison_names):
        g=([r for r in u1_long if r['objective']==obj] if name=='U1 EELS' else
           [r for r in long_runs if config_name(r)==name and r['config']['objective']==obj])
        vals=[int(r['novel_unique']) for r in g]
        ax.scatter(j+np.linspace(-.13,.13,len(vals)),vals,color=colors[j],s=40)
        ax.plot([j-.23,j+.23],[np.mean(vals)]*2,color='black',lw=2,label='Group average' if j==0 else None)
        ax.plot(j,np.median(vals),'D',color='crimson',ms=5,label='Group median' if j==0 else None)
    ax.set(title=obj.capitalize(),xticks=range(len(comparison_names)),xticklabels=comparison_names,yscale='symlog')
    ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.25);ax.legend()
axes[0].set_ylabel('New distinct feasible points absent from initial data')
fig.suptitle('Same 400 input points and objectives, n=8, 50 seconds, one CPU')
fig.tight_layout();savefig(fig,'protes_u1_parameter_comparison_50s');plt.show()
fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharex=True,sharey=True)
for ax,obj in zip(axes,['linear','quadratic']):
    for j,name in enumerate(comparison_names):
        if name=='U1 EELS':
            g=sorted([r for r in u1_data if r['objective']==obj and float(r['budget'])==5],key=lambda r:int(r['n']))
            xs=[int(r['n']) for r in g]
        else:
            g=sorted([r for r in short_runs if config_name(r)==name and r['config']['objective']==obj],key=lambda r:r['config']['n'])
            xs=[r['config']['n'] for r in g]
        ax.plot(xs,[int(r['novel_unique']) for r in g],'o-',label=name,color=colors[j])
    ax.set(title=obj.capitalize(),xlabel='Assignment size n',yscale='symlog',xticks=range(4,11));ax.grid(alpha=.25);ax.legend(fontsize=8)
axes[0].set_ylabel('New distinct feasible points (one matched trial)')
fig.suptitle('Same initial data and objectives, 5 seconds, one CPU')
fig.tight_layout();savefig(fig,'protes_u1_parameter_comparison_5s');plt.show()
table(['n','Objective','Seed index','Budget','Actual seconds','Generated','New unique','Iterations started/completed','Initial fit complete','Initial best','Final best'],
      [[r['n'],r['objective'],r['repetition'],r['budget'],f'{float(r["elapsed_seconds"]):.3f}',r['generated'],r['novel_unique'],
        f'{r["iterations"]}/{r["completed_iterations"]}',r['initial_training_complete'],f'{float(r["initial_c_min"]):.3f}',f'{float(r["c_min"]):.3f}'] for r in u1_data])
''')
md('''## 11. Границы интерпретации

Сравнение полных моделей проверяет эффективность хранения и конкретных штатных сэмплеров при одинаковом равномерном распределении. Оно не измеряет качество обученной модели или время достижения хорошей целевой функции. Более компактное хранение само по себе не гарантирует более быстрый сэмплер: словари, выделение памяти, JIT и векторизация влияют на результат.

Сравнение по выборке отдельно проверяет число новых допустимых точек за общий вычислительный бюджет. U(1)-EELS использует ограничения в представлении, а PROTES в этом протоколе их не получает. На малых пространствах число новых точек ограничено оставшимися решениями, на больших — новизна не означает хорошую стоимость. Поэтому сохранены также лучшие значения целевой функции и фактические времена.
''')
nb.write(book,path)
print('Updated full-model and matched-data comparison sections.')
