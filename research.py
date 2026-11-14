# %% Imports
from pathlib import Path
import json, time, sys, subprocess, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from IPython.display import display, Markdown

OUT = Path('results'); OUT.mkdir(exist_ok=True)
TRAIN_SEEDS = [11, 22, 33]
VALID_SEEDS = list(range(10000, 10010))
TEST_SEEDS = list(range(20000, 20020))
EPISODES, HORIZON, CAPACITY = 6000, 60, 16
SCENARIOS = {'ID 4':(4,4), 'Seen 6':(6,6), 'OOD 8':(8,8), 'Shock 4→8':(4,8), 'Drop 4→2':(4,2)}
assert not set(TRAIN_SEEDS) & set(VALID_SEEDS)
assert not set(VALID_SEEDS) & set(TEST_SEEDS)
plt.rcParams.update({'figure.dpi':120, 'axes.spines.top':False, 'axes.spines.right':False})

# %% Environment
class InventoryEnv(gym.Env):
    """Lost-sales inventory. Hidden demand regime; public rounded demand estimate."""
    metadata = {'render_modes':[]}
    def __init__(self, randomized=False, schedule=None, horizon=HORIZON):
        super().__init__()
        self.randomized, self.schedule, self.horizon = randomized, schedule, horizon
        self.action_space = spaces.Discrete(CAPACITY+1)
        self.observation_space = spaces.MultiDiscrete([CAPACITY+1, CAPACITY+1])
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if self.schedule is None:
            level = int(self.np_random.choice([2,4,6])) if self.randomized else 4
            levels = np.full(self.horizon, level)
        else:
            levels = np.array([self.schedule[0] if t < self.horizon//2 else self.schedule[1] for t in range(self.horizon)])
        # All exogenous randomness is generated before any action.
        self.demands = self.np_random.poisson(levels)
        self.stock = int(options['stock']) if 'stock' in options else int(self.np_random.integers(CAPACITY+1))
        self.estimate, self.t = 4, 0
        return np.array([self.stock, self.estimate], dtype=np.int64), {}
    def step(self, action):
        assert self.action_space.contains(action)
        if self.t >= self.horizon:
            raise RuntimeError('reset required after truncation')
        order = max(0, int(action)-self.stock)
        available = self.stock + order
        demand = int(self.demands[self.t])
        sold = min(available, demand)
        lost = demand-sold
        self.stock = available-sold
        cost = float(self.stock + 5*lost + 2*(order>0))
        self.estimate = int(np.clip(np.rint(0.7*self.estimate+0.3*demand), 0, CAPACITY))
        self.t += 1
        info = dict(day=self.t, demand=demand, sold=sold, lost=lost, stock=self.stock,
                    available=available, order=order, cost=cost, estimate=self.estimate)
        return np.array([self.stock,self.estimate], dtype=np.int64), -cost, False, self.t >= self.horizon, info

def td_target(reward, next_q, terminated, gamma=0.95):
    return reward + gamma*(not terminated)*np.max(next_q)

# %% Independent checks
check_env(InventoryEnv(), skip_render_check=True)
e = InventoryEnv(horizon=2)
e.reset(seed=1, options={'stock':3}); e.demands = np.array([7,0])
obs,r,term,trunc,info = e.step(5)
assert (info['order'],info['sold'],info['lost'],info['stock'],r)==(2,5,2,0,-12)
assert not term and not trunc
obs,r,term,trunc,info = e.step(4)
assert r == -6 and info['stock']==4 and trunc and not term
assert td_target(1,np.array([10,3]),False,0.5)==6
assert td_target(2,np.array([10,3]),True,0.5)==2
for seed in [1,2,3]:
    e1,e2 = InventoryEnv(schedule=(4,8)),InventoryEnv(schedule=(4,8))
    e1.reset(seed=seed,options={'stock':0}); e2.reset(seed=seed,options={'stock':16})
    assert np.array_equal(e1.demands,e2.demands)
    for t in range(HORIZON):
        _,r,_,_,x=e1.step(t%17)
        assert 0<=x['stock']<=16 and x['sold']+x['lost']==x['demand']
        assert x['available']-x['sold']==x['stock'] and r==-x['cost']
print('PASS: Gymnasium API; demand coupling; inventory conservation; hand-computed cost; terminal/bootstrap handling.')

# %% Evaluation helpers

def rollout(policy, schedule, seed, stock=8):
    env = InventoryEnv(schedule=schedule)
    obs,_ = env.reset(seed=seed,options={'stock':stock})
    rng = np.random.default_rng(seed+500000)
    rows=[]
    for _ in range(HORIZON):
        if isinstance(policy, np.ndarray):
            action=int(np.argmax(policy[tuple(obs)]))
        elif policy == 'random':
            action=int(rng.integers(17))
        elif policy == 'adaptive':
            action=int(np.clip(np.ceil(obs[1]+np.sqrt(max(obs[1],1))),0,16))
        else:
            action=int(policy)
        obs,_,term,trunc,info=env.step(action)
        rows.append(info)
        if term or trunc: break
    return pd.DataFrame(rows)

def episode_metrics(frame):
    return dict(cost_day=frame.cost.mean(), post_cost=frame.cost.iloc[30:].mean(),
                fill_rate=frame.sold.sum()/max(frame.demand.sum(),1),
                stock=frame.stock.mean(), stockout_rate=(frame.lost>0).mean(),
                episode_return=-frame.cost.sum())

def validation_return(q, level):
    return np.mean([-rollout(q,(level,level),s).cost.sum() for s in VALID_SEEDS])

# %% Training

def train(condition, seed):
    env=InventoryEnv(randomized=(condition=='randomized'))
    rng=np.random.default_rng(seed)
    q=np.zeros((17,17,17),dtype=np.float64)
    history=[]
    start=time.perf_counter()
    for episode in range(EPISODES+1):
        if episode%300==0:
            vals={str(level):validation_return(q,level) for level in [4,6]}
            history.append(dict(condition=condition,seed=seed,steps=episode*HORIZON,
                                validation_4=vals['4'],validation_6=vals['6']))
            if episode%1500==0:
                print(condition,seed,'episodes',episode,'validation returns',vals,flush=True)
            np.savez_compressed(OUT/f'{condition}_{seed}.npz',q=q,episodes=episode)
            pd.DataFrame(history).to_csv(OUT/f'learning_{condition}_{seed}.csv',index=False)
        if episode==EPISODES: break
        obs,_=env.reset(seed=seed*1000000+episode)
        epsilon=max(0.05,1-0.95*episode/(0.8*EPISODES))
        for step in range(HORIZON):
            state=tuple(obs)
            action=int(rng.integers(17)) if rng.random()<epsilon else int(np.argmax(q[state]))
            nxt,reward,terminated,truncated,_=env.step(action)
            idx=state+(action,)
            q[idx] += 0.15*(td_target(reward,q[tuple(nxt)],terminated)-q[idx])
            obs=nxt
            if terminated or truncated: break
    seconds=time.perf_counter()-start
    return q,history,dict(condition=condition,seed=seed,seconds=seconds,steps=EPISODES*HORIZON)

models={}; histories=[]; timing=[]
for condition in ['fixed','randomized']:
    for seed in TRAIN_SEEDS:
        q,h,t=train(condition,seed)
        models[(condition,seed)]=q; histories.extend(h); timing.append(t)
learning=pd.DataFrame(histories); runtime=pd.DataFrame(timing)
learning.to_csv(OUT/'learning.csv',index=False); runtime.to_csv(OUT/'runtime.csv',index=False)
display(runtime)

# %% Baseline validation — no test data used
base_validation=[]
for target in range(17):
    costs=[rollout(target,(4,4),s,stock).cost.mean() for s in VALID_SEEDS for stock in [0,8,16]]
    base_validation.append(dict(target=target,validation_cost=np.mean(costs)))
base_validation=pd.DataFrame(base_validation)
BASE_TARGET=int(base_validation.loc[base_validation.validation_cost.idxmin(),'target'])
base_validation.to_csv(OUT/'base_stock_validation.csv',index=False)
print('Base-stock target selected only on ID validation:',BASE_TARGET)

# %% Frozen final evaluation
records=[]
policies=[(condition,seed,q) for (condition,seed),q in models.items()]
policies += [('random',-1,'random'),('base_stock',-1,BASE_TARGET),('adaptive',-1,'adaptive')]
for name,training_seed,policy in policies:
    for scenario,schedule in SCENARIOS.items():
        for demand_seed in TEST_SEEDS:
            for initial_stock in [0,8,16]:
                frame=rollout(policy,schedule,demand_seed,initial_stock)
                records.append(dict(policy=name,training_seed=training_seed,scenario=scenario,
                                    demand_seed=demand_seed,initial_stock=initial_stock,**episode_metrics(frame)))
results=pd.DataFrame(records)
results.to_csv(OUT/'episodes.csv',index=False)
metrics=['cost_day','post_cost','fill_rate','stock','stockout_rate','episode_return']
per_seed=results.groupby(['policy','training_seed','scenario'])[metrics].mean().reset_index()
per_seed.to_csv(OUT/'seed_means.csv',index=False)
summary=per_seed.groupby(['policy','scenario'])[metrics].agg(['mean','std'])
summary.to_csv(OUT/'summary.csv')
# lower tail includes episode stochasticity and initial stock; not independent training replicates
q10=results.groupby(['policy','scenario']).episode_return.quantile(0.1).rename('return_q10')
q10.to_csv(OUT/'return_q10.csv')
display(summary.round(3))
main=per_seed[(per_seed.scenario=='Shock 4→8') & per_seed.policy.isin(['fixed','randomized'])]
paired=main.pivot(index='training_seed',columns='policy',values='post_cost')
paired['saving']=paired.fixed-paired.randomized
paired.to_csv(OUT/'primary_effect.csv')
display(paired.round(3))
print('PRIMARY: paired savings mean ± sample SD:',paired.saving.mean(),paired.saving.std(ddof=1))

# %% Learning curves
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for ax,level in zip(axes,[4,6]):
    for condition,color in [('fixed','#2563eb'),('randomized','#ea580c')]:
        g=learning[learning.condition==condition].groupby('steps')[f'validation_{level}'].agg(['mean','std'])
        x=g.index.to_numpy(); m=g['mean'].to_numpy(); s=g['std'].to_numpy()
        ax.plot(x,m,label=condition,color=color); ax.fill_between(x,m-s,m+s,color=color,alpha=.15)
    ax.set(title=f'Validation return, demand λ={level}',xlabel='Environment interactions',ylabel='Return / 60-day episode (cost units)')
    ax.legend(); ax.grid(alpha=.2)
fig.suptitle('Greedy evaluation: mean ± sample SD across 3 training seeds; 10 episodes per seed')
fig.savefig(OUT/'learning.png'); plt.show()

# %% Comparison plots
fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
colors={'fixed':'#2563eb','randomized':'#ea580c','base_stock':'#16a34a','adaptive':'#9333ea','random':'#64748b'}
for ax,metric,label in zip(axes,['cost_day','fill_rate'],['Cost per day (units), lower is better','Demand fill rate (fraction), higher is better']):
    for i,name in enumerate(colors):
        g=per_seed[per_seed.policy==name].groupby('scenario')[metric].agg(['mean','std']).reindex(SCENARIOS)
        pos=np.arange(len(SCENARIOS))+(i-2)*.15
        ax.bar(pos,g['mean'],width=.14,label=name,color=colors[name],yerr=g['std'].fillna(0),capsize=2)
    ax.set_xticks(np.arange(len(SCENARIOS)),SCENARIOS,rotation=20)
    ax.set_ylabel(label); ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Held-out evaluation: 20 demand seeds × 3 initial stocks; bars ± SD across training seeds')
fig.savefig(OUT/'comparison.png'); plt.show()

# %% Predetermined behavior example — no best-episode selection
fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
trajectory=[]
for name,policy in [('fixed',models[('fixed',11)]),('randomized',models[('randomized',11)]),('base_stock',BASE_TARGET),('adaptive','adaptive')]:
    f=rollout(policy,(4,8),20000,8); f['policy']=name; trajectory.append(f)
    axes[0].plot(f.day,f.available,label=name,color=colors[name],alpha=.85)
    axes[1].plot(f.day,f.cost.cumsum(),label=name,color=colors[name])
    axes[2].plot(f.day,f.lost.cumsum(),label=name,color=colors[name])
axes[0].step(f.day,f.demand,label='Realized demand',color='black',alpha=.5,where='mid')
for ax in axes:
    ax.axvline(30.5,color='black',ls='--',alpha=.4); ax.grid(alpha=.2)
axes[0].set_ylabel('Units before sales'); axes[0].legend(ncol=5,fontsize=8)
axes[1].set_ylabel('Cumulative cost (units)'); axes[2].set_ylabel('Cumulative lost demand (units)'); axes[2].set_xlabel('Day')
fig.suptitle('Demand shock 4→8: fixed training seed 11, demand seed 20000, initial stock 8')
fig.savefig(OUT/'trajectory.png'); plt.show()
pd.concat(trajectory).to_csv(OUT/'trajectory.csv',index=False)

# %% Policy diagnostics
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,condition in zip(axes,['fixed','randomized']):
    policy=models[(condition,11)].argmax(axis=2)
    im=ax.imshow(policy.T,origin='lower',vmin=0,vmax=16,cmap='viridis',aspect='auto')
    ax.set(title=f'{condition}, training seed 11',xlabel='Current inventory (units)',ylabel='Rounded demand estimate (units/day)')
fig.colorbar(im,ax=axes,label='Chosen target inventory (units)')
fig.suptitle('Policy diagnostics — Q-learning has no monotonicity constraint')
fig.savefig(OUT/'policy.png'); plt.show()

# %% Numeric analysis and reproducibility
mean_effect=float(paired.saving.mean()); sd_effect=float(paired.saving.std(ddof=1))
relative=100*mean_effect/paired.fixed.mean()
id_means=per_seed[per_seed.scenario=='ID 4'].groupby('policy').cost_day.mean()
shock_means=per_seed[per_seed.scenario=='Shock 4→8'].groupby('policy').post_cost.mean()
verdict='поддерживается по направлению среднего эффекта' if mean_effect>0 else 'не поддерживается по направлению среднего эффекта'
report=f'''## Результаты и интерпретация

Основная гипотеза **{verdict}**: после скачка спроса 4→8 средняя парная экономия randomized против fixed составляет **{mean_effect:.3f} ± {sd_effect:.3f} единицы издержек/день** (sample SD по 3 training seeds), или {relative:.1f}% стоимости fixed. Все три парные разности: {paired.saving.round(3).tolist()}.

Обычный спрос λ=4: fixed {id_means['fixed']:.3f}, randomized {id_means['randomized']:.3f} единицы/день. Цена рандомизации в ID: {id_means['randomized']-id_means['fixed']:+.3f} единицы/день.

После скачка 4→8: fixed {shock_means['fixed']:.3f}, randomized {shock_means['randomized']:.3f}, base-stock {shock_means['base_stock']:.3f}, adaptive {shock_means['adaptive']:.3f}, random {shock_means['random']:.3f} единицы/день. Превосходство над random само по себе не доказывает полезность RL: важны сравнения с двумя эвристиками.

Рандомизация расширяет покрытие состояний с высоким спросом. Но Q-таблица может плохо оценивать редкие пары «состояние–действие», а фиксированный коэффициент обучения сохраняет шум оценок. Карты политики показывают, насколько решения согласованы в соседних состояниях. Из одного графика нельзя установить механизм причинно; для этого нужен дополнительный эксперимент с посещаемостью состояний или памятью.

Основной эффект относится только к этим шести запускам и синтетической среде. Три training seed — небольшая выборка; показанный SD не является доверительным интервалом. Эпизоды с разным начальным запасом разделяют один поток спроса и не считаются независимыми повторами обучения. Базовая эвристика настроена только на ID validation; адаптивная эвристика задана до теста. Нельзя утверждать превосходство над лучшей возможной эвристикой.

### Ограничения и следующие проверки

- Режим спроса скрыт; округлённая оценка спроса не гарантирует марковость. Здесь Q-learning — практический метод, а не применение теоремы сходимости к MDP.
- Полный спрос наблюдается, включая потерянные продажи; реальная система часто видит только продажи. Нет lead time, закупочной цены, возвратов, сезонности и ограничений поставщика.
- Учебная цель — операционные издержки; это не денежная прибыль реального бизнеса.
- Влияние обучения при λ=6 неотделимо от улучшения покрытия высоких оценок спроса. Это часть исследуемого изменения распределения, но не доказательство универсальной устойчивости.
- Следующий эксперимент: больше training seeds, задержка поставки и цензурированный спрос; отдельно сравнить более длинную память. Не менять постановку после просмотра этого теста.

### Воспроизводимость

Все модели — последние checkpoints после {EPISODES*HORIZON:,} взаимодействий; отбор по test не проводился. Общий бюджет — {6*EPISODES*HORIZON:,} взаимодействий, без шагов отдельной оценки. Время шести запусков: {runtime.seconds.sum():.1f} секунды на текущей машине. Финальная таблица включает {len(results)} эпизодов. Обучение/оценка CPU, без платных API. Trajectory зафиксирована заранее и не выбиралась по красоте.
'''
display(Markdown(report)); (OUT/'analysis.md').write_text(report,encoding='utf-8')
from importlib.metadata import distributions
freeze = '\n'.join(sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata['Name']})) + '\n'
(OUT/'pip-freeze.txt').write_text(freeze)
metadata={'python':sys.version,'platform':platform.platform(),'training_seeds':TRAIN_SEEDS,'validation_seeds':VALID_SEEDS,'test_seeds':TEST_SEEDS,'episodes':EPISODES,'horizon':HORIZON,'gymnasium':gym.__version__,'numpy':np.__version__,'pandas':pd.__version__}
(OUT/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
print(json.dumps(metadata,ensure_ascii=False,indent=2))
print('Full pip freeze saved to results/pip-freeze.txt')
