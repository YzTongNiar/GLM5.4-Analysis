import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import {
  ArrowRight,
  BookOpen,
  Check,
  ChevronDown,
  CircleHelp,
  Download,
  ExternalLink,
  GitBranch,
  Gauge,
  Moon,
  Pause,
  Play,
  Printer,
  RotateCcw,
  Search,
  Sparkles,
  Sun,
  TreePine,
  Zap,
} from 'lucide-react'
import guideMarkdown from '../../../reports/loopfill-reading-guide.md?raw'
import './App.css'

type SectionProps = {
  id: string
  index: string
  eyebrow: string
  title: string
  intro: string
  children: ReactNode
}

function Section({ id, index, eyebrow, title, intro, children }: SectionProps) {
  return (
    <section id={id} className="section">
      <div className="section-heading">
        <div className="section-index">{index}</div>
        <div>
          <span className="eyebrow">{eyebrow}</span>
          <h2>{title}</h2>
          <p>{intro}</p>
        </div>
      </div>
      {children}
    </section>
  )
}

const laneData = [
  { token: 'to', source: '浅层 + 检索', progress: [0, 1, 2, 3, 3, 3] },
  { token: 'join', source: '后缀检索', progress: [0, 0, 1, 2, 3, 3] },
  { token: 'Lean', source: '后缀检索', progress: [0, 0, 1, 2, 3, 3] },
  { token: 'Club', source: '后缀检索', progress: [0, 0, 1, 2, 3, 3] },
  { token: '\\n', source: '浅层预测', progress: [0, 0, 0, 1, 2, 3] },
]

const stepDetails = [
  '已验证前缀是 “I would like”。后缀检索与浅层预测先把候选组织成一棵树。',
  '根节点进入第 1 次循环。后缀链上的未来 Token 也已占位，等待进入同一批次。',
  '根节点继续向完整深度推进；to、join、Lean 等候选在不同深度并行前进。',
  '完整深度验证提交 to。匹配分支被保留，已经完成的循环不会重算。',
  'join、Lean、Club 在同一拍完成验证并连续提交；换行符还在继续计算。',
  '最后的换行符完成验证。5 个 Token 共用 5 个 Loop Steps，而普通 AR 需要 15 步。',
]

function PipelineDemo() {
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    if (!playing) return
    const timer = window.setInterval(() => {
      setStep((current) => {
        if (current >= 5) {
          setPlaying(false)
          return current
        }
        return current + 1
      })
    }, 1200)
    return () => window.clearInterval(timer)
  }, [playing])

  return (
    <div className="demo-card pipeline-demo">
      <div className="demo-toolbar">
        <div>
          <span className="demo-label">交互实验 01</span>
          <h3>把图 2(b) 变成可逐拍推进的流水线</h3>
        </div>
        <div className="demo-actions">
          <button className="icon-button" onClick={() => { setStep(0); setPlaying(false) }} aria-label="重置演示" title="重置">
            <RotateCcw size={17} />
          </button>
          <button
            className="primary-button compact"
            onClick={() => {
              if (step >= 5) setStep(0)
              setPlaying((value) => !value)
            }}
          >
            {playing ? <Pause size={16} /> : <Play size={16} />}
            {playing ? '暂停' : step >= 5 ? '重播' : '播放'}
          </button>
        </div>
      </div>

      <div className="pipeline-context">
        <span>已验证前缀</span>
        <code>I would like</code>
        <ArrowRight size={16} />
        <strong>目标续写：to join Lean Club \n</strong>
      </div>

      <div className="lane-head">
        <span>输出 Token</span>
        <span>Loop 1</span>
        <span>Loop 2</span>
        <span>Loop 3 · 验证</span>
        <span>状态</span>
      </div>
      <div className="lanes">
        {laneData.map((lane) => {
          const progress = lane.progress[step]
          return (
            <div className={`token-lane ${progress >= 3 ? 'complete' : progress > 0 ? 'active' : ''}`} key={lane.token}>
              <div className="token-name">
                <strong>{lane.token}</strong>
                <small>{lane.source}</small>
              </div>
              {[1, 2, 3].map((loop) => (
                <div className={`loop-cell ${progress >= loop ? 'filled' : ''}`} key={loop}>
                  {progress >= loop ? loop === 3 ? <Check size={16} /> : <span>{loop}</span> : <span className="empty-dot" />}
                </div>
              ))}
              <span className="lane-state">
                {progress === 0 ? '等待' : progress >= 3 ? '已提交' : `进行中 ${progress}/3`}
              </span>
            </div>
          )
        })}
      </div>

      <div className="step-controller">
        <input
          aria-label="流水线时间步"
          type="range"
          min="0"
          max="5"
          step="1"
          value={step}
          onChange={(event) => { setStep(Number(event.target.value)); setPlaying(false) }}
        />
        <div className="step-markers">
          {[0, 1, 2, 3, 4, 5].map((value) => <span className={value === step ? 'current' : ''} key={value}>{value}</span>)}
        </div>
      </div>
      <div className="step-narration" aria-live="polite">
        <span>STEP {step}/5</span>
        <p>{stepDetails[step]}</p>
      </div>

      <div className="method-comparison">
        {[
          ['普通自回归', 15, '1.0×'],
          ['LoopSpec', 9, '1.7×'],
          ['SuffixDecoding', 9, '1.7×'],
          ['LoopFill', 5, '3.0×'],
        ].map(([name, steps, speedup]) => (
          <div className={name === 'LoopFill' ? 'highlight' : ''} key={name}>
            <span>{name}</span>
            <strong>{steps} 步</strong>
            <small>{speedup}</small>
          </div>
        ))}
      </div>
      <p className="demo-note">依据论文图 2(b) 重绘的教学示意，用于理解调度，不是通用性能模拟器。</p>
    </div>
  )
}

const draftModes = [
  {
    label: '仅后缀草稿',
    length: '1.34',
    title: '检索给出一段历史续写',
    description: '后缀树能一次提出多个 Token，但历史没有匹配时，正确路径很快就会中断。',
  },
  {
    label: '+ 浅层草稿',
    length: '1.73',
    title: '模型补上检索缺失的节点',
    description: '浅层循环把 to 加入同一棵树；它不等待完整深度验证，因此能缩短相邻 Token 的启动间隔。',
  },
  {
    label: '+ 二次检索',
    length: '2.28',
    title: '沿着新草稿继续向后查',
    description: '浅层提出 to 后，再以 “I would like to” 查询后缀树，继续加入 join → Lean，正确草稿路径进一步延长。',
  },
]

function DraftTreeDemo() {
  const [mode, setMode] = useState(2)
  const current = draftModes[mode]
  return (
    <div className="demo-card tree-demo">
      <div className="demo-toolbar">
        <div>
          <span className="demo-label">交互实验 02</span>
          <h3>两种草稿源如何在同一棵树里接力？</h3>
        </div>
        <div className="path-metric"><strong>{current.length}</strong><span>Token</span><small>平均正确草稿路径</small></div>
      </div>
      <div className="segmented three" role="tablist" aria-label="草稿源配置">
        {draftModes.map((item, index) => (
          <button className={mode === index ? 'selected' : ''} onClick={() => setMode(index)} key={item.label} role="tab" aria-selected={mode === index}>
            {item.label}
          </button>
        ))}
      </div>

      <div className="tree-stage">
        <div className="tree-root"><BookOpen size={17} /><span>I would like</span><small>已验证根节点</small></div>
        <div className="tree-branches">
          <div className="tree-branch muted-branch">
            <div className="tree-node suffix"><Search size={14} /><span>a</span></div>
            <div className="tree-connector" />
            <div className="tree-node suffix"><Search size={14} /><span>coffee</span></div>
            <small>检索候选，后来会被完整模型排除</small>
          </div>
          {mode >= 1 && (
            <div className="tree-branch correct-branch">
              <div className="tree-node shallow"><Zap size={14} /><span>to</span><em>浅层</em></div>
              {mode >= 2 && (
                <>
                  <div className="tree-connector" />
                  <div className="tree-node suffix"><Search size={14} /><span>join</span><em>再检索</em></div>
                  <div className="tree-connector" />
                  <div className="tree-node suffix"><Search size={14} /><span>Lean</span><em>再检索</em></div>
                </>
              )}
              <small>正确分支可继续向完整深度推进</small>
            </div>
          )}
        </div>
      </div>
      <div className="demo-explanation">
        <GitBranch size={20} />
        <div><strong>{current.title}</strong><p>{current.description}</p></div>
      </div>
      <div className="score-formula">
        <span>路径评分</span>
        <code>s(v) = s(parent(v)) · w(v)</code>
        <span>后缀边权</span>
        <code>w(v) = min(1, c · q(v))</code>
      </div>
      <p className="demo-note">1.34、1.73、2.28 来自论文图 1(b) 的 Ouro-2.6B 统计；树形图用于解释候选合并关系。</p>
    </div>
  )
}

function SliderField({ label, value, min, max, step = 1, onChange, display }: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
  display: string
}) {
  return (
    <label className="slider-field">
      <span><strong>{label}</strong><output>{display}</output></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  )
}

function TheoryCalculator() {
  const [T, setT] = useState(4)
  const [d, setD] = useState(1)
  const [ps, setPs] = useState(0.51)
  const [alpha, setAlpha] = useState(0.796)
  const gammaSuffix = 1 / (1 - ps)
  const gammaShallow = T / d
  const gammaLF = gammaSuffix * T / (alpha * d + (1 - alpha) * T)
  const beatsShallow = d * ps > (T - d) * (1 - ps) * (1 - alpha)

  return (
    <div className="demo-card theory-demo">
      <div className="demo-toolbar">
        <div><span className="demo-label">交互实验 03</span><h3>调一调参数，看加速上限如何变化</h3></div>
        <button className="text-button" onClick={() => { setT(4); setD(1); setPs(0.51); setAlpha(0.796) }}><RotateCcw size={15} />恢复论文示例</button>
      </div>
      <div className="theory-layout">
        <div className="slider-stack">
          <SliderField label="完整循环深度 T" value={T} min={2} max={32} display={`${T}`} onChange={(value) => { setT(value); if (d >= value) setD(value - 1) }} />
          <SliderField label="浅层草稿深度 d" value={d} min={1} max={T - 1} display={`${d}`} onChange={setD} />
          <SliderField label="后缀命中率 pₛ" value={ps} min={0} max={0.95} step={0.01} display={`${Math.round(ps * 100)}%`} onChange={setPs} />
          <SliderField label="后缀未中时浅层命中率 α" value={alpha} min={0} max={1} step={0.01} display={`${Math.round(alpha * 100)}%`} onChange={setAlpha} />
        </div>
        <div className="theory-results">
          <div className="gamma-primary">
            <span>LoopFill 理论 γ</span>
            <strong>{gammaLF.toFixed(2)}</strong>
            <small>每 T 个串行循环步生成的 Token 数</small>
          </div>
          <div className="gamma-grid">
            <div><span>仅后缀</span><strong>{gammaSuffix.toFixed(2)}</strong></div>
            <div><span>浅层上限 T/d</span><strong>{gammaShallow.toFixed(2)}</strong></div>
          </div>
          <div className={`condition ${beatsShallow ? 'pass' : 'warn'}`}>
            {beatsShallow ? <Check size={17} /> : <CircleHelp size={17} />}
            <span>{beatsShallow ? '当前参数满足公式（8），理论上可超过纯浅层草稿上限。' : '当前参数不满足公式（8），后缀收益不足以保证超过纯浅层草稿。'}</span>
          </div>
        </div>
      </div>
      <div className="formula-strip">
        <span>γ<sub>LF</sub></span>
        <strong>=</strong>
        <span>1 / (1 − p<sub>s</sub>)</span>
        <strong>×</strong>
        <span>T / [αd + (1 − α)T]</span>
      </div>
      <p className="demo-note">这是论文在单请求、容量充足假设下的理想分析；γ 不等于实测墙钟加速比 η，也不包含高并发下的草稿竞争。</p>
    </div>
  )
}

type ModelName = 'Ouro-2.6B' | 'Huginn-3.5B'
type Temperature = 'greedy' | 'sampling'

const latency: Record<ModelName, Record<Temperature, { ar: number; ahead: number; fill: number }>> = {
  'Ouro-2.6B': {
    greedy: { ar: 1, ahead: 4.19, fill: 5.85 },
    sampling: { ar: 1, ahead: 3.25, fill: 3.50 },
  },
  'Huginn-3.5B': {
    greedy: { ar: 1, ahead: 4.12, fill: 5.13 },
    sampling: { ar: 1, ahead: 2.37, fill: 2.60 },
  },
}

const throughput: Record<ModelName, { b: number; ar: number; fill: number; ratio: number }[]> = {
  'Ouro-2.6B': [
    { b: 1, ar: 5.53, fill: 27.12, ratio: 4.91 },
    { b: 2, ar: 10.65, fill: 51.17, ratio: 4.80 },
    { b: 4, ar: 20.75, fill: 86.22, ratio: 4.16 },
    { b: 8, ar: 38.72, fill: 140.94, ratio: 3.64 },
    { b: 16, ar: 67.48, fill: 189.56, ratio: 2.81 },
  ],
  'Huginn-3.5B': [
    { b: 1, ar: 5.89, fill: 26.17, ratio: 4.44 },
    { b: 2, ar: 11.11, fill: 40.73, ratio: 3.67 },
    { b: 4, ar: 19.00, fill: 59.50, ratio: 3.13 },
    { b: 8, ar: 32.99, fill: 94.08, ratio: 2.85 },
    { b: 16, ar: 48.34, fill: 129.18, ratio: 2.67 },
  ],
}

function ResultsExplorer() {
  const [model, setModel] = useState<ModelName>('Ouro-2.6B')
  const [temperature, setTemperature] = useState<Temperature>('greedy')
  const [batch, setBatch] = useState(16)
  const current = latency[model][temperature]
  const currentThroughput = throughput[model].find((item) => item.b === batch)!
  const bars = [
    { name: 'AR', value: current.ar, className: 'ar' },
    { name: 'LoopAhead', value: current.ahead, className: 'ahead' },
    { name: 'LoopFill', value: current.fill, className: 'fill' },
  ]
  const improvement = Math.round((current.fill / current.ahead - 1) * 100)

  return (
    <div className="demo-card results-demo">
      <div className="demo-toolbar">
        <div><span className="demo-label">交互实验 04</span><h3>切换模型、温度与并发，核对论文数字</h3></div>
        <span className="source-chip">论文表 2、表 3</span>
      </div>
      <div className="result-controls">
        <div className="segmented">
          {(['Ouro-2.6B', 'Huginn-3.5B'] as ModelName[]).map((item) => <button className={model === item ? 'selected' : ''} onClick={() => setModel(item)} key={item}>{item}</button>)}
        </div>
        <div className="segmented">
          <button className={temperature === 'greedy' ? 'selected' : ''} onClick={() => setTemperature('greedy')}>贪心 · T=0</button>
          <button className={temperature === 'sampling' ? 'selected' : ''} onClick={() => setTemperature('sampling')}>采样 · T=1</button>
        </div>
      </div>

      <div className="results-grid">
        <div className="latency-panel">
          <div className="panel-title"><div><span>单请求解码延迟</span><h4>七基准平均加速 η</h4></div><span className="gain-badge">较 LoopAhead +{improvement}%</span></div>
          <div className="bar-chart">
            {bars.map((bar) => (
              <div className="bar-row" key={bar.name}>
                <span>{bar.name}</span>
                <div className="bar-track"><div className={`bar-value ${bar.className}`} style={{ width: `${Math.max(8, bar.value / 6 * 100)}%` }} /></div>
                <strong>{bar.value.toFixed(2)}×</strong>
              </div>
            ))}
          </div>
          <p>温度升高后，输出更难预测，草稿失配增加，因此采样模式的平均加速明显回落。</p>
        </div>

        <div className="throughput-panel">
          <div className="panel-title"><div><span>端到端输出吞吐量</span><h4>SpecBench · Temperature 1.0</h4></div></div>
          <div className="batch-selector">
            {[1, 2, 4, 8, 16].map((value) => <button className={batch === value ? 'selected' : ''} onClick={() => setBatch(value)} key={value}>B={value}</button>)}
          </div>
          <div className="throughput-number"><strong>{currentThroughput.fill.toFixed(2)}</strong><span>tokens/s</span></div>
          <div className="throughput-ratio">相当于同并发 AR 的 <strong>{currentThroughput.ratio.toFixed(2)}×</strong></div>
          <div className="ar-reference">AR：{currentThroughput.ar.toFixed(2)} tokens/s</div>
          <p>并发升高后，LoopFill 的绝对吞吐量仍上升，但相对 AR 的倍数下降，说明草稿节点开始与其他请求竞争计算。</p>
        </div>
      </div>
      {batch === 16 && (
        <div className="results-caveat"><Gauge size={18} /><span>在 B=16 时，相对次优方法的优势为：Ouro-2.6B <strong>1.6%</strong>，Huginn-3.5B <strong>9.9%</strong>。不要把单请求贪心场景下的“+40%”直接套用到高并发吞吐场景。</span></div>
      )}
    </div>
  )
}

const quizItems = [
  {
    question: '为什么纯浅层预测会受到 T/d 的限制？',
    answer: '浅层预测一次通常只提出一个 Token。相邻 Token 的正确前缀计算至少间隔 d 个循环步，因此每 T 步最多推进约 T/d 个 Token；草稿被拒绝时还会产生额外重启。',
  },
  {
    question: '为什么两种草稿要共享一棵树，而不是先后独立运行？',
    answer: '共享树让后缀候选能沿着浅层草稿继续检索，浅层模型也能为检索节点补充分支；相同 Token 还可以合并节点，共享计算、验证与状态复用。',
  },
  {
    question: 'LoopFill 为什么不会改变最终输出？',
    answer: '草稿只决定哪些前缀提前计算。真正输出始终由根节点完成 T 次循环后的 pT 决定；贪心取完整模型 argmax，采样从完整模型分布产生。',
  },
  {
    question: '为什么更大的 N 会同时带来更快单请求和更低高并发吞吐？',
    answer: '单请求时，空闲 GPU 容量可以承载更多草稿，减少串行等待；高并发时，额外草稿会抢占其他请求的 Loop Pass，而被剪枝分支的计算无法转化为有效输出。',
  },
]

function Quiz() {
  const [open, setOpen] = useState<number | null>(0)
  return (
    <div className="quiz-list">
      {quizItems.map((item, index) => (
        <div className={`quiz-item ${open === index ? 'open' : ''}`} key={item.question}>
          <button onClick={() => setOpen(open === index ? null : index)} aria-expanded={open === index}>
            <span>{String(index + 1).padStart(2, '0')}</span>
            <strong>{item.question}</strong>
            <ChevronDown size={19} />
          </button>
          {open === index && <p>{item.answer}</p>}
        </div>
      ))}
    </div>
  )
}

function App() {
  const [dark, setDark] = useState(() => window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false)
  const [progress, setProgress] = useState(0)
  const navItems = useMemo(() => [
    ['problem', '问题背景'],
    ['pipeline', '流水线演示'],
    ['tree', '双草稿树'],
    ['theory', '理论计算'],
    ['results', '实验结果'],
    ['review', '评价与自测'],
  ], [])

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  useEffect(() => {
    const updateProgress = () => {
      const height = document.documentElement.scrollHeight - window.innerHeight
      setProgress(height > 0 ? Math.min(100, window.scrollY / height * 100) : 0)
    }
    updateProgress()
    window.addEventListener('scroll', updateProgress, { passive: true })
    window.addEventListener('resize', updateProgress)
    return () => {
      window.removeEventListener('scroll', updateProgress)
      window.removeEventListener('resize', updateProgress)
    }
  }, [])

  const downloadMarkdown = () => {
    const blob = new Blob([guideMarkdown], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'loopfill-reading-guide.md'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="app-shell">
      <div className="reading-progress" style={{ width: `${progress}%` }} />
      <header className="site-header">
        <a className="brand" href="#top" aria-label="返回顶部"><span><GitBranch size={19} /></span>LoopFill<span className="brand-sub">论文导读</span></a>
        <nav aria-label="章节导航">
          {navItems.map(([id, label]) => <a href={`#${id}`} key={id}>{label}</a>)}
        </nav>
        <div className="header-actions">
          <button className="icon-button" onClick={() => window.print()} aria-label="打印或保存 PDF" title="打印 / 保存 PDF"><Printer size={17} /></button>
          <button className="icon-button" onClick={() => setDark((value) => !value)} aria-label="切换深浅色" title="切换深浅色">{dark ? <Sun size={18} /> : <Moon size={18} />}</button>
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <div className="paper-tag"><Sparkles size={15} />ARXIV:2611.19649 · 交互式中文导读</div>
            <h1>填满循环解码的<br /><em>流水线空档</em></h1>
            <p className="hero-lead">LoopFill 将后缀树的多 Token 检索与浅层循环预测合并到同一棵草稿树：一边提前组织未来计算，一边由完整深度模型兜底验证。</p>
            <div className="hero-actions">
              <a className="primary-button" href="#pipeline"><Play size={17} />开始交互演示</a>
              <button className="secondary-button" onClick={downloadMarkdown}><Download size={17} />下载 Markdown</button>
            </div>
            <div className="paper-meta">Siqi Zhu · Suozhi Huang · Yi Zhao · Jiaxuan You<span>2026-11-14 · v1</span></div>
          </div>
          <div className="hero-visual" aria-label="LoopFill 双草稿源示意">
            <div className="orbit orbit-one" />
            <div className="orbit orbit-two" />
            <div className="hero-root"><span>已验证前缀</span><strong>I would like</strong></div>
            <div className="hero-source source-suffix"><Search size={19} /><div><strong>后缀树</strong><span>一次提出多个 Token</span></div></div>
            <div className="hero-source source-shallow"><Zap size={19} /><div><strong>浅层循环</strong><span>补充下一 Token</span></div></div>
            <div className="hero-tree">
              <span>to</span><i /><span>join</span><i /><span>Lean</span><i /><span>Club</span>
            </div>
            <div className="hero-verify"><Check size={18} /><span>完整深度验证</span><strong>输出不变</strong></div>
          </div>
        </section>

        <section className="stat-strip" aria-label="论文关键数字">
          <div><strong>5.85×</strong><span>Ouro-2.6B 贪心平均加速</span></div>
          <div><strong>2</strong><span>互补草稿来源</span></div>
          <div><strong>0</strong><span>额外训练</span></div>
          <div><strong>7</strong><span>评测基准</span></div>
          <div><strong>p<sub>T</sub></strong><span>完整深度最终验证</span></div>
        </section>

        <div className="content-wrap">
          <Section id="problem" index="01" eyebrow="THE BOTTLENECK" title="先理解：慢的不是一层，而是重复串行" intro="Looped Transformer 用共享模块的反复执行换取参数效率与测试时计算，但每个 Token 都要等待 T 次循环完成。">
            <div className="concept-grid">
              <article className="concept-card dark-card">
                <span className="card-kicker">普通自回归</span>
                <div className="loop-equation">H<sup>(t)</sup> = f(H<sup>(t−1)</sup>, E)</div>
                <p>当前 Token 没跑完 T 次循环，下一个 Token 就无法开始。生成 5 个 Token、T=3 时，需要 15 个串行 Loop Steps。</p>
                <div className="serial-blocks">{Array.from({ length: 15 }, (_, index) => <i className={index % 3 === 2 ? 'verify' : ''} key={index} />)}</div>
              </article>
              <article className="concept-card">
                <span className="card-kicker">已有浅层投机</span>
                <h3>提前猜一个，边验证边向后算</h3>
                <p>第 d 层先提出下一个 Token，让后续位置提前开工；当前位置继续到 T 层验证。问题是每次只猜一个，加速受到 T/d 限制，猜错还会出现流水气泡。</p>
                <div className="limit-badge"><Gauge size={17} /><span>纯浅层草稿上限</span><strong>T / d</strong></div>
              </article>
              <article className="concept-card accent-card">
                <span className="card-kicker">LoopFill 的突破口</span>
                <h3>从“逐 Token 提前”升级为“整段未来提前”</h3>
                <p>后缀检索负责一次拉起多个未来 Token，浅层预测负责补洞与延伸。两者优化的是不同方向，却在同一棵树上汇合。</p>
                <div className="axis-pair"><span><Search size={16} />序列方向 · 多 Token</span><span><Zap size={16} />深度方向 · 少等待</span></div>
              </article>
            </div>
          </Section>

          <Section id="pipeline" index="02" eyebrow="PIPELINE IN MOTION" title="逐拍看懂：草稿先行，完整模型最后拍板" intro="点击播放或拖动时间轴。注意不同 Token 可以同时处于不同循环深度，验证接受后直接复用已有进度。">
            <PipelineDemo />
            <div className="four-steps">
              {[
                ['01', '提出', '本地与全局后缀树检索历史续写；浅层状态提出 Top-k 候选。'],
                ['02', '选择', '按路径分数保留 Top-N 节点，控制额外计算与分支膨胀。'],
                ['03', '推进', '不同请求、不同候选、不同循环深度在同一个 Batch 中前进。'],
                ['04', '验证与复用', '根节点完成 T 层后决定输出，匹配子树提升为新根并保留状态。'],
              ].map(([number, title, text]) => <article key={number}><span>{number}</span><h3>{title}</h3><p>{text}</p></article>)}
            </div>
          </Section>

          <Section id="tree" index="03" eyebrow="TWO DRAFTERS, ONE TREE" title="核心创新：不是两个草稿器，而是一棵协同草稿树" intro="切换三种配置，观察浅层草稿与二次检索如何逐步延长正确路径。相同候选合并后，只计算一次。">
            <DraftTreeDemo />
          </Section>

          <Section id="theory" index="04" eyebrow="A SMALL THEORY LAB" title="理论直觉：后缀命中零等待，浅层命中只等 d 步" intro="公式并不神秘：后缀命中时下一 Token 立即启动；后缀失败但浅层命中时等待 d；两者都失败才等待完整 T。">
            <TheoryCalculator />
          </Section>

          <Section id="results" index="05" eyebrow="READ THE NUMBERS IN CONTEXT" title="实验结果：先分清延迟、吞吐量与运行场景" intro="论文结果很强，但不同设置的数字不能混用。下面直接切换原始论文中的模型、温度和并发配置。">
            <ResultsExplorer />
            <div className="budget-grid">
              <article><span className="card-kicker">草稿来源消融</span><h3>两种草稿确实都在贡献</h3><p>SpecBench 上，后缀草稿贡献约 38%–43% 的输出 Token，浅层草稿贡献约 55%–57%。</p><div className="source-bar"><i style={{ width: '41%' }} /><i style={{ width: '56%' }} /><i style={{ width: '3%' }} /></div><div className="source-legend"><span>后缀 38%–43%</span><span>浅层 55%–57%</span></div></article>
              <article><span className="card-kicker">节点预算消融</span><h3>N 不是越大越好</h3><p>Huginn-3.5B 上，N 从 16 增至 64：单请求解码速度约提升 19.5%，但 B=16 的端到端吞吐量下降 43%。</p><div className="budget-tradeoff"><span><ArrowRight size={15} />单请求延迟更低</span><span className="negative"><ArrowRight size={15} />高并发吞吐更低</span></div></article>
            </div>
          </Section>

          <Section id="review" index="06" eyebrow="CRITICAL READING" title="最后带走的，不应只有 5.85×" intro="LoopFill 的价值在于连接序列方向与循环深度方向的投机；它的代价则藏在可预测性、运行时复杂度和候选预算里。">
            <div className="review-grid">
              <article className="review-card strength"><span>真正值得借鉴</span><h3>免训练 + 多 Token + 完整深度验证</h3><ul><li>后缀检索与浅层预测具有明确互补性；</li><li>共享树统一了候选合并、计算、验证和复用；</li><li>草稿只改变调度，不替完整模型做最终决定。</li></ul></article>
              <article className="review-card limits"><span>需要保留的判断</span><h3>免训练不等于零系统成本</h3><ul><li>后缀树维护、动态调度和 KV 状态复用都需要引擎支持；</li><li>高随机采样、高并发与低可预测文本会压缩收益；</li><li>目前证据集中于两类 1.4B–3.5B 模型和单张 L20X。</li></ul></article>
            </div>
            <div className="takeaway"><TreePine size={26} /><p><strong>一句话回收全文：</strong>LoopFill 用检索和浅层模型共同组织投机计算，再用完整深度模型兜底；它优化的是串行等待，而不是取消验证。</p></div>
            <div className="quiz-heading"><CircleHelp size={23} /><div><span>读后自测</span><h3>四个问题，检查是否真正读懂</h3></div></div>
            <Quiz />
          </Section>
        </div>
      </main>

      <footer>
        <div><GitBranch size={20} /><strong>LoopFill 交互式论文导读</strong><span>基于 arXiv:2611.19649v1 整理</span></div>
        <div className="footer-actions">
          <button onClick={downloadMarkdown}><Download size={15} />Markdown 导读</button>
          <a href="https://arxiv.org/abs/2611.19649" target="_blank" rel="noreferrer">论文原文<ExternalLink size={14} /></a>
        </div>
      </footer>
    </div>
  )
}

export default App
