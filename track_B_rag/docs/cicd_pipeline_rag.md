# CI/CD-конвеєр для Acme Cloud RAG Assistant (Track B)

**Система:** Acme Cloud RAG (`rag_sut.py`) — e5 → Chroma → Qwen2.5-1.5B-Instruct 
**База для рішень:** `test_strategy.md`, `reports/results.md` (прогін 36 кейсів), SLA з ДЗ 8, `release_notes_rag.md`, дефекти D-01…D-05  
**Контекст:** проєктування конвеєра таким чином, ніби RC завтра йде в production.

---

## 0. Короткий контекст SUT (навіщо саме такий CI)

Пайплайн запиту: питання → multilingual-e5 → `similarity_search(k=2)` → склейка контексту → слабкий system prompt → greedy Qwen → `{answer, sources}`.

Після eval-прогону (вердикт **no-ship**):

| Метрика | RC1 факт | Target SLA | Block? |
|---|---:|---:|---|
| Hit@2 | 0.870 | ≥ 0.90 | так |
| Recall@2 | 0.739 | ≥ 0.80 | так |
| MRR | 0.783 | ≥ 0.80 | так |
| NDCG@2 | 0.717 | ≥ 0.80 | так |
| FactCorrectness (happy) | 0.733 | ≥ 0.70 | так |
| SafeRefusal | 0.429 | ≥ 0.90 | так |
| CrossLingualAgreement | 0.429 | ≥ 0.85 | PR: warning / release: block |
| InjectionOK | 0.857 | = 1.00 | так |
| F1@2 | 0.594 | — | warning |

Відомі open-дефекти, які конвеєр має тримати видимими: **D-01** (Free 2GB vs 5GB), **D-02** (refusal), **D-03** (Recall@2), **D-04** (UA `PWNED`), **D-05** (EN/UA drift).

---

## 1. Тригери

### Що запускає конвеєр
1. **Pull Request** у `main` / `release/*` — швидкий AI quality gate (`ai-pr-gate.yml`).
2. **Push у `main`** після merge — той самий офлайн-гейт + публікація артефактів baseline.
3. **Nightly (cron 02:00 UTC)** — повна генерація на GPU-runner + `--n-runs 5` на флакі-зрізі.
4. **Manual / release tag** (`v*`) — повний pre-release suite + підпис ship/no-ship.
5. **Зміна секрету/моделі в registry** (workflow_dispatch) — smoke на 12 кейсах після ротації.

### Які зміни є тригерами

| Тип зміни | Приклад у нашому репо | Тригер |
|---|---|---|
| Код SUT / адаптер | `rag_sut.py`, `src/system_under_test.py`, `src/generate.py` | PR + обов’язковий regenerate на nightly |
| Промпт / політика відповіді | system prompt у `RagSUT.ask` | PR (smoke generate) + nightly full |
| Корпус KB | `_CORPUS` у `rag_sut.py` (d1…d8) | PR + full eval (D-01 чутливий до d2) |
| Конфіг retrieval | `_TOP_K`, `_CHUNK_SIZE`, модель e5/Qwen | PR block + nightly |
| Golden-сет | `data/eval_dataset.jsonl` | PR: schema + offline eval; nightly: повна генерація |
| Метрики / гейти | `tests/test_eval.py`, `src/metrics/custom_metrics.py` | PR (сам гейт) |
| Залежності | `requirements.txt` | PR |
| Release notes / контракт | `release_notes_rag.md` | PR review + sync порогів у стратегії |

### Що трапляється найчастіше → має запускати перевірки першим
Для цього SUT найчастіші зміни в практиці курсу/команди:
1. **правка промпта / відмов** (після D-02),
2. **правка корпусу** (після D-01 — видалення/канон d2),
3. **розширення golden-сету** (нові регресійні кейси з дефектів),
4. рідше — зміна моделі чи K.

Тому path-filters у PR-гейті обов’язково покривають `rag_sut.py`, `data/eval_dataset.jsonl`, `src/**`, `tests/**`.

---

## 2. Яруси (Stages)

### A. На кожен Pull Request (~8–15 хв, без GPU)
1. Lint/schema: `test_functional.py` (≥30 кейсів, метадані, наявність `output` у generations).
2. **Офлайн eval** над закоміченим `outputs/generations.json`: `bash run_eval.sh`.
3. Якщо змінено `rag_sut.py` або промпт-шлях — **smoke regenerate** на 12 кейсах (`HP-FREE-EN/UA`, `NEG-ENTERPRISE*`, `ADV-INJECT-*`, `EDGE-FREE-CONFLICT`, `HP-PRO-PRICE-*`) на GPU self-hosted / Colab runner; інакше skip.
4. Публікація артефактів: JUnit/pytest log, таблиця метрик, diff vs baseline RC1.

Орієнтовна тривалість для *нашого* проєкту: офлайн частина **&lt;1 хв** (уже виміряно); smoke generate 12 кейсів на T4/Colab — **~10–20 хв** (cold start моделей).

### B. Nightly (~45–90 хв)
1. Повна генерація: `TRACK=B python src/generate.py --n-runs 1` на всіх 36 кейсах.
2. Стабільність: `--n-runs 5` на зрізі negative + adversarial + 7 EN/UA пар (≈20 кейсів × 5).
3. Повний `run_eval` + оновлення `reports/nightly_metrics.json`.
4. Порівняння з baseline (див. §4); алерт, якщо відкрився новий fail поза xfail-списком D-01…D-05.

Тривалість: повний generate 36 кейсів локально зайняв **~7 хв після завантаження моделей**; з cold download e5+Qwen — **~15–40 хв**. Nightly з n-runs=5 на зрізі — ще **~20–40 хв**.

### C. Лише перед релізом (~2–3 год wall-clock)
1. Увесь nightly + **без xfail**: target SLA з §3 мають бути зелені (інакше no-ship).
2. Регресія конфігурації (патерн ДЗ 8): `chunk_size` / `_TOP_K` sensitivity на retrieval-метриках.
3. Ручний review карток D-01…D-05: усі P1 closed або accepted risk підписаний PM.
4. Canary prep: фіксація model digests (e5 + Qwen) і хешу `_CORPUS`.
5. Підпис **ship/no-ship** у release notes (зараз за даними RC1 — **no-ship**).

---

## 3. Гейти (Quality Gates)

Пороги взято з `test_strategy.md` §4; **фактичні RC1-числа** — з `reports/results.md`. Baseline обґрунтування: SLA ДЗ 8 (Hit@3≥0.9, NDCG@3≥0.8, MRR≥0.8) адаптовано під `_TOP_K=2` і брудний корпус Acme.

### Block merge (PR не мерджиться, якщо нижче порога *після* зняття xfail / на нових регресіях)

| Метрика | Поріг | RC1 | Джерело порога |
|---|---:|---:|---|
| Hit@2 | ≥ 0.90 | 0.870 | ДЗ 8 SLA Hit@K≥0.9 |
| Recall@2 | ≥ 0.80 | 0.739 | послаблено vs Hit через multi-gold@K=2 |
| MRR | ≥ 0.80 | 0.783 | ДЗ 8 |
| NDCG@2 | ≥ 0.80 | 0.717 | ДЗ 8 NDCG≥0.8 |
| FactCorrectness (happy) | ≥ 0.70 | 0.733 (pass) | бізнес-факти тарифів (жорсткі оракули) |
| SafeRefusal (negative) | ≥ 0.90 | 0.429 | acceptance #3 release notes |
| InjectionOK (P1 adv) | = 1.00 | 0.857 | D-04 / R-05; нульова толерантність до `PWNED` |

На поточному RC1 ці гейти **свідомо червоні** і задокументовані як `xfail` у pytest, щоб `run_eval` був відтворюваний для лектора, але **релізний гейт знімає xfail**.

### Warning only (не блокують merge, блокують/ескалюють перед релізом)

| Перевірка | Поріг / правило | RC1 | Навіщо warning |
|---|---|---:|---|
| F1@2 | падіння &gt; 0.05 від baseline 0.594 | 0.594 | шум vs покриття при конфлікті d1/d2 |
| CrossLingualAgreement | ≥ 0.85 на release; на PR — warn якщо &lt; 0.70 | 0.429 | дорогий повний EN/UA на кожен PR |
| Precision@2 / зайвий `d2` у Free-кейсах | будь-який ріст частки відповідей «2 GB only» | D-01 | ранній сигнал регресії корпусу |
| Δ NDCG після зміни chunk/K | \|\|Δ\|\| &gt; 0.03 vs попередній nightly | — | як у ДЗ 8 (chunk 300→50, Δ≈0.011 був шум) |

### Політика xfail
- Дозволені лише дефекти з картками в `reports/results.md` (D-0X) і `strict=False`.
- Новий fail поза списком → **block merge**.
- Закриття дефекта → тест має стати зеленим (XPASS = сигнал прибрати xfail).

---

## 4. Статистична перевірка (регресія vs шум)

Наш golden-сет **малий (36 кейсів)** і частково детермінований (greedy). Тому класичний bootstrap на тисячах семплів не дає великої потужності; комбінуємо кілька простих правил, узгоджених із `PASS_RATE_THRESHOLD=0.8` у стратегії.

### Підходи саме для цього сету

1. **Повторні прогони + pass-rate (основний анти-шум)**  
   Nightly: `--n-runs 5` на ~20 флакі-чутливих кейсах (negative, adversarial, EN/UA пари).  
   Кейс «стабільний fail/pass», якщо pass-rate ≤0.2 або ≥0.8.  
   Зона **0.2–0.8** = шум/нестабільна політика (як EN vs UA refusal у D-02) → не рахуємо як чисту регресію коду без ручного look.

2. **Мінімальний розмір ефекту (effect size gate)**  
   Регресія метрики на повному сеті (36) зараховується лише якщо падіння ≥:
   - **0.03** абсолютних для Hit/Recall/MRR/NDCG (більше за Δ=0.011 з регресії chunk_size у ДЗ 8),
   - **0.05** для SafeRefusal / CrossLingual / FactCorrectness.  
   Менші гойдання трактуємо як шум малого N.

3. **Bootstrap-порівняння (нічний, спрощений)**  
   1000 resamples по кейсах (with replacement) для Δ(metric_new − metric_baseline).  
   Сигнал регресії: 95% CI для Δ повністю нижче 0 **і** \|mean Δ\| ≥ мінімального ефекту з п.2.  
   На N=36 CI широкий — тому bootstrap **підтверджує**, але не єдиний гейт.

4. **Стратифікація**  
   Окремо дивимось зрізи: Free-storage, negative, adversarial-UA.  
   Приклад: падіння лише на `ADV-INJECT-UA` при стабільному EN — це D-04-клас, а не «глобальний Hit».

### Як відрізнити реальну регресію від шуму (правило)
- **Регресія:** effect size ≥ порога **і** (pass-rate зсунувся за 0.8/0.2 **або** bootstrap CI &lt; 0) **і** відтворюється на greedy n-runs=1.  
- **Шум:** \|\|Δ\|\| &lt; effect size, або pass-rate в сірій зоні без зміни медіани зрізу.  
- Для **greedy-only** змін корпусу/промпта шум малий — достатньо n-runs=1 + effect size; multi-run потрібен після будь-якого sampling або зміни quant/device.

---

## 5. Вартість і час виконання

### Як контролюємо вартість
- **PR за замовчуванням без LLM-генерації** — лише офлайн `run_eval` на закоміченому `generations.json` (\$0 API; CPU секунди).
- GPU/generate — **smoke 12 кейсів** лише якщо paths зачепили SUT/промпт/KB.
- Nightly — 1 повний generate + обмежений n-runs=5 зріз, не 36×5 завжди.
- LLM-as-judge **не в block-гейті** (див. Компроміси); локальні оракули/формула retrieval.
- Кеш моделей HF на runners; cache pip; артефакт `generations.json` не перегенеровується без потреби.

### Оптимізації
| Механізм | Застосування |
|---|---|
| Paths filters | як у `ai-pr-gate.yml` |
| Кеш моделей / pip | GitHub Actions cache + HF hub cache на self-hosted |
| Sampling / smoke slice | 12 кейсів на PR замість 36 |
| Offline-first | метрики з `sources`/`output` без повторного виклику Qwen |
| xfail known defects | не витрачаємо релізний час на «шукати» вже відоме, але тримаємо видимим |

### Орієнтовна вартість повного nightly (якби платний API)
Припущення: замість локальної Qwen — `gpt-4o-mini` для генерації відповіді (не judge).

- 36 кейсів × 1 run ≈ 36 викликів; зріз стабільності 20 × 5 = 100; **разом ~136 completion**.  
- Середній prompt+context ~800 tokens in / ~80 out → грубо **~120k input + ~11k output tokens**.  
- За публічними тарифами порядку \$0.15 / 1M in і \$0.60 / 1M out: **≈ \$0.02–0.05 за nightly** лише на generation.  
- Якщо додати LLM-judge на всі 36 (2 виклики/кейс) — ще ≈\$0.05–0.15.  
- **Висновок:** для нашого маленького сету \$ не проблема; дорожчий ресурс — **GPU-хвилини й час інженера**, тому економимо через offline-first і smoke, а не через відмову від eval.

Локальний/Colab шлях (як зараз): **\$0 API**; вартість = квота Colab GPU / електрика.

---

## 6. Робота із секретами

Проєкт **за замовчуванням без ключів** (локальні e5 + Qwen). Для production-конвеєра з опційним API:

### Де зберігаються ключі
- **GitHub Actions Secrets / Environments** (`OPENAI_API_KEY`, опційно `HF_TOKEN` для rate limit) — лише на рівні environment `nightly` і `release`.
- Локально / Colab: `.env` (у `.gitignore`), ніколи в ноутбуках як hardcoded string.
- `check_submission.py` сканує репо на патерни ключів (із пропуском `.venv`).

### Хто має доступ
- **PR runners:** без API-секретів (офлайн-гейт).
- **Nightly/release environment:** secrets доступні лише jobs з `environment:` approval (lead QA + 1 engineer).
- Читання значень секретів у логах заборонене (masking); заборонено `echo` / артефакти з env.

### Як уникнути потрапляння в репо / ноутбуки
1. `.env` у `.gitignore`; у `.env.example` — лише імена змінних.  
2. Ноутбуки (`capstone_colab.ipynb`, `student_rag_SUT.ipynb`) читають `os.environ`, не вставляють ключ у клітинку.  
3. Pre-commit / `check_submission.py` на PR.  
4. `outputs/generations.json` комітиться **без** метаданих запитів до платного API (лише `output`/`sources`).  
5. Для нашого треку B ключ **не потрібен** для зеленого `run_eval` — це навмисний дизайн зниження ризику витоку.

---

## 7. Сигналізація

### Куди надходять результати
- Статус checks у GitHub PR (`ai-pr-gate`).
- Nightly: канал **Slack/Teams `#acme-rag-qa`** + summary у GitHub Actions run.
- Реліз: коментар у release ticket + оновлення `reports/results.md`.

### Хто отримує повідомлення
| Подія | Одержувачі |
|---|---|
| PR gate red (новий fail ≠ xfail) | автор PR + CODEOWNERS (`tests/`, `rag_sut.py`) |
| Nightly failed / регресія effect-size | черговий QA + DS owner RAG |
| InjectionOK &lt; 1.0 або новий P1 | QA lead + Security on-call (пагінг) |
| Release no-ship | PM + Tech lead |

### Як дізнаємось про невдалий nightly
1. GitHub Actions failure email підписникам workflow.  
2. Slack webhook: короткий блок «Hit/Recall/SafeRefusal/Injection vs SLA + Δ vs baseline».  
3. Якщо падіння на зрізі adversarial — окремий severity=high тег.

### Артефакти після виконання
- `outputs/generations.json` (і `generations.nightly.json` у artifact store)
- pytest JUnit XML + лог `run_eval`
- `reports/results.md` / `nightly_metrics.json` (Hit, Recall, MRR, NDCG, F1, Fact, Refusal, CrossLingual, Injection)
- список відкритих xfail ↔ D-0X
- model digests + hash `_CORPUS`

---

## 8. Після деплою

Уявімо, що Acme Cloud RAG уже в production (чат підтримки тарифів).

### Guardrail-метрики (2–3)
1. **Grounding/refusal proxy (online):** частка відповідей із маркером відмови на запитах із низьким retrieval-score (proxy D-02) — ціль ≥ 0.90 на out-of-KB семплі.  
2. **Fact-critical error rate:** частка сесій, де відповідь містить «2 GB» для Free *або* вигадану Enterprise-ціну (регулярні оракули як у `custom_metrics`) — ціль **&lt; 1%** за годину.  
3. **Safety breach rate:** частка відповідей із forbidden token / leak system prompt (клас D-04) — ціль **= 0** за 15-хв вікно.

Додатково моніторимо між релізами: latency p95 `ask()`, частка порожніх `sources`, drift розподілу мов EN/UA, hit-rate онлайн-ретрівала проти періодичного golden replay.

### Правило автоматичного rollback (конкретна політика)

> **Rollback Policy RP-RAG-1.**  
> Якщо протягом будь-яких **15 послідовних хвилин** після деплою виконується **хоча б одна** умова:  
> (a) `safety_breach_rate > 0` (будь-який `PWNED` / leak system prompt у прод-логах), **або**  
> (b) `fact_critical_error_rate ≥ 3%` на ≥ 50 запитах про тарифи Free/Pro/Enterprise, **або**  
> (c) `online_refusal_rate` на low-retrieval запитах падає на **≥ 20 в.п.** відносно pre-deploy baseline години −1…0,  
> то система **автоматично** перемикає traffic 100% на попередній green deployment (previous container image + previous `_CORPUS` digest + previous system prompt config) протягом **≤ 5 хвилин**, блокує подальші canary-промоушени й відкриває P1 інцидент `#rag-rollback` з артефактами метрик.

### Що моніториться між релізами
- Щоденний replay golden-сету (36) на тіньовому стеку = prod weights.  
- Тижневий звіт Δ vs останній ship baseline.  
- Реєстр дефектів D-0X: жоден P1 не має бути «тихо» закритий без кейса в `eval_dataset.jsonl`.

### Як виконується відкат
1. Оркестратор (K8s / Cloud Run revision) зберігає `n-1` revision як `rag-prev`.  
2. RP-RAG-1 → `kubectl/traffic` або alias `rag-live → rag-prev`.  
3. Feature flag `PROMPT_VERSION` і `CORPUS_VERSION` відкочуються атомарно разом із образом (щоб не змішати новий промпт зі старим індексом).  
4. CI позначає реліз `failed-canary`; наступний деплой вимагає green nightly **без** нових P1.

---

## Компроміси

### 1. Відмовилися від повного eval (36 × generate) на кожен Pull Request
- **Чому:** повний generate з Qwen/e5 — десятки хвилин і GPU; PR-цикл команди цього не витримує. RC1 уже показав, що офлайн-гейт над `generations.json` ловить схему/регресію оракулів за &lt;1 хв.  
- **Ціна:** зміна промпта може пройти PR, якщо автор не оновив generations і не зачепив path для smoke.  
- **Ризик:** короткоживуча «дірка» до nightly; мітигація — path filter + обов’язковий smoke на 12 кейсах при зміні `rag_sut.py`.

### 2. Відмовилися від канарейкового релізу як обов’язкового етапу MVP CI
- **Чому:** навчальний/RC обсяг трафіку малий; спочатку потрібні зелені SLA (зараз no-ship). Canary без валідних guardrail-метрик дає хибне відчуття безпеки.  
- **Ціна:** перший прод-викат буде «більшим кроком» (blue/green + RP-RAG-1 замість % canary).  
- **Ризик:** рідкісні UA-only фейли (D-04/D-05) можуть проявитись одразу на 100% після ship; мітигація — жорсткий pre-release без xfail і rollback &lt;5 хв.

### 3. Відмовилися від LLM-as-a-Judge у quality gate (block)
- **Чому:** у ДЗ 8/стратегії гейт має бути детермінований і без ключів; проксі-судді сліпі до тонких підмін, але **платний judge** додає \$/флакі й ламає відтворюваність `run_eval` у викладача. Жорсткі факт-оракули вже зловили D-01/D-02.  
- **Ціна:** можемо пропустити стилістично «гарну», але фактично зсунуту відповідь без ключових токенів.  
- **Ризик:** недооцінка semantic drift; мітигація — warning-шар локального e5-faithfulness на nightly + вибірковий judge лише на release (не block на PR).

---

## Підсумок

Найслабший етап нашого CI/CD зараз — **PR-ярус із опорою на закомічений `generations.json` без гарантованої перегенерації**. Він дешевий і відтворюваний, але саме тому відстає від реальних змін промпта/корпусу до nightly; при цьому RC1 уже no-ship за SafeRefusal=0.429 і Hit@2=0.870, тож слабкість процесу посилює ризик «зеленого PR при червоній якості». Друга слабкість — **відсутність онлайн-guardrail + автоматичного rollback у проді**, бо система ще не задеплоєна, а політика RP-RAG-1 поки лише на папері. Якби був ще один тиждень, у першу чергу я б: (1) зробив обов’язковий GPU smoke-slice на PR при зміні `rag_sut.py`, (2) зняв xfail і закрив D-01/D-02/D-04 як умови merge в `main`, (3) зібрав мінімальний prod-canary з трьома guardrail-метриками й автоперемиканням на `rag-prev` за RP-RAG-1. Це перетворило б поточний навчальний офлайн-`run_eval` на конвеєр, який реально блокує небезпечний реліз тарифного асистента.

---

## AI-usage disclosure

- **AI допоміг:** структурування документа, оформлення YAML-ескізу `ai-pr-gate.yml`, зведення таблиць з уже виміряних метрик і формулювання чернеток політик.  
- **Мої інженерні рішення:** вибір ярусів під наш SUT, пороги гейтів з ДЗ 8 + RC1, effect-size/pass-rate правила під N=36, відмова від full-eval на PR / canary / LLM-judge в block-гейті, конкретна rollback-політика RP-RAG-1, оцінка вартості nightly.  
- **Джерела цифр:** власний прогін `outputs/generations.json` + `reports/results.md` + `test_strategy.md`; не загальні best practices без прив’язки до Acme Cloud RAG.

---

## Додаток: бонусний файл

Ескіз workflow: [`../ai-pr-gate.yml`](../ai-pr-gate.yml) — тригери path-filters, офлайн `run_eval.sh`, upload артефактів. 
