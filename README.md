# Курсовая работа: Zhu-Net

Репозиторий содержит техническое описание датасетов и копию комплекта обучения Zhu-Net. Полная статистика данных приведена в [docs/datasets.md](docs/datasets.md). CPU-образ для описания данных отделён от окружения обучения.

## Оглавление

- [Структура](#структура)
- [Комплект обучения](#комплект-обучения)
- [Dockerfiles](#dockerfiles)
- [Подготовка PowerShell](#подготовка-powershell)
- [Сборка CPU-образа](#сборка-cpu-образа)
- [Запуск baseline](#запуск-baseline)
- [Запуск expanded](#запуск-expanded)
- [Проверка результатов](#проверка-результатов)
- [Фактические результаты Docker](#фактические-результаты-docker)
- [Повторная дополнительная проверка](#повторная-дополнительная-проверка)
- [Сохранённые результаты обучения](#сохранённые-результаты-обучения)
- [Документация и дальнейшие задачи](#документация-и-дальнейшие-задачи)

## Структура

```text
zhu-net-coursework/
├── README.md
├── Dockerfile.dataset
├── .dockerignore
├── .gitignore
├── .gitattributes
├── requirements.txt
├── scripts/
│   ├── describe_dataset.py
│   └── verify_publication.py
├── tests/
│   ├── test_describe_dataset.py
│   └── test_d4_data.py
├── docs/
│   ├── datasets.md
│   ├── tasks.md
│   ├── results.md
│   └── d4-training.md
├── training/
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── contract.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── runtime_info.py
│   ├── MODEL_SOURCE.json
│   ├── requirements.lock
│   ├── environment.lock
│   ├── expanded_config.json
│   ├── config/
│   │   ├── full.yaml
│   │   └── pilot.yaml
│   └── kernels/
│       ├── SRM3_3.npy
│       └── SRM5_5.npy
├── training_d4/                     # отдельный комплект D4; см. docs/d4-training.md
└── reports/                         # создаётся локально, исключена из Git
    ├── baseline/                    # отчёты первого этапа
    ├── expanded/                    # отчёты первого этапа
    └── docker/
        ├── baseline-20/
        │   ├── images.csv
        │   └── summary.json
        ├── expanded-20/
        │   ├── images.csv
        │   └── summary.json
        └── verification.json        # результат проверки на компьютере
```

## Dockerfiles

| Файл | Назначение | Контекст сборки | Зависимости |
| --- | --- | --- | --- |
| `Dockerfile.dataset` | Описание датасетов на CPU | корень репозитория `.` | корневой `requirements.txt`: Pillow 13.0.0 |
| `training/Dockerfile` | Исходное окружение обучения Zhu-Net | папка `training/` | `training/requirements.lock`, базовый образ PyTorch по digest |
| `training_d4/Dockerfile` | Окружение expanded D4 | папка `training_d4/` | `training_d4/requirements.lock`, базовый образ PyTorch по digest |

`Dockerfile.dataset` использует Python 3.14 slim и запускает `scripts/describe_dataset.py` через ENTRYPOINT. Датасеты и отчёты подключаются при запуске. В образ копируются только скрипт и корневой `requirements.txt`; корневой `.dockerignore` ограничивает контекст сборки этими файлами и Dockerfile. Запуск идёт от пользователя `1000:1000`, без GPU.

У `training/Dockerfile` собственный `.dockerignore` в контексте `training/`. Его команда по умолчанию показывает `train.py --help`.

## Подготовка PowerShell

Команды проверены в Windows PowerShell 5.1 с Docker Desktop в режиме Linux containers, контекст `desktop-linux`. Docker Desktop должен иметь доступ к дискам C: и D:. Docker CLI указан полным путём, поэтому добавлять его в PATH не требуется.

```powershell
Set-Location 'C:\steganalysis\zhu-net-coursework'
$dockerExe = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$baselineDataset = 'C:\steganalysis\dataset_for_check\dataset'
$expandedDataset = 'D:\steganalysis_expanded\dataset'
$reportRoot = Join-Path (Get-Location) 'reports'
New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
```

Если Desktop ещё не запущен:

```powershell
Start-Process -FilePath 'C:\Program Files\Docker\Docker\Docker Desktop.exe' -WindowStyle Hidden
```

Дождитесь готовности Engine. Следующая команда должна показать секцию Server с ОС `linux`:

```powershell
& $dockerExe version
```

## Сборка CPU-образа

Из корня репозитория, с переменными из блока подготовки:

```powershell
& $dockerExe build --file Dockerfile.dataset --tag zhu-net-dataset:stage2 .
if ($LASTEXITCODE -ne 0) { throw 'Dataset image build failed' }
```

При первой сборке нужен доступ к Docker Hub и индексу пакетов Python. При запуске готового образа сеть отключена. Тег базового Python-образа может обновляться; фактические версии и идентификатор проверенного образа приведены ниже.

## Запуск baseline

```powershell
& $dockerExe run --rm --read-only --network none `
    --mount "type=bind,source=$baselineDataset,target=/datasets/baseline,readonly" `
    --mount "type=bind,source=$reportRoot,target=/reports" `
    zhu-net-dataset:stage2 /datasets/baseline /reports/docker/baseline-20 --limit-pairs 20
if ($LASTEXITCODE -ne 0) { throw 'Baseline container failed' }
```

## Запуск expanded

Baseline подключается дополнительно для `--reference-dataset`:

```powershell
& $dockerExe run --rm --read-only --network none `
    --mount "type=bind,source=$expandedDataset,target=/datasets/expanded,readonly" `
    --mount "type=bind,source=$baselineDataset,target=/datasets/baseline,readonly" `
    --mount "type=bind,source=$reportRoot,target=/reports" `
    zhu-net-dataset:stage2 /datasets/expanded /reports/docker/expanded-20 `
    --limit-pairs 20 --reference-dataset /datasets/baseline
if ($LASTEXITCODE -ne 0) { throw 'Expanded container failed' }
```

Флаг `readonly` у каждого входного bind mount запрещает запись в датасет. Отдельный bind mount `/reports` доступен для записи, поэтому результаты остаются на компьютере после удаления контейнера (`--rm`). Флаг `--read-only` задаёт режим только чтения для файловой системы образа и не блокирует подключённую папку отчётов.

Лимит выбирает первые 20 разных `pair_id` в порядке манифеста; в обоих наборах это BOSSBase-пары из train. Структура манифеста проверяется целиком, изображения декодируются для выбранных пар. Для проверки всех изображений уберите `--limit-pairs 20` и укажите другую выходную папку. Полные локальные прогоны первого этапа описаны в `docs/datasets.md`.

## Проверка результатов

Следующий блок читает файлы с компьютера, проверяет ожидаемые поля JSON и число строк CSV:

```powershell
foreach ($datasetName in @('baseline', 'expanded')) {
    $reportDir = Join-Path $reportRoot "docker\$datasetName-20"
    $summary = Get-Content -LiteralPath (Join-Path $reportDir 'summary.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $csvRows = @(Import-Csv -LiteralPath (Join-Path $reportDir 'images.csv') -Encoding UTF8)
    if ($summary.inspected.pairs -ne 20 -or $summary.inspected.images -ne 40 -or
        $summary.complete -ne $false -or $summary.healthy -ne $true -or $csvRows.Count -ne 40) {
        throw "Unexpected report: $datasetName"
    }
    [pscustomobject]@{
        Dataset = $datasetName
        Pairs = $summary.inspected.pairs
        Images = $summary.inspected.images
        CsvRows = $csvRows.Count
        Complete = $summary.complete
        Healthy = $summary.healthy
    }
}
```
## Фактические результаты Docker

Образ: `zhu-net-dataset:stage2`. Фактический ID: `sha256:ccaa52b79d7a265cd3f7f515e4c6553dfa1d3efb0d843860ed33dd619cfd67ec`.

| Набор                         | Код завершения | Пар | Изображений | Строк CSV | complete | healthy |
| ----------------------------- | -------------: | --: | ----------: | --------: | -------- | ------- |
| baseline                      |              0 |  20 |          40 |        40 | false    | true    |
| expanded + baseline reference |              0 |  20 |          40 |        40 | false    | true    |

Оба `summary.json` и оба `images.csv` прочитаны с компьютера после завершения контейнеров. Отсутствующих и нечитаемых изображений, неполных пар нет. В выбранных изображениях подтверждены PGM P5, 256 × 256, режим L.

Для expanded `reference_comparison` подтвердил хеш метаданных baseline, наличие всех 19 988 строк его манифеста, отсутствие изменённых или пропавших строк и совпадение хешей 40 выбранных изображений. Это пробная проверка содержимого изображений; полные результаты первого этапа находятся в `docs/datasets.md`.

Локальная машинная запись проверки: `reports/docker/verification.json`. Сводки: `reports/docker/baseline-20/summary.json`, `reports/docker/expanded-20/summary.json`. Таблицы: `reports/docker/baseline-20/images.csv`, `reports/docker/expanded-20/images.csv`. Эти генерируемые файлы исключены из Git; команды выше создают их локально.

## Повторная дополнительная проверка

В прежней одноразовой проверке после успешных Docker-прогонов упало условие:

```python
assert tasks.count('**Описание:**') == tasks.count('**Критерий готовности:**') == 10
```

Причина — передача Python-кода с кириллицей через pipeline Windows PowerShell 5.1 при `$OutputEncoding = US-ASCII`: строковые литералы заменялись знаками `?`. В самом `docs/tasks.md`, прочитанном как UTF-8, найдено по 10 описаний и критериев; исправление содержимого задач не требуется.

Проверка сохранена в UTF-8-файле и запускается напрямую. Она проверяет документацию, ссылки на публикуемые файлы, комплект обучения и правила Git. Дополнительный флаг сверяет локальные Docker JSON/CSV, хеши с прежними host-прогонами и запись verification:

```powershell
& 'C:\steganalysis\.venv\Scripts\python.exe' -B scripts\verify_publication.py --local-reports
if ($LASTEXITCODE -ne 0) { throw 'Publication verification failed' }
```

Без локальных отчётов, например после клонирования, используйте `python -B scripts/verify_publication.py` из активного окружения. Флаг `--local-reports` требует ранее созданные отчёты; обучение и Docker эта проверка не запускает.

## Сохранённые результаты обучения

[docs/results.md](docs/results.md) содержит сохранённые метрики baseline и expanded D4 с явным разделением validation/test, пути и SHA-256 источников и сравнение результатов. Expanded D4 подтверждён JSON из скачанного архива; прежние baseline reference в расширенном комплекте остаются копиями baseline. Реализация и пределы подтверждения её происхождения описаны в [docs/d4-training.md](docs/d4-training.md).

## Документация и дальнейшие задачи

[docs/datasets.md](docs/datasets.md) содержит полные таблицы разбиений, технические характеристики изображений, происхождение данных и описание расширения train. [docs/tasks.md](docs/tasks.md) содержит 10 заготовок GitLab Issues с описаниями и критериями готовности.

`.gitignore` исключает локальные датасеты, PGM-файлы, архивы, веса, checkpoints, выходные файлы экспериментов, кэш Python, локальные окружения и всю папку `reports/`. SRM-фильтры `training/kernels/*.npy` и `training_d4/kernels/*.npy` сохраняются в составе исходников. Генерируемые CSV/JSON остаются локальными; для получения их в новой копии репозитория выполните команды выше или полные команды из `docs/datasets.md`.

- [Протокол экспериментов и ответы на вопросы](docs/experiment_protocol.md)
