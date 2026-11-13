# Реализация и запуск expanded D4

В `training_d4/` сохранён отдельный комплект из `D:\steganalysis_expanded\lab_submission\source`. Исходные 14 файлов скопированы побайтно; их SHA-256 перечислены в [SOURCE_MANIFEST.json](../training_d4/SOURCE_MANIFEST.json). Baseline остаётся в `training/`. Результаты уже выполненного expanded D4 находятся в [results.md](results.md); новые обучение, inference и сборка образа при подготовке репозитория не выполнялись.
## Конфигурация и происхождение

| Параметр             | Архивный `outputs/train/config_used.json:config` |
| -------------------- | ------------------------------------------------ |
| phase                | expanded-d4                                      |
| epochs               | 400                                              |
| seed                 | 20261025                                         |
| batch_pairs          | 8                                                |
| expected_pair_counts | train 13997, validation 999, test 4998           |
| evaluate_test        | true                                             |

## Команды полного запуска

Команды ниже — PowerShell из корня репозитория. Это инструкция для отдельного запуска. Нужен Docker с NVIDIA GPU и доступом к датасету. Для нового эксперимента используйте новую папку outputs.

```powershell
$dockerExe = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$datasetDir = 'D:\steganalysis_expanded\dataset'
$outputDir = Join-Path (Get-Location) 'outputs\expanded-d4-new'
$scratchDir = Join-Path (Get-Location) 'scratch\expanded-d4-new'
New-Item -ItemType Directory -Path $outputDir, $scratchDir -Force | Out-Null

& $dockerExe build --file training_d4/Dockerfile --tag zhu-expanded-d4:coursework training_d4
if ($LASTEXITCODE -ne 0) { throw 'D4 image build failed' }

& $dockerExe run --rm --gpus all --shm-size=2g --network none `
    --mount "type=bind,source=$datasetDir,target=/datasets,readonly" `
    --mount "type=bind,source=$outputDir,target=/outputs" `
    --mount "type=bind,source=$scratchDir,target=/scratch" `
    zhu-expanded-d4:coursework python -u /app/train.py `
    --dataset-dir /datasets --output-dir /outputs --scratch-dir /scratch `
    --config /app/config/full-archive.json --device cuda --resume auto
if ($LASTEXITCODE -ne 0) { throw 'D4 training failed' }
```

Повтор последней команды продолжает только собственный совместимый checkpoint из этой папки. Не подключайте исторические веса архива как состояние нового запуска: совместимость signature не подтверждена. Для пилота используйте `/app/config/pilot.json` и отдельную папку outputs; test в пилоте отключён. Полный конфиг из архива включает финальный test выбранной по validation модели.

`training_d4/.dockerignore` ограничивает контекст исходниками, конфигами, lock-файлами и двумя SRM-фильтрами. Архивы, датасеты, checkpoints и отчёты не нужны для сборки. Правила `.gitattributes` сохраняют байты комплектов обучения при checkout, поскольку SHA-256 кода входит в signature.

## Проверка сохранённых результатов без обучения

```powershell
& 'C:\steganalysis\.venv\Scripts\python.exe' -B scripts/verify_publication.py `
    --local-reports `
    --results-archive 'C:\Users\Honor\Downloads\zhuk2-20261107-results.tar(1).gz' `
    --d4-source 'D:\steganalysis_expanded\lab_submission\source' `
    --dataset-metadata 'D:\steganalysis_expanded\dataset\metadata.json'
if ($LASTEXITCODE -ne 0) { throw 'Publication verification failed' }
& 'C:\steganalysis\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
```

Проверка читает JSON из архива, сверяет его SHA-256, 400 эпох, счётчики шагов, расписание learning rate и метрики по матрицам ошибок. Дополнительные флаги сверяют все 14 внешних исходных файлов с манифестом и копией в репозитории, а также хеш локального `metadata.json` и его содержимое без `files` с архивными метаданными. Она не извлекает и не загружает веса. После клонирования без внешних артефактов запускается `python -B scripts/verify_publication.py`; при этом проверяются публикуемые файлы и хеши копии исходников.
