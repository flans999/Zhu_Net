# Сохранённые результаты обучения

## Baseline: validation и test

По BC, `config.expected_pair_counts`: train — 3 997 пар, validation — 999 пар, test — 4 998 пар. Источник — BOSSBase с WOW 0.4 bpp (`BC.dataset_metadata.description`). В B сохранено 400 записей `history`; завершение обучения подтверждает BR. Это сведения о запуске.

Лучший checkpoint выбран по максимальной `validation.accuracy`: эпоха 120 (`B.history[119].epoch`), она же `T.checkpoint_epoch`. Размер validation — 1 998 изображений (`B.history[119].validation.images`); размер test — 9 996 изображений (`B.test.images` и `T.test.images`).

В таблице метрики качества указаны в долях, loss — средняя cross-entropy. Все значения приведены из соответствующего объекта JSON без округления.

| Метрика | Validation, лучшая эпоха: B `history[119].validation` | Test, после обучения: B `test` | Test, сохранённая CPU-оценка: T `test` |
| --- | ---: | ---: | ---: |
| `accuracy` | 0.8493493493493494 | 0.8468387354941976 | 0.8468387354941976 |
| `loss` | 0.3974366347203861 | 0.42797833280515457 | 0.42797825646652354 |
| `precision_stego` | 0.8304924242424242 | 0.8257846269498215 | 0.8257846269498215 |
| `recall_stego` | 0.8778778778778779 | 0.8791516606642658 | 0.8791516606642658 |
| `f1_stego` | 0.8535279805352798 | 0.8516329101657137 | 0.8516329101657137 |

Для accuracy это **84.9349% validation** (B, `best_validation_accuracy`, округлено) и **84.6839% test** (B/T, `test.accuracy`, округлено). Последняя эпоха не является лучшей: validation accuracy на эпохе 400 равна 0.8188188188188188 (B, `history[399].validation.accuracy`).
## Expanded D4: результаты из архива

| ID  | Путь внутри архива                   | SHA-256 содержимого                                                |
| --- | ------------------------------------ | ------------------------------------------------------------------ |
| X   | `outputs/train/metrics.json`         | `95a9070027676e0c295ca55a62dec531c57475f8f3355a88b995fb4cf3f6ca81` |
| XR  | `outputs/train/training_result.json` | `36e672d18344e9a69c11e6a6eb13cc9799ddd171202b6fecda7a600375639fca` |
| XC  | `outputs/train/config_used.json`     | `954092178e94aa2774bfd477953fbce34aff43b88a129516d15d7b18dbd1b28a` |

XR подтверждает `status: success`, `success: true`, `training_completed: true`, `phase: expanded-d4` и `completed_epochs: 400`. В X ровно 400 последовательных записей `history`. Лучший validation checkpoint — эпоха 368 (`X.history[367].epoch`), а не последняя эпоха.

Повторная проверка подтвердила согласованность всех 400 записей: при batch 8 виртуальных пар каждая эпоха содержит 111 976 / 8 = 13 997 шагов, итоговый `global_step` равен 5 598 800 и совпадает в X и XR. Learning rate каждой эпохи соответствует архивным `learning_rate`, `milestones` и `gamma`; accuracy, precision, recall и F1 всех validation и итогового test совпадают с расчётом по сохранённым матрицам ошибок. Это проверка JSON, без повторного запуска модели.

XC фиксирует `augmentation: d4-exhaustive-v1`, `physical_train_pairs: 13997`, `virtual_train_pairs: 111976`. По `XC.config.expected_pair_counts`, validation содержит 999 пар, test — 4 998 пар. Значения `images` подтверждают 1 998 изображений validation (`X.history[367].validation.images`) и 9 996 изображений test (`X.test.images`). Это метаданные запуска и размеры соответствующих выборок.

| Метрика | Validation, лучшая эпоха: X `history[367].validation` | Test: X `test` |
| --- | ---: | ---: |
| `accuracy` | 0.9064064064064065 | 0.9000600240096038 |
| `loss` | 0.3600008315269596 | 0.48854393315559136 |
| `precision_stego` | 0.9043824701195219 | 0.9003003003003003 |
| `recall_stego` | 0.908908908908909 | 0.8997599039615847 |
| `f1_stego` | 0.9066400399400899 | 0.9000300210147103 |

Итого: **90.6406% validation** на эпохе 368 (`X.best_validation_accuracy` и `X.history[367].validation.accuracy`, округлено) и **90.0060% test** на 9 996 изображениях (`X.test.accuracy`, `X.test.images`, округлено).
## Сравнение и пределы подтверждения

| Показатель | Baseline, B | Expanded D4, X | Разница, процентных пунктов |
| --- | ---: | ---: | ---: |
| Лучшая validation accuracy | 84.9349% | 90.6406% | +5.7057 |
| Test accuracy | 84.6839% | 90.0060% | +5.3221 |

Источники: validation — `B.best_validation_accuracy` и `X.best_validation_accuracy`; test — `B.test.accuracy` и `X.test.accuracy`. Проценты и разницы вычислены из исходных долей и округлены. Сравниваются сохранённые одиночные запуски.