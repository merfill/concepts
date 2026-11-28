# Эксперимент № 018 — 2026-11-28

## Центроиды понятий по распределениям: различают ли они понятия? (шаг 1)

**Статус:** шаг 1 прогнан на двух моделях. Ниже — пред-регистрация и результат (§6).
**Место:** отдельная линия после 017. Вопрос не про «носитель» и не про синтетические признаки,
а про то, можно ли по распределениям токенов выделить центроид понятия, свободный от формы, и
различить по нему понятия.
**Вопрос шага 1:** если для каждого понятия усреднить его переформулировки в смысловом
подпространстве, различаются ли полученные центроиды разных понятий?

---

## 1. Почему шаг 1

Если центроиды не различают понятия, то словесное выражение, контроли и валидация бессмысленны.
Поэтому сначала только это: выделить центроиды и проверить, что по ним одно понятие отличимо от
другого. Остальное — следующими шагами.

## 2. Понятия и переформулировки

20 понятий, по 5 переформулировок. Переформулировки обозначают то же понятие, различаются словами
и формой и не имеют слова, общего для всех пяти и уникального для понятия (иначе это лексический
якорь). Перекрытие: среднее Jaccard внутри понятия `0.0032`, между `0.0019`.

Рамка: `You can describe {X} as`.

| понятие | переформулировки |
|---|---|
| dog | a dog; a domestic canine; a barking quadruped kept as a pet; a loyal household animal; man's best friend |
| bear | a bear; a large furry omnivore; a heavy clawed forest mammal; a hibernating woodland creature; a big grizzly or black animal |
| owl | an owl; a nocturnal raptor; a hooting night bird; a round-eyed feathered hunter; a wise-looking creature of the dark |
| eagle | an eagle; a large soaring raptor; a broad-winged bird of prey; a sharp-eyed winged hunter; a bald national emblem of flight |
| apple | an apple; a round red pome; a crisp orchard fruit; a common tree-grown snack; a hand-sized sweet crop |
| banana | a banana; a long yellow tropical produce; a curved peelable snack; a soft starchy finger fruit; a potassium-rich bunch item |
| carrot | a carrot; an orange taproot; a crunchy root vegetable; a rabbit's favourite crop; a tapered edible garden plant |
| onion | an onion; a layered bulb; a pungent round vegetable; a tear-inducing kitchen staple; a papery-skinned allium |
| bed | a bed; a piece of furniture for sleeping; a mattress on a frame; a bedroom resting place; somewhere to lie down at night |
| couch | a couch; a sofa; a long upholstered seat; a living-room bench for several people; a padded seating furniture |
| car | a car; an automobile; a four-wheeled motor vehicle; a passenger road machine; a family transport with an engine |
| boat | a boat; a small vessel; a watercraft; a floating craft for people; a hull that travels on water |
| screwdriver | a screwdriver; a slotted turning implement; a hand tool with a shaped tip; a driving device with a handle; a fastener-twisting gadget |
| broom | a broom; a sweeping implement; a brush on a long handle; a floor-cleaning tool; a bundle of bristles for tidying |
| spear | a spear; a long pointed weapon; a throwing lance; a shaft with a sharp tip; a primitive thrusting polearm |
| whip | a whip; a lash on a handle; a leather striking cord; a long flexible thong; a braided instrument of discipline |
| jeans | jeans; a pair of jeans; denim trousers; blue five-pocket pants; casual cotton twill bottoms |
| sweater | a sweater; a knitted pullover; a warm jumper; a woollen top; a long-sleeved winter garment |
| guitar | a guitar; a six-stringed instrument; a strummed wooden chordophone; a plucked music-maker with a neck; a fretted box for accompaniment |
| violin | a violin; a fiddle; a bowed treble chordophone; a small four-stringed instrument; a high-pitched orchestral strings member |

## 3. Канал и метод

1. Каждая переформулировка подставляется в рамку `You can describe {X} as`; читается **полное
   распределение следующего токена** `p` (весь словарь).
2. `p → sqrt(p)` (расстояние Хеллингера становится евклидовым); PCA на обучающей части
   (строгий held-out).
3. В этих координатах — Фишер: направления, где между понятиями далеко, а переформулировки одного
   понятия близко.
4. Центроид понятия = среднее его переформулировок в смысловом подпространстве.

## 4. Проверка шага 1

**Отложенная переформулировка.** Для каждого понятия прячем одну переформулировку; направления
Фишера и центроиды строим по остальным (80 предъявлений); спрятанную проецируем и относим к
ближайшему центроиду. Повторяем 5 фолдов, всего 100 проверок.

**Метрика:** accuracy ближайшего центроида. Случайность при 20 понятиях — 5%.

**Baseline:**
- без Фишера — ближайший центроид по сырым координатам Хеллингера;
- мешок слов — TF-IDF по тексту переформулировки, без модели.

**Нуль:** перемешать метки понятий, повторить ту же машину.

**Размерность:** `k ∈ {5, 10, 15, 19}`; основная — `k = 10`, остальные как кривая.

## 5. Критерий успеха и провала

- **Успех:** accuracy ≥ 0.5 (в 10 раз выше шанса), выше обоих baseline, permutation `p < 0.01`;
  на перемешанных метках — около 0.05.
- **Провал:** accuracy около шанса или не выше baseline; тогда меняем канал/рамку/дизайн, а не
  натягиваем.

## 6. Результат шага 1 и лексический контроль

**Шаг 1.** Случайность при 20 понятиях — 0.05. Фишер `k=10`: **0.58** (distilgpt2) и **0.69**
(Qwen); лучшее `k`: **0.61** (`k=15`) и **0.80** (`k=19`). Baseline: сырые PCA-10 0.35/0.44;
Hellinger-центроид 0.45/0.60; TF-IDF-центроид 0.05/0.05. Нуль (перемешанные метки): 0.04,
`p ≈ 0.005`. Порог шага (≥ 0.5, выше baseline, `p < 0.01`) пройден.

**Лексический контроль.**
- Диагностика якорей: ни у одного понятия нет слова, общего для всех пяти и уникального.
- TF-IDF-Фишер (лексическая машина) — **0.08/0.08**, уровень шанса.
- После вычитания лексически предсказуемой части координат: **0.39/0.48**; нуль 0.045/0.048,
  `p ≈ 0.005`.
- RDM: модель ↔ понятие `+0.15/+0.16`, модель ↔ лексика `+0.31/+0.18`, частная при контроле
  лексики `−0.07/+0.02`.

**Вывод.** Понятийный сигнал не сводится к точным словам (лексическая машина на уровне шанса), но
он слабый и низкоразмерный: в сырой геометрии доминирует форма/лексика, а понятие вытаскивает
только обученный Фишер. Qwen сильнее distilgpt2.

**Оговорки.** 20 подобранных понятий; train и test делят одни и те же понятия (проверяется перенос
на новую переформулировку, не на новое понятие); TF-IDF ловит точные слова, не синонимию;
резидуализация консервативна. Английский, CPU.

## 7. Что осталось

- словесное выражение понятия из центроида;
- контроли частоты/длины/синтаксиса и кросс-рамка;
- held-out понятие, человеческое сходство.

## 8. Артефакты

- `analysis/src/tc/concept_centroids.py` — предъявления, Хеллингер-Gram, Фишер, центроиды,
  leave-one-out, лексический контроль;
- `analysis/scripts/run_concept_centroids.py` — прогон на двух моделях, JSON;
- `analysis/tests/test_concept_centroids.py` — синтетика;
- `analysis/results/concept_centroids_*.json`.
