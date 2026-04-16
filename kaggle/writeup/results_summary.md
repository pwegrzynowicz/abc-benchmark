# ABC Selective Attention Benchmark — Results Summary

## Scope

This summary is based on the attached benchmark bundle only.  
The bundle contains **8 tasks** and results for **14 models** 
(one model not included yet - Gemini 3.1 Pro Preview).

### Task set
- feature-sensitive text filtering
- feature-sensitive text counting
- feature-sensitive visual filtering
- feature-sensitive visual counting
- structure-sensitive text filtering
- structure-sensitive text counting
- structure-sensitive visual filtering
- structure-sensitive visual counting

## Overall ranking

Average accuracy across all 8 tasks:

1. **Gemini 3 Flash Preview** — **91.91%**
2. **Gemini 2.5 Pro** — **91.40%**
3. **Gemini 2.5 Flash** — **89.48%**
4. **GPT-5.4** — **83.14%**
5. **Claude Opus 4.6** — **80.64%**
6. Claude Opus 4.5 — 74.90%
7. Claude Sonnet 4.6 — 68.33%
8. Claude Sonnet 4.5 — 67.59%
9. Gemini 2.0 Flash — 62.50%
10. GPT-5.4 mini — 61.68%
11. GLM-5 — 50.08%
12. Qwen3 235B — 29.33%
13. DeepSeek V3.2 — 25.93%
14. Qwen3 Next 80B — 21.90%

## Clear performance tiers

### Top tier
The **Gemini family** clearly dominates the benchmark:
- Gemini 3 Flash Preview
- Gemini 2.5 Pro
- Gemini 2.5 Flash

These models are the strongest and most balanced across both text and visual tasks.

### Strong upper-middle tier
- GPT-5.4
- Claude Opus 4.6

These models are strong overall, but they show more visible degradation on harder structure-sensitive slices than the Gemini models.

### Middle tier
- Claude Opus 4.5
- Claude Sonnet 4.6
- Claude Sonnet 4.5
- Gemini 2.0 Flash
- GPT-5.4 mini

These models show mixed profiles: useful competence, but clear weaknesses by modality or task family.

### Lower tier
- GLM-5
- Qwen3 235B
- DeepSeek V3.2
- Qwen3 Next 80B

This group either collapses on visual tasks, structure-sensitive tasks, or both.

## Average task difficulty

Average accuracy across all 14 models:

1. **feature-sensitive text filtering** — **82.07%**
2. **feature-sensitive text counting** — **74.02%**
3. **structure-sensitive text counting** — **70.77%**
4. **structure-sensitive text filtering** — **69.83%**
5. **feature-sensitive visual filtering** — **61.43%**
6. **feature-sensitive visual counting** — **55.82%**
7. **structure-sensitive visual counting** — **51.70%**
8. **structure-sensitive visual filtering** — **40.74%**

## Main patterns

### 1. Selective attention is not monolithic
The benchmark separates at least two different regimes:
- **local feature-based selection**
- **structure-sensitive selection**

Across models, performance is strongest when relevance is defined by explicit local features and weakest when relevance depends on higher-level organization such as grouping, continuity, common region, or scope-like structure.

### 2. Text is easier than vision
All four text tasks are easier on average than all four visual tasks.  
This is one of the strongest global patterns in the results.

### 3. Feature-sensitive attention is easier than structure-sensitive attention
The feature-sensitive tasks are consistently easier than the structure-sensitive ones, especially in vision.

### 4. Structure-sensitive visual filtering is the hardest setting
This is the single hardest task family in the benchmark, with an average accuracy of only **40.74%**.  
That makes it one of the benchmark’s most discriminative slices.

## Best model by task

- **feature-sensitive text filtering**: Gemini 2.5 Pro — **100.00%**
- **feature-sensitive text counting**: GLM-5 — **100.00%**
- **feature-sensitive visual filtering**: GPT-5.4 — **97.94%**
- **feature-sensitive visual counting**: Gemini 2.5 Pro — **91.75%**
- **structure-sensitive text filtering**: Gemini 2.5 Flash — **95.90%**
- **structure-sensitive text counting**: Gemini 2.5 Flash — **93.85%**
- **structure-sensitive visual filtering**: Gemini 3 Flash Preview — **98.33%**
- **structure-sensitive visual counting**: Gemini 3 Flash Preview — **96.25%**

## Most important model-profile observations

### Gemini is the strongest family overall
Gemini models form the top tier, but their strengths are not identical:
- **Gemini 3 Flash Preview** is the strongest on the hardest visual-structure tasks
- **Gemini 2.5 Flash** is especially strong on structure-sensitive text
- **Gemini 2.5 Pro** is extremely strong and well-balanced across the board

### GPT-5.4 is strong, especially on visual feature-sensitive tasks
GPT-5.4 is not the overall winner, but it is extremely strong on **feature-sensitive visual filtering** and remains a strong all-round model.

### Claude Opus 4.6 is good but less balanced than the top Gemini models
Claude Opus 4.6 performs strongly overall, but it degrades more on structure-sensitive slices, especially in vision.

### GLM-5 has the strangest profile in the benchmark
GLM-5 is extremely strong in text:
- feature-sensitive text counting: **100.00%**
- feature-sensitive text filtering: **99.93%**
- structure-sensitive text counting: **84.10%**
- structure-sensitive text filtering: **83.59%**

But it nearly collapses in vision:
- average text accuracy: **91.84%**
- average visual accuracy: **8.31%**

This makes GLM-5 the clearest example of a model that can handle symbolic text selection well but does not meaningfully handle the visual side of the benchmark.

### Qwen and DeepSeek collapse on visual tasks
The weakest open-model profiles are dominated by very poor visual results, especially on structure-sensitive visual tasks.

## Hardest dimensions and variants

The hardest benchmark slices:

- **structure-sensitive visual filtering / principle = continuity** — **15.48%**
- **feature-sensitive visual counting / combined = hard** — **21.79%**
- **structure-sensitive visual counting / principle = continuity** — **29.52%**
- **feature-sensitive visual filtering / combined = hard** — **29.74%**
- **structure-sensitive visual filtering / combined = hard** — **31.90%**
- **structure-sensitive text counting / principle = scope_indentation** — **31.90%**
- **structure-sensitive text filtering / principle = scope_indentation** — **39.05%**

These are exactly the variants with the most value for discrimination: they apply either strong grouping pressure, strong composition pressure, or both.
Scope indentation failure may have happened due parsing and lost context/spaces during parsing.

## Baseline observations

Baseline variants are important because they help distinguish:
- inability to perform the task at all
- vs. failure caused by additional attentional pressure

Average baseline accuracy by task family:

- feature-sensitive text counting — **96.41%**
- feature-sensitive text filtering — **96.41%**
- structure-sensitive text counting — **95.00%**
- structure-sensitive text filtering — **86.90%**
- structure-sensitive visual counting — **70.24%**
- feature-sensitive visual counting — **68.97%**
- feature-sensitive visual filtering — **68.21%**
- structure-sensitive visual filtering — **53.10%**

This supports an important interpretation: many visual failures should not automatically be attributed to selective attention alone. In weaker models, some of the collapse likely reflects weak underlying visual perception or scene parsing. The harder variants become most informative once baseline competence is established.

## Conclusions

### What the benchmark shows clearly
- The benchmark has **strong discriminatory power**: it does not collapse into all-0% or all-100%.
- The strongest models are clearly separated from the middle and lower tiers.
- The benchmark distinguishes **local feature matching** from **attention over structure**.
- Visual structure remains substantially harder than text or local-feature selection.

### What the benchmark is best at revealing
ABC is strongest when it probes:
- distractor suppression
- rule maintenance under composition
- grouping / continuity / common-region binding

### Central takeaway
Many models are much better at **matching the right local cue** than at **attending through structure**.  
That is the main capability gap this benchmark exposes.
