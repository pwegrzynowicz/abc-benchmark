### Project Name



ABC: Attention Benchmark for Cognition

### Your Team

**Team XYZ**  
Patrycja Węgrzynowicz, Ksawery Kopeć

### Problem Statement

Current LLM and multimodal benchmarks measure broad capabilities such as knowledge, reasoning, instruction following, and general multimodal competence, but they rarely isolate **attention** as a capability in its own right. In particular, they say little about whether a model can keep its focus on the correct targets when plausible distractors compete for attention.

ABC addresses this gap by evaluating **selective attention under interference**.

The broader ABC design space spans four axes:
- **attention family**: selective, sustained, shifting, divided
- **attentional basis**: feature-sensitive, structure-sensitive
- **modality**: text, visual, mixed
- **task type**: filtering, counting

This submission instantiates one slice of that space:
- **family**: selective attention
- **attentional basis**: feature-sensitive and structure-sensitive
- **modality**: text and visual
- **task type**: filtering and counting

The key conceptual distinction is between:
- **feature-sensitive attention**, where relevance is determined by local attributes or logical combinations of attributes;
- **structure-sensitive attention**, where relevance depends on higher-level organization such as grouping, region membership, continuity, or scope.

A model may follow explicit local rules well while still failing when the correct answer depends on belonging to the right group, line, region, or subset. ABC is designed to expose that gap.

### Task & benchmark construction

ABC is organized as a structured benchmark space rather than a flat collection of tasks. Each combination of **attentional basis × modality × task type** defines a **task group** with its own input form, rule logic, and output format.

Each example has the same abstract structure:
- an **input**: a text instance or visual scene containing targets and distractors,
- a **selection rule**: the logic defining relevance,
- an **output**: either the matching items (**filtering**) or their number (**counting**).

This keeps the benchmark conceptually consistent across modalities. What changes is the source of attentional pressure.

In **feature-sensitive** tasks, pressure comes from distractors, compositional rules, and rule maintenance. The model must identify the rule-relevant features, apply the correct conjunction, negation, or disjunction, and ignore plausible confounds.

In **structure-sensitive** tasks, pressure comes from organization. This branch is inspired by Gestalt principles such as **proximity**, **similarity**, **common region**, and **continuity**. In vision, these appear directly through scene layout and grouping. In text, they are adapted through grouping, spacing, indentation, and layout. The task is not just to find a locally matching item, but to attend to the correct **unit of organization**.

Within each task group, the dataset contains multiple **dimensions** and **variants**. Dimensions define the main source of attentional pressure; variants are concrete realizations of that pressure. This supports diagnostic analysis: instead of observing only that a model scored lower, we can often identify whether the failure is driven by distractor pressure, unstable rule maintenance, weak structural binding, or inability to perform the basic form of the task.

A special role is played by **baseline** dimensions. These are intentionally simple versions of each task used to test whether the model can perform the underlying operation before additional attentional pressure is introduced. Baseline is therefore not just an easier difficulty setting; it is an interpretive anchor.

This matters especially in visual tasks. If a model fails a difficult visual example, that failure may reflect weak scene understanding or structural parsing rather than selective attention itself. Baseline variants help separate these cases:
- if a model fails baseline, later failures cannot be cleanly attributed to attention;
- if a model passes baseline but fails once distractors or grouping pressure are added, the evidence for an attentional breakdown is much stronger.

At the single-task level, ABC tests whether the model can:
- perform the basic operation required by the task,
- apply the rule correctly,
- preserve the rule over multiple candidates,
- suppress distractors,
- and, in structure-sensitive settings, maintain attention over the correct group, region, line, or scope.

### Dataset

The dataset is fully synthetic and programmatically generated. For this benchmark, that is a strength, not a limitation.

Synthetic generation makes it possible to control the source of difficulty, construct deliberate near-miss distractors, and guarantee deterministic gold answers. It also allows systematic variation of attentional pressure across dimensions and variants while keeping the evaluation auditable.

The dataset is organized by task group, where each group corresponds to one combination of **attentional basis × modality × task type**. Within each group, the dataset includes baseline settings as well as harder distractor-heavy or structure-heavy variants.

The evaluated benchmark bundle contains:
- **1410** feature-sensitive text examples
- **630** feature-sensitive visual examples
- **390** structure-sensitive text examples
- **360** structure-sensitive visual examples

Each row includes the metadata required for exact evaluation and later analysis, such as task group, dimension, variant, prompt, gold answer, and task-specific target or distractor statistics. Because every example is generated from an explicit underlying state, provenance is straightforward and the dataset is fully auditable.

### Technical details

ABC was built around three technical requirements: **deterministic gold generation**, **strict output checking**, and **diagnostic evaluation**.

**Deterministic gold generation.** Each example is generated from an explicit underlying state: a structured record set in text or an object-and-group scene specification in vision. Gold answers are computed from that state and the task rule itself, not inferred from rendered output or post-hoc annotation. This guarantees correctness by construction.

**Strict output checking.** The benchmark uses constrained, machine-checkable outputs rather than open-ended answers. In counting tasks, the model must return an exact integer in a fixed schema. In filtering tasks, it must return the exact structured subset, which is validated deterministically against the gold selection. This avoids vague free-form answers and keeps evaluation robust.

**Diagnostic evaluation.** Every example is labeled by **task group**, **dimension**, and **variant**. This supports analysis at multiple levels: by modality, by attentional basis, by task type, and by specific source of pressure. Baseline dimensions are part of this design: they help distinguish failures caused by weak perceptual or structural readout from failures caused by distractor pressure, rule maintenance, or grouping-based attention.

Together, these choices make ABC not only automatically scorable, but also interpretable. The benchmark can reveal not just that a model failed, but whether it failed because it could not solve the basic task, could not preserve the rule, could not suppress distractors, or could not maintain attention over the correct structured unit.

### Results, insights, and conclusions

We evaluated **15 models** spanning frontier multimodal systems and open models. The benchmark produces clear performance tiers rather than collapsing into a trivial all-pass or all-fail regime.

The strongest models form a clear top tier, with the **Gemini family** showing the most balanced performance across both text and visual tasks. A second group of strong frontier models performs well overall but shows clearer degradation on harder structure-sensitive slices. Weaker models often remain competent on easier text tasks while degrading sharply, sometimes catastrophically, on harder visual-structure tasks.

The central result is that **selective attention is not monolithic**.

Across models, performance is strongest when relevance is defined by explicit local features and weakest when relevance depends on higher-level organization such as grouping, continuity, common region, or scope-like structure. Strong local matching is therefore not the same as robust attention over structure.

This is the main insight revealed by ABC. In feature-sensitive tasks, a model can often succeed by matching explicit attributes and suppressing obvious distractors. In structure-sensitive tasks, that is not enough. The model must first identify the correct group, region, line, or subset, preserve that structural binding, and only then apply the rule. Many models that appear strong on surface-level selection and general task execution are substantially less reliable in this regime.

The benchmark also reveals distinct failure modes. Adversarial-confound errors indicate weak distractor suppression. Negation and disjunction failures indicate unstable rule maintenance. Continuity and common-region failures indicate weak structural binding. Scope-like text failures indicate flattening of hierarchy rather than preservation of grouped structure. ABC therefore diagnoses **how** attention fails, not just **which** model scores higher.

The visual results require careful interpretation. Poor performance on harder visual slices does not always imply a purely attentional failure; in some models it likely reflects weak underlying scene understanding. This is exactly why baseline variants are important. They establish whether the model can solve the simple form of a visual task at all, making later failures on confound-heavy or grouping-heavy variants much more interpretable.

ABC supports three main conclusions:
1. **Selective attention decomposes into multiple sub-capabilities rather than forming a single unified skill.**
2. **Structure-sensitive attention remains a major weakness even in otherwise strong models.**
3. **Benchmarking attention through grouping, interference, and structural dependence reveals failures that broad general-purpose benchmarks often hide.**

In short, many models are much better at **matching the right local cue** than at **attending through structure**. That is the capability gap ABC is designed to expose.

### Organizational affiliations

Independent

### References & citations
- Wagemans, J. et al. (2012). A Century of Gestalt Psychology in Visual Perception: Perceptual Grouping and Figure–Ground Organization.
