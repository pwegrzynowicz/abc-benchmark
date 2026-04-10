# Benchmark name
ABC-mini: Attention Benchmark for Cognition

# Main claim

We evaluate attention in AI along two dimensions: core attentional operations and attentional basis.
Core operations follow cognitive science distinctions such as sustained attention, selective/perceptual inhibition, shifting, and stimulus-driven capture, while the basis axis distinguishes feature-based from structure-sensitive selection.
The benchmark’s main novelty is testing whether models can selectively attend not only to simple features but also to organized structure.

## Primary axis

- sustained attention
- selective attention / perceptual inhibition
- attention shifting
- stimulus-driven attention

## Secondary axis

- feature-based
- structure-sensitive

# MVP cells

1. sustained × feature-based
2. selective × feature-based
3. selective × structure-sensitive

# Task: cluster_constrained_counting_v1

## Status
MVP / frozen for now

## Purpose
Selective × structure-sensitive (cluster-constrained counting)

The model must count target items that belong to the same spatial cluster as the anchor item while ignoring feature-matched distractors outside that cluster.

## Current strengths
- deterministic scoring
- clear model differentiation
- visible difficulty effect
- works in Kaggle benchmark pipeline

## Known limitations
- medium and hard are not yet cleanly separated
- some run-to-run instability for a subset of models
- only 60 items in current MVP set
- may still contain a few borderline items

## Planned future improvements
- expand to 90 items or more
- review borderline failures manually
- tighten medium/hard calibration
- test on a second independently generated dataset

## Stimulus type

A simple image containing multiple objects arranged into several spatial clusters.

Each object has local features such as:
- shape
- color
- anchor marker

One object is designated as the anchor item --- the anchor is always shown as a starred outline marker.

## Prompt format

A prompt presents a single visual stimulus and asks the model to count only the target items that belong to the same spatial cluster as the anchor item.

Example prompt:

> Count the red circles in the same cluster as the starred item. Ignore all other red circles. Respond with a number only.

The instruction should be minimal, explicit, and stable across items.

## Allowed outputs

A valid response must contain exactly one non-negative integer with no explanation.

Examples of valid outputs:
- 0
- 3

Examples of invalid outputs:
- three
- The answer is 3
- 3 red circles

## Gold label

The gold label is the exact number of target items that satisfy both conditions:
- they match the requested local feature criteria, and
- they belong to the same spatial cluster as the anchor item.

Example:
- if the anchor belongs to a cluster containing exactly 3 red circles, then gold label = 3

### How gold label should be computed

The item generator should:
- generate a scene containing multiple spatial clusters,
- designate one anchor item,
- identify which objects belong to the same spatial cluster as the anchor under the generator’s clustering rule,
- count how many of those objects match the requested target feature,
- store that count as the gold label.

## Generation and clustering rules

For the first task version, spatial cluster membership is defined by sampling objects around cluster centers 
with bounded within-cluster radius `r_in`, while cluster centers are separated by at least `d_out`, 
where `d_out` is sufficiently larger than `r_in` to make cluster boundaries visually unambiguous.

Moreover:
- no overlapping clusters
- no bridge points between clusters
- no singleton anchor cluster unless intentionally allowed
- objects do not overlap each other
- objects stay within image bounds
- anchor marker never overlaps neighboring objects

## Dataset balancing constraints

- answers should be within a bounded range like 0–5 or 1–5
- roughly balance counts across the dataset
- no strong skew toward 1 or 2

## Sanity-check requirement

Before inclusion, each generated item should pass an automatic sanity check verifying 
that the anchor’s spatial cluster is unique, visually separated, 
and that the gold label satisfies all anti-shortcut constraints.

## Human clarity requirement

Each item should be visually interpretable by a typical human observer with minimal effort. 
The anchor’s cluster should be uniquely identifiable without requiring abstract rule discovery or ambiguous clustering judgments.

## Difficulty knobs

- clutter
- distractor similarity
- number of spatial clusters
- number of objects per spatial cluster
- distance between spatial clusters
- within-cluster spacing
- overlap between target-feature items inside vs outside the anchor spatial cluster
- cluster separation strength
- local-feature confound

## What this task should measure

- selective attention under structured distractors
- ability to use perceptual organization to constrain selection
- resistance to distractors that share the same local features but belong to irrelevant spatial clusters
- structure-sensitive filtering rather than simple global counting

## What would invalidate it

- the answer can be obtained by counting all target-feature items globally without using clustering
- the relevant cluster is ambiguous or visually unclear
- multiple clustering interpretations are equally plausible
- the anchor does not clearly identify a unique target cluster
- local shortcuts solve the task without attending to structure
- the prompt wording does most of the work and the visual structure contributes little
- rendering artifacts correlate with the correct count
- distractors are too weak, making the task simple perception or counting rather than selective attention

## Shortcut risks

- global count cue
- color-count cue
- shape-count cue
- positional bias
- anchor proximity artifact
- trivial density cue
- answer distribution bias

## Anti-shortcut constraints

For each item:
- at least one distractor item matching the target feature must appear outside the anchor’s cluster
- at least one non-target item must appear inside the anchor’s cluster
- global counting of the target feature must differ from the correct answer
- the answer should not be inferable from cluster size alone
- anchor position should vary across items

## Why this is an attention task

This task is intended to measure selective attention, not just perception or generic counting, 
because the model must focus on task-relevant items while suppressing feature-matched distractors in irrelevant spatial clusters. 
That matches the attention framing in your documents: attention involves focusing cognitive resources 
on relevant information and filtering competing information, while selective attention specifically involves 
maintaining focus on task-relevant stimuli in the presence of distractors.

### Example item concept

Scene contains:
- several clusters of mixed shapes and colors
- one starred anchor item inside one cluster
- red circles both inside and outside the anchor’s cluster

Prompt:
- Count the red circles in the same cluster as the starred item. Ignore all other red circles. Respond with a number only.

Correct reasoning:
- identify the anchor’s cluster,
- restrict attention to that spatial cluster,
- count only the red circles inside it,
- ignore red circles elsewhere.