# Bigram Crunch Lab

This is a small Go program I used while working on the `bigram_crunch` funbox
for Monkeytype.

The first version I had running on my own deployment at
`typing.martinnn.com` was intentionally simple. It mostly used hard-coded
weights and a direct score for letter pairs that I mistyped. That was enough to
prove the idea was interesting, but it was not a very good way to decide what
the scoring constants should be, or whether the word-selection strategy was
actually doing what I wanted.

This repo is my attempt to make that process a bit more explicit. The aim is to
define some controlled word samples and typing histories, then compare a few
scoring and selection strategies to see which ones are more likely to push known
weak bigrams toward the top.

## What I am trying to test

Bigram Crunch is meant to do two related things:

- notice bigrams that are mistyped more often
- notice bigrams that are typed correctly but slowly

The second point is important because otherwise the feature becomes too similar
to a normal weak-spot/mistake mode. I wanted the algorithm to pick up cases
where a pair is technically correct, but still slows the typist down.

The simulation gives a synthetic typist a few known weak bigrams, for example
`th`, `he`, `ng`, `mb`, and `qu`. Those bigrams are given a higher miss rate
and an extra timing delay. The program then compares scoring variants and checks
whether those known weak bigrams are recovered near the top.

The output is evidence for tuning, not proof.

## Things I tried

The current run compares a few rough directions:

- a miss-heavy score
- a more balanced miss/timing score
- a timing-forward score
- stricter timing outlier filtering
- faster or slower confidence ramp-up
- random word selection
- sampled word selection
- indexed word selection from `bigram -> words`

One useful early result was that greedy indexed selection can get stuck. It may
find a few weak pairs early, then keep selecting words for those pairs and fail
to discover the remaining weak pairs. That pushed me toward keeping a small
amount of random exploration even after the algorithm has found some weak
bigrams.

That is the kind of result I wanted from this repo: not "the algorithm is now
solved", but "this direction seems less likely to fail in an obvious way".

## How the algorithm works

The harness has two main parts: a bigram scorer and a word selector.

### Bigram scoring

Each bigram keeps a small stats record:

```go
type Stats struct {
    Attempts  int
    Misses    int
    AverageMS float64
    Score     float64
}
```

Every time the synthetic typist types a bigram, the program updates:

- `Attempts`: how many times this bigram has been seen
- `Misses`: how many times it was mistyped
- `AverageMS`: the moving average for the key-to-key timing
- `Score`: the current estimate of how weak this bigram is

The score is currently calculated like this:

```text
missRate = misses / attempts

timingPenalty = clamp(
  (averageMS - timingBaselineMS) / timingBaselineMS,
  0,
  2
)

confidence = min(1, attempts / confidenceAttempts)

score = confidence * (
  missRate * missWeight +
  timingPenalty * timingWeight
)
```

The confidence term is there because I do not want one early mistake to dominate
the ranking forever. The score is allowed to ramp up as the bigram gets enough
attempts to be more meaningful.

Timing samples below `minTimingMS` or above `maxTimingMS` are ignored. The
reason for this is practical rather than theoretical: very small timings and
large pauses usually say more about input noise or distraction than about a
specific letter pair.

### Word selection

Once bigrams have scores, the selector tries to choose words that contain the
highest-scoring bigrams.

The indexed selector builds a map like this:

```text
th -> thing, think, other, three
ng -> thing, bring, strong, language
mb -> bemba, mumba, chomba
```

It then looks at the current weak bigrams and chooses from words that contain
those pairs. The simulator also keeps a short recent-word list so it can avoid
repeating the same words too aggressively.

One thing the simulator made obvious is that this cannot be purely greedy. If
the selector only practises the first weak bigrams it finds, it can fail to
discover other weak bigrams. That is why the indexed variants include an
`explorationRate`, where some percentage of words are still selected randomly.

## What is being optimised

This is an optimisation problem in a limited sense. The program is not training
a model with gradient descent. It is doing a constrained grid search.

The search tries combinations of values like:

```text
missWeight
timingWeight
confidenceAttempts
maxTimingMS
explorationRate
```

For each combination, the program runs the same synthetic typing setup and
computes an objective score. A candidate name like this:

```text
grid-m0.55-t0.45-c5-max1200-x0.20
```

means:

```text
missWeight = 0.55
timingWeight = 0.45
confidenceAttempts = 5
maxTimingMS = 1200
explorationRate = 0.20
```

The current objective is:

```text
objective =
  recall@10 * 100
  - falsePositive10 * 1.5
  - meanWeakRank * 0.25
  - repeatWordRatio * 20
```

You can think of this as minimising a simple loss where:

```text
loss = -objective
```

The largest reward is for recovering the known weak bigrams. The other terms are
smaller penalties to avoid a top 10 full of noise, weak bigrams ranked too low,
or a word stream that repeats too much.

This objective is deliberately simple and probably not final. It gives me a way
to compare settings under the same assumptions. It does not prove that the top
candidate is universally optimal.

## Run

```bash
go run .
```

Example with more sessions and CSV output:

```bash
go run . -sessions 250 -words 150 -csv results.csv
```

By default the program also runs a constrained parameter search over indexed
selection variants. It is meant to show which combinations performed well under
the same synthetic setup.

```bash
go run . -search=true -search-top 10
```

To only run the named hand-picked variants:

```bash
go run . -search=false
```

Useful flags:

```bash
go run . -weak th,he,ng,mb,qu -pause-rate 0.02 -weak-miss 0.08 -weak-extra-ms 95
```

## Tests

The real assertions are in `main_test.go` and can be run with:

```bash
go test ./...
```

Those tests use `testify/require`. I kept each test deliberately self-contained:
the variant settings, input history, action being tested, and expected result
are all inside the test function. That makes the tests a little more verbose,
but it also means I do not need to jump between helper scenarios just to see
what a case is checking.

The word samples used by the tests are also listed in `testdata/words.md` as
documentation. The tests do not load that file; it is only there to make the
data easier to inspect in one place.

The current assertion tests cover:

- bigram extraction only returns adjacent ASCII `a-z` pairs
- equal histories should not create an artificial weak bigram
- a miss-heavy bigram should move to the top
- confidence should ramp scores up as attempts accumulate
- scoring weights should control whether misses or timing dominate
- a slow-but-correct bigram should move to the top
- a large timing outlier should not change the timing signal
- timing samples should only affect averages inside the configured bounds
- word scores should use only the top three scored bigrams in a word
- indexed selection should fall back before any bigram has a score
- indexed selection should prefer words containing high-scoring bigrams
- recent words should be avoided when alternatives are available
- evaluation should compute recall, false positives, rank, and repeat metrics
- the objective should reward recall and penalise noise, rank, and repetition
- indexed selection needs some exploration to avoid getting stuck
- parameter search should return the highest-objective candidates first

The program also runs a few deterministic scenario checks by default. These are
small hand-shaped cases that helped me check the assumptions before looking at
the larger simulation. They are printed for readability when running the
program, but they are not a replacement for `go test ./...`:

- **equal history**: when all bigram histories are the same, no pair should be
  promoted
- **one missy bigram**: when one bigram has more misses, it should rise to the
  top
- **one slow but correct bigram**: when one bigram is slow but mostly correct,
  it should still rise to the top
- **timing outlier**: when a timing sample is an obvious pause, it should be
  ignored
- **greedy indexed selection**: when indexed selection is greedy, it can miss
  weak pairs it has not discovered yet

You can skip those checks with:

```bash
go run . -scenarios=false
```

## Metrics

- `recall@10`: fraction of known weak bigrams found in the top 10 scores.
- `false+10`: top-10 entries that are not known weak bigrams.
- `meanRank`: average rank of the known weak bigrams. Lower is better.
- `meanScore`: average score of top-10 bigrams.
- `uniqueWords`: unique selected words divided by total selected words.
- `repeatWords`: adjacent repeated words divided by total transitions.

A typical run currently looks like this:

```text
Scenario checks

scenario                           pass    expected                                               actual
equal history                      yes     no bigram should materially outrank the others         top=th bottom=mb scoreDelta=0.000000
one missy bigram                   yes     ng should move to the top                              top=ng
one slow but correct bigram        yes     mb should move to the top                              top=mb
timing outlier                     yes     a single pause above the timing cap should not change the score before=0.000000 after=0.000000
greedy indexed selection           yes     exploration should recover more known weak bigrams than greedy indexing greedy recall@10=0.80 explore recall@10=1.00

variant                         recall@10   false+10     meanRank
miss-heavy-random                    1.00          5         3.00
balanced-sample                      1.00          5         3.00
timing-forward-indexed-greedy        0.80          6        20.80
timing-forward-indexed-explore       1.00          5         3.00
strict-outlier-indexed-explore       1.00          5         3.00
fast-confidence-indexed-explore      1.00          5         3.00
```

The exact numbers will vary with the seed and flags, but the main observation
from this run is that indexed selection without exploration can miss weak
bigrams that it does not sample often enough.

The search output prints an `objective` score. The objective is deliberately
simple: it rewards recovering the known weak bigrams, then applies smaller
penalties for false positives, poor weak-bigram rank, and repeated adjacent
words.

Near the end, the program prints the values that are worth copying into the
Monkeytype implementation:

```text
Suggested values to copy into Monkeytype:
  missRateWeight = 0.55
  timingWeight = 0.45
  confidenceAttempts = 5
  maxTimingMs = 1200
  explorationRate = 0.20
```

Those are the candidate parameters from the highest-scoring grid result. The
exact numbers can change when the seed, synthetic typing model, or search ranges
change, so I would treat them as tuned starting values rather than permanent
truths.

## Limitations

There are a few important limitations to keep in mind:

- This is not a full model of human typing. It uses a synthetic typist rather
  than real Monkeytype data, so the results should be treated as a controlled
  approximation.
- The default word list is small compared with real Monkeytype language files.
- The search is a constrained grid search, not a global optimisation.
- The objective function is hand-shaped and may need adjustment.
- Several candidates can tie or nearly tie under the current setup.
- The current algorithm only models adjacent ASCII `a-z` bigrams.

## Related Work I Looked At

The original scoring implementation lived on my own Monkeytype deployment at
`typing.martinnn.com`. At that point it was mostly a joke version of the idea:
I was switching keyboard layouts to Colemak, so I structured it around bigrams I
kept mistyping and did not include timing at all.

Since I am now trying to push this toward a wider audience, I wanted to revisit
the scoring part and make it a little more considered. The scoring is a large
part of the feature, so I figured it was worth taking a look at existing
research around bigrams, typing behaviour, and keystroke timing. A lot of what
I found was adjacent rather than directly applicable, but a few papers were
still useful to think through.

- Dhakal, Feit, Kristensson, and Oulasvirta, "Observations on Typing from 136
  Million Keystrokes", CHI 2018.
  The useful idea here is that letter-pair behaviour and inter-key intervals can
  say something meaningful about typing performance. Bigram Crunch is much
  simpler than their analysis, but this paper made the timing part feel worth
  trying rather than relying only on mistakes.
  https://doi.org/10.1145/3173574.3174220

- Killourhy and Maxion, "Comparing Anomaly-Detection Algorithms for Keystroke
  Dynamics", DSN 2009.
  This is not really about typing practice. It is about identifying whether a
  typing sample looks like it came from a genuine user or an impostor. Still,
  their setup was useful to think about because they compare several approaches
  under the same dataset and evaluation procedure. That is close to the kind of
  thing I wanted here: not necessarily to use their algorithms, but to avoid
  changing the test conditions every time I changed the scoring rule.
  https://doi.org/10.1109/DSN.2009.5270346

  I also found this implementation repo, which tries some of the common detector
  ideas from that work:
  https://github.com/vinigarg/Comparing-Anomaly-Detection-Algorithms-for-Keystroke-Dynamics

  Some of the ideas are adjacent, especially the use of timing features and
  repeated evaluation. But I do not think the ML-style anomaly-detection setup
  would have helped much for Bigram Crunch. I am not trying to classify a user
  or detect an impostor. I am trying to rank letter pairs for practice and then
  choose useful words from that ranking. A simpler scoring simulation felt like
  the more direct thing to try first.

I also looked at some broader keystroke-dynamics survey material. Most of it is
focused on authentication, which is adjacent but not the same goal. Bigram
Crunch is not trying to identify a user; it is trying to choose useful practice
words for one user.

## Notes For Monkeytype

The current Monkeytype implementation should probably stay conservative:

- track adjacent ASCII `a-z` bigrams first
- ignore very small and very large timing samples
- combine miss rate and timing penalty with a confidence ramp
- avoid a purely greedy indexed selector
- keep some random exploration so undiscovered weak bigrams can still surface
- keep reset/inspection as console helpers until the feature stabilizes

The main thing I would take from this experiment right now is not a single magic
constant. It is that word selection needs to balance exploitation and
exploration. If it only practises the bigrams it already knows about, it can
miss other weak pairs.
