package main

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBigramsOnlyReturnsAdjacentASCIIAlphabeticPairs(t *testing.T) {
	// GIVEN words with uppercase letters, punctuation, short length, and non-ASCII
	// characters.
	normalWord := "Thing"
	punctuationWord := "A!b?cd"
	nonASCIIWord := "café"

	// WHEN bigrams are extracted.
	normalBigrams := bigrams(normalWord)
	punctuationBigrams := bigrams(punctuationWord)
	nonASCIIBigrams := bigrams(nonASCIIWord)
	emptyBigrams := bigrams("")
	shortBigrams := bigrams("a")

	// THEN only adjacent ASCII a-z pairs should be returned.
	require.Equal(t, []string{"th", "hi", "in", "ng"}, normalBigrams)
	require.Equal(t, []string{"cd"}, punctuationBigrams)
	require.Equal(t, []string{"ca", "af"}, nonASCIIBigrams)
	require.Empty(t, emptyBigrams)
	require.Empty(t, shortBigrams)
}

func TestScoreDoesNotPromoteEqualHistories(t *testing.T) {
	variant := Variant{
		MissWeight:         0.55,
		TimingWeight:       0.45,
		ConfidenceAttempts: 10,
		TimingBaselineMS:   180,
		MinTimingMS:        20,
		MaxTimingMS:        1500,
		ExplorationRate:    0.15,
		Selector:           SelectorIndexed,
	}
	stats := map[string]*Stats{}

	// GIVEN several bigrams with exactly the same typing history.
	for _, bigram := range []string{"th", "he", "ng", "mb"} {
		for range 20 {
			updateStats(stats, bigram, true, 145, variant)
		}
	}

	// WHEN the scorer ranks them.
	ranked := rankStats(stats)

	// THEN none of them should be meaningfully promoted over the others.
	require.Len(t, ranked, 4)
	require.InDelta(t, ranked[0].Score, ranked[3].Score, 0.000001)
}

func TestScorePromotesMissyBigram(t *testing.T) {
	variant := Variant{
		MissWeight:         0.55,
		TimingWeight:       0.45,
		ConfidenceAttempts: 10,
		TimingBaselineMS:   180,
		MinTimingMS:        20,
		MaxTimingMS:        1500,
		ExplorationRate:    0.15,
		Selector:           SelectorIndexed,
	}
	stats := map[string]*Stats{}

	// GIVEN several bigrams with the same timing, but one pair has mistakes.
	for _, bigram := range []string{"th", "he", "ng", "mb"} {
		for attempt := range 30 {
			correct := !(bigram == "ng" && attempt < 5)
			updateStats(stats, bigram, correct, 145, variant)
		}
	}

	// WHEN the scorer ranks them.
	ranked := rankStats(stats)

	// THEN the miss-heavy bigram should be treated as the weakest pair.
	require.Equal(t, "ng", ranked[0].Bigram)
	require.Greater(t, ranked[0].Score, ranked[1].Score)
}

func TestScoreRampsUpWithConfidenceAttempts(t *testing.T) {
	variant := Variant{
		MissWeight:         1,
		TimingWeight:       0,
		ConfidenceAttempts: 10,
		TimingBaselineMS:   180,
		MinTimingMS:        20,
		MaxTimingMS:        1500,
		ExplorationRate:    0.15,
		Selector:           SelectorIndexed,
	}

	// GIVEN two histories with the same miss rate but different attempt counts.
	earlyHistory := Stats{Attempts: 2, Misses: 1, AverageMS: 145}
	confidentHistory := Stats{Attempts: 10, Misses: 5, AverageMS: 145}

	// WHEN the scorer applies the confidence ramp.
	earlyScore := score(earlyHistory, variant)
	confidentScore := score(confidentHistory, variant)

	// THEN the low-sample history should be dampened more heavily.
	require.InDelta(t, 0.10, earlyScore, 0.000001)
	require.InDelta(t, 0.50, confidentScore, 0.000001)
	require.Greater(t, confidentScore, earlyScore)
}

func TestScoreWeightsChangeWhetherMissesOrTimingDominate(t *testing.T) {
	missyHistory := Stats{Attempts: 20, Misses: 3, AverageMS: 180}
	slowHistory := Stats{Attempts: 20, Misses: 0, AverageMS: 260}

	missFocused := Variant{
		MissWeight:         0.90,
		TimingWeight:       0.10,
		ConfidenceAttempts: 10,
		TimingBaselineMS:   180,
		MinTimingMS:        20,
		MaxTimingMS:        1500,
		ExplorationRate:    0.15,
		Selector:           SelectorIndexed,
	}
	timingFocused := Variant{
		MissWeight:         0.10,
		TimingWeight:       0.90,
		ConfidenceAttempts: 10,
		TimingBaselineMS:   180,
		MinTimingMS:        20,
		MaxTimingMS:        1500,
		ExplorationRate:    0.15,
		Selector:           SelectorIndexed,
	}

	// GIVEN one bigram with misses and another typed correctly but slowly.
	// WHEN the scoring weights are changed.
	missFocusedMissScore := score(missyHistory, missFocused)
	missFocusedSlowScore := score(slowHistory, missFocused)
	timingFocusedMissScore := score(missyHistory, timingFocused)
	timingFocusedSlowScore := score(slowHistory, timingFocused)

	// THEN the ranking pressure should move toward the heavier signal.
	require.Greater(t, missFocusedMissScore, missFocusedSlowScore)
	require.Greater(t, timingFocusedSlowScore, timingFocusedMissScore)
}

func TestScorePromotesSlowButCorrectBigram(t *testing.T) {
	variant := Variant{
		MissWeight:         0.55,
		TimingWeight:       0.45,
		ConfidenceAttempts: 10,
		TimingBaselineMS:   180,
		MinTimingMS:        20,
		MaxTimingMS:        1500,
		ExplorationRate:    0.15,
		Selector:           SelectorIndexed,
	}
	stats := map[string]*Stats{}

	// GIVEN several bigrams typed correctly, but one pair is consistently slow.
	for _, bigram := range []string{"th", "he", "ng", "mb"} {
		for range 30 {
			spacing := 145.0
			if bigram == "mb" {
				spacing = 260
			}
			updateStats(stats, bigram, true, spacing, variant)
		}
	}

	// WHEN the scorer ranks them.
	ranked := rankStats(stats)

	// THEN the slow pair should rise to the top even without mistakes.
	require.Equal(t, "mb", ranked[0].Bigram)
	require.Greater(t, ranked[0].Score, ranked[1].Score)
}

func TestScoreIgnoresLargeTimingOutlier(t *testing.T) {
	variant := Variant{
		MissWeight:         0.55,
		TimingWeight:       0.45,
		ConfidenceAttempts: 10,
		TimingBaselineMS:   180,
		MinTimingMS:        20,
		MaxTimingMS:        1500,
		ExplorationRate:    0.15,
		Selector:           SelectorIndexed,
	}
	stats := map[string]*Stats{}

	// GIVEN a normal bigram with a stable timing history.
	for range 30 {
		updateStats(stats, "he", true, 145, variant)
	}
	beforeScore := stats["he"].Score
	beforeAverage := stats["he"].AverageMS

	// WHEN one timing sample is far above the configured timing cap.
	updateStats(stats, "he", true, variant.MaxTimingMS+500, variant)

	// THEN the whole update should be skipped so the pause does not change the
	// bigram history.
	require.Equal(t, 30, stats["he"].Attempts)
	require.Equal(t, beforeAverage, stats["he"].AverageMS)
	require.InDelta(t, beforeScore, stats["he"].Score, 0.000001)
}

func TestUpdateStatsOnlyUsesTimingSamplesInsideConfiguredBounds(t *testing.T) {
	variant := Variant{
		MissWeight:         0.55,
		TimingWeight:       0.45,
		ConfidenceAttempts: 10,
		TimingBaselineMS:   180,
		MinTimingMS:        20,
		MaxTimingMS:        1500,
		ExplorationRate:    0.15,
		Selector:           SelectorIndexed,
	}
	stats := map[string]*Stats{}

	// GIVEN timing samples below, on, and above the configured timing bounds.
	// WHEN the histories are updated.
	updateStats(stats, "lo", true, variant.MinTimingMS-1, variant)
	updateStats(stats, "mn", true, variant.MinTimingMS, variant)
	updateStats(stats, "mx", true, variant.MaxTimingMS, variant)
	updateStats(stats, "hi", true, variant.MaxTimingMS+1, variant)

	// THEN only in-range timings should create or update bigram histories.
	require.Nil(t, stats["lo"])
	require.Equal(t, variant.MinTimingMS, stats["mn"].AverageMS)
	require.Equal(t, variant.MaxTimingMS, stats["mx"].AverageMS)
	require.Nil(t, stats["hi"])
}

func TestWordScoreUsesOnlyTopThreeScoredBigrams(t *testing.T) {
	stats := map[string]*Stats{
		"ab": {Score: 10},
		"bc": {Score: 20},
		"cd": {Score: 30},
		"de": {Score: 40},
	}

	// GIVEN a word with four scored bigrams.
	// WHEN the selector scores the word.
	scoredWord := wordScore("abcde", stats)
	unseenWord := wordScore("zz", stats)

	// THEN only the three highest bigram scores should be counted.
	require.Equal(t, 90.0, scoredWord)
	require.Zero(t, unseenWord)
}

func TestIndexedSelectorFallsBackWhenNoScoredBigramsExist(t *testing.T) {
	words := []string{"alpha", "bravo", "charlie"}
	variant := Variant{
		MissWeight:         0.55,
		TimingWeight:       0.45,
		ConfidenceAttempts: 10,
		TimingBaselineMS:   180,
		MinTimingMS:        20,
		MaxTimingMS:        1500,
		ExplorationRate:    0,
		Selector:           SelectorIndexed,
	}
	sim := Simulator{
		Rng:       rand.New(rand.NewSource(1)),
		Words:     words,
		WordIndex: buildWordIndex(words),
	}

	// GIVEN indexed selection before any bigram has a score.
	// WHEN the selector is asked for a word.
	selected := sim.selectWord(variant, map[string]*Stats{})

	// THEN it should fall back to a normal word from the configured list.
	require.Contains(t, words, selected)
}

func TestIndexedSelectorPrefersWordsContainingHighScoringBigrams(t *testing.T) {
	words := []string{"plain", "thing"}
	variant := Variant{
		MissWeight:         0.55,
		TimingWeight:       0.45,
		ConfidenceAttempts: 10,
		TimingBaselineMS:   180,
		MinTimingMS:        20,
		MaxTimingMS:        1500,
		ExplorationRate:    0,
		Selector:           SelectorIndexed,
	}
	stats := map[string]*Stats{
		"th": {Attempts: 10, Score: 10},
	}
	sim := Simulator{
		Rng:       rand.New(rand.NewSource(1)),
		Words:     words,
		WordIndex: buildWordIndex(words),
	}

	// GIVEN a high-scoring bigram that only appears in one available word.
	// WHEN indexed selection chooses from the word index.
	selected := sim.selectWord(variant, stats)

	// THEN the selected word should be the one containing the weak bigram.
	require.Equal(t, "thing", selected)
}

func TestSelectorAvoidsRecentWordsWhenPossible(t *testing.T) {
	stats := map[string]*Stats{
		"th": {Attempts: 10, Score: 10},
	}
	sim := Simulator{
		Rng:         rand.New(rand.NewSource(1)),
		Words:       []string{"thing", "think"},
		RecentWords: []string{"thing"},
	}

	// GIVEN two equally useful candidates where one was just used.
	// WHEN the selector scores candidate words directly.
	selected := sim.pickScoredWord([]string{"thing", "think"}, stats)

	// THEN it should pick the non-recent candidate.
	require.Equal(t, "think", selected)
}

func TestEvaluateComputesRecallRankAndRepeatMetrics(t *testing.T) {
	ranked := []RankedBigram{
		{Bigram: "th", Stats: Stats{Score: 0.9}},
		{Bigram: "ab", Stats: Stats{Score: 0.8}},
		{Bigram: "ng", Stats: Stats{Score: 0.7}},
		{Bigram: "zz", Stats: Stats{Score: 0.6}},
	}
	weakBigrams := []string{"th", "ng", "mb"}
	selectedWords := []string{"one", "one", "two", "two", "two"}

	// GIVEN ranked bigrams, known weak targets, and a word stream with repeats.
	// WHEN the evaluation metrics are computed.
	result := evaluate("metric-check", ranked, weakBigrams, selectedWords)

	// THEN recall, false positives, weak ranks, and word repetition should be
	// derived from those inputs.
	require.Equal(t, "metric-check", result.Variant)
	require.InDelta(t, 2.0/3.0, result.RecallAt10, 0.000001)
	require.Equal(t, 2, result.FalsePositive10)
	require.InDelta(t, 3.0, result.MeanWeakRank, 0.000001)
	require.InDelta(t, 0.40, result.UniqueWordRatio, 0.000001)
	require.InDelta(t, 0.75, result.RepeatWordRatio, 0.000001)
}

func TestObjectiveRewardsRecallAndPenalizesNoiseRankAndRepetition(t *testing.T) {
	// GIVEN a baseline candidate and variants that are worse in one dimension.
	baseline := objectiveScore(3, 0, 6, 0, 3, 100)
	lowerRecall := objectiveScore(2, 0, 6, 0, 3, 100)
	moreFalsePositives := objectiveScore(3, 2, 6, 0, 3, 100)
	worseWeakRank := objectiveScore(3, 0, 30, 0, 3, 100)
	moreRepeats := objectiveScore(3, 0, 6, 20, 3, 100)

	// WHEN the objective is compared.
	// THEN lower recall, noisy top results, worse ranks, and repeated words should
	// all reduce the score.
	require.Greater(t, baseline, lowerRecall)
	require.Greater(t, baseline, moreFalsePositives)
	require.Greater(t, baseline, worseWeakRank)
	require.Greater(t, baseline, moreRepeats)
}

func TestIndexedSelectionNeedsExploration(t *testing.T) {
	words := defaultWords()
	weakBigrams := []string{"th", "he", "ng", "mb", "qu"}
	model := TypistModel{
		BaseMissRate: 0.006,
		WeakMissRate: 0.08,
		BaseMeanMS:   145,
		BaseStdMS:    35,
		WeakExtraMS:  95,
		PauseRate:    0.01,
		PauseMeanMS:  1800,
		WeakBigrams:  toSet(weakBigrams),
	}

	greedy := Variant{
		Name:               "greedy",
		MissWeight:         0.55,
		TimingWeight:       0.45,
		ConfidenceAttempts: 10,
		TimingBaselineMS:   180,
		MinTimingMS:        20,
		MaxTimingMS:        1500,
		ExplorationRate:    0,
		Selector:           SelectorIndexed,
	}
	exploring := Variant{
		Name:               "exploring",
		MissWeight:         0.55,
		TimingWeight:       0.45,
		ConfidenceAttempts: 10,
		TimingBaselineMS:   180,
		MinTimingMS:        20,
		MaxTimingMS:        1500,
		ExplorationRate:    0.15,
		Selector:           SelectorIndexed,
	}

	// GIVEN a synthetic typist with several known weak bigrams.
	// WHEN indexed selection is run with and without random exploration.
	greedyResult := runVariant(greedy, words, weakBigrams, 100, 120, 42, model)
	exploreResult := runVariant(exploring, words, weakBigrams, 100, 120, 42, model)

	// THEN the exploratory variant should recover more known weak pairs.
	require.Less(t, greedyResult.RecallAt10, exploreResult.RecallAt10)
	require.Equal(t, 0.80, greedyResult.RecallAt10)
	require.Equal(t, 1.00, exploreResult.RecallAt10)
}

func TestParameterSearchReturnsBestCandidatesFirst(t *testing.T) {
	words := []string{
		"thing", "other", "bring", "strong", "bemba", "mumba", "quick", "quiet",
		"plain", "table", "later", "sound",
	}
	weakBigrams := []string{"th", "ng", "mb", "qu"}
	model := TypistModel{
		BaseMissRate: 0.006,
		WeakMissRate: 0.08,
		BaseMeanMS:   145,
		BaseStdMS:    35,
		WeakExtraMS:  95,
		PauseRate:    0.01,
		PauseMeanMS:  1800,
		WeakBigrams:  toSet(weakBigrams),
	}

	// GIVEN a small synthetic setup for the constrained grid search.
	// WHEN the parameter search runs.
	results := runParameterSearch(words, weakBigrams, 2, 20, 7, model, SearchOptions{Workers: 1})

	// THEN candidates should be returned from highest objective to lowest, with
	// the selected parameters preserved on each result.
	require.Len(t, results, 400)
	for index := 1; index < len(results); index++ {
		require.LessOrEqual(t, results[index].Objective, results[index-1].Objective)
	}
	require.Equal(t, SelectorIndexed, results[0].Parameters.Selector)
	require.Equal(t, results[0].Variant, results[0].Parameters.Name)
	require.Contains(t, results[0].Variant, "grid-")
}
