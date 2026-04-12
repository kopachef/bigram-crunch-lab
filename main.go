package main

import (
	"cmp"
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"strings"
)

type Variant struct {
	Name               string
	MissWeight         float64
	TimingWeight       float64
	ConfidenceAttempts int
	TimingBaselineMS   float64
	MinTimingMS        float64
	MaxTimingMS        float64
	ExplorationRate    float64
	Selector           SelectorKind
}

type SelectorKind string

const (
	SelectorRandom  SelectorKind = "random"
	SelectorSample  SelectorKind = "sample"
	SelectorIndexed SelectorKind = "indexed"
)

type Stats struct {
	Attempts  int
	Misses    int
	AverageMS float64
	Score     float64
}

type Result struct {
	Variant          string
	Parameters       Variant
	Objective        float64
	RecallAt10       float64
	FalsePositive10  int
	MeanWeakRank     float64
	MeanTopScore     float64
	UniqueWordRatio  float64
	RepeatWordRatio  float64
	TopBigrams       []RankedBigram
	SelectedWordRuns int
}

type ScenarioResult struct {
	Name        string
	Expected    string
	Actual      string
	Passed      bool
	Observation string
	TopBigrams  []RankedBigram
}

type RankedBigram struct {
	Bigram string
	Stats
}

type TypistModel struct {
	BaseMissRate float64
	WeakMissRate float64
	BaseMeanMS   float64
	BaseStdMS    float64
	WeakExtraMS  float64
	PauseRate    float64
	PauseMeanMS  float64
	WeakBigrams  map[string]bool
}

type Simulator struct {
	Rng           *rand.Rand
	Words         []string
	WeakBigrams   []string
	WordIndex     map[string][]string
	Model         TypistModel
	WordsPerRun   int
	RecentWords   []string
	RecentWordCap int
}

func main() {
	var (
		seed           = flag.Int64("seed", 42, "random seed")
		sessions       = flag.Int("sessions", 100, "number of sessions to simulate per variant")
		wordsPerRun    = flag.Int("words", 120, "words typed in each simulated session")
		outputCSV      = flag.String("csv", "", "optional path for CSV summary output")
		weakList       = flag.String("weak", "th,he,ng,mb,qu", "comma-separated synthetic weak bigrams")
		pauseRate      = flag.Float64("pause-rate", 0.01, "chance that a timing sample is a pause/outlier")
		weakMissRate   = flag.Float64("weak-miss", 0.08, "miss probability for known weak bigrams")
		weakExtraMS    = flag.Float64("weak-extra-ms", 95, "extra mean timing delay for known weak bigrams")
		showTopBigrams = flag.Int("top", 10, "number of top bigrams to print per variant")
		runScenarios   = flag.Bool("scenarios", true, "run deterministic scenario checks before aggregate simulation")
		runSearch      = flag.Bool("search", true, "search a constrained parameter grid and print the best configurations")
		searchTopN     = flag.Int("search-top", 8, "number of optimisation candidates to print")
	)
	flag.Parse()

	weakBigrams := parseWeakBigrams(*weakList)
	words := defaultWords()
	variants := defaultVariants()
	recommended := variants[3]

	if *runScenarios {
		printScenarioResults(runScenarioChecks(recommended))
		fmt.Println()
	}

	results := make([]Result, 0, len(variants))
	for _, variant := range variants {
		result := runVariant(variant, words, weakBigrams, *sessions, *wordsPerRun, *seed, TypistModel{
			BaseMissRate: 0.006,
			WeakMissRate: *weakMissRate,
			BaseMeanMS:   145,
			BaseStdMS:    35,
			WeakExtraMS:  *weakExtraMS,
			PauseRate:    *pauseRate,
			PauseMeanMS:  1800,
			WeakBigrams:  toSet(weakBigrams),
		})
		results = append(results, result)
	}

	printResults(results, *showTopBigrams)

	if *runSearch {
		fmt.Println()
		searchResults := runParameterSearch(words, weakBigrams, *sessions, *wordsPerRun, *seed, TypistModel{
			BaseMissRate: 0.006,
			WeakMissRate: *weakMissRate,
			BaseMeanMS:   145,
			BaseStdMS:    35,
			WeakExtraMS:  *weakExtraMS,
			PauseRate:    *pauseRate,
			PauseMeanMS:  1800,
			WeakBigrams:  toSet(weakBigrams),
		})
		printSearchResults(searchResults, *searchTopN)
	}

	if *outputCSV != "" {
		if err := writeCSV(*outputCSV, results); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write CSV: %v\n", err)
			os.Exit(1)
		}
	}
}

func runParameterSearch(words []string, weakBigrams []string, sessions, wordsPerRun int, seed int64, model TypistModel) []Result {
	var results []Result

	missWeights := []float64{0.45, 0.55, 0.60, 0.65, 0.70}
	confidenceAttempts := []int{5, 10, 15, 20}
	maxTimingValues := []float64{750, 900, 1200, 1500}
	explorationRates := []float64{0.05, 0.10, 0.15, 0.20, 0.30}

	for _, missWeight := range missWeights {
		for _, confidence := range confidenceAttempts {
			for _, maxTiming := range maxTimingValues {
				for _, exploration := range explorationRates {
					variant := Variant{
						Name: fmt.Sprintf(
							"grid-m%.2f-t%.2f-c%d-max%.0f-x%.2f",
							missWeight,
							1-missWeight,
							confidence,
							maxTiming,
							exploration,
						),
						MissWeight:         missWeight,
						TimingWeight:       1 - missWeight,
						ConfidenceAttempts: confidence,
						TimingBaselineMS:   180,
						MinTimingMS:        20,
						MaxTimingMS:        maxTiming,
						ExplorationRate:    exploration,
						Selector:           SelectorIndexed,
					}
					result := runVariant(variant, words, weakBigrams, sessions, wordsPerRun, seed, model)
					results = append(results, result)
				}
			}
		}
	}

	slices.SortFunc(results, func(a, b Result) int {
		if b.Objective != a.Objective {
			return cmp.Compare(b.Objective, a.Objective)
		}
		if b.RecallAt10 != a.RecallAt10 {
			return cmp.Compare(b.RecallAt10, a.RecallAt10)
		}
		if a.MeanWeakRank != b.MeanWeakRank {
			return cmp.Compare(a.MeanWeakRank, b.MeanWeakRank)
		}
		return cmp.Compare(a.RepeatWordRatio, b.RepeatWordRatio)
	})

	return results
}

func runScenarioChecks(variant Variant) []ScenarioResult {
	// These are human-readable checks for the CLI output. The real test
	// assertions live in main_test.go so failures are machine-readable.
	return []ScenarioResult{
		scenarioEqualHistory(variant),
		scenarioOneMissyBigram(variant),
		scenarioOneSlowBigram(variant),
		scenarioTimingOutlierIgnored(variant),
		scenarioGreedySelectionCanGetStuck(),
	}
}

func scenarioEqualHistory(variant Variant) ScenarioResult {
	stats := map[string]*Stats{}
	// GIVEN several bigrams with exactly the same typing history.
	for _, bg := range []string{"th", "he", "ng", "mb"} {
		for range 20 {
			updateStats(stats, bg, true, 145, variant)
		}
	}
	// WHEN the scorer ranks them.
	ranked := rankStats(stats)
	// THEN none of them should be meaningfully promoted over the others.
	passed := len(ranked) == 4 && almostEqual(ranked[0].Score, ranked[3].Score, 0.000001)
	return ScenarioResult{
		Name:        "equal history",
		Expected:    "no bigram should materially outrank the others",
		Actual:      fmt.Sprintf("top=%s bottom=%s scoreDelta=%.6f", ranked[0].Bigram, ranked[3].Bigram, ranked[0].Score-ranked[3].Score),
		Passed:      passed,
		Observation: "This is a sanity check: identical histories should not create an artificial weak pair.",
		TopBigrams:  ranked,
	}
}

func scenarioOneMissyBigram(variant Variant) ScenarioResult {
	stats := map[string]*Stats{}
	// GIVEN several bigrams with the same timing, but one pair has mistakes.
	for _, bg := range []string{"th", "he", "ng", "mb"} {
		for i := range 30 {
			correct := !(bg == "ng" && i < 5)
			updateStats(stats, bg, correct, 145, variant)
		}
	}
	// WHEN the scorer ranks them.
	ranked := rankStats(stats)
	return ScenarioResult{
		Name:        "one missy bigram",
		Expected:    "ng should move to the top",
		Actual:      fmt.Sprintf("top=%s", ranked[0].Bigram),
		Passed:      ranked[0].Bigram == "ng",
		Observation: "Mistakes still need to matter, otherwise the mode ignores the most obvious weak spots.",
		TopBigrams:  ranked,
	}
}

func scenarioOneSlowBigram(variant Variant) ScenarioResult {
	stats := map[string]*Stats{}
	// GIVEN several bigrams typed correctly, but one pair is consistently slow.
	for _, bg := range []string{"th", "he", "ng", "mb"} {
		for range 30 {
			spacing := 145.0
			if bg == "mb" {
				spacing = 260
			}
			updateStats(stats, bg, true, spacing, variant)
		}
	}
	// WHEN the scorer ranks them.
	ranked := rankStats(stats)
	return ScenarioResult{
		Name:        "one slow but correct bigram",
		Expected:    "mb should move to the top",
		Actual:      fmt.Sprintf("top=%s", ranked[0].Bigram),
		Passed:      ranked[0].Bigram == "mb",
		Observation: "This is the main difference from a pure mistake tracker: correct-but-slow pairs should still be visible.",
		TopBigrams:  ranked,
	}
}

func scenarioTimingOutlierIgnored(variant Variant) ScenarioResult {
	stats := map[string]*Stats{}
	// GIVEN two normal bigrams with a stable history.
	for _, bg := range []string{"th", "he"} {
		for range 30 {
			updateStats(stats, bg, true, 145, variant)
		}
	}
	before := stats["he"].Score
	// WHEN one timing sample is far above the configured timing cap.
	updateStats(stats, "he", true, variant.MaxTimingMS+500, variant)
	after := stats["he"].Score
	// THEN it should be treated as a pause/outlier, not as typing weakness.
	return ScenarioResult{
		Name:        "timing outlier",
		Expected:    "a single pause above the timing cap should not change the score",
		Actual:      fmt.Sprintf("before=%.6f after=%.6f", before, after),
		Passed:      almostEqual(before, after, 0.000001),
		Observation: "This guards against a pause or distraction turning a normal bigram into a long-term target.",
		TopBigrams:  rankStats(stats),
	}
}

func scenarioGreedySelectionCanGetStuck() ScenarioResult {
	words := defaultWords()
	weakBigrams := []string{"th", "he", "ng", "mb", "qu"}
	// GIVEN a synthetic typist with several known weak bigrams.
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
	greedy := defaultVariants()[2]
	exploring := defaultVariants()[3]
	// WHEN indexed selection is run with and without random exploration.
	greedyResult := runVariant(greedy, words, weakBigrams, 100, 120, 42, model)
	exploreResult := runVariant(exploring, words, weakBigrams, 100, 120, 42, model)
	// THEN the exploratory variant should be less likely to starve undiscovered weak pairs.
	return ScenarioResult{
		Name:        "greedy indexed selection",
		Expected:    "exploration should recover more known weak bigrams than greedy indexing",
		Actual:      fmt.Sprintf("greedy recall@10=%.2f explore recall@10=%.2f", greedyResult.RecallAt10, exploreResult.RecallAt10),
		Passed:      exploreResult.RecallAt10 > greedyResult.RecallAt10,
		Observation: "In this seed, greedy indexed selection over-focuses on early discoveries. Exploration keeps the selector from starving unseen weak pairs.",
		TopBigrams:  exploreResult.TopBigrams,
	}
}

func defaultVariants() []Variant {
	return []Variant{
		{
			Name:               "miss-heavy-random",
			MissWeight:         0.80,
			TimingWeight:       0.20,
			ConfidenceAttempts: 10,
			TimingBaselineMS:   180,
			MinTimingMS:        20,
			MaxTimingMS:        1500,
			ExplorationRate:    1,
			Selector:           SelectorRandom,
		},
		{
			Name:               "balanced-sample",
			MissWeight:         0.70,
			TimingWeight:       0.30,
			ConfidenceAttempts: 10,
			TimingBaselineMS:   180,
			MinTimingMS:        20,
			MaxTimingMS:        1500,
			ExplorationRate:    0.20,
			Selector:           SelectorSample,
		},
		{
			Name:               "timing-forward-indexed-greedy",
			MissWeight:         0.55,
			TimingWeight:       0.45,
			ConfidenceAttempts: 10,
			TimingBaselineMS:   180,
			MinTimingMS:        20,
			MaxTimingMS:        1500,
			ExplorationRate:    0,
			Selector:           SelectorIndexed,
		},
		{
			Name:               "timing-forward-indexed-explore",
			MissWeight:         0.55,
			TimingWeight:       0.45,
			ConfidenceAttempts: 10,
			TimingBaselineMS:   180,
			MinTimingMS:        20,
			MaxTimingMS:        1500,
			ExplorationRate:    0.15,
			Selector:           SelectorIndexed,
		},
		{
			Name:               "strict-outlier-indexed-explore",
			MissWeight:         0.60,
			TimingWeight:       0.40,
			ConfidenceAttempts: 15,
			TimingBaselineMS:   180,
			MinTimingMS:        20,
			MaxTimingMS:        900,
			ExplorationRate:    0.15,
			Selector:           SelectorIndexed,
		},
		{
			Name:               "fast-confidence-indexed-explore",
			MissWeight:         0.60,
			TimingWeight:       0.40,
			ConfidenceAttempts: 5,
			TimingBaselineMS:   180,
			MinTimingMS:        20,
			MaxTimingMS:        1500,
			ExplorationRate:    0.15,
			Selector:           SelectorIndexed,
		},
	}
}

func runVariant(variant Variant, words []string, weakBigrams []string, sessions, wordsPerRun int, seed int64, model TypistModel) Result {
	stats := map[string]*Stats{}
	selectedWords := []string{}
	rng := rand.New(rand.NewSource(seed))
	sim := Simulator{
		Rng:           rng,
		Words:         words,
		WeakBigrams:   weakBigrams,
		WordIndex:     buildWordIndex(words),
		Model:         model,
		WordsPerRun:   wordsPerRun,
		RecentWordCap: 25,
	}

	for range sessions {
		sim.RecentWords = sim.RecentWords[:0]
		for range wordsPerRun {
			word := sim.selectWord(variant, stats)
			selectedWords = append(selectedWords, word)
			sim.typeWord(word, stats, variant)
		}
	}

	ranked := rankStats(stats)
	result := evaluate(variant.Name, ranked, weakBigrams, selectedWords)
	result.Parameters = variant
	return result
}

func (s *Simulator) selectWord(variant Variant, stats map[string]*Stats) string {
	switch variant.Selector {
	case SelectorRandom:
		return s.rememberWord(s.Words[s.Rng.Intn(len(s.Words))])
	case SelectorSample:
		if s.Rng.Float64() < variant.ExplorationRate {
			return s.rememberWord(s.Words[s.Rng.Intn(len(s.Words))])
		}
		candidates := make([]string, 0, 40)
		for range 40 {
			candidates = append(candidates, s.Words[s.Rng.Intn(len(s.Words))])
		}
		return s.pickScoredWord(candidates, stats)
	case SelectorIndexed:
		if s.Rng.Float64() < variant.ExplorationRate {
			return s.rememberWord(s.Words[s.Rng.Intn(len(s.Words))])
		}
		ranked := rankStats(stats)
		candidateSet := map[string]bool{}
		for _, item := range ranked[:min(10, len(ranked))] {
			for _, word := range s.WordIndex[item.Bigram] {
				candidateSet[word] = true
				if len(candidateSet) >= 120 {
					break
				}
			}
			if len(candidateSet) >= 120 {
				break
			}
		}
		candidates := make([]string, 0, len(candidateSet))
		for word := range candidateSet {
			candidates = append(candidates, word)
		}
		slices.Sort(candidates)
		return s.pickScoredWord(candidates, stats)
	default:
		return s.rememberWord(s.Words[s.Rng.Intn(len(s.Words))])
	}
}

func (s *Simulator) pickScoredWord(candidates []string, stats map[string]*Stats) string {
	if len(candidates) == 0 {
		return s.rememberWord(s.Words[s.Rng.Intn(len(s.Words))])
	}

	type candidate struct {
		word  string
		score float64
	}
	scored := []candidate{}
	for _, word := range candidates {
		if contains(s.RecentWords, word) {
			continue
		}
		score := wordScore(word, stats)
		if score > 0 {
			scored = append(scored, candidate{word: word, score: score})
		}
	}

	if len(scored) == 0 {
		for _, word := range candidates {
			score := wordScore(word, stats)
			if score > 0 {
				scored = append(scored, candidate{word: word, score: score})
			}
		}
	}

	if len(scored) == 0 {
		return s.rememberWord(s.Words[s.Rng.Intn(len(s.Words))])
	}

	slices.SortFunc(scored, func(a, b candidate) int {
		if b.score != a.score {
			return cmp.Compare(b.score, a.score)
		}
		return cmp.Compare(a.word, b.word)
	})
	if len(scored) > 5 {
		scored = scored[:5]
	}

	total := 0.0
	for _, candidate := range scored {
		total += candidate.score
	}
	target := s.Rng.Float64() * total
	for _, candidate := range scored {
		target -= candidate.score
		if target <= 0 {
			return s.rememberWord(candidate.word)
		}
	}
	return s.rememberWord(scored[len(scored)-1].word)
}

func (s *Simulator) rememberWord(word string) string {
	s.RecentWords = append(s.RecentWords, word)
	if len(s.RecentWords) > s.RecentWordCap {
		s.RecentWords = s.RecentWords[1:]
	}
	return word
}

func (s *Simulator) typeWord(word string, stats map[string]*Stats, variant Variant) {
	for _, bigram := range bigrams(word) {
		isWeak := s.Model.WeakBigrams[bigram]
		missRate := s.Model.BaseMissRate
		mean := s.Model.BaseMeanMS
		if isWeak {
			missRate = s.Model.WeakMissRate
			mean += s.Model.WeakExtraMS
		}

		missed := s.Rng.Float64() < missRate
		spacing := math.Max(1, s.Rng.NormFloat64()*s.Model.BaseStdMS+mean)
		if s.Rng.Float64() < s.Model.PauseRate {
			spacing += s.Model.PauseMeanMS
		}

		updateStats(stats, bigram, !missed, spacing, variant)
	}
}

func updateStats(all map[string]*Stats, bigram string, correct bool, spacing float64, variant Variant) {
	if spacing < variant.MinTimingMS || spacing > variant.MaxTimingMS {
		return
	}

	stats := all[bigram]
	if stats == nil {
		stats = &Stats{}
		all[bigram] = stats
	}

	previousAttempts := stats.Attempts
	stats.Attempts++
	if !correct {
		stats.Misses++
	}
	stats.AverageMS = movingAverage(stats.AverageMS, spacing, previousAttempts, 50)
	stats.Score = score(*stats, variant)
}

func movingAverage(current, next float64, previousAttempts, window int) float64 {
	if previousAttempts == 0 || current == 0 {
		return next
	}
	count := min(previousAttempts+1, window)
	rate := 1 / float64(count)
	return next*rate + current*(1-rate)
}

func score(stats Stats, variant Variant) float64 {
	if stats.Attempts == 0 {
		return 0
	}
	missRate := float64(stats.Misses) / float64(stats.Attempts)
	timingPenalty := clamp((stats.AverageMS-variant.TimingBaselineMS)/variant.TimingBaselineMS, 0, 2)
	confidence := min(1, float64(stats.Attempts)/float64(variant.ConfidenceAttempts))
	return confidence * (missRate*variant.MissWeight + timingPenalty*variant.TimingWeight)
}

func wordScore(word string, stats map[string]*Stats) float64 {
	scores := []float64{}
	for _, bigram := range bigrams(word) {
		if stat := stats[bigram]; stat != nil && stat.Score > 0 {
			scores = append(scores, stat.Score)
		}
	}
	slices.SortFunc(scores, func(a, b float64) int {
		return cmp.Compare(b, a)
	})
	total := 0.0
	for _, score := range scores[:min(3, len(scores))] {
		total += score
	}
	return total
}

func evaluate(name string, ranked []RankedBigram, weakBigrams, selectedWords []string) Result {
	weak := toSet(weakBigrams)
	topN := ranked[:min(10, len(ranked))]
	found := 0
	falsePositives := 0
	topScore := 0.0
	for _, item := range topN {
		topScore += item.Score
		if weak[item.Bigram] {
			found++
		} else {
			falsePositives++
		}
	}

	rankTotal := 0
	for _, target := range weakBigrams {
		rank := len(ranked) + 1
		for index, item := range ranked {
			if item.Bigram == target {
				rank = index + 1
				break
			}
		}
		rankTotal += rank
	}

	uniqueWords := map[string]bool{}
	repeats := 0
	for index, word := range selectedWords {
		uniqueWords[word] = true
		if index > 0 && selectedWords[index-1] == word {
			repeats++
		}
	}

	return Result{
		Variant:          name,
		Objective:        objectiveScore(found, falsePositives, rankTotal, repeats, len(weakBigrams), len(selectedWords)),
		RecallAt10:       float64(found) / float64(len(weakBigrams)),
		FalsePositive10:  falsePositives,
		MeanWeakRank:     float64(rankTotal) / float64(len(weakBigrams)),
		MeanTopScore:     topScore / float64(max(1, len(topN))),
		UniqueWordRatio:  float64(len(uniqueWords)) / float64(max(1, len(selectedWords))),
		RepeatWordRatio:  float64(repeats) / float64(max(1, len(selectedWords)-1)),
		TopBigrams:       topN,
		SelectedWordRuns: len(selectedWords),
	}
}

func objectiveScore(found, falsePositives, rankTotal, repeats, weakCount, selectedWordCount int) float64 {
	recall := float64(found) / float64(max(1, weakCount))
	meanRank := float64(rankTotal) / float64(max(1, weakCount))
	repeatRatio := float64(repeats) / float64(max(1, selectedWordCount-1))

	// This objective is intentionally simple. It mostly rewards finding the known
	// weak bigrams, then nudges the search away from noisy top-10 lists, poor
	// weak-bigram ranks, and repetitive word streams.
	return recall*100 -
		float64(falsePositives)*1.5 -
		meanRank*0.25 -
		repeatRatio*20
}

func rankStats(stats map[string]*Stats) []RankedBigram {
	ranked := make([]RankedBigram, 0, len(stats))
	for bigram, stat := range stats {
		ranked = append(ranked, RankedBigram{Bigram: bigram, Stats: *stat})
	}
	slices.SortFunc(ranked, func(a, b RankedBigram) int {
		if b.Score != a.Score {
			return cmp.Compare(b.Score, a.Score)
		}
		return cmp.Compare(b.Attempts, a.Attempts)
	})
	return ranked
}

func bigrams(word string) []string {
	word = strings.ToLower(word)
	out := []string{}
	for i := 0; i < len(word)-1; i++ {
		bg := word[i : i+2]
		if bg[0] >= 'a' && bg[0] <= 'z' && bg[1] >= 'a' && bg[1] <= 'z' {
			out = append(out, bg)
		}
	}
	return out
}

func buildWordIndex(words []string) map[string][]string {
	index := map[string][]string{}
	for _, word := range words {
		seen := map[string]bool{}
		for _, bg := range bigrams(word) {
			if seen[bg] {
				continue
			}
			seen[bg] = true
			index[bg] = append(index[bg], word)
		}
	}
	return index
}

func parseWeakBigrams(input string) []string {
	parts := strings.Split(input, ",")
	out := []string{}
	for _, part := range parts {
		part = strings.ToLower(strings.TrimSpace(part))
		if len(part) == 2 {
			out = append(out, part)
		}
	}
	if len(out) == 0 {
		return []string{"th", "he", "ng", "mb", "qu"}
	}
	return out
}

func toSet(items []string) map[string]bool {
	set := map[string]bool{}
	for _, item := range items {
		set[item] = true
	}
	return set
}

func contains(items []string, target string) bool {
	for _, item := range items {
		if item == target {
			return true
		}
	}
	return false
}

func almostEqual(a, b, tolerance float64) bool {
	return math.Abs(a-b) <= tolerance
}

func clamp(value, low, high float64) float64 {
	return math.Max(low, math.Min(high, value))
}

func printScenarioResults(results []ScenarioResult) {
	fmt.Println("Scenario checks")
	fmt.Println()
	fmt.Printf("%-34s %-7s %-54s %s\n", "scenario", "pass", "expected", "actual")
	for _, result := range results {
		pass := "no"
		if result.Passed {
			pass = "yes"
		}
		fmt.Printf("%-34s %-7s %-54s %s\n", result.Name, pass, result.Expected, result.Actual)
		fmt.Printf("  %s\n", result.Observation)
	}
}

func printResults(results []Result, topN int) {
	fmt.Println("Bigram Crunch scoring simulation")
	fmt.Println()
	fmt.Printf("%-34s %10s %10s %10s %12s %12s %12s %12s\n", "variant", "objective", "recall@10", "false+10", "meanRank", "meanScore", "uniqueWords", "repeatWords")
	for _, result := range results {
		fmt.Printf("%-34s %10.2f %10.2f %10d %12.2f %12.4f %12.2f %12.4f\n",
			result.Variant,
			result.Objective,
			result.RecallAt10,
			result.FalsePositive10,
			result.MeanWeakRank,
			result.MeanTopScore,
			result.UniqueWordRatio,
			result.RepeatWordRatio,
		)
	}

	fmt.Println()
	for _, result := range results {
		fmt.Printf("%s top bigrams:\n", result.Variant)
		for _, item := range result.TopBigrams[:min(topN, len(result.TopBigrams))] {
			fmt.Printf("  %-2s score=%0.4f attempts=%d misses=%d avgMs=%0.1f\n",
				item.Bigram, item.Score, item.Attempts, item.Misses, item.AverageMS)
		}
		fmt.Println()
	}
}

func printSearchResults(results []Result, topN int) {
	if len(results) == 0 {
		return
	}

	fmt.Println("Parameter search")
	fmt.Println()
	fmt.Println("The search is a constrained grid over indexed-selection variants. The objective rewards recall first, then penalises false positives, weak-bigram rank, and repeated adjacent words.")
	fmt.Println()
	fmt.Printf("%-34s %10s %10s %10s %12s %12s %12s\n", "candidate", "objective", "recall@10", "false+10", "meanRank", "uniqueWords", "repeatWords")
	for _, result := range results[:min(topN, len(results))] {
		fmt.Printf("%-34s %10.2f %10.2f %10d %12.2f %12.2f %12.4f\n",
			result.Variant,
			result.Objective,
			result.RecallAt10,
			result.FalsePositive10,
			result.MeanWeakRank,
			result.UniqueWordRatio,
			result.RepeatWordRatio,
		)
	}

	best := results[0]
	fmt.Println()
	fmt.Println("Best candidate:")
	fmt.Printf("  %s\n", best.Variant)
	fmt.Printf("  objective=%.2f recall@10=%.2f false+10=%d meanRank=%.2f repeatWords=%.4f\n",
		best.Objective,
		best.RecallAt10,
		best.FalsePositive10,
		best.MeanWeakRank,
		best.RepeatWordRatio,
	)
	fmt.Println()
	fmt.Println("Suggested values to copy into Monkeytype:")
	fmt.Printf("  missRateWeight = %.2f\n", best.Parameters.MissWeight)
	fmt.Printf("  timingWeight = %.2f\n", best.Parameters.TimingWeight)
	fmt.Printf("  confidenceAttempts = %d\n", best.Parameters.ConfidenceAttempts)
	fmt.Printf("  maxTimingMs = %.0f\n", best.Parameters.MaxTimingMS)
	fmt.Printf("  explorationRate = %.2f\n", best.Parameters.ExplorationRate)
}

func writeCSV(path string, results []Result) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	if err := writer.Write([]string{
		"variant",
		"objective",
		"recall_at_10",
		"false_positive_10",
		"mean_weak_rank",
		"mean_top_score",
		"unique_word_ratio",
		"repeat_word_ratio",
	}); err != nil {
		return err
	}
	for _, result := range results {
		if err := writer.Write([]string{
			result.Variant,
			fmt.Sprintf("%.6f", result.Objective),
			fmt.Sprintf("%.4f", result.RecallAt10),
			fmt.Sprintf("%d", result.FalsePositive10),
			fmt.Sprintf("%.4f", result.MeanWeakRank),
			fmt.Sprintf("%.6f", result.MeanTopScore),
			fmt.Sprintf("%.6f", result.UniqueWordRatio),
			fmt.Sprintf("%.6f", result.RepeatWordRatio),
		}); err != nil {
			return err
		}
	}
	return nil
}

func defaultWords() []string {
	return []string{
		"about", "above", "after", "again", "along", "among", "angle", "answer", "another", "around",
		"because", "before", "begin", "being", "below", "between", "better", "beyond", "branch", "bring",
		"called", "cannot", "change", "charge", "check", "choose", "church", "clear", "close", "could",
		"danger", "degree", "different", "during", "early", "earth", "either", "enough", "every", "example",
		"family", "father", "figure", "final", "follow", "friend", "front", "general", "given", "going",
		"great", "group", "hand", "happen", "having", "heard", "heart", "heavy", "house", "human",
		"important", "include", "inside", "instead", "language", "large", "later", "learn", "letter", "little",
		"machine", "matter", "maybe", "member", "method", "middle", "might", "minute", "money", "mother",
		"number", "often", "other", "paper", "people", "person", "place", "point", "possible", "present",
		"question", "quick", "quiet", "quite", "rather", "reason", "record", "right", "round", "school",
		"second", "should", "simple", "small", "something", "sound", "state", "still", "strong", "system",
		"table", "taken", "their", "there", "thing", "think", "those", "though", "three", "through",
		"together", "toward", "under", "until", "using", "water", "where", "which", "while", "without",
		"world", "would", "write", "young", "zambia", "bemba", "ubuntu", "mumba", "chomba", "kalumba",
		"ngoma", "ngulu", "thing", "bring", "string", "strong", "queen", "quote", "query", "quite",
	}
}
