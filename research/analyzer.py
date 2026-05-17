import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

SCORE_COLUMNS = ["custom", "flesch-kincaid-ease", "flesch-kincaid-level", "smog", "coleman-liau", "spache", "linsear-write", "gunning-fog", "dale-chall"]

# Feature names corresponding to SHAP value indices, matching the features dict keys in order
FEATURE_NAMES = [
    "nouns", "verbs", "adjectives", "adverbs", "pronouns",
    "first_person_pronouns", "third_person_pronouns",
    "avg_syllables_per_word", "avg_word_frequency_log", "avg_content_word_frequency_log",
    "avg_age_of_acquisition", "avg_concreteness", "avg_imagery", "avg_familiarity", "avg_polysemy",
    "negations", "causal_verbs", "intentional_actions",
    "word_count", "sentence_length", "function_to_content_ratio",
    "connectives_total", "causal_connectives", "temporal_connectives",
    "logical_connectives", "additive_connectives", "adversative_connectives",
    "dependency_depth", "modifiers_per_np", "words_before_main_verb",
    "passive_constructions",
    "content_overlap_adjacent", "content_overlap_all",
    "noun_overlap_adjacent", "argument_overlap_adjacent", "stem_overlap_all",
    "lsa_overlap_adjacent", "lsa_overlap_all", "lsa_given_new",
    "lsa_overlap_mean", "lsa_overlap_std", "lsa_overlap_max", "lsa_overlap_min",
    "lsa_novelty", "lsa_shift",
    "pos_dissimilarity_prev", "word_dissimilarity_prev", "new_word_ratio",
    "verb_overlap_adjacent",
    "embedding_norm", "embedding_norm_diff", "type_token_ratio",
    "lexical_diversity_all", "lexical_diversity_verbs",
    "sentence_length_cohesion", "sentence_length_log",
]


def _parse_shap(shap_val):
    """Parse SHAP values from string or array."""
    if isinstance(shap_val, str):
        # Try numpy fromstring first
        try:
            return np.fromstring(shap_val.strip('[]'), sep='\n').reshape(-1)
        except:
            # Fallback: split by spaces
            cleaned = shap_val.strip('[]').replace('\n', ' ')
            values = cleaned.split()
            return np.array([float(x) for x in values])
    return np.array(shap_val).reshape(-1)


class Analyzer:
    """Comprehensive analyzer for hospital financial assistance documents."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.score_cols = SCORE_COLUMNS

    # ======================== OVERVIEW & SUMMARY STATISTICS ========================

    def summary_statistics(self):
        """Generate comprehensive summary statistics for all scores."""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS FOR ALL DOCUMENTS")
        print("="*60)
        print(f"Total documents analyzed: {len(self.df)}")
        print(f"Total hospitals: {self.df['name'].nunique()}")
        print(f"\nScore Statistics:")
        print(self.df[self.score_cols].describe())
        return self.df[self.score_cols].describe()

    def hospital_document_count(self):
        """Count and display documents per hospital."""
        print("\n" + "="*60)
        print("DOCUMENT COUNT BY HOSPITAL")
        print("="*60)
        counts = self.df.groupby('name').size().sort_values(ascending=False)
        for hospital, count in counts.items():
            print(f"{hospital}: {count} documents")
        return counts

    # ======================== READABILITY ANALYSIS ========================

    def readability_rankings(self):
        """Rank documents by overall readability (based on custom metric)."""
        print("\n" + "="*60)
        print("READABILITY RANKINGS (Custom Metric)")
        print("="*60)
        sorted_df = self.df.sort_values('custom', ascending=False)
        for idx, row in sorted_df.head(20).iterrows():
            print(f"{row['name']:20s} | {row['title']:40s} | Custom Score: {row['custom']:.4f}")
        return sorted_df[['name', 'title', 'custom']]

    def most_readable_documents(self, score_type='custom', top=10):
        """Find the most readable documents."""
        print(f"\n" + "="*60)
        print(f"TOP {top} MOST READABLE DOCUMENTS ({score_type})")
        print("="*60)
        top_docs = self.df.nlargest(top, score_type)[['name', 'title', score_type]]
        for idx, (i, row) in enumerate(top_docs.iterrows(), 1):
            print(f"{idx}. {row['name']:20s} - {row['title']:35s} ({row[score_type]:.4f})")
        return top_docs

    def least_readable_documents(self, score_type='custom', top=10):
        """Find the least readable documents."""
        print(f"\n" + "="*60)
        print(f"TOP {top} LEAST READABLE DOCUMENTS ({score_type})")
        print("="*60)
        bottom_docs = self.df.nsmallest(top, score_type)[['name', 'title', score_type]]
        for idx, (i, row) in enumerate(bottom_docs.iterrows(), 1):
            print(f"{idx}. {row['name']:20s} - {row['title']:35s} ({row[score_type]:.4f})")
        return bottom_docs

    def hospital_average_readability(self):
        """Calculate average readability scores for each hospital."""
        print("\n" + "="*60)
        print("AVERAGE READABILITY BY HOSPITAL")
        print("="*60)
        hospital_avg = self.df.groupby('name')[self.score_cols].mean()
        print(hospital_avg)
        return hospital_avg

    def hospital_readability_comparison(self):
        """Compare hospitals on custom metric with visualization."""
        print("\n" + "="*60)
        print("HOSPITAL READABILITY COMPARISON")
        print("="*60)
        hospital_avg = self.df.groupby('name')['custom'].mean().sort_values(ascending=False)
        for hospital, score in hospital_avg.items():
            print(f"{hospital:20s}: {score:.4f}")

        plt.figure(figsize=(10, 6))
        hospital_avg.plot(kind='barh', color='steelblue')
        plt.xlabel('Average Custom Readability Score')
        plt.title('Hospital Readability Comparison')
        plt.tight_layout()
        plt.show()
        return hospital_avg

    # ======================== SCORE ANALYSIS ========================

    def plot_individual_reports_scores(self, index=-1):
        """Plot individual document scores across all metrics."""
        scores_series = self.df.loc[index, self.score_cols]
        numeric_scores = scores_series.dropna()

        X = np.arange(len(numeric_scores))
        Y = numeric_scores.to_numpy(dtype=float)
        X_labels = numeric_scores.index.to_list()

        plt.figure(figsize=(12, 6))
        plt.bar(X, Y, color="skyblue")
        plt.xticks(X, X_labels, rotation=45, ha="right")
        plt.ylabel("Score")
        plt.xlabel("Metrics")
        plt.title(f"Score Report for {self.df.loc[index, 'title']} ({self.df.loc[index, 'name']})")
        plt.tight_layout()
        plt.show()

    def score_distribution_by_hospital(self, score_type='custom'):
        """Visualize score distributions across hospitals."""
        print(f"\n" + "="*60)
        print(f"SCORE DISTRIBUTION: {score_type}")
        print("="*60)

        plt.figure(figsize=(12, 6))
        self.df.boxplot(column=score_type, by='name', figsize=(12, 6))
        plt.suptitle(f'{score_type} Distribution by Hospital')
        plt.xlabel('Hospital')
        plt.ylabel(score_type)
        plt.tight_layout()
        plt.show()

    def score_variance_analysis(self):
        """Analyze variance in scores within and across hospitals."""
        print("\n" + "="*60)
        print("SCORE VARIANCE ANALYSIS")
        print("="*60)

        print("\nOverall Variance by Score Type:")
        variances = self.df[self.score_cols].var()
        for score, var in variances.items():
            print(f"  {score:20s}: {var:.6f}")

        print("\nWithin-Hospital Variance (Std Dev):")
        hospital_var = self.df.groupby('name')[self.score_cols].std()
        print(hospital_var)
        return hospital_var

    def metric_correlation_analysis(self):
        """Analyze correlations between different readability metrics."""
        print("\n" + "="*60)
        print("METRIC CORRELATION ANALYSIS")
        print("="*60)

        corr_matrix = self.df[self.score_cols].corr()
        print(corr_matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title('Readability Metric Correlation Matrix')
        plt.tight_layout()
        plt.show()
        return corr_matrix

    # ======================== FEATURE ANALYSIS ========================

    def extract_feature_dict(self, index):
        """Extract and parse features dictionary from document."""
        try:
            features = self.df.loc[index, 'features']
            if isinstance(features, str):
                return json.loads(features.replace("'", '"'))
            return features
        except Exception:
            return {}

    def most_common_linguistic_patterns(self, feature_category='nouns', top=15):
        """Analyze most common linguistic patterns across documents."""
        print(f"\n" + "="*60)
        print(f"MOST COMMON {feature_category.upper()} PATTERNS")
        print("="*60)

        pattern_counts = Counter()
        for idx in range(len(self.df)):
            features = self.extract_feature_dict(idx)
            if feature_category in features:
                if isinstance(features[feature_category], dict):
                    for word, count in features[feature_category].items():
                        if isinstance(count, (int, float)):
                            pattern_counts[word] += count

        top_patterns = pattern_counts.most_common(top)
        for pattern, count in top_patterns:
            print(f"  {pattern:25s}: {count}")
        return top_patterns

    def feature_statistics_by_hospital(self):
        """Analyze feature statistics by hospital."""
        print("\n" + "="*60)
        print("FEATURE STATISTICS BY HOSPITAL")
        print("="*60)

        hospital_features = {}
        for hospital in self.df['name'].unique():
            hospital_df = self.df[self.df['name'] == hospital]
            hospital_features[hospital] = {
                'num_docs': len(hospital_df),
                'avg_custom_score': hospital_df['custom'].mean()
            }
            print(f"\n{hospital}:")
            print(f"  Documents: {hospital_features[hospital]['num_docs']}")
            print(f"  Avg Custom Score: {hospital_features[hospital]['avg_custom_score']:.4f}")

        return hospital_features

    # ======================== SHAP / FEATURE IMPORTANCE ========================

    def top_important_features_by_shap(self, index=-1):
        """Show top important features for predicting readability (via SHAP values)."""
        print("\n" + "="*60)
        print(f"TOP IMPORTANT FEATURES FOR: {self.df.loc[index, 'title']} ({self.df.loc[index, 'name']})")
        print(f"Custom Score: {self.df.loc[index, 'custom']:.4f}")
        print("="*60)

        shap_values = _parse_shap(self.df.loc[index, 'shap'])
        n_features = min(len(shap_values), len(FEATURE_NAMES))
        shap_values = shap_values[:n_features]
        names = FEATURE_NAMES[:n_features]

        ranked = sorted(zip(names, shap_values), key=lambda x: abs(x[1]), reverse=True)[:10]
        print(f"\n{'Feature':<35} {'SHAP Value':>12}  {'Direction'}")
        print("-" * 60)
        for feat, val in ranked:
            direction = "▲ harder" if val < 0 else "▼ easier"
            print(f"  {feat:<33} {val:>+12.6f}  {direction}")

        return shap_values

    def average_shap_importance(self):
        """Calculate average SHAP importance across all documents."""
        print("\n" + "="*60)
        print("AVERAGE FEATURE IMPORTANCE (SHAP)")
        print("="*60)

        all_shap = []
        for idx in range(len(self.df)):
            sv = _parse_shap(self.df.loc[idx, 'shap'])
            all_shap.append(np.abs(sv))

        # Trim to shortest (in case lengths differ slightly)
        min_len = min(len(s) for s in all_shap)
        all_shap = [s[:min_len] for s in all_shap]
        avg_importance = np.mean(all_shap, axis=0)

        n = min(min_len, len(FEATURE_NAMES))
        ranked = sorted(zip(FEATURE_NAMES[:n], avg_importance[:n]), key=lambda x: x[1], reverse=True)

        print(f"\n{'Rank':<6} {'Feature':<35} {'Avg |SHAP|':>12}")
        print("-" * 55)
        for i, (feat, imp) in enumerate(ranked[:20], 1):
            print(f"  {i:<4} {feat:<35} {imp:>12.6f}")

        plt.figure(figsize=(12, 7))
        top_names = [r[0] for r in ranked[:20]]
        top_vals = [r[1] for r in ranked[:20]]
        colors = ['coral'] * 20
        plt.barh(top_names[::-1], top_vals[::-1], color=colors)
        plt.xlabel('Average |SHAP Value|')
        plt.title('Top 20 Features by Average Importance Across All Documents')
        plt.tight_layout()
        plt.show()

        return avg_importance

    # ======================== KEY FIX: most_relevant_features ========================

    def most_relevant_features(self, top=20, difficulty='hard'):
        """
        Show which features drive readability difficulty (or ease).

        Parameters
        ----------
        top : int
            Number of top features to show.
        difficulty : str
            'hard'  — analyse the least readable documents (lowest custom score).
            'easy'  — analyse the most readable documents (highest custom score).
            'both'  — show hard and easy side by side.
        """
        if difficulty == 'both':
            self.most_relevant_features(top=top, difficulty='hard')
            self.most_relevant_features(top=top, difficulty='easy')
            return

        ascending = (difficulty == 'hard')  # hard → lowest scores first
        label = "LEAST READABLE (HARDEST)" if difficulty == 'hard' else "MOST READABLE (EASIEST)"
        n_sample = max(3, len(self.df) // 5)   # bottom/top 20 % of corpus

        # Select the hard or easy subset
        subset = self.df.nsmallest(n_sample, 'custom') if difficulty == 'hard' \
            else self.df.nlargest(n_sample, 'custom')

        print("\n" + "="*60)
        print(f"FEATURES DRIVING {label} DOCUMENTS")
        print(f"  Analysing {n_sample} documents  |  custom score range: "
              f"{subset['custom'].min():.3f} – {subset['custom'].max():.3f}")
        print("="*60)

        # Collect SHAP vectors for this subset
        shap_matrix = []
        for idx in subset.index:
            sv = _parse_shap(self.df.loc[idx, 'shap'])
            shap_matrix.append(sv)

        min_len = min(len(s) for s in shap_matrix)
        shap_matrix = np.array([s[:min_len] for s in shap_matrix])  # shape (n_docs, n_features)
        n = min(min_len, len(FEATURE_NAMES))
        names = FEATURE_NAMES[:n]
        shap_matrix = shap_matrix[:, :n]

        # Mean signed SHAP — negative = pushes score down (harder), positive = pushes up (easier)
        mean_signed = shap_matrix.mean(axis=0)
        mean_abs    = np.abs(shap_matrix).mean(axis=0)

        # For hard docs we care about features with the most negative contribution (making it harder)
        # For easy docs we care about features with the most positive contribution (making it easier)
        if difficulty == 'hard':
            ranked = sorted(zip(names, mean_signed, mean_abs),
                            key=lambda x: x[1])   # most negative first
        else:
            ranked = sorted(zip(names, mean_signed, mean_abs),
                            key=lambda x: -x[1])  # most positive first

        ranked = ranked[:top]

        print(f"\n{'Rank':<6} {'Feature':<35} {'Mean SHAP':>10}  {'Mean |SHAP|':>12}  {'Effect'}")
        print("-" * 75)
        for i, (feat, signed, magnitude) in enumerate(ranked, 1):
            effect = "makes HARDER ↓" if signed < 0 else "makes EASIER ↑"
            print(f"  {i:<4} {feat:<35} {signed:>+10.5f}  {magnitude:>12.6f}  {effect}")

        # Bar chart: signed mean SHAP
        feat_names   = [r[0] for r in ranked]
        signed_vals  = [r[1] for r in ranked]
        bar_colors   = ['#e05c5c' if v < 0 else '#5cb85c' for v in signed_vals]

        plt.figure(figsize=(12, 7))
        plt.barh(feat_names[::-1], signed_vals[::-1], color=bar_colors[::-1])
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xlabel('Mean SHAP Value  (negative = harder, positive = easier)')
        plt.title(f'Top {top} Features Driving {label} Documents')
        plt.tight_layout()
        plt.show()

        return ranked

    # ======================== OUTLIER & ANOMALY DETECTION ========================

    def find_outlier_documents(self, score_type='custom', std_threshold=2):
        """Find documents that are statistical outliers."""
        print(f"\n" + "="*60)
        print(f"OUTLIER DOCUMENTS ({score_type}, > {std_threshold} std devs)")
        print("="*60)

        mean = self.df[score_type].mean()
        std  = self.df[score_type].std()

        outliers = self.df[
            (self.df[score_type] < mean - std_threshold * std) |
            (self.df[score_type] > mean + std_threshold * std)
        ]

        print(f"Found {len(outliers)} outliers:")
        for idx, row in outliers.iterrows():
            z_score = (row[score_type] - mean) / std
            print(f"  {row['name']:20s} - {row['title']:35s} ({row[score_type]:.4f}, z={z_score:.2f})")

        return outliers

    def inconsistency_detection(self):
        """Find documents with inconsistent scores (high variance across metrics)."""
        print("\n" + "="*60)
        print("INCONSISTENT SCORING ANALYSIS")
        print("="*60)

        self.df['score_cv'] = self.df[self.score_cols].std(axis=1) / (self.df[self.score_cols].mean(axis=1) + 1e-8)
        inconsistent = self.df.nlargest(10, 'score_cv')[['name', 'title', 'score_cv']]

        print("Documents with most inconsistent scores across metrics:")
        for idx, (i, row) in enumerate(inconsistent.iterrows(), 1):
            print(f"  {idx}. {row['name']:20s} - {row['title']:35s} (CV: {row['score_cv']:.4f})")

        return inconsistent

    # ======================== COMPARATIVE ANALYSIS ========================

    def document_type_analysis(self):
        """Analyze readability by document type (extracted from title)."""
        print("\n" + "="*60)
        print("ANALYSIS BY DOCUMENT TYPE")
        print("="*60)

        doc_types = []
        for title in self.df['title']:
            t = title.lower()
            if 'privacy' in t:
                doc_types.append('Privacy')
            elif 'consent' in t or 'authorization' in t:
                doc_types.append('Consent')
            elif 'guidelines' in t or 'policy' in t:
                doc_types.append('Guidelines/Policy')
            elif 'rights' in t:
                doc_types.append('Patient Rights')
            elif 'terms' in t or 'conditions' in t:
                doc_types.append('Terms/Conditions')
            elif 'application' in t:
                doc_types.append('Application')
            elif 'assistance' in t:
                doc_types.append('Financial Assistance')
            else:
                doc_types.append('Other')

        self.df['doc_type'] = doc_types

        type_stats = self.df.groupby('doc_type')[self.score_cols].mean()
        print(type_stats)

        plt.figure(figsize=(12, 6))
        self.df.groupby('doc_type')['custom'].mean().sort_values(ascending=False).plot(
            kind='bar', color='lightgreen')
        plt.title('Average Readability by Document Type')
        plt.ylabel('Custom Readability Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        return type_stats

    def hospital_rank_vs_readability(self):
        """Analyze relationship between hospital rank and document readability."""
        print("\n" + "="*60)
        print("HOSPITAL RANK VS READABILITY CORRELATION")
        print("="*60)

        rank_readability = self.df.groupby('name').agg({
            'rank': 'first',
            'custom': 'mean'
        }).sort_values('rank')

        print(rank_readability)

        corr = rank_readability['rank'].corr(rank_readability['custom'])
        print(f"\nCorrelation (Rank vs Avg Readability): {corr:.4f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(rank_readability['rank'], rank_readability['custom'], s=100, alpha=0.6, color='purple')
        for idx, row in rank_readability.iterrows():
            plt.annotate(idx, (row['rank'], row['custom']), fontsize=9)
        plt.xlabel('Hospital Rank')
        plt.ylabel('Average Readability Score')
        plt.title('Hospital Rank vs Document Readability')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return rank_readability

    # ======================== FORMATTING ARTIFACT DIAGNOSIS ========================

    def diagnose_formatting_artifacts(self):
        """Flag documents with corrupted linguistic features due to parsing artifacts."""
        print("\n" + "="*60)
        print("FORMATTING/PARSING ARTIFACT DIAGNOSIS")
        print("="*60)

        flagged_docs = []
        reasons = []

        for idx, row in self.df.iterrows():
            flags = []
            
            # Parse features dict
            try:
                features_str = row['features']
                if isinstance(features_str, str):
                    features = json.loads(features_str.replace("'", '"'))
                else:
                    features = features_str
            except:
                flags.append("features_parse_error")
                flagged_docs.append(idx)
                reasons.append(flags)
                continue

            # Check word_count avg < 5
            if 'word_count' in features and features['word_count']['avg'] < 5:
                flags.append("word_count_avg_low")

            # Check avg_syllables_per_word unusual
            if 'avg_syllables_per_word' in features:
                syl_avg = features['avg_syllables_per_word']['avg']
                if syl_avg > 3.5 or syl_avg < 1:
                    flags.append("syllables_unusual")

            # Check sentence_length avg extreme
            if 'sentence_length' in features:
                sent_avg = features['sentence_length']['avg']
                if sent_avg < 3 or sent_avg > 100:
                    flags.append("sentence_length_extreme")

            # Check avg_word_frequency_log near 0
            if 'avg_word_frequency_log' in features and abs(features['avg_word_frequency_log']['avg']) < 0.01:
                flags.append("word_freq_near_zero")

            # Check sd == 0 for many features (suggests parsed as single block)
            zero_sd_count = 0
            total_features = 0
            for feat_name, stats in features.items():
                if isinstance(stats, dict) and 'sd' in stats:
                    total_features += 1
                    if stats['sd'] == 0:
                        zero_sd_count += 1
            
            if total_features > 0 and zero_sd_count / total_features > 0.5:
                flags.append("many_zero_sd")

            # Check flesch-kincaid-ease extreme
            if 'flesch-kincaid-ease' in row and (row['flesch-kincaid-ease'] < -20 or row['flesch-kincaid-ease'] > 100):
                flags.append("flesch_extreme")

            if flags:
                flagged_docs.append(idx)
                reasons.append(flags)

        print(f"Found {len(flagged_docs)} potentially corrupted documents out of {len(self.df)} total")

        if flagged_docs:
            print("\nFlagged Documents:")
            print(f"{'Index':<5} {'Title':<40} {'Hospital':<20} {'Flags'}")
            print("-" * 80)
            for i, idx in enumerate(flagged_docs):
                row = self.df.loc[idx]
                title = str(row.get('title', ''))[:39]
                hospital = str(row.get('name', ''))[:19]
                flags_str = ', '.join(reasons[i])
                print(f"{idx:<5} {title:<40} {hospital:<20} {flags_str}")

        return flagged_docs, reasons

    # ======================== LSA MULTICOLLINEARITY ANALYSIS ========================

    def analyze_lsa_multicollinearity(self):
        """Compute pairwise correlations for LSA overlap features."""
        print("\n" + "="*60)
        print("LSA FEATURE MULTICOLLINEARITY ANALYSIS")
        print("="*60)

        lsa_features = [
            'lsa_overlap_adjacent', 'lsa_overlap_all', 'lsa_overlap_mean',
            'lsa_overlap_std', 'lsa_overlap_max', 'lsa_overlap_min',
            'lsa_given_new', 'lsa_novelty', 'lsa_shift'
        ]

        # Extract avg values for each LSA feature
        lsa_data = {}
        for feat in lsa_features:
            lsa_data[feat] = []

        for idx, row in self.df.iterrows():
            try:
                features_str = row['features']
                if isinstance(features_str, str):
                    features = json.loads(features_str.replace("'", '"'))
                else:
                    features = features_str
                
                for feat in lsa_features:
                    if feat in features:
                        lsa_data[feat].append(features[feat]['avg'])
                    else:
                        lsa_data[feat].append(np.nan)
            except:
                for feat in lsa_features:
                    lsa_data[feat].append(np.nan)

        # Create correlation matrix
        df_lsa = pd.DataFrame(lsa_data)
        corr_matrix = df_lsa.corr()

        print("Pairwise Pearson correlations for LSA features (avg values):")
        print(corr_matrix.round(3))

        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('LSA Feature Correlations')
        plt.tight_layout()
        plt.show()

        return corr_matrix

    # ======================== ENHANCED MOST RELEVANT FEATURES ========================

    def most_relevant_features(self, top=20, difficulty='hard', deduplicate_groups=False):
        """
        Show which features drive readability difficulty (or ease).

        Parameters
        ----------
        top : int
            Number of top features to show.
        difficulty : str
            'hard'  — analyse the least readable documents (lowest custom score).
            'easy'  — analyse the most readable documents (highest custom score).
            'both'  — show hard and easy side by side.
        deduplicate_groups : bool
            If True, deduplicate highly correlated feature groups, keeping only
            the single feature with highest mean |SHAP| from each group.
        """
        if difficulty == 'both':
            self.most_relevant_features(top=top, difficulty='hard', deduplicate_groups=deduplicate_groups)
            self.most_relevant_features(top=top, difficulty='easy', deduplicate_groups=deduplicate_groups)
            return

        ascending = (difficulty == 'hard')  # hard → lowest scores first
        label = "LEAST READABLE (HARDEST)" if difficulty == 'hard' else "MOST READABLE (EASIEST)"
        n_sample = max(3, len(self.df) // 5)   # bottom/top 20 % of corpus

        # Select the hard or easy subset
        subset = self.df.nsmallest(n_sample, 'custom') if difficulty == 'hard' \
            else self.df.nlargest(n_sample, 'custom')

        print("\n" + "="*60)
        print(f"FEATURES DRIVING {label} DOCUMENTS")
        print(f"  Analysing {n_sample} documents  |  custom score range: "
              f"{subset['custom'].min():.3f} – {subset['custom'].max():.3f}")
        if deduplicate_groups:
            print("  Feature deduplication: ENABLED")
        print("="*60)

        # Collect SHAP vectors for this subset
        shap_matrix = []
        for idx in subset.index:
            sv = _parse_shap(self.df.loc[idx, 'shap'])
            shap_matrix.append(sv)

        min_len = min(len(s) for s in shap_matrix)
        shap_matrix = np.array([s[:min_len] for s in shap_matrix])  # shape (n_docs, n_features)
        n = min(min_len, len(FEATURE_NAMES))
        names = FEATURE_NAMES[:n]
        shap_matrix = shap_matrix[:, :n]

        # Mean signed SHAP — negative = pushes score down (harder), positive = pushes up (easier)
        mean_signed = shap_matrix.mean(axis=0)
        mean_abs    = np.abs(shap_matrix).mean(axis=0)

        # Create initial ranking
        features_with_shap = list(zip(names, mean_signed, mean_abs))

        if deduplicate_groups:
            # Define feature groups
            feature_groups = {
                'lsa_overlap': ['lsa_overlap_adjacent', 'lsa_overlap_all', 'lsa_overlap_mean', 
                               'lsa_overlap_std', 'lsa_overlap_max', 'lsa_overlap_min', 
                               'lsa_given_new', 'lsa_novelty', 'lsa_shift'],
                'lexical_overlap': ['content_overlap_adjacent', 'content_overlap_all', 
                                   'noun_overlap_adjacent', 'argument_overlap_adjacent', 
                                   'stem_overlap_all', 'verb_overlap_adjacent'],
                'pronoun_types': ['first_person_pronouns', 'third_person_pronouns'],
                'connective_types': ['causal_connectives', 'temporal_connectives', 
                                    'logical_connectives', 'additive_connectives', 'adversative_connectives']
            }

            # Deduplicate: keep only highest |SHAP| from each group
            deduplicated = []
            used_features = set()
            
            for group_name, group_features in feature_groups.items():
                group_candidates = [(name, signed, abs_val) for name, signed, abs_val in features_with_shap 
                                   if name in group_features and name not in used_features]
                if group_candidates:
                    # Keep the one with highest mean |SHAP|
                    best = max(group_candidates, key=lambda x: x[2])
                    deduplicated.append(best)
                    used_features.update([best[0]])
                    print(f"Deduplicated {group_name}: kept {best[0]} (|{best[1]:.3f}|), dropped {len(group_candidates)-1} others")
            
            # Add non-grouped features
            for name, signed, abs_val in features_with_shap:
                if name not in used_features:
                    deduplicated.append((name, signed, abs_val))
            
            features_with_shap = deduplicated

        # Sort and rank
        if difficulty == 'hard':
            ranked = sorted(features_with_shap, key=lambda x: x[1])   # most negative first
        else:
            ranked = sorted(features_with_shap, key=lambda x: -x[1])  # most positive first

        ranked = ranked[:top]

        print(f"\n{'Rank':<6} {'Feature':<35} {'Mean SHAP':>10}  {'Mean |SHAP|':>12}  {'Effect'}")
        print("-" * 75)
        for i, (feat, signed, magnitude) in enumerate(ranked, 1):
            effect = "makes HARDER ↓" if signed < 0 else "makes EASIER ↑"
            print(f"  {i:<4} {feat:<35} {signed:>+10.5f}  {magnitude:>12.6f}  {effect}")

        # Bar chart: signed mean SHAP
        feat_names   = [r[0] for r in ranked]
        signed_vals  = [r[1] for r in ranked]
        bar_colors   = ['#e05c5c' if v < 0 else '#5cb85c' for v in signed_vals]

        plt.figure(figsize=(12, 7))
        plt.barh(feat_names[::-1], signed_vals[::-1], color=bar_colors[::-1])
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xlabel('Mean SHAP Value  (negative = harder, positive = easier)')
        plt.title(f'Top {top} Features Driving {label} Documents' + 
                 (' (Deduplicated)' if deduplicate_groups else ''))
        plt.tight_layout()
        plt.show()

        return ranked

    # ======================== FEATURE SANITY CHECK ========================

    def feature_sanity_check(self, top=10, difficulty='hard'):
        """
        Compare actual feature avg values between hard vs easy documents
        for the top features from most_relevant_features.
        """
        print("\n" + "="*60)
        print("FEATURE-LEVEL SANITY CHECK")
        print(f"Comparing top {top} features between hard vs easy documents")
        print("="*60)

        # Get top features from most_relevant_features
        top_features = self.most_relevant_features(top=top, difficulty=difficulty, deduplicate_groups=True)
        feature_names = [f[0] for f in top_features]

        # Get hard and easy subsets
        n_sample = max(3, len(self.df) // 5)
        hard_subset = self.df.nsmallest(n_sample, 'custom')
        easy_subset = self.df.nlargest(n_sample, 'custom')

        print(f"Hard documents: {len(hard_subset)} (custom <= {hard_subset['custom'].max():.3f})")
        print(f"Easy documents: {len(easy_subset)} (custom >= {easy_subset['custom'].min():.3f})")

        # Extract avg values for each feature
        results = []
        
        for feat_name in feature_names:
            hard_avgs = []
            easy_avgs = []
            
            # Hard documents
            for idx in hard_subset.index:
                try:
                    features_str = self.df.loc[idx, 'features']
                    if isinstance(features_str, str):
                        features = json.loads(features_str.replace("'", '"'))
                    else:
                        features = features_str
                    
                    if feat_name in features:
                        hard_avgs.append(features[feat_name]['avg'])
                except:
                    pass
            
            # Easy documents
            for idx in easy_subset.index:
                try:
                    features_str = self.df.loc[idx, 'features']
                    if isinstance(features_str, str):
                        features = json.loads(features_str.replace("'", '"'))
                    else:
                        features = features_str
                    
                    if feat_name in features:
                        easy_avgs.append(features[feat_name]['avg'])
                except:
                    pass
            
            if hard_avgs and easy_avgs:
                hard_mean = np.mean(hard_avgs)
                easy_mean = np.mean(easy_avgs)
                diff = easy_mean - hard_mean  # positive means easy docs have higher values
                
                results.append({
                    'feature': feat_name,
                    'hard_avg': hard_mean,
                    'easy_avg': easy_mean,
                    'difference': diff
                })

        # Print comparison table
        print(f"\n{'Feature':<35} {'Hard Avg':>10} {'Easy Avg':>10} {'Diff (E-H)':>12} {'SHAP Effect'}")
        print("-" * 85)
        
        for res in results:
            feat = res['feature']
            # Find SHAP effect for this feature
            shap_info = next((f for f in top_features if f[0] == feat), None)
            if shap_info:
                shap_signed = shap_info[1]
                effect = "makes HARDER" if shap_signed < 0 else "makes EASIER"
            else:
                effect = "N/A"
            
            print(f"{res['feature']:<35} {res['hard_avg']:>10.3f} {res['easy_avg']:>10.3f} "
                  f"{res['difference']:>+12.3f} {effect}")

        # Check for mismatches
        mismatches = []
        for res in results:
            feat = res['feature']
            shap_info = next((f for f in top_features if f[0] == feat), None)
            if shap_info:
                shap_signed = shap_info[1]
                actual_diff = res['difference']
                
                # If SHAP says makes harder (negative) but hard docs have LOWER values, that's a mismatch
                if shap_signed < 0 and actual_diff > 0:
                    mismatches.append(f"{feat}: SHAP says makes harder, but hard docs have lower values")
                elif shap_signed > 0 and actual_diff < 0:
                    mismatches.append(f"{feat}: SHAP says makes easier, but easy docs have lower values")

        if mismatches:
            print(f"\n⚠️  POTENTIAL SHAP INTERPRETATION ISSUES ({len(mismatches)}):")
            for m in mismatches:
                print(f"   {m}")
        else:
            print("\n✅ All top features show directionally consistent SHAP interpretations.")

        return results

    # ======================== COMPREHENSIVE REPORTS ========================

    def generate_full_report(self):
        """Generate a comprehensive analysis report."""
        print("\n\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*78 + "║")
        print("║" + "COMPREHENSIVE RESEARCH ANALYSIS REPORT".center(78) + "║")
        print("║" + "Hospital Financial Assistance Documentation Quality Study".center(78) + "║")
        print("║" + " "*78 + "║")
        print("╚" + "="*78 + "╝")

        self.summary_statistics()
        self.hospital_document_count()
        self.hospital_readability_comparison()
        self.most_readable_documents(top=5)
        self.least_readable_documents(top=5)
        self.hospital_average_readability()
        self.metric_correlation_analysis()
        self.score_variance_analysis()
        self.find_outlier_documents()
        self.inconsistency_detection()
        self.document_type_analysis()
        self.hospital_rank_vs_readability()
        self.average_shap_importance()
        self.diagnose_formatting_artifacts()
        self.analyze_lsa_multicollinearity()
        self.most_relevant_features(top=20, difficulty='hard', deduplicate_groups=True)
        self.most_relevant_features(top=20, difficulty='easy', deduplicate_groups=True)
        self.feature_sanity_check(top=10, difficulty='hard')

        print("\n" + "="*60)
        print("REPORT GENERATION COMPLETE")
        print("="*60)