# Attribution of classifier errors in the GaiaDR3
## Project Abstract
In this project, we create a machine learning classifier that predicts stellar luminosity from the Gaia Data Release 3 photometry and astrometry. This is done by using spectroscopic GSP-Spec surface gravities as independent training labels. A volume-limited sample of about 524,917 stars within 200pc was constructed along with a gradient-boosted model trained on color, absolute magnitude, and tangential velocity. Using HEALPix-blocked spatial cross-validation, the model achieved a macro F1 of 0.983 for dwarf-evolved distinction, with well-calibrated probabilities. It recovered the Hyades red clump and produced no in-domain false domains in the Pleiades. Because spectroscopic and photometric measurements constrain the same physical quantity, about 205 confident out-of-fold misclassifications were understood as diagnostics of measurement interference. Each was matched against five color-magnitude-matched controls. Three possible causes were identified. Unresolved binarity is suggested by the monotonic increase in error rate across RUWE bins not included in training {0.11% to 0.99%; odds ratio 9.3, p = 4.5 × 10⁻¹⁷}. Metal-poor subdwarfs with (M/H) 0.215 dex lower than their controls (p = 5×10⁻³). Lastly, it could be due to a labeling threshold artifact, which was shown by luminosity excesses beyond the 0.753 mag maximum attainable by an unresolved equal-mass binary. The training labels ranged from 3563- 7604 K

## Introduction
The Gaia DR3 survey measures every star through two independent chains. The astrometric and photometric chains give us position, parallax, and proper motion. The Radial Velocity Spectrometer provides the spectrum, which lets us build a GSP-Spec pipeline that derives effective temperature, surface gravity, and metallicity without the astrometric and photometric chains.

They follow the same laws of physics in both chains. Absolute magnitude at a given color fixes the star's radius. Logically we could test if a classifier trained *solely on the photometric one should be able to predict the spectrometric reading*. The objects we are testing for are the disagreements, ie those the classifier gets confidently wrong.

For example, a star sitting where a dwarf is supposed to be, but the spectrum contradicts this by stating it is evolved, raises two physically possible explanations. It could be an unresolved binary that contributes the light of two stars from a point source and sits above the main sequence, or a metal-poor star with a less opaque atmosphere that is bluer and fainter.

This project aims to determine which of these explanations accounts for the errors in the star classification and to test them through a controlled experiment.

## Methods
A volume limited sample was taken within 200pc such ϖ > 5 mas. This required parallax_over_error >, ruwe < 1.4, and a BP/RP excess factor of (1.0, 1.5). Of 1,241,209 sources, 524,914 were retrieved.

We chose a volume-limited sample rather than a magnitude-limited sample for the following reasons. A sphere of 200 pc is already a complete sample, so it does not require a Malmquist-type correction. Additionally, 200pc is within the local dust-free bubble, so extinction correction is negligible compared to other systematics

Subsampling was done via sky bias. Gaia's source_id encodes HEALPix position in its upper parts, so an unordered TOP N returns a contiguous patch of sky. This biases kinematics and the giant fraction, and anything else varies with galactic latitude.

| Diagnostic Tested  | Value  | Expected  |
|--------------------|------------|-----------|
| Rows Retrieved     |524,917     |-525,000   |
| Median Fractional ϖ err| 0.5%   | <5% |
| Median Distance    |155.5pc      |158.7pc   |
|BP-RP Range         |-0.62 to 6.08|White Dwarf to Late Main|

Median distance is a crucial check here because, for a uniform-density, volume-limited sample, the expected value is about R/2^(1/3) = 158.7 pc. The 2% difference does make sense when accounting for the faintest M dwarfs at the outer edge.

Using standard error propagation.A 0.5% parallax error becomes 0.011 mag in M_G, so the unresolved binary sequence sits 0.753 mag above the main sequence. This allows for clean resolving

### Labels
Labels originate from the GSP-Spec giving use Effective Temperature (T_eff), Log g and the (M/H) from the radial velocity spectra alone (Recio-Blanco et al. 2023). As such, it is a completely different pipeline from the photometric processing, making the errors falsifiable rather than relying on circular logic

Of the 524,917 stars, 109,074 (-20.8%) have GSP-Spec parameters, and 43,175 pass all recommended quality filters. Median G is about 11.26 for labeled, compared with 16.13 for unlabelled. The parent sample is volume-limited, but the* labeled set is magnitude-limited*.

This was done based on luminosity class rather than spectral class. Spectral classes are a relabeling of effective temperature, i.e., color. This creates a circular logic loop. Luminosity class encodes solely the evolutionary state, which is not a function of the surface temperature

One point to note is that the main sequence is not quite horizontal. The initial fixed cut at log g= 4.0 produces about 18.7% subgiants, which is unlikely for a brief evolutionary phase on a cosmic scale. The Kiel diagram we made gave us a better understanding in that the surface gravity changes with mass and temperature. Any horizontal cut risks mislabeling a hot dwarf as a subgiant while missing all the evolved cool stars. The boundary was re-evaluated relative to a fitted ridge in each 150K bin; the modal log g locates the main sequence, and a cubic with those modes gives the ridge (T_eff).

The delta log-g distribution shows no visible valley. Instead, it falls monotonically from the main-sequence peak, with no separate subgiant mode. The post-calibration scatter is about 0.175 dex, which is about the physical main-sequence and subgiant separation. The only note is that the boundary is precision-limited. Subgiants and giants were then merged into an evolved class.

| Class | n  | Source  |
|------|---|-----|
| Dwarf | 37,306| Spectroscopic|
| Evolved| 3,445 | Spectroscopic|
| Ambigious|2,424 |Within Buffer Threshold |
| White Dwarf |20,867|Geometric|

Given that white dwarfs are not present on the GSP-SPec due to their faintness for the RVS method, they are labeled geometrically. As such, these are the only examples of a circular class. This will act as a pipeline check, a control and not a result

### Features
Five feautures which include: bp_rp, bp_g, g_rp, mg, log_v_tan

Tangential velocity is calculated as v_tan = 4.74047μ/ϖ. Raw proper motion is an artifact of distance. The observed median of 32.5km/s matches local thin disc dispersion

Parallax was not included. The labeled set is limited by magnitudes; such distance correlates with whether a star can even be labeled. A model with parallax only understands the selection function rather than stellar physics, which does not carry over to clusters at a fixed distance

The factors being withheld later for the attribution step include: RUWE, BP/RP excess factor, radial velocity, galactic latitude, variability flag, and M/H

### Key Thresholds
We derived adjustable constants from the data and tested their redundancy and impact on results. This took the longest to fine-tune.


| Constant  | Value  | Reasoning  |
|-----------|---|---|
| WD Boundary  | +6.13  |density min between populations/bin   |
| SUBGIANT_DELTA  | -0.35  | swept over 5vals  |
| BUFFER | 0.12  | swept over 6 values  |
| LOGG_SCATTER  |0.175 dex  |  16-84 percentile half range |
| BRIGHT_EXCESS  | 0.368   |2 standard deviations of 0.184 mag p   |
| MAX_BINARY_EXCESS  | 0.937  | 2.5 log10(2) + 1σ  |

Additionally, the WD boundary was measured. The M_G distribution is bimodal, with a main-sequence and a cooling-track component. A kernel density estimate locates the minima and a weighted fit through fourteen such minima gives a boundary shown above

The Buffer band sets how far a star must sit from the dwarf and evolved thresholds to obtain a label.

### Controlling Leakage
Only the sky blocks are being split, not stars. Objects such as wide binaries, co-moving pairs, and cluster members are identical across all features. A random split puts one candidate in train and its corresponding twin in the test, making the model 'predict' a star it has memorized. The split uses HEALPix level-3 patches from source_id, which is about 768 patches of about 54 deg^2 each. Patch overlap between train and test is asserted to be zero.

## Pipeline Explained
Nineteen modules in the source code run in a specific order. Each stage writes a parquet and JSON provenance to logs so any stage can be re-run independently.

### Foundations
**config.py**: The blueprint for all the sample cuts, constants, and holds items such as PARALLAX_MIN, RUWE_MAX, EXCESS_FACTOR_RANGE, KAPPA, the FEATURES  and DIAGNOSTICS lists that define the visibility of the model, along with Log Provenance as a digital diary of each page

**setup_verify.py**: It ensures the environment is sanitized, requiring Python>= 3.10, the nine required packages, writability of all project directories, Gaia Credentials with chmod 600 permissions, and an anonymous TAP query. Exits non-zero on failure and highlights the failure condition.

**domain.py**: Defines where the model is applicable. domain_measure() returns the 0.6-99.5 percentile bounds in color and absolute magnitude for both per class and globally; in_domain() tests candidacy; prediction_is_supported() applies the harsher test of checking a star's applicability within an assigned envelope and additionally names the failure mode

### Acquiring Data
**query_gaia.py**: This retrieves the field sample by making an ADQL WHERE clause once between count and data queries. The number reported is the sample downloaded. random_index chunks such a timeout only costs one_chunk rather than the entire run. Cache each chunk to parquet so an interrupted query can resume.

**query_labels.py**: Retrieves the GSP-SPEC parameters for the same stars. No cross-match is needed, since a server-side JOIN reproduces the stars generated in the Gaia query phase. random_index chunk cuts from gaia query is understood so the samples are the same

**query_binaries.py**: This retrieves the complement of the labels phase, same volume. However, this goes through 1.4 < RUWE < 3.0. It recovers an unresolved binary population without redownloading any stars. The upper bound of 3 keeps out excess astrometric noise that indicates companionship

### Labelling
**build_labels_mod.py**: Turns spectroscopic parameters into luminosity classes through the following method
1) logg_calibrate() applies the Recio-Blanco et al.(2023) Table 3 polynomial that corrects a documented systematic resulting from the giant branch effect
2) ridge_fit() locates the main sequence with respect to temperature. This is done via a modal log g per 150K bin across 4200- 6800 K, then a degree 3 polynomial. No extrapolation outside fitted range
3) class_assign() labels each star wrt offset from the ridge with a BUFFER band around the threshold

### Feature Construction
**build_feautures.py**: Derives the quantities not covered within Gaia and assembles the model-ready table. derive() computes M_G, v_tan, log v_tan and distance.flag_white_dwarfs() applies measured geometric cut. Spectroscopic labels take precedence where both exist

### Visualisations
**hr_plot.py**: Four Figures which include: full-sample HR diagram, labeled classes over field density, white dwarf cuts with residual distribution, main sequence zoom with binary sequence residual

### Modelling
**train_model.py**: Sky-blocked split and three modes of increasing strictness: Majority-class baseline, logistic regression on color and magnitude ONLY, LIGHTGBM on all five features. Gives a confusion matrix, feature importances, and performance with respect to color.

**calibrate.py**: Uses four-fold group-aware cross-validation with a nested calibration split. Each star receives a prediction from a model that has never observed its sky patch. This fits with respect to isotonic regression, builds a reliability diagram, and reports the Brier score and log loss for both raw and calibrated probabilities

### Validation and Error Analysis
**validate_clusters_mod.py** Distribution shift test retrieves the Hyades and Pleiades members via a cone search plus parallax and proper motion cuts. Then they are ran through the frozen calibrated model while repoprting the in domain and out-of-domain behavior separately

**extract_anomalies.py**: Isolates the confident errors and creates the control sample. Filters to spectroscopic classes, restricts to the training domaina nd keeps misclassifications above the confidence threshold; matches each star to at least five correctly classified stars of the same true class within 0.1 mag in color and 0.4 mag in M_G.

**attribute.py**:Analysis that goes through three pieces of evidence
1) Dose Response Tests: passes high RUWE complement through the unchanged model and measures the respective error rates in the RUWE bins, with the Wilson confidence itnervals and a Fisher exactness test
2) Two-sample KS tests of the anomalies against the matched controls on the metallicity, kinematics, RUWE, and the BP/RP excess factor
3) Attribution: This assigns each anomaly to a probable cause in priority order. The attributed fraction cannot be inflated by counting a star under several headings.

Additionally, five independent modules outside the pipeline justify the numbers the pipeline uses. These include:
1) **wd_cut-calibrate.py**: White dwarf boundary measured wrt density minimum
2) **audit_thresholds.py**: Hand-set constants matching wrt data present
3) **threshold_sweep.py**: Result's dependence on the SUBGIANT_DELTA or confidence cut
4) **buffer-sweep.py**: Does the result depend on the ambiguous bandwidth.
5) **ruwe-bin_check.py**: Does the result depend on the location of RUWE being placed?

## Results
### HR Diagram
Full-Sample resolves the main sequence, white dwarf cooling sequence, the red clump, the subgiant branch, M dwarf overdensity near BP-RP=3.

When the running median is subtracted from the main sequence, the residuals are strongly asymmetric: about 200 stars per bin at -1.0 mag versus 30 at +1.0 mag. Single stars scatter symmetrically about the ridge, with only unresolved binaries producing a one-sided excess. A shoulder appears at about -0.753 mag, the approximate offset for an equal-mass pair

The white dwarf gap region contains about 11 stars plotted against 20,867 candidates, which is a contamination indicator of about 0.001

### Classification
About 61,618 labeled stars with complete features are split into 46,135 training (576 patches) and 15.483 test (192 patches), with zero patch overlap.

| Model | Macro F1 (dwarf vs evolved) |
|--------|----------------------------|
| Majority class | 0.477 |
| Logistic regression, `bp_rp` + `mg` only | 0.971 |
| LightGBM |0.986 |
 
| Class | Precision | Recall | F1 | n |
|--------|----------|--------|----|---|
| dwarf | 0.997 | 0.998 | 0.997 | 9,315 |
| evolved | 0.975 | 0.973 | 0.974 | 891 |
| white dwarf| 1.000 | 1.000 | 1.000 | 5,277 |

Yields a balanced accuracy of 0.990 with an evolved recall of 0.973 at a 10:8:1 class imbalance.

LIGHTGBM can be seen beating the two-feature logistic regression by about 0.015. The Color and absolute magnitude have all the discriminating signal. Given that the second and third color indices contribute nearly nothing, a simpler model would have sufficed

When binned by color, the dwarf/evolved F1 is at 0.876 at BP-RP 0.5-1.0, rising above 0.99 redward. This bin is a main-sequence turn-off where dwarfs and subgiants tend to overlap in luminosity due to stellar structure, regardless of measurement. The errors occur in the region where class labels are ambiguous.

### RUWE Results
The model was not retrained, as it was trained on clean astrometric models and has not seen an unresolved binary. Passing this population through the unchanged model makes RUWE the only variable, making a controlled test.
|RUWE | n | errors | rate | 95% CI|
|------|---|--------|-------|-------|
|1.0–1.2 | 14,549 | 15 | 0.10% | 0.06, 0.17|
|1.2–1.4 | 3,319 | 4 | 0.12% | [0.05, 0.31|
|1.4–1.7 | 1,815 | 10 | 0.55% | [0.30, 1.01|
|1.7–2.0 | 899 | 10 | 1.11% | [0.61, 2.04|
|2.0–2.4 | 743 | 6 | 0.81% | [0.37, 1.75|
|2.4–3.0 | 805 | 16 | 1.99% | [1.23, 3.20|

Misclassification rate of spectroscopic dwarfs as evolved rose from 0.11% below RUWE 1.4 to 0.99% in the range 1.4-3, an odds ratio of 9.3. The rate increased across the range, from 0.10% to 1.99%. This apparent dip at RUWE 2.0-2.4 rests on six instances, and its interval overlaps the adjacent bin, consistent with counting noise. Wilson intervals were used given their increased sensitivity near 0.

Because the model has not seen a star with RUWE>1.4 during training, the rising error rate cannot be attributed to the training process. It would have to be monotone in RUWE across the given range. The result suggests an association with elevelated RUWE and main sequence over luminosity shown by Belokurov et Al. (2020)


### Cluster Validation
| | Hyades (700 Myr) | Pleiades (125 Myr) |
|-|------------------|--------------------|
| members retrieved | 378 | 855 |
| inside training domain | 188 (49.7%) | 360 (42.1%) |
| predicted evolved, in-domain | 14 | 28 |
| of which red (BP−RP > 0.7) | 5 | 0 |
| expected from stellar evolution | 4 | 0 |

The Hyades red clump was recovered in the data, which strongly supports this. The five red stars classified as Evolved have M_G = 0.03, 0.05, 0.08, 0.3, and 0.55, with a BP-RP range of 1.16-1.24 and G = 3.28-4.07, with p = 1.000. Four are γ, δ¹, ε, and θ¹ Tauri. Given their occupancy of a box 0.27 mag tall and 0.05 mag wide, this indicates helium ignition at a constant core mass. Conversely, a model-fitting noise distribution cannot produce that level of concentration. The Pleiades also showed zero red false positives, which is theoretically sound given the cluster's young age.

The only point to note is the systematic error in the Pleiades classification. GSP-SPEC scans F, G, K only, so there are no A or B stars in the labels. The only stars that might seem blue and luminous are the giant branch, which results in 28 Pleiades B stars being classified as evolved despite being on the main sequence. Because Tree cannot extrapolate outside the training range, it assigns the next closest label.

### Understanding Misclassifications
We reported 68 confident errors to five controls. Withheld diagnostics yield null values. Given this sample size, no metallicity or kinematic signature is detected. The earlier run on an unlabelled set yielded a metal-poor signal
| Attributed cause | n | % |
|------------------|---|---|
| close to label boundary | 40 | 58.8 |
| over-luminous beyond any binary | 20 | 29.4 |
| unexplained | 6 | 8.8 |
| metal-poor subdwarf | 2 | 2.9 |

The over luminous beyond binary thresholds a star's excess above the main sequence ridge against a 0.753 mag binary ceiling. The Excess is flat at 1.9mag across every RUWE bin, including those below 1.4 where no binaries exist. It therefore measures the model's decision-boundary location, not an intrinsic property.

## Discussion
The findings suggest this is a reinterpretation of the common objection that introducing binaries causes misclassification. We address this in the model's design rather than statistically. The model is frozen before introducing the binary population by fitting on stars with RUWE < 1.4 and has no parameter informed by stars above that boundary.

Binarity increases the misclassification rate by a factor of nine. Whereas the individual misclassified stars tend to be label-boundary cases, 59% sit within 0.175 dex of the classification threshold. Binaries thus make errors more likely without identifying the specific stars that cause them.

Interestingly, comparing observed luminosity excess with the hard physical companion showed that the boundary population would have been attributed to binarity and the compounding effect binaries have. Given an equal-mass pair is twice as luminous, 0.753 mag is an upper bound such that no unresolved binary can exceed

An error to be noted in this process was that RUWE was limited to RUWE < 1.4 to ensure the cleanliness of astrometric solutions. RUWE elevation occurs when an unresolved companion perturbs the photocentre. As such, the cut removed the entire population set the analysis aimed to test. The first attribute resulted in a RUWE ratio of 0.99 between anomalies and control, which is not a null result, but the relevant stars were clearly not present. No re-querying was needed; we only corrected the complement. A 'null' result can indicate the cut was too drastic and should be revised further along

Half of each validation cluster falls outside the training domain, and the model behaves systematically there, given how tree ensembles assign the nearest leaf. This shows the classifier can be trusted and traces the mistake to a property of the label source, i.e., a systematic error. Once the red edge half of the failure had disappeared once the white dwarf boundary was measured from the data, it demonstrates the derivation of threshold-change outcomes rather than being solely grounded in rigor

## Conclusions
Individual misclassified stars largely stem from label-boundary cases, with delta log g within the measurement scatter of the threshold. Binarity governs the rate rather than the identity of the errors. The classifier transfers well to clusters it has not seen, such as Hyades and Pleiades, notably recovering the Hyades red clump along with zero red false positives against the Pleiades

Outside its training domain, most errors are systematic due to the aforementioned tree-ensemble behavior linked to label-source (GSP-SPEC) bias in temperature coverage. This failure was eliminated by considering the white dwarf boundary of the data.This sets the precedent that every classification threshold is derived or tested in some form, with respective sensitivities included. However, there is no solid metallicity or kinematic signature that is detected amongst the anomalies in n=68, perhaps suggesting a pathway to more comprehensive tests

## Recreating this Project
### Requirements
We need Python>= 3.10, which can be done with help from the requirements.txt file
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
### Gaia Credentials
A free ESA Gaia archive account allows for higher query limits. Storing account details outside the repository is ideal.
```bash
printf 'USERNAME\nPASSWORD\n' > ~/.gaia_credentials
chmod 600 ~/.gaia_credentials
```
### Run Order
 
```bash
python src/query_gaia.py            # field sample, takes time
python src/query_labels.py          # GSP-Spec labels, takes more time!
python src/build_labels_mod.py      # luminosity classes
python src/build_features.py        # model-ready table
python src/hr_plot.py               # HR diagrams (4)
python src/train_model.py           # baselines +LightGBM
python src/calibrate.py             # isotonic +out-of-fold
python src/query_binaries.py        # high-RUWE complement  
python src/validate_clusters_mod.py # Hyades + Pleiades Validation
python src/extract_anomalies.py     # confident errors +controls
python src/attribute.py             # dose-response +attribution
```
 
Sensitivity checks, run separately:
 
```bash
python src/audit_thresholds.py      # constants v measured values
python src/wd_cut_calibrate.py      # white dwarf boundary
python src/ruwe_bin_check.py        # RUWE split sensitivity
python src/buffer_sweep.py 0.12     # one BUFFER value per invocation
python src/threshold_sweep.py       # SUBGIANT_DELTA x confidence 
```
 Note the query stages cache every chunk so that an interrupted download resumes rather than restarting. The query stages HAVE to populate /data

## Data and Software Acknowledgments
This work used data acquired by the European Space Agency (ESA) mission Gaia and processed by the Gaia Data Processing and Analysis consortium.

Software: astropy,astroquery, scikit-learn, pandas, numpy, scipy, matplotlib

## Key References
Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1 — Gaia DR3
Recio-Blanco et al. 2023, A&A 674, A29 
Belokurov et al. 2020, MNRAS 496, 1922 
Lindegren et al. 2021, A&A 649, A2 
Gentile Fusillo et al. 2021, MNRAS 508, 3877 

## License
MIT for source code; Gaia DR3 data is not here and is only available with ESA/DPAC

## Contribution
I designed the project, developed the methodology, conducted the scientific reading, made the decisions, and wrote the code. AI Assistance (Claude) helped me debug code, fix errors, and expand my coding knowledge in this format.
