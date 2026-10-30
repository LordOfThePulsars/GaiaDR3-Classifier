# Attribution of classifier errors in the GaiaDR3
Creating a calibrated classifer that acts as a detector for photometric-spectroscopic disagreement in a volume limited sample of 524,917 stars within 200 parsecs

## Project Abstract
A gradient boosted classifer that trains only on stars that have a RUWE<1.4 . Ergo, stars with no astrometric sign of an unresolved companion, is applied to 

## Introduction
The Gaia DR3 survey measures every star through two independent chains. The astrometric and photometric chain gives us position, parallax and proper motion. The Radial Velocity Spectrometer gives the spectrum which allows us to make a GSP-Spec piepline that derrives effective temperature, surface gravity and metallicity that does not need the astrometric and photometric chains.

They follow the same laws of physics in bothc chains. Absolute magnitude at a certain colour fixes the radius of the star. Logically we could test if a classifier trained *solely on the photometric one should be able to predict the spectrometric reading*. The objects we are testing for are the disagreements, ie those the classifer gets confidently wrong.

For example a star sitting where a dwarf is supposed to be but the spectrum contradicts by stating it is evolved raises two physical possible explanations. It could be an unresolved binary that contributes the light of two stars from a point source and sits above the main seuqence; Or a metal poor star that has a less opaque atmosphere which is bluer and fainter.

What this project aims to do is understand which of these explanations actually account for the errors in the star classification and attempts a controlled experiment.

## Methods
A volume limited sample was taken within 200pc such ϖ > 5 mas. This needed a parallax_over_error >20 and ruwe <1.4 and a BP/RP excess factor of (1.0,1.5) Of 1,241,209 sources, 524,914 were retrieved.

A Volume limited sample was chosen rather than a magnitude limited sample for the following reasons. A sphere of 200pc stellar sphere already has a present mix rather than needing a Malmquist type correction. Addionally, 200pc is within the local dust free bubble such extinction correction is negligible against other systematics

Subsampling was done via sky bias. Gaia's source_id encodes HEALPix position in its upper parts so an unordered TOP N returns a contigent patch of sky. This biases kinematics and the giant fraction and anything else was varies on galactic latitude.

| Diagnostic Tested  | Value  | Expected  |
|--------------------|------------|-----------|
| Rows Retrieved     |524,917     |-525,000   |
| Median Fractional ϖ err| 0.5%   | <5% |
| Median Distance    |155.5pc      |158.7pc   |
|BP-RP Range         |-0.62 to 6.08|White Dwarf to Late Main|

Median distance is a crucial check hear since for a uniform density volume limited sample, the Expected value is about R/2^(1/3) = 158.7 pc. The 2% difference does make sense when accounting for the faintest M dwarfs at the outer edge.

Using standard error propagation.A 0.5% parallax error becomes 0.011mag in M_G such the unresolved binary sequence sits 0.753 mag above the main sequence. THis allows for clean resolving

### Labels
Labels originate from the GSP-Spec giving use Effective Temperature (T_eff), Log g and the (M/H) from the radial velocity spectra alone (Recio-Blanco et al. 2023). As such a completely different pipeline from the photometric processing, this makes confident errors falsifiable rather than relying on circular logic

Processing the 524,917 stars, 109,074 (-20.8%) have the GSP-Spec parameters and 43,175 pass all the recommended quality filter. Median G is about 11.26 for labelled against the 16.13 unlabelled. The parent sample is volume-limited but the *labelled set is magnitude limited*.

This was done on the basis of luminosity class rather than spectral class. Spectral classes are a relabelling of the Effective temperature, ie color. This creates a circular loop of logic. Luminosity class encodes solely the evolutionary state which is not a function of the surface temperature

One point to not is the main sequnce is not quite horizontal. The initial fixed cut at log g= 4.0 produces about 18.7% subgiants which is unlikely for a brief evolutionary phase on a cosmic scale. The Kiel diagram made gave us a better understanding in that the surface gravity changes with mass and temperature. Any horizontal cut runs into the error of say mislabelling a hot dwarf as a subgiant while missing all the volved cool stars. The boundary was re evaluated relative to a fitted ridge in each 150K bin the modal log g locates the main sequence and a cubic with those modes give the ridge (T_eff).

The delta log-g distribution shows no visible valley. It instad falls monotonically from the main sequence peak with no seperate subgiant mode. The post calibration scatter is a result of about 0.175 dex which is about physical main sequence and subgiant seperation. Only note being the boundary is precision limited. Subgiants and giants were then merged into an evovled class.

| Class | n  | Source  |
|------|---|-----|
| Dwarf | 37,306| Spectroscopic|
| Evolved| 3,445 | Spectroscopic|
| Ambigious|2,424 |Within Buffer Threshold |
| White Dwarf |20,867|Geometric|

Given White dwarfs are not present on the GSP-SPec given their faintness for the RVS method, they are labelled geometrically. As such this is the only examples of a circular class. This will act as a pipeline check, a control and not a result

### Feautures
Five feautures which include: bp_rp, bp_g, g_rp, mg, log_v_tan

Tagential velocity is calculated as v_tan=4.74047μ/ϖ. Raw proper motion is an artefact of distance. The observed median of 32.5km/s matches local thin disc dispersion

Parallax was not included. The labelled set is limited by magnitudes such distance correlates with whether a star can even be labelled. A model with parallax only understands the selection function rather than stellar physics which does not carry over to clusters at a fixed distance

The factors being witheld later for the attribution step include: RUWE, BP/RP excess vactor, radial velocity, galactic latitue, variability flag and M/H

### Key Thresholds
Adjustable constants were derived from the data and tested for their redundancy and results. This took the longest to fine tune.


| Constant  | Value  | Reasoning  |
|-----------|---|---|
| WD Boundary  | +6.13  |density min between populations/bin   |
| SUBGIANT_DELTA  | -0.35  | swept over 5vals  |
| BUFFER | 0.12  | swept over 6 values  |
| LOGG_SCATTER  |0.175 dex  |  16-84 percentile half range |
| BRIGHT_EXCESS  | 0.368   |2 standard deviations of 0.184 mag p   |
| MAX_BINARY_EXCESS  | 0.937  | 2.5 log10(2) + 1σ  |

Additionally the WD boundary was measured. The M_G distribution is bimodal with main sequence and cooling track. A kernel density estimate locates the minima and a weighted fit through fourteen such minima gives a boundary shown above

The Buffer band sets how far the dwarf and evolved threshold a star must sit before it obtains a label.

### Controlling Leakage
Only the sky blocks are being split, not stars. Objects such as wide binaries, co moving pairs and cluster memebrs are identical in every feauture. A random split puts one candidate in train and its corresponding twin in the test, making the model 'predict' a star it has memorised. The split uses a HEALPix level-3 patches from source_id, which is about 768 patches of about 54 deg^2 each. Patch overlap between train and test is asserted to be zero.

## Pipeline Explained
Nineteen modules in the source code which runs in a specific order. Each stage writes a parquet and JSON provenance to logs so any stage can be re-run independently.

### Foundations
**config.py**: The blueprint for all the sample cuts, constants and holds items such as PARALLAX_MIN,RUWE_MAX, EXCESS_FACTOR_RANGE, KAPPA, the FEAUTURES and DIAGNOSTICS lists that define the visibility of the model, along with Log Provenance as a digitial diary of each page

**setup_verify.py**: It ensures the environment is sanitised, requiring Python>= 3.10, the nine required package, writability of all project directories, Gaia Credentials with chmod 600 permissions, anonymous TAP query. Exits non-zero should something fail and highlights condition of failure.

**domain.py**: Defines where the model is applicable. domain_measure() returns the 0.6-99.5 percentile bounds in colour and absolute magnitude for both per class and globally, in_domain() tests candidacy, prediction_is_supported() applies the harsher test of checking a star's applicability within an assigned envelope and additionally names failure mode

### Acquiring Data
**query_gaia.py**: This retrieves the field sample by making an ADQL WHERE clause once between count and data queries. Number reported is the sample downloaded. random_index chunks such a timeout only costs one_chunk rather than the entire run. Caching each chunk to parquet such an interrupted query can resume.

**query_labels.py**: Retrieves the GSP-SPEC parameters for the same stars.There is no need for a cross match given a server side JOIN reproduces the stars generated the in the gaia query phase. random_index chunk cuts from gaia query is understood so the samples are the same

**query_binaries.py**: This retrieves the complement of the labels phase, same volume. however this goes through a 1.4< RUWE <3.0 . It recovers an unresolved binary population without redownlaoding any stars. The upper bound of 3 keeps out excess astrometric noise that indicates companionship

### Labelling
**build_labels_mod.py**: Turns spectroscopic parameters into luminosity classes through the following method
1) logg_calibrate() applies the Recio-Blanco et al.(2023) Table 3 polynomial that corrects a documented systematic resullting from giant branch effect
2) ridge_fit() locates the mainsequence with respect to temperature. This is done via a modal log g per 150K bin across 4200-6800K then a degree 3 polynomial. No extrapolation outside fitted range
3) class_assign() labels each star wrt offset from the ridge with a BUFFER band around the threshold

### Feauture Construction
**build_feautures.py**: Derives the quantities not covered within Gaia and assembles the model-ready table. derive() computes M_G, v_tan, log v_tan and distance.flag_white_dwarfs() applies measured geometric cut. Spectroscopic labels take precendence where both exist

### Visualisations
**hr_plot.py**: Four Figures which include: full-sample HR diagram, labelled classes over field density, white dwarf cuts with residual distribution, main sequence zoom with binary sequence residual

### Modelling
**train_model.py**: Sky blocked split and three modes of increasing strictness: Majority-class baseline, logisitc regression on colour and magnitude ONLY, LIGHTGBM on all five feautures. Gives a confusion matrix, feauture importances and perfomances with respect to colour.

**calibrate.py**: Uses four fold group aware cross validation with a nested calibration split. Each star receives a prediction from a model that has never observed its sky patch. This fits with respect to isotonic regression, builds a reliability diagram and reports the brier score and log loss for both raw and calibrated probailities

### Validation and Error Analysis
**validate_clusters_mod.py** Distribution shift test retrieves the Hyades and Pleiades members via a cone search plus parallax and proper motion cuts. Then they are ran through the frozen calibrated model while repoprting the in domain and out of domain behaviour seperately

**extract_anomalies.py**: Isolates the confident errors and creates the control sample. Filters to spectroscopic classes, restricts to training domaina nd keeps misclassifcations above the confidence threshold, matches each star to atleast five correctly classified stars of the same true class within 0.1 mag inc olour and 0.4 mag in M_G.

**attribute.py**:Analysis that goes through three pieces of evidence
1) Dose Response Tests: passes high RUWE complement throught he unchanged model and measures the respective error rates in the RUWE bins, with the Wilson confidence itnervals and a Fisher exactness test
2) Two Sample KS tests of the anomalies against the matched controls on the metallicity, kinematics, RUWE and the BP/RP excess factor
3) Attribution: This assigns each anomaly to a probably cause in priority order. The attributed fraction cannot be inflated by counting a star under several headings.

There are additionally five independent modules that are not part of the pipeline that exist to justify the numbers the pipeline uses. These include:
1) **wd_cut-calibrate.py**: White dwarf boundary measured wrt density minimum
2) **audit_thresholds.py**: Hand set constants matching wrt data present
3) **threshold_sweep.py**: Result's dependence on the SUBGIANT_DELTA or confidence cut
4) **buffer-sweep.py**: Does the result depend on the ambiguious band width.
5) **ruwe-bin_check.py**: Does result depend on the location of RUWE being placed.

## Results
### HR Diagram
Full-Sample resolves the main sequence, white dwarf cooling sequence, the red clump, the subgiant branch, M dwarf overdensity near BP-RP=3.

When the runing median is subtracted from the main sequence, a strong asymmetric residual of about 200 stars per bin at -1.0 mag againt 30 at +1.0 mag. Single stars scatter symmetrically about the ridge with only unresolved binaries producing a one sided excess. A shoulder apperas about -0.753 mag, the approxiamte offset for an equal mass pair

the white dwarf gap region contains about 11 stars plotted against 20,867 candidates which is a contamination indicator of about 0.001

### Classification
About 61,618 labelled stars with complete feautures split such into 46,135 training (576 patches) and 15.483 test (192 patches) with zero patch overlap.

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

Yields a abalanced accuracy of 0.990 with an evolved recall of 0.973 at an 10:8:1 class imbalance.

LIGHTGBM can be seen beating th two feauture logistic regression by about 0.015. The Colour and absolute magnitude have all the discriminating signal. Given the second and third colour indices contribute nearly nothing a simpler model would have sufficed

When binned by colour, the dwarf/evolved F1 is at 0.876 at BP-RP 0.5-1.0, rising above 0.99 redward. This bin is a main sequence turn-off where dwarfs and subgiants tend to overlap in luminosity due to the stellar structure, not heeding to measurement. The errors occur in the region where class labels are ambigious.

### Cluster Validation
| | Hyades (700 Myr) | Pleiades (125 Myr) |
|-|------------------|--------------------|
| members retrieved | 378 | 855 |
| inside training domain | 188 (49.7%) | 360 (42.1%) |
| predicted evolved, in-domain | 14 | 28 |
| of which red (BP−RP > 0.7) | 5 | 0 |
| expected from stellar evolution | 4 | 0 |

The Hyades red clump was recovered in the data which is a strong indication. The five red stars classified as Evolved with M_G= 0.03,0.05, 0.08, 0.3, 0.55 with a BP-RP rnage of 1.16-1.24 and G= 3.28-4.07 with a p=1.000. Four are γ, δ¹, ε and θ¹ Tauri. Given their occupance of a box 0.27mag tall and 0.05 mag wide, it indicates helium ignition at a constant core mass. Conversely, a model fitting noise cannot produce that level of concentration. The pleiades also showed zero red false positives, theoretically sound given the young age of the cluster.

Only point to note is the systematic error in the Pleiades classifcation. GSP-SPEC scans F,G,K only so there are no A or B in the labels. The only stars that might seem blue and luminous is the giant branch which results in 28 Pleaides B stars classified as evolved despite being part of the main sequence. Given Tree cannot extrapolate outside the training range, they give it the next closest label.

### Understanding Misclassifications
68 confident errros were reported to five controls. Witheld diagnostics yield null values. No metallicity or kinematic signature is detected given this sample size. The earlier run on an unlabelled set yielded a metal poor signal
| Attributed cause | n | % |
|------------------|---|---|
| close to label boundary | 40 | 58.8 |
| over-luminous beyond any binary | 20 | 29.4 |
| unexplained | 6 | 8.8 |
| metal-poor subdwarf | 2 | 2.9 |

The over luminous beyond binary thresholds a star's excess above the main sequence ridge against a 0.753 mag binary ceiling. The Excess is flat at 1.9mag across every RUWE bin, including those below 1.4 where no binaries exist. IT as such measures the model's decision boundary location not an intrinsic property.

## Discussion
The findings suggest this is a re-understanding of the common objection introduction of binaries cause misclassifcation. That is addressed here in the nature of the model rather than statistically being addressed. The model is frozen before the introduction of the binary population by fitting on stars with a RUWE<1.4 and has no parameter informed by a star above that boundary.

Binarity raises the rate of misclassifcation by a factor of nine. Whereas the individual misclassified stars tend to be label-boundary situations, 59%  sit within the 0.175 dex scatter of the classification threshold. Binaries as such make errors more likely without determining the specific stars that cause errors.

Interestingly, checking observed luminosity excess against the hard physical companion gave us the understanding the boundary population would have been attributed to binarity and the compounding the effect binaries have. Given an equal mass pair is twice as luminous, 0.753 mag is an upper bound such no unresolved binary can exceed

An error while doing this process to be noted was ruwe being limited such ruwe<1.4 to ensure the cleanliness of astrometric solutions. RUWE's elevation occurs as a result of an unresolved companion perturbing the photocentre. AS such that cut removed the entire population set the analysis was aiming to test. the first attribute resulted ina. ruwe ratio of 0.99 between anomalies and control which is not a null result, but the relevant stars were clearly not present. No re-querying was needed, solely rectifying the complement. A 'null' result can indicate the cut was too drastic and should be revised further along

Half of each validation cluster falls outside the training domain and the model behabiour there is systematic given how tree ensembles assign the nearest leaf. This quantifies that the classifier can be trusted and traces the mistake to a property of the label source, ie a systematic error. Once the red edge half of the failure had disappeared once the white dwarf boundary was measured from the data, it demonstrates the derivation of threshold changes outcomes rather than being solely grounded in rigor

## Conclusions
Individual misclassified stars are largely due to label-boundary cases with the delta log g within the measurement scatter of the threshold. Binarity governs the rate rather than the identity of the errors. The classifier transfers well to clusters it has not seen, such as Hyades and Pleiades noteably recovering the Hyades red clump along with zero red false positives against the pleiades

Outside its training domain, most errors are systematic in nature due to the aforemtioned tree-ensemble behaviour linked to the label source (GSP-SPEC) bias in the temperature coverage. This failure was eliminated by considering the white dwarf boundary of the data.This sets the precendent of every classifcaiton threshold being derived or tested in some form with respective sensitivities being included. However there is no solid mettalicity or kinematic signature that is detected amongst the anomalies in n=68, perhaps suggesting a pathway to more comprehensive tests

## Recreating this Project
### Requirements
We need Python>= 3.10 which cna be done with help from the requirements.txt file
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
### Gaia Credentials
A free ESA Gaia archive account allows for higher query limits. Storing account details outside repository is ideal.
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
 Note the query stages cache every chunk such an interrupted downlaod resumes rather than restarting. The query stages HAVE to populate /data

## Data and Software Acknowledgements
This work made use of the data acquired from the European Space Agency (ESA) mission Gaia proccesed by the Gaia Data Processing and Analysis consortium.

Software: astropy,astroquery, scikit-learn, pandas, numpy, scipy, matplotlib, pandas

## Key References
Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1 — Gaia DR3
Recio-Blanco et al. 2023, A&A 674, A29 
Belokurov et al. 2020, MNRAS 496, 1922 
Lindegren et al. 2021, A&A 649, A2 
Gentile Fusillo et al. 2021, MNRAS 508, 3877 

## License
MIT for source code, Gaia DR3 data is not here and is only available with ESA/DPAC

## Contribution
The design of the project, methodology, scientific reading and decisions, code-writing were my own. AI Assistance (Claude) assisted me in debugging code, rectifying errors and expanding my knowledge on coding in this format.
