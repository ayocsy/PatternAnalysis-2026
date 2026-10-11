# ConvNeXt-Tiny for ADNI classification: accuracy, confidence and computational cost

**Author:** Sheung Yan (Gaster) Chiang — s4862630  
**Course:** COMP3710, 2026  
**Task:** Hard difficulty, AD versus normal control (NC; also called CN in the brief)

> **Draft status:** Results below are from completed Rangpur experiments. Before submission, fill in the exact environment versions, verify the training configuration against the saved configuration files, and replace the feasibility section with the review actually discussed with the tutor. The feasibility check-off date and feedback have not been supplied.

## 1. Problem and main finding

This project compares a single-channel ConvNeXt-Tiny classifier with a smaller conventional CNN on ADNI brain MRI slices. The prototype predicts Alzheimer’s disease (AD) or normal control (NC), combines predictions across slices, and tests whether calibrated probabilities can identify cases needing human review. The engineering question is whether the larger architecture improves useful predictions and confidence enough to justify its extra computing cost.

In this experiment, the baseline achieved better overall test results with substantially fewer parameters and lower GPU cost. With validation-selected thresholds, patient accuracy was **82.35% for baseline and 71.57% for ConvNeXt**. ConvNeXt correctly classified one confidently misclassified baseline case, but that example did not translate into stronger aggregate performance. Both models achieved about 83.8% accuracy among accepted test patients, below the experimental 90% target. The rejection rules therefore do not justify unattended triage.

The assignment specifies a minimum 80% test accuracy for ConvNeXt. **The current ConvNeXt run does not meet that target at slice, scan or patient level.** The baseline passing a threshold does not satisfy the ConvNeXt requirement. This report retains the observed results rather than changing thresholds after examining test performance.

## 2. Working principles

A convolution learns small spatial filters that respond to local image patterns. Stacking layers combines these into more complex features. The baseline uses four convolution, BatchNorm, ReLU and max-pooling blocks with 32, 64, 128 and 256 channels, followed by global average pooling, dropout (0.30) and a two-class linear output.

ConvNeXt modernizes this approach using residual blocks, large depthwise convolutions and channel mixing [1]. Each implemented block applies a 7×7 depthwise convolution, LayerNorm, fourfold channel expansion, GELU, channel reduction, layer scaling and a residual connection with stochastic depth. The Tiny configuration has stage depths (3, 3, 9, 3) and channels (96, 192, 384, 768). A 4×4 stride-4 stem and three downsampling layers reduce spatial resolution; global average pooling, final normalization and a linear layer produce two logits. The implementation uses a one-channel input and two-class output, and is trained from scratch without pretrained weights.

![Implemented ConvNeXt and prediction pipeline](figures/working_principles.png)

*Figure 1. Architecture and prediction flow. Calibration is fitted on validation patients; the same settings are then applied to held-out test patients.*

For a slice with logits z_NC and z_AD, its AD probability is:

```text
p_AD = sigmoid(z_AD - z_NC)
p_AD_calibrated = sigmoid((z_AD - z_NC) / T)
```

Here T is a positive temperature fitted using validation patient negative log-likelihood (NLL), following the temperature-scaling idea in [2]. Calibration is applied to each slice before averaging. Scan and patient probabilities are the arithmetic mean of all their slice probabilities. Patients with more scans contribute more training slices, although patient evaluation counts each patient once. With 20 slices per scan, patient averaging is also equivalent to averaging that patient's scan means.

A prediction is AD when its probability reaches the threshold selected for that aggregation level. Temperature controls probability sharpness; the decision threshold controls the class decision. These are separate operations. A positive temperature preserves slice ranking and the slice decision at threshold 0.5, but averaging transformed probabilities can change scan or patient ranking and decisions.

## 3. Data audit, split and preprocessing

Data are read from the course-provided ADNI files:

```text
/home/groups/comp3710/ADNI/AD_NC
/home/groups/comp3710/ADNI/meta_data_with_label.json
```

The audit identified **30,520 JPEG slices, 1,526 scans and 680 patients**, with 20 slices per scan. Images are 256×240 pixels (width × height). Filenames identify the scan and slice; metadata link scans to patients and labels. Metadata label 0 (NC) is mapped to output 0 and label 2 (AD) to output 1. No conflicting patient labels were identified in the audit.

The supplied training and testing folders had no shared scans but shared **216 patients**. Using those folders directly would allow the model to see the same person during training and testing. The implementation therefore pools their records and constructs a seeded, class-stratified patient split. All scans and slices belonging to one patient remain in one split. Assertions check that both patient and scan sets are disjoint.

| Split | Patients | NC patients | AD patients | Scans | Slices |
|---|---:|---:|---:|---:|---:|
| Train | 476 | 321 | 155 | 1,065 | 21,300 |
| Validation | 102 | 69 | 33 | 228 | 4,560 |
| Test | 102 | 69 | 33 | 233 | 4,660 |
| Total | 680 | 459 | 221 | 1,526 | 30,520 |

The approximate ratio is 70/15/15, using seed **3710**. Patient classification is the primary operating unit; slice and scan results are also reported. The test set contains 69 NC and 33 AD patients, so accuracy alone can conceal poor AD recall. Slices from the same patient are correlated and should not be treated as independent clinical cases.

Preprocessing converts each image to grayscale, resizes it to 224×210, pads seven black pixels above and below, converts it to a tensor, and normalizes with mean 0.5 and standard deviation 0.5. This preserves the original aspect ratio. Training additionally uses random affine augmentation: rotation within ±5°, translation within 2%, and scale 0.98–1.02. Validation and test transformations are deterministic. This is a project-specific preprocessing choice, not a claim of reproducing the original ConvNeXt ImageNet recipe. No new skull stripping, intensity standardization across scans or 3D registration is performed by this code.

## 4. Feasibility review

**Status: retrospective draft for review; replace with the original tutor-reviewed one-page review before submission.** Completion of the required consultation cannot be established from the experiment logs. Do not present this reconstructed section as evidence of an earlier check-off.

### User need and scope

The intended user is an imaging specialist reviewing screening predictions. The prototype investigates AD/NC classification and whether confidence can flag ambiguous cases. It processes supplied 2D JPEG slices and evaluates patient-level outputs. It does not establish clinical diagnosis or validated clinical automation.

| Testable criterion for this review | Evidence or present status |
|---|---|
| No shared patients or scans across train, validation and test | Split assertions and audited counts support this criterion |
| ConvNeXt test accuracy at least 0.80, as required by the brief | Not met; highest reported test accuracy across the measured variants is below 0.80 |
| Compare patient accuracy, macro-F1, per-class metrics and AUROC against the same-split baseline | Completed; baseline is stronger overall |
| Experimental accepted-case accuracy at least 0.90 with coverage at least 0.50 | Met during validation rule selection, not met on test |
| Proposed peak allocated training VRAM budget below 4 GiB on one A100 | Observed peaks below 4 GiB; this retrospective budget is not evidence of a pre-agreed criterion |

### Model choice and course concepts

ConvNeXt-Tiny is the hard-difficulty model specified for ADNI. Convolution provides a useful local spatial bias; residual connections and normalization support deeper optimization. The smaller CNN provides a lower-cost comparison. Cross-entropy, gradient-based learning, augmentation, regularization, model selection and patient-separated evaluation connect the implementation to course concepts. A larger network does not automatically generalize better on a limited patient cohort.

### Evidence, risks, computing budget and fallback

The data audit makes the experiment feasible but exposes patient leakage in the supplied folder split. Both complete training runs fit on one A100, with recorded training-loop times of approximately 5.31 minutes for baseline and 19.65 minutes for ConvNeXt. The main risks are limited independent patients, correlated slices, image-quality variation, overfitting and confident errors. Only one seed per architecture has been measured.

A practical next experiment would investigate validation performance under stronger regularization or a smaller ConvNeXt configuration, with a fixed GPU budget and all decisions made without adapting to these exposed test results. Any new evaluation should acknowledge that the current test set has already been examined; fresh independent patients are needed for a new confirmatory claim. The fallback is to retain the simpler baseline as a research prototype and require human review, rather than claim that its acceptance rule meets the 90% test target.

**Consultation record — fill in from your actual check-off:** date, tutor, submitted review, feedback and response.

## 5. Training and evaluation protocol

Both models use the same records, preprocessing, seed and full evaluation splits. Training uses cross-entropy and AdamW with cosine learning-rate decay. The training budget is 30 epochs, batch size 32, two data-loader workers, 224×224 input, CUDA mixed precision and early stopping after eight epochs without improvement. Checkpoints are selected by validation patient macro-F1 at the training threshold 0.5.

The following optimizer values are the supplied code defaults. **Verify them against each saved `configuration.json` before treating this table as the exact run configuration.**

| Setting | Baseline | ConvNeXt |
|---|---:|---:|
| Initial learning rate | 0.001 | 0.0003 |
| Weight decay | 0.0001 | 0.05 |
| Selected epoch | 6 | 18 |
| Last training epoch | 14 | 26 |
| Input channels / output classes | 1 / 2 | 1 / 2 |
| Pretraining | None | None |

ConvNeXt's last training epoch reached slice accuracy 0.9822 while validation loss remained 1.0724, consistent with a substantial generalization gap. This does not prove its exact cause. The selected epoch, rather than the last epoch, supplies the weights for final evaluation.

The Rangpur training version uses `--skip-test` so that training does not evaluate test patients. Calibration is subsequently fitted using validation patients only. A scalar temperature minimizes validation patient NLL. Separate thresholds for slice, scan and patient predictions maximize validation macro-F1 over a grid from 0 to 1 in steps of 0.005, with ties resolved toward 0.5. The same validation set also selected the checkpoint, so validation calibration and threshold performance are optimistic; there is no independent calibration set.

| Validation-fitted setting | Baseline | ConvNeXt |
|---|---:|---:|
| Temperature | 1.010741 | 1.485609 |
| Slice AD threshold | 0.300 | 0.380 |
| Scan AD threshold | 0.365 | 0.420 |
| Patient AD threshold | 0.380 | 0.400 |
| Patient acceptance cutoff | 0.500 | 0.730 |

For patient rejection, assigned-class probability is p_AD for an AD prediction or 1−p_AD for an NC prediction. Validation selects the highest-coverage rule reaching 90% accepted accuracy and at least 50% coverage. A patient is routed to review when its assigned-class probability is below the saved cutoff. With a non-0.5 decision threshold, this quantity can be below 0.5 for the assigned AD class; it is not automatically the maximum class probability.

Checkpoint hashes, split manifests and inference precision are stored with calibration settings. Final testing checks these identities and applies the settings unchanged. Existing calibration and test reports are protected against accidental overwrite. The example demo subsequently loads the same weights and settings; it does not fit new parameters.

## 6. Held-out test results

**Raw:** uncalibrated probabilities with threshold 0.5. **Tuned:** temperature-scaled probabilities with validation-selected thresholds. Accuracy, recall and F1 are fractions. Each patient contributes one patient-level prediction.

| Model | Level | Variant | Accuracy | Macro-F1 | AUROC | AD recall |
|---|---|---|---:|---:|---:|---:|
| Baseline | Slice | Raw | 0.7655 | 0.7625 | 0.8807 | 0.6462 |
| Baseline | Slice | Tuned | 0.8174 | 0.8172 | 0.8807 | 0.8403 |
| Baseline | Scan | Raw | 0.7768 | 0.7733 | 0.9122 | 0.6441 |
| Baseline | Scan | Tuned | 0.8670 | 0.8669 | 0.9125 | 0.8729 |
| Baseline | Patient | Raw | 0.7941 | 0.7444 | 0.8595 | 0.5455 |
| Baseline | Patient | Tuned | 0.8235 | 0.7984 | 0.8595 | 0.7273 |
| ConvNeXt | Slice | Raw | 0.7292 | 0.7285 | 0.7996 | 0.7691 |
| ConvNeXt | Slice | Tuned | 0.7296 | 0.7276 | 0.7996 | 0.8042 |
| ConvNeXt | Scan | Raw | 0.7597 | 0.7591 | 0.8479 | 0.7966 |
| ConvNeXt | Scan | Tuned | 0.7682 | 0.7655 | 0.8481 | 0.8644 |
| ConvNeXt | Patient | Raw | 0.7353 | 0.7153 | 0.8406 | 0.7273 |
| ConvNeXt | Patient | Tuned | 0.7157 | 0.7005 | 0.8432 | 0.7576 |

The baseline's tuned patient threshold improves AD recall from 18/33 to 24/33, while NC false positives increase from six to nine. ConvNeXt tuning increases correctly classified AD patients from 24 to 25 but increases NC false positives from 18 to 21. Its validation threshold therefore does not improve test patient accuracy. These results are reported without selecting a new operating threshold on test labels.

### Patient per-class metrics

| Model / variant | NC precision | NC recall | NC F1 | AD precision | AD recall | AD F1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline / raw | 0.8077 | 0.9130 | 0.8571 | 0.7500 | 0.5455 | 0.6316 |
| Baseline / tuned | 0.8696 | 0.8696 | 0.8696 | 0.7273 | 0.7273 | 0.7273 |
| ConvNeXt / raw | 0.8500 | 0.7391 | 0.7907 | 0.5714 | 0.7273 | 0.6400 |
| ConvNeXt / tuned | 0.8571 | 0.6957 | 0.7680 | 0.5435 | 0.7576 | 0.6329 |

Tuned patient confusion counts:

| Model | True NC → NC | True NC → AD | True AD → NC | True AD → AD |
|---|---:|---:|---:|---:|
| Baseline | 60 | 9 | 9 | 24 |
| ConvNeXt | 48 | 21 | 8 | 25 |

### Calibration and accepted-case performance

| Model | Patient NLL raw → calibrated | Brier raw → calibrated | AD-probability ECE raw → calibrated |
|---|---|---|---|
| Baseline | 0.45009 → 0.44985 | 0.14305 → 0.14304 | 0.09881 → 0.10290 |
| ConvNeXt | 0.49348 → 0.48086 | 0.16564 → 0.16246 | 0.12761 → 0.13583 |

Lower NLL and Brier scores are better. Temperature scaling improves both here, especially for ConvNeXt. The reported ECE is a **classwise AD-probability calibration error**, comparing mean P(AD) with the observed AD fraction within bins. It is not the usual maximum-class-confidence ECE. Its increase at patient level shows that calibration improvements depend on the metric and do not establish reliable rejection.

| Model | Validation accepted / total | Validation accepted accuracy | Test accepted / total | Test coverage | Test accepted accuracy | Test accepted errors |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 93 / 102 | 0.9032 | 93 / 102 | 0.9118 | 0.8387 | 15 |
| ConvNeXt | 70 / 102 | 0.9000 | 68 / 102 | 0.6667 | 0.8382 | 11 |

Both rules met the validation target, but neither reached 90% accepted accuracy on test. ConvNeXt routed one third of test patients to review while achieving almost the same accepted accuracy as baseline, which routed fewer than one tenth. There is no demonstrated operational benefit from the more restrictive ConvNeXt rule in this run.

## 7. Computational cost

| Measurement | Baseline | ConvNeXt |
|---|---:|---:|
| Trainable parameters | 388,834 | 27,818,594 |
| Training-loop time, seconds | 318.60 | 1,179.01 |
| Peak allocated training GPU memory, MiB | 643.59 | 2,289.56 |
| Peak allocated inference GPU memory, MiB | 253.81 | 1,000.25 |
| Batched model inference, ms per slice | 0.1099 | 0.4605 |
| Throughput, slices per second | 9,099.88 | 2,171.32 |
| Measured inference samples | 640 | 640 |

Profiling used CUDA on the A100 partition, batch size 32 and mixed precision. Twenty measured batches followed three warm-up batches. Timing synchronizes CUDA around the model forward pass and excludes loading, preprocessing, host-to-device transfer and patient aggregation. The per-slice value is batch time divided by sample count, not interactive single-image latency. Memory is PyTorch peak allocated memory, not total GPU utilization or reserved memory. The JSON field names use “MB”, but the code divides bytes by 1024², so this table correctly uses MiB.

Training time includes the epoch loop, validation and checkpoint/plot writing, but excludes initial dataset/model setup and the later inference profile. The peak inference figure can include allocations already resident from training; these are not separate clean-process deployment memory measurements. ConvNeXt has about **71.5×** as many parameters, **3.56×** peak allocated training memory and **4.19×** batched latency. Total training time is about **3.70×** longer, also reflecting different numbers of completed epochs.

## 8. Qualitative failure analysis

The failure figures show selected confident patient-level errors. A representative slice is chosen near its patient's mean probability; this does not identify a causal anatomical feature. These examples are deliberately selected for error analysis and do not estimate population error frequency. Visual hypotheses below need further checks against the source images and complete scans.

![Baseline confident test errors](figures/baseline_test_failure_cases.png)

![ConvNeXt confident test errors](figures/convnext_test_failure_cases.png)

| Case | Observed result | Failure mode and interpretation | Follow-up needed |
|---|---|---|---|
| 168_S_6843, true AD | Baseline P(AD) ≈0.036; ConvNeXt ≈0.118; both predict NC | Shared confident false negative. Agreement between architectures does not establish correctness, and both saved rules accept the incorrect predictions. | Inspect every scan/slice and verify metadata; test whether image quality or weak learned disease features explain the error. |
| 032_S_6600, true AD | Baseline P(AD) ≈0.062; ConvNeXt ≈0.119; both predict NC | A second shared false negative. The failure survives both aggregation and rejection; averaging is not enough when model evidence is consistently wrong. | Review per-scan variation and compare with correctly classified AD cases without changing test thresholds. |
| 168_S_6086, true NC | Baseline P(AD)=0.9032, wrong; ConvNeXt P(AD)=0.0117, correct | Confident architecture disagreement. The displayed slices are unusually dark or noisy. Quality sensitivity is a plausible hypothesis, not an established cause. | Inspect original JPEGs, intensities and the full scan; assess quality effects using validation data. |
| 041_S_6292, true NC | Baseline P(AD) ≈0.842; predicts AD | Confident false positive accepted by the baseline rule. A high AD score is not sufficient evidence of disease. No causal image feature is established from the figure. | Compare slice and scan scores, check image quality and test for acquisition-related shortcuts. |
| 127_S_4198, true NC | ConvNeXt P(AD) ≈0.896; predicts AD | Confident false positive accepted by the ConvNeXt rule. Lower visible sharpness in the representative image warrants a quality check but does not prove the trigger. | Review complete scan images and quality distribution relative to correctly classified NC validation patients. |

### Detailed disagreement demonstration

![Baseline patient demo](figures/baseline_prediction_demo.png)

![ConvNeXt patient demo](figures/convnext_prediction_demo.png)

For patient **168_S_6086**, scan **917090**, the CSV audit includes 20 slices. Baseline's calibrated AD probabilities range from **0.693 to 0.990**, with a mean about 0.903; ConvNeXt's range from **0.000 to 0.084**, with a mean about 0.012. Every baseline slice exceeds its saved AD slice threshold 0.30; every ConvNeXt slice is below its threshold 0.38. The disagreement therefore extends across the whole scan rather than one selected slice. Both patient predictions pass their saved acceptance cutoffs, even though baseline is wrong.

The demo figures display evenly spaced slices 94, 104 and 113. They are original image examples, not attention maps or explanations of the model's internal decision. ConvNeXt's success on this case is useful evidence of complementary behavior, but its worse overall results remain the basis for the engineering recommendation.

## 9. Recommendation and limitations

For this dataset, split and training budget, retain the **baseline as the preferred experimental model**. It achieves stronger overall test metrics with much lower cost. Do not deploy either saved acceptance rule as validated autonomous screening: both miss the test accepted-accuracy target and retain confident false negatives.

The next development priorities are an image-quality audit, stronger control of overfitting, and patient-balanced training or sampling evaluated on validation data. A separate calibration cohort would reduce the optimism caused by reusing validation patients for checkpoint, temperature, threshold and rejection selection. Repeated seeds and patient-level uncertainty intervals would help establish whether architecture differences are stable. These experiments have not been performed and should not be presented as completed improvements.

This study has one seed per architecture, only 102 test patients, no external cohort and no verified acquisition/site-stratified analysis. Patient separation prevents identity leakage but does not guarantee robustness across sites or scanners. The prototype uses 2D JPEGs rather than a complete 3D diagnostic pipeline. Its results are an engineering comparison on the course dataset, not clinical validation.

## 10. Files and reproduction

| File | Purpose |
|---|---|
| `modules.py` | Baseline CNN and single-channel ConvNeXt-Tiny implementations |
| `dataset.py` | Metadata parsing, patient split, leakage checks and transforms |
| `train.py` | Training, validation checkpoint selection and resource profiling |
| `evaluate_models.py` | Validation-only calibration and frozen final evaluation |
| `evaluate_models.slurm` | GPU evaluation jobs for both checkpoints |
| `predict.py` | Selected-patient inference using saved weights and calibration |
| `predict_demo.slurm` | Baseline/ConvNeXt example on patient 168_S_6086 |
| `figures/` | Report figures |

### Environment — pending exact versions

Dependencies are Python, PyTorch, torchvision, NumPy, SciPy, scikit-learn, Matplotlib and Pillow. **Exact versions must be filled in from the experiment environment.** Do not substitute package versions from the machine used to draft this report. On Rangpur:

```bash
source ~/comp3710/venvs/convnext-report/bin/activate
cd ~/comp3710/PatternAnalysis-2026/recognition/ConvNeXt_s4862630
python --version
python -m pip freeze > requirements-lock.txt
python -c 'import torch; print("PyTorch", torch.__version__); print("CUDA runtime", torch.version.cuda); print("cuDNN", torch.backends.cudnn.version())'
cat outputs/baseline_seed3710/configuration.json
cat outputs/convnext_seed3710/configuration.json
```

Record the Python version, CUDA/cuDNN versions and package versions from these outputs. Capture GPU model/driver with `nvidia-smi` inside a GPU allocation. `requirements-lock.txt` records packages; reproducing a compatible CUDA installation still requires the matching PyTorch wheel source/platform.

### Training

Submit GPU work through Slurm, not on the login node. The following are reproduction arguments using the defaults described above; compare with the actual saved run configuration first. Run each within an A100 allocation, with four CPUs and two workers. Rangpur scripts omit an explicit `--mem` directive because the cluster rejected the earlier memory configuration.

```bash
python -u train.py --model baseline --epochs 30 --batch-size 32 \
  --workers 2 --seed 3710 --patience 8 --device cuda --skip-test \
  --output-dir outputs/baseline_seed3710

python -u train.py --model convnext --epochs 30 --batch-size 32 \
  --workers 2 --seed 3710 --patience 8 --device cuda --skip-test \
  --output-dir outputs/convnext_seed3710
```

Use a fresh output directory for a reproduction run so the recorded experiment is preserved. Both commands require the current Rangpur `train.py` version supporting `--skip-test`. Seeding and deterministic cuDNN settings improve repeatability but do not promise identical results across hardware or library versions.

### Calibration, testing and demo

```bash
mkdir -p logs
sbatch evaluate_models.slurm calibrate
# Wait for successful calibration before submitting final evaluation.
sbatch evaluate_models.slurm test
# Existing calibration/test results are protected; do not rerun over them.
sbatch predict_demo.slurm 168_S_6086
```

The recorded calibration and final evaluation are already complete. To reproduce them, use fresh output locations in a copy of the evaluation job script. The demo uses a job-specific output directory and can be run without overwriting the original evaluation. It depends on `evaluate_models.py` being in the same project directory.

Example recorded output (job 648666):

```text
Model=baseline epoch=6 patients=['168_S_6086']
Patient 168_S_6086 | true=NC predicted=AD P(AD)=0.9032 action=accept_prediction
Model=convnext epoch=18 patients=['168_S_6086']
Patient 168_S_6086 | true=NC predicted=NC P(AD)=0.0117 action=accept_prediction
```

Training produces checkpoints, history, curves, configuration and resource profiles. Evaluation produces calibration settings, per-level metrics, slice CSVs, reliability/confusion figures, patient ROC/coverage plots and selected failure figures. Prediction produces `predictions.json`, slice/scan/patient CSV files and `prediction_examples.png`. Keep datasets and model checkpoints outside Git; commit code, environment records and the figures used in the report.

## 11. Artificial Intelligence Usage Disclosure

ChatGPT/Codex assisted with interpreting the assignment, explaining CNN and ConvNeXt concepts, scaffolding and debugging the Python implementation, designing the patient-level split and leakage checks, writing Slurm scripts, developing calibration/evaluation and prediction utilities, interpreting experiment outputs and drafting this report.

Verification included inspecting the supplied code and data-audit outputs, running the models and evaluation jobs on Rangpur, comparing printed metrics with saved JSON/CSV outputs, inspecting qualitative examples, and running the selected-patient demo with both checkpoints. The assistant performed local syntax, numerical and mocked orchestration checks for the evaluation and prediction utilities; those local checks did not exercise actual CUDA inference. The Rangpur runs supplied the real GPU evidence. No claim is made that a comprehensive independent test suite or clinical validation was completed.

**Before submission, confirm that this disclosure matches the tools you actually used and the checks you personally completed.** The submitted implementation, numerical claims and final interpretation remain the student's responsibility.

## 12. References

1. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T. and Xie, S. (2022). *A ConvNet for the 2020s*. CVPR, pp. 11976–11986. [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html). The architecture follows the ConvNeXt design with task-specific input/output changes and a smaller project training recipe.
2. Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML, PMLR 70, pp. 1321–1330. [Paper](https://proceedings.mlr.press/v70/guo17a.html). Temperature scaling is adapted here to minimize validation patient NLL after slice aggregation.
3. COMP3710 teaching team (2026). *Working Prototype Project in the Era of Artificial Intelligence*, supplied draft v2 assignment brief. Requirements and domain investigation are taken from this supplied document; check the current course submission instructions before submitting.
4. Course-provided ADNI JPEG subset and `meta_data_with_label.json`, available under `/home/groups/comp3710/ADNI`. Dataset counts and split statistics are from the project audit; this report does not infer additional acquisition provenance beyond the supplied metadata.
