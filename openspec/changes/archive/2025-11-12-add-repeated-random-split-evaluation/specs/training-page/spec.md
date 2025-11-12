## ADDED Requirements
### Requirement: Repeated Random Split Evaluation Option
The system SHALL allow users to choose between single evaluation and repeated random split evaluation.

#### Scenario: Evaluation method selection
- **WHEN** a user is ready to train a model
- **THEN** the system displays radio buttons or dropdown for evaluation method selection:
  - Single evaluation (default)
  - Repeated random split evaluation
- **AND** each method has a brief description explaining its characteristics

#### Scenario: Repeated split configuration
- **WHEN** repeated random split evaluation is selected
- **THEN** the system displays a number input for number of repetitions
- **AND** the repetition range is 3 to 20 (default: 5)
- **AND** a description explains that more repetitions provide more stable results but take longer

#### Scenario: Evaluation method information display
- **WHEN** repeated random split evaluation is selected
- **THEN** the system displays information explaining:
  - What repeated random split evaluation does
  - How it differs from single evaluation
  - That it provides more stable and reliable results
  - That it takes longer than single evaluation

### Requirement: Repeated Random Split Evaluation Execution
The system SHALL perform repeated random split evaluation when selected.

#### Scenario: Execute repeated evaluations
- **WHEN** repeated random split evaluation is selected with N repetitions
- **THEN** the system performs N independent random splits and model training
- **AND** each repetition uses a different random seed to ensure independence
- **AND** evaluation metrics (R², MSE, MAE) are calculated for each repetition

#### Scenario: Calculate aggregate statistics
- **WHEN** all repetitions complete
- **THEN** the system calculates mean and standard deviation for each metric
- **AND** the statistics are calculated across all repetitions
- **AND** the results are displayed in a clear format

#### Scenario: Display evaluation results
- **WHEN** repeated random split evaluation completes
- **THEN** the system displays:
  - Mean and standard deviation for each metric (R², MSE, MAE)
  - Format: "Mean ± Std" (e.g., "0.85 ± 0.03")
  - Optionally: detailed results table showing each repetition's metrics
- **AND** the results are clearly labeled and easy to interpret

#### Scenario: Progress indication
- **WHEN** repeated random split evaluation is in progress
- **THEN** a progress indicator is displayed showing:
  - Current repetition number
  - Total number of repetitions
  - Estimated time remaining (if possible)
- **AND** the user understands that evaluation is in progress

#### Scenario: Single evaluation fallback
- **WHEN** single evaluation is selected
- **THEN** the system performs evaluation as before (single random split)
- **AND** the evaluation results are displayed in the same format as before
- **AND** no changes to existing behavior

