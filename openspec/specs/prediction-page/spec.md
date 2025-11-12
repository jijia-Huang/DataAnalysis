# prediction-page Specification

## Purpose
TBD - created by archiving change add-prediction-page. Update Purpose after archive.
## Requirements
### Requirement: Model Selection for Prediction
The system SHALL allow users to select a model for prediction from either the training page session state or saved model files.

#### Scenario: Use model from training page
- **WHEN** a user navigates to the prediction page after training a model
- **THEN** the system automatically detects the trained model in session state
- **AND** the system displays model information (model type, target variables, feature names)
- **AND** the user can use this model for prediction without reloading

#### Scenario: Load saved model file
- **WHEN** no model is available in session state
- **THEN** the system displays a model file selector showing all saved models
- **AND** the system displays model information for each saved model (model name, type, target variables, modified time)
- **AND** the user can select a model file to load
- **AND** after selection, the model is loaded and ready for prediction

#### Scenario: Model information display
- **WHEN** a model is selected (from session state or file)
- **THEN** the system displays model information including:
  - Model type (Linear Regression, Gradient Descent)
  - Target variable names
  - Feature names
  - Number of features
- **AND** the information is clearly formatted and easy to understand

### Requirement: Single Data Point Prediction
The system SHALL allow users to input a single data point and get predictions.

#### Scenario: Single prediction input form
- **WHEN** a model is selected for prediction
- **THEN** the system dynamically generates an input form based on the model's feature names
- **AND** for numeric features, the system provides number input fields
- **AND** for categorical features, the system provides select boxes with available categories
- **AND** each input field is clearly labeled with the feature name

#### Scenario: Single prediction execution
- **WHEN** a user fills in all required feature values and clicks "Predict"
- **THEN** the system validates the input data (data types, value ranges)
- **AND** if validation passes, the system executes prediction
- **AND** if validation fails, clear error messages are displayed with correction suggestions

#### Scenario: Single prediction result display
- **WHEN** a single prediction completes successfully
- **THEN** the system displays:
  - Input feature values
  - Predicted values for all target variables
- **AND** the results are clearly formatted and easy to read
- **AND** for multiple target variables, each prediction is clearly labeled

### Requirement: Batch Prediction from CSV File
The system SHALL allow users to upload a CSV file and get batch predictions.

#### Scenario: CSV file upload
- **WHEN** a user wants to perform batch prediction
- **THEN** the system provides a file uploader for CSV files
- **AND** the system validates the CSV file format after upload
- **AND** if the file format is invalid, clear error messages are displayed

#### Scenario: CSV data validation
- **WHEN** a CSV file is uploaded
- **THEN** the system checks that the CSV contains all required feature columns
- **AND** the system checks that feature names match the model's feature names (order can differ)
- **AND** the system checks that data types match (numeric, categorical)
- **AND** for categorical features, the system checks that values were seen during training
- **AND** if validation fails, clear error messages indicate which columns or values are problematic

#### Scenario: Batch prediction execution
- **WHEN** CSV data validation passes
- **THEN** the system executes batch prediction for all rows
- **AND** the system displays a progress indicator during prediction
- **AND** if prediction fails, clear error messages are displayed

#### Scenario: Batch prediction result display
- **WHEN** batch prediction completes successfully
- **THEN** the system displays a table containing:
  - Original input data columns
  - Predicted values for all target variables (as new columns)
- **AND** the table is sortable and scrollable
- **AND** the results are clearly formatted and easy to read

### Requirement: Prediction Result Export
The system SHALL allow users to export prediction results to CSV format.

#### Scenario: Export single prediction result
- **WHEN** a single prediction is completed
- **THEN** the system provides a download button to export results
- **AND** the exported CSV contains input features and predicted values
- **AND** the file is named appropriately (e.g., "prediction_results.csv")

#### Scenario: Export batch prediction results
- **WHEN** batch prediction is completed
- **THEN** the system provides a download button to export results
- **AND** the exported CSV contains all original input data columns and predicted value columns
- **AND** the file is named appropriately (e.g., "batch_prediction_results.csv")

### Requirement: Input Data Validation
The system SHALL validate input data before prediction to ensure compatibility with the model.

#### Scenario: Feature name validation
- **WHEN** input data is provided for prediction
- **THEN** the system checks that all required features are present
- **AND** the system checks that feature names match the model's feature names
- **AND** if features are missing or names don't match, clear error messages are displayed

#### Scenario: Data type validation
- **WHEN** input data is provided for prediction
- **THEN** the system checks that numeric features contain numeric values
- **AND** the system checks that categorical features contain valid category values
- **AND** if data types don't match, clear error messages are displayed

#### Scenario: Categorical value validation
- **WHEN** categorical features are provided for prediction
- **THEN** the system checks that all category values were seen during training
- **AND** if unknown category values are found, clear error messages indicate which values are problematic
- **AND** the error message suggests valid category values

#### Scenario: Missing value handling
- **WHEN** input data contains missing values
- **THEN** the system displays a clear error message indicating missing values
- **AND** the error message indicates which features have missing values
- **AND** the system prevents prediction until missing values are filled

### Requirement: User-Friendly Prediction Interface
The system SHALL provide a clear, intuitive interface for non-technical users.

#### Scenario: Step-by-step guidance
- **WHEN** a user visits the prediction page
- **THEN** clear step-by-step instructions are displayed
- **AND** the instructions use simple, non-technical language
- **AND** each step is clearly numbered or organized

#### Scenario: Input method selection
- **WHEN** a model is selected
- **THEN** the system provides options to choose between single prediction and batch prediction
- **AND** each option has a clear description of when to use it
- **AND** the user can switch between methods easily

#### Scenario: Visual feedback
- **WHEN** a user interacts with the prediction interface
- **THEN** visual feedback confirms selections (highlighting, checkmarks)
- **AND** the current step in the prediction process is clearly indicated
- **AND** disabled options are grayed out appropriately

#### Scenario: Error prevention
- **WHEN** a user makes an invalid selection or input
- **THEN** the system prevents proceeding to the next step
- **AND** helpful hints are displayed explaining what needs to be corrected

