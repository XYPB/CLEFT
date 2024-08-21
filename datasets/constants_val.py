import os
from pathlib import Path


DATA_BASE_DIR = '~/project/CLEFT/data' 
DATA_BASE_DIR = os.path.expanduser(DATA_BASE_DIR)
MY_API_TOKEN = "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"


# #############################################
# CHEXPERT constants
# #############################################
CHEXPERT_DATA_DIR = DATA_BASE_DIR + "/CheXpert/CheXpert-v1.0"
CHEXBERT_TRAIN_CSV = DATA_BASE_DIR + "/CheXpert/train_cheXbert_clean_95.csv"
CHEXBERT_TRAIN_5_CSV = DATA_BASE_DIR + "/CheXpert/train_cheXbert_95_five_classes.csv"
CHEXPERT_VALID_CSV = DATA_BASE_DIR + "/CheXpert/train_cheXbert_clean_5.csv"
CHEXPERT_VALID_5_CSV = DATA_BASE_DIR + "/CheXpert/train_cheXbert_5_five_classes.csv"
CHEXPERT_TEST_CSV = DATA_BASE_DIR + "/CheXpert/chexpert_5x200.csv"
CHEXPERT_PATH_COL = "Path"
CHEXPERT_VIEW_COL = "AP/PA"
CHEXPERT_FINDINGS = [
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices',
    'No Finding',
]
CHEXPERT_FINDINGS_5 = [
    'Atelectasis',
    'Edema',
    'Cardiomegaly',
    'Pleural Effusion',
    'Consolidation',
]
CHEXPERT_BASE_CAPTION = "This is a chest X ray of a patient with "
CHEXPERT_BASE_CAPTION_EMBEDDING = DATA_BASE_DIR + "/CheXpert/chexpert_base_caption_embedding.pt"
CHEXPERT_NEGATIVE_CAPTION = "but this image has no "


# #############################################
# RSNA constants
# #############################################
RSNA_DATA_DIR = DATA_BASE_DIR + "/RSNA Pneumonia"
RSNA_TRAIN_DATA_PATH = RSNA_DATA_DIR + "/stage_2_train_images"
RSNA_TEST_DATA_PATH = RSNA_DATA_DIR + "/stage_2_test_images"
RSNA_TRAIN_CSV = RSNA_DATA_DIR + "/stage_2_train_balance.csv"
RSNA_TEST_CSV = RSNA_DATA_DIR + "/stage_2_test_balance.csv"


# #############################################
# EMBED constants
# #############################################
EMBED_DATA_DIR = DATA_BASE_DIR + "/Embed"
EMBED_DATA_PATH = EMBED_DATA_DIR + "/images"
EMBED_TRAIN_META_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_train.csv"
EMBED_TEST_META_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_test.csv"
EMBED_VALID_META_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_valid.csv"
# Read the full annotation for calcification information
EMBED_ANNO_CSV_REDUCED = EMBED_DATA_DIR + "/tables/EMBED_OpenData_clinical_reduced.csv"
EMBED_ANNO_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_clinical.csv"
EMBED_LEGENDS_CSV = EMBED_DATA_DIR + "/tables/AWS_Open_Data_Clinical_Legend.csv"
EMBED_BALANCED_TEST_PATH = EMBED_DATA_DIR + "/test_7x200_path2label.pickle"
EMBED_BALANCED_TRAIN_PATH = EMBED_DATA_DIR + "/train_7x550_path2label.pickle"
EMBED_BALANCED_DEN_TEST_PATH = EMBED_DATA_DIR + "/test_4x500_path2density.pickle"
EMBED_BALANCED_DEN_TRAIN_PATH = EMBED_DATA_DIR + "/train_4x1000_path2density.pickle"
EMBED_TRAIN_PATH2DENSITY = EMBED_DATA_DIR + "/train_path2density.pickle"
EMBED_VALID_PATH2DENSITY = EMBED_DATA_DIR + "/valid_path2density.pickle"
EMBED_TEST_PATH2DENSITY = EMBED_DATA_DIR + "/test_path2density.pickle"

EMBED_IMAGE_TYPE_COL = "FinalImageType"
EMBED_PATH_COL = "anon_dicom_path"
EMBED_PID_COL = 'empi_anon'
EMBED_SID_COL = 'acc_anon'
EMBED_SIDE_COL = 'ImageLateralityFinal'
EMBED_FINDING_SIDE_COL = 'side'
EMBED_VIEW_COL = 'ViewPosition'
EMBED_DENSITY_COL = 'tissueden'
EMBED_BIRADS_COL = 'asses'
EMBED_PROCEDURE_COL = 'StudyDescription'
EMBED_MASS_SHAPE_COL = 'massshape'
EMBED_MASS_DENSITY_COL = 'massdens'
EMBED_CALC_FIND_COL = 'calcfind'
EMBED_CALC_DIST_COL = 'calcdistri'
EMBED_AGE_COL = 'age_at_study'
EMBED_RACE_COL = 'RACE_DESC'
EMBED_ETHNIC_COL = 'ETHNIC_GROUP_DESC'
EMBED_PATH_TRANS_FUNC = lambda x: x.replace("/mnt/NAS2/mammo/anon_dicom", EMBED_DATA_PATH)
EMBED_PROCEDURE2REASON_FUNC = lambda x: "screening" if "screen" in x.lower() else "diagnostic" if "diag" in x.lower() else ""
# Normal caption constants
BREAST_BASE_CAPTION = "This is a breast 2D full-field digital mammogram of a patient "
BREAST_SIDE_CAPTION = "on side " # Make the caption more grammarly correct
BREAST_VIEW_CAPTION = "with view "
BREAST_DENSITY_CAPTION = "with breast tissue density "
BREAST_BIRADS_CAPTION = "with BIRADS score "
# TODO: Add more findings according to the EMBED dataset structure
# Natural Captions
EMBED_NATURE_BASE_CAPTION = "This is a breast 2D full-field digital {{REASON}} mammogram of a patient. "
EMBED_NATURE_IMAGE_CAPTION = "This mammogram is for {{SIDE}} breast with {{VIEW}} view. "
# Structural Captions
EMBED_PROCEDURE = 'Procedure reported: ' # EMBED_PROCEDURE_COL
EMBED_REASON = 'Reason for procedure: ' # Screening / Diagnostic, maybe add more details later
EMBED_PATIENT = 'Patient info: ' # AGE + RACE + ETHNIC
EMBED_IMAGE = 'Image info: ' # EMBED_IMAGE_TYPE_COL + EMBED_SIDE_COL + EMBED_VIEW_COL
EMBED_DENSITY = 'Breast composition: ' # EMBED_DENSITY_COL + extra description
EMBED_FINDINGS = 'Findings: ' # EMBED_MASS info + EMBED_CALC_FIND_COL + extra description
EMBED_IMPRESSIONS = 'Impressions: ' # EMBED_BIRADS_COL + extra description
EMBED_ASSESSMENT = 'Overall Assessment: ' # EMBED_BIRADS_COL number

EMBED_PATIENT_INFO_CAPTION = "This patient is {{RACE}}, {{ETHNIC}}, and {{AGE}} years old. "
EMBED_IMAGE_INFO_CAPTION = "This is a {{IMAGE_TYPE}} full-field digital mammogram of the {{SIDE}} breast with {{VIEW}} view. "
EMBED_BREAST_COMPOSITION_CAPTION = "The breast is {{DENSITY}}. "
EMBED_DENSITY_EXTRA_CAPTION = {
    3: "This may lower the sensitivity of mammography. ",
    4: "This may lower the sensitivity of mammography. ",
}
EMBED_FINDS_CAPTION = "The mammogram shows that "
EMBED_MASS_CAPTION = {
    'A': "an additional imaging is recommended. ",
    'N': "no significant masses, calcification, or other abnormalities are present. ",
    'B': "a benign finding is present. ",
    'P': "a probably benign finding is present. ",
    'S': "a suspicious abnormality is present. ",
    'M': "a highly suggestive of malignancy is present, a biopsy is recommended. ",
    'K': "a known biopsy-proven malignant mass is present. ",
}
EMBED_MASS_EXTRA_CAPTION = 'The mass is {{SHAPE}} and {{DENSITY}}. '
EMBED_CALC_FINDS_CAPTION = 'A {{DISTRI}} {{SHAPE}} calcification is present. '
EMBED_IMPRESSION_CAPTION = "BI-RADS Category {{BIRADS}}: {{BIRADS_DESC}}. "
EMBED_ASSESSMENT_CAPTION = {
    'A': "Additional imaging is recommended. ",
    'N': "Negative. ",
    'B': "Benign. ",
    'P': "Probably benign. ",
    'S': "Suspicious abnormality. ",
    'M': "Highly suggestive of malignancy. ",
    'K': "Known biopsy-proven malignancy. ",
}
EMBED_SIDES_DESC = {
    'L': 'left',
    'R': 'right',
    'B': 'bilateral',
}
EMBED_DENSITY_DESC = {
    1: "almost entirely fat",
    2: "scattered fibroglandular densities",
    3: "heterogeneously dense",
    4: "extremely dense",
    5: "normal male dense",
}
EMBED_LETTER_TO_BIRADS = {
    "A": 0,
    "N": 1,
    "B": 2,
    "P": 3,
    "S": 4,
    "M": 5,
    "K": 6,
}
EMBED_BIRADS_DESC = {
    'A': "additional imaging required",
    'N': "negative",
    'B': "benign finding",
    'P': "probably benign finding",
    'S': "suspicious abnormality",
    'M': "highly suggestive of malignancy",
    'K': "known biopsy-proven malignancy",
}