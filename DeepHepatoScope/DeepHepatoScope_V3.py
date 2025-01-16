#pip install dash scanpy scikit-learn umap-learn numpy pandas plotly pillow seaborn matplotlib scipy gseapy statsmodels tensorflow keras
import dash
from dash import dcc, html, Input, Output, State, callback_context
from dash import html
import dash_bootstrap_components as dbc
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import base64
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"  # Limit threads to 1 to reduce noise
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)  # Use 'spawn' for Mac if needed
import matplotlib.image as mpimg
from io import BytesIO
from PIL import Image
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import io
import matplotlib.pyplot as plt
import traceback
from scipy.sparse import issparse
import gseapy as gp
from gseapy import prerank
from gseapy import gseaplot
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
import plotly.tools as tls
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "DeepHepatoScope"

#Leave the specific order as this works for both scRNA and spatial h5ad files
def encode_anndata(anndata):
    dict = {
            'X': anndata.X.toarray() if hasattr(anndata.X, "toarray") else anndata.X,
            'obs': anndata.obs.to_dict(orient='index'), #orient='index'
            'var_names': anndata.var_names.tolist(),
            'var': anndata.var.to_dict() #orient='index
        }
    return dict

def decode_dict(dict):
    if isinstance(dict['X'], list):
        dict['X'] = np.array(dict['X'])  # Convert list to NumPy array
    elif isinstance(dict['X'], np.ndarray) and dict['X'].ndim == 1:
        dict['X'] = dict['X'].reshape(-1, 1)  # Ensure 2D array
    obs_df = pd.DataFrame.from_dict(dict['obs'], orient='index')
    var_df = pd.DataFrame.from_dict(dict['var'], orient='index')
    if var_df.shape[0] != dict['X'].shape[1]:
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(dict['X'].shape[1])])  # Placeholder
    anndata = sc.AnnData(X=dict['X'], obs=obs_df, var=var_df)
    anndata.var_names = dict['var_names']
    return anndata

# Default RF Parameters
default_rf_params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
}

# Default parameters for Neural Network
default_nn_params = {
    "hidden_layers": 3,
    "neurons_per_layer": 64,
    "activation_function": "relu",
}

# Load the saved image and convert it to base64 for Dash rendering
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

first_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("GSE189903", className="card-title"),
            html.P("Ma et al. (2022)"),
            dbc.Button("Select", color="primary", id="card1-button"),
            dbc.Button("Advanced", color="grey", id="advanced1-link", className="ms-2"),
        ]
    ),
    id = "card1"
)

second_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("GSE151530", className="card-title"),
            html.P("Ma et al. (2021)"),
            dbc.Button("Select", color="primary", id="card2-button"),
            dbc.Button("Advanced", color="grey", id="advanced2-link", className="ms-2"),
        ]
    ),
    id = "card2"
)

third_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("GSE125449", className="card-title"),
            html.P("Ma et al. (2019)"),
            dbc.Button("Select", color="primary", id="card3-button"),
            dbc.Button("Advanced", color="grey", id="advanced3-link", className="ms-2"),
        ]
    ),
    id = "card3"
)

# App layout
app.layout = dbc.Container([
    dcc.Store(id="target-data-store", storage_type="memory"),
    dcc.Store(id="coordinates-store", storage_type="memory"),
    dcc.Store(id="image-store", storage_type="memory"),
    dcc.Store(id="target-data-sub-store", storage_type="memory"),
    dcc.Store(id="second-data-store", storage_type="memory"),
    dcc.Store(id="second-coordinates-store", storage_type="memory"),
    dcc.Store(id="second-image-store", storage_type="memory"),
    html.H1("DeepHepatoScope", className="text-center mt-4 mb-4"),
    dcc.Store(id="second-data-sub-store", storage_type="memory"),
    
    # Pre-loaded dataset buttons
    dbc.Row([
        dbc.Col(html.H4("Choose a Pre-loaded Reference Dataset:"), width=12),
        dbc.Col(first_card, width=4),
        dbc.Col(second_card, width=4),
        dbc.Col(third_card, width=4),
        dcc.Store(id="selected-button", data=None),  # Store for tracking selected button
        html.Div(id="advanced-section", style={"display": "none", "overflow": "hidden", "height": "0px", "transition": "height 0.5s ease-in-out", "marginTop": "20px"}),
        dbc.Col(html.Div(id="card-error-message", className="text-danger mt-2")),
    ], className="mb-4"),

    dcc.Store(id="model-settings-store", storage_type="memory"),
    
    # Radio buttons to choose between RF or NN model
    dbc.Row([
        dbc.Col(html.H4("Choose a Model:")),
        dbc.Col(
            dbc.ButtonGroup(
                [
                    dbc.RadioItems(
                        id="model-selection",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-primary",
                        labelCheckedClassName="active",
                        options=[
                            {"label": "Random Forest", "value": "RF"},
                            {"label": "Neural Network", "value": "NN"}
                        ],
                        value="RF",  # Default selection
                        inline=True,  # Ensure it behaves as a group of buttons
                    ),
                ],
                size="sm",  # Optional: Control the size of the buttons
                style={"margin-top": "10px"}
            ),
        ),
    ], className="mb-4"),

    # Accordion for Advanced Settings - default shown for Random Forest
    dbc.Row([
        dbc.Col([
            # Random Forest Accordion
            dbc.Accordion([
                dbc.AccordionItem(
                    [
                        dbc.Row([
                            dbc.Col(html.Label("Number of Estimators (n_estimators):")),
                            dbc.Col(dcc.Input(id="n_estimators-input", value=default_rf_params["n_estimators"], type="number", min=1, step=1))
                        ]),
                        dbc.Row([
                            dbc.Col(html.Label("Maximum Depth (max_depth):")),
                            dbc.Col(dcc.Input(id="max_depth_RF-input", value=default_rf_params["max_depth"], type="number", min=1, step=1))
                        ]),
                        dbc.Row([
                            dbc.Col(html.Label("Min Samples Split (min_samples_split):")),
                            dbc.Col(dcc.Input(id="min_samples_split-input", value=default_rf_params["min_samples_split"], type="number", min=1, step=1))
                        ]),
                        dbc.Row([
                            dbc.Col(html.Label("Min Samples Leaf (min_samples_leaf):")),
                            dbc.Col(dcc.Input(id="min_samples_leaf-input", value=default_rf_params["min_samples_leaf"], type="number", min=1, step=1))
                        ])
                    ],
                    title="Advanced Random Forest Classifier Settings",
                    id="rf-accordion",
                ),
            ], id="rf-accordion-container", className="mb-4"),

            # Neural Network Accordion
            dbc.Accordion([
                dbc.AccordionItem(
                    [
                        dbc.Row([
                            dbc.Col(html.Label("Number of Hidden Layers:")),
                            dbc.Col(dcc.Input(id="hidden_layers-input", value=default_nn_params["hidden_layers"], type="number", min=1, step=1))
                        ]),
                        dbc.Row([
                            dbc.Col(html.Label("Neurons per Layer:")),
                            dbc.Col(dcc.Input(id="neurons_per_layer-input", value=default_nn_params["neurons_per_layer"], type="number", min=1, step=1))
                        ]),
                        dbc.Row([
                            dbc.Col(html.Label("Activation Function:")),
                            dbc.Col(dcc.Dropdown(
                                id="activation_function-input",
                                options=[
                                    {"label": "ReLU", "value": "relu"},
                                    {"label": "Sigmoid", "value": "sigmoid"},
                                    {"label": "Tanh", "value": "tanh"}
                                ],
                                value=default_nn_params["activation_function"]
                            ))
                        ])
                    ],
                    title="Advanced Neural Network Settings",
                    id="nn-accordion",
                ),
            ], id="nn-accordion-container", className="mb-4"),
        ])
    ], className="mb-4"),

    dcc.Store(id="lowres-scale-factor-store", storage_type="memory"),

    # Upload target dataset
    dbc.Row([
        dbc.Col([
            html.H4("Upload Target Dataset"),
            dcc.Upload(
                id="upload-scrna",
                children=html.Div(["Drag and Drop or Click to Upload Target .h5ad"]),
                style={"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px"},
                multiple=False
            ),
            dcc.Upload(
                id="upload-h5ad",
                children=html.Div(["Drag and Drop or Click to Upload Target .h5ad"]),
                style={"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "none"},
                multiple=False
            ),
            dcc.Upload(
                id="upload-coordinates",
                children=html.Div(["Drag and Drop or Click to Upload Coordinates .csv"]),
                style={"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "none"},
                multiple=False
            ),
            dcc.Upload(
                id="upload-image",
                children=html.Div(["Drag and Drop or Click to Upload Image .png"]),
                style={"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "none"},
                multiple=False
            ),
            dcc.Input(id="lowres-scale-factor", type="number", placeholder="Lowres scale factor", step="any", style={"display": "none"}),
            dbc.RadioItems(
                id="spatial-data-check",
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options=[
                    {"label": "scRNA-Seq", "value": "scrna"},
                    {"label": "Spatial Transcriptomics", "value": "spatial"},
                ],
                value="scrna",
            ),
            dbc.Progress(id="target-progress", striped=True, animated=True, style={"margin-top": "10px", "display": "none"}),
            html.P(id="target-upload-status", className="text-success"),
        ])
    ], className="mb-4"),

    dcc.Store(id="target-data-settings-store", storage_type="memory"),

    dbc.Row([
        dbc.Col([
            html.Div(
                dbc.Accordion(
        [
            dbc.AccordionItem(
                [
         dbc.Row([
              dbc.Col(html.Label("Number of variable genes to use in PCA:")),
               dbc.Col(dcc.Input(id="variable_genes-input", value=2000, type="number", min=1, step=1))
            ]),
            dbc.Row([
                dbc.Col(html.Label("Number of PCA dimensions to use in Clustering:")),
                dbc.Col(dcc.Input(id="pca_dims-input", value=50, type="number", min=1, step=1))
            ]),
            dbc.Row([
                dbc.Col(html.Label("Normalisation method:")),
                dbc.Col(dcc.Dropdown(id="normalisation-dropdown", options=[{"label": "Z-score", "value": "zscore"}, {"label": "LogNormalize", "value": "lognormalize"}], value="zscore"))
            ]),
        dbc.Row([
                dbc.Col(html.Label("What to calculate?")),
                dbc.Col(dcc.Checklist(id="calculation-checklist", options=[{"label": "UMAP", "value": "umap"}, {"label": "t-SNE", "value": "tsne"}])),
            ])
    ],
                title="Advanced Clustering Settings",
            ),
        ],
    )
)
        ])
    ], className="mb-4"),
    
    # Train and classify button
    dbc.Row([
        dbc.Col(html.Button("Train RF Model and Classify Target", id="train-button", className="btn btn-primary"), width=4),
        dbc.Col(dbc.Progress(id="train-progress", striped=True, animated=True, style={"margin-top": "10px", "display": "none"})),
        dbc.Col(html.P(id="classification-status", className="text-danger")),
    ], className="mb-4"),
    
    # t-SNE plot
    dcc.Store(id="umap-plot-store", storage_type="memory"),
    dcc.Store(id="tsne-plot-store", storage_type="memory"),
    dcc.Store(id="marker-gene-store", storage_type="memory"),

    dbc.Row([
        dbc.Col([
            html.H4("Data visualisation"),
            dbc.Accordion([
                dbc.AccordionItem(
                    [
                        dbc.Row([
                            dbc.Col(html.Pre(id="classification-report", className="mt-2", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"})),
                            dbc.Col(dcc.Graph(id="cm-plot")),
                            #dbc.Col(dcc.Graph(id="roc-auc-plot"))
                        ]),
                    ],
                    title="Advanced Classifier Results",
                    id="classifier-plots",
                ),
                dbc.AccordionItem(
                    [
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="heatmap-plot")),
                        ]),
                    ],
                    title="Advanced Target Data Analyses",
                    id="target-plots",
                ),
            ], id="classifier-plots-container", className="mb-4"),
            dcc.Graph(id="spatial-plot", style={'display': 'none'}),
            dcc.Graph(id="umap-plot", style={'display': 'none'}),
            dcc.Graph(id="tsne-plot", style={'display': 'none'}),
            dcc.Input(id="marker_gene-input", placeholder="Enter marker gene name", type="text"),
            dbc.Button("Show Canonical Marker Genes", id="marker-button", className="btn btn-secondary mt-2"),
            dbc.Button("Return to original plot", id="return-button", className="btn btn-secondary mt-2", style={'display': 'none'}),
            html.Div(id="marker-gene-status", className="mt-2")
        ])
    ]),
    dbc.Row([
    dbc.Col([
        html.H4("Downstream GSEA analyses"),
        dbc.DropdownMenu(
            label="Select reference dataset to classify second dataset",
            children=[
                dbc.DropdownMenuItem("GSE189903", id="second-GSE189903"),
                dbc.DropdownMenuItem("GSE151530", id="second-GSE151530"),
                dbc.DropdownMenuItem("GSE125449", id="second-GSE125449"),
            ],
        ),
        html.Div(id="second-reference-output-div"),
        dcc.Store(id="second-reference-store"),
        html.P(id="second-upload-status", className="text-success")
    ])
]),
    dcc.Store(id="second-model-settings-store", storage_type="memory"),
    dbc.Row([
        dbc.Col(html.H4("Choose a Model:")),
        dbc.Col(
            dbc.ButtonGroup(
                [
                    dbc.RadioItems(
                        id="second-model-selection",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-primary",
                        labelCheckedClassName="active",
                        options=[
                            {"label": "Random Forest", "value": "RF"},
                            {"label": "Neural Network", "value": "NN"}
                        ],
                        value="RF",  # Default selection
                        inline=True,  # Ensure it behaves as a group of buttons
                    ),
                ],
                size="sm",  # Optional: Control the size of the buttons
                style={"margin-top": "10px"}
            ),
        ),
    ], className="mb-4"),

    # Accordion for Advanced Settings - default shown for Random Forest
    dbc.Row([
        dbc.Col([
            # Random Forest Accordion
            dbc.Accordion([
                dbc.AccordionItem(
                    [
                        dbc.Row([
                            dbc.Col(html.Label("Number of Estimators (n_estimators):")),
                            dbc.Col(dcc.Input(id="n_estimators-input-2", value=default_rf_params["n_estimators"], type="number", min=1, step=1))
                        ]),
                        dbc.Row([
                            dbc.Col(html.Label("Maximum Depth (max_depth):")),
                            dbc.Col(dcc.Input(id="max_depth_RF-input-2", value=default_rf_params["max_depth"], type="number", min=1, step=1))
                        ]),
                        dbc.Row([
                            dbc.Col(html.Label("Min Samples Split (min_samples_split):")),
                            dbc.Col(dcc.Input(id="min_samples_split-input-2", value=default_rf_params["min_samples_split"], type="number", min=1, step=1))
                        ]),
                        dbc.Row([
                            dbc.Col(html.Label("Min Samples Leaf (min_samples_leaf):")),
                            dbc.Col(dcc.Input(id="min_samples_leaf-input-2", value=default_rf_params["min_samples_leaf"], type="number", min=1, step=1))
                        ])
                    ],
                    title="Advanced Random Forest Classifier Settings",
                    id="rf-accordion-2",
                ),
            ], id="rf-accordion-container-2", className="mb-4"),

            # Neural Network Accordion
            dbc.Accordion([
                dbc.AccordionItem(
                    [
                        dbc.Row([
                            dbc.Col(html.Label("Number of Hidden Layers:")),
                            dbc.Col(dcc.Input(id="hidden_layers-input-2", value=default_nn_params["hidden_layers"], type="number", min=1, step=1))
                        ]),
                        dbc.Row([
                            dbc.Col(html.Label("Neurons per Layer:")),
                            dbc.Col(dcc.Input(id="neurons_per_layer-input-2", value=default_nn_params["neurons_per_layer"], type="number", min=1, step=1))
                        ]),
                        dbc.Row([
                            dbc.Col(html.Label("Activation Function:")),
                            dbc.Col(dcc.Dropdown(
                                id="activation_function-input-2",
                                options=[
                                    {"label": "ReLU", "value": "relu"},
                                    {"label": "Sigmoid", "value": "sigmoid"},
                                    {"label": "Tanh", "value": "tanh"}
                                ],
                                value=default_nn_params["activation_function"]
                            ))
                        ])
                    ],
                    title="Advanced Neural Network Settings",
                    id="nn-accordion-2",
                ),
            ], id="nn-accordion-container-2", className="mb-4"),
        ])
    ], className="mb-4"),
    dcc.Store(id="second-lowres-scale-factor-store", storage_type="memory"),

    # Upload target dataset
    dbc.Row([
        dbc.Col([
            html.H4("Upload Target Dataset"),
            dcc.Upload(
                id="upload-scrna-2",
                children=html.Div(["Drag and Drop or Click to Upload Target .h5ad"]),
                style={"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px"},
                multiple=False
            ),
            dcc.Upload(
                id="upload-h5ad-2",
                children=html.Div(["Drag and Drop or Click to Upload Target .h5ad"]),
                style={"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "none"},
                multiple=False
            ),
            dcc.Upload(
                id="upload-coordinates-2",
                children=html.Div(["Drag and Drop or Click to Upload Coordinates .csv"]),
                style={"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "none"},
                multiple=False
            ),
            dcc.Upload(
                id="upload-image-2",
                children=html.Div(["Drag and Drop or Click to Upload Image .png"]),
                style={"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "none"},
                multiple=False
            ),
            dcc.Input(id="lowres-scale-factor-2", type="number", placeholder="Lowres scale factor", step="any", style={"display": "none"}),
            dbc.RadioItems(
                id="spatial-data-check-2",
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options=[
                    {"label": "scRNA-Seq", "value": "scrna"},
                    {"label": "Spatial Transcriptomics", "value": "spatial"},
                ],
                value="scrna",
            ),
            dbc.Progress(id="target-progress-2", striped=True, animated=True, style={"margin-top": "10px", "display": "none"}),
            html.P(id="target-upload-status-2", className="text-success"),
        ])
    ], className="mb-4"),

    dcc.Store(id="second-target-data-settings-store", storage_type="memory"),

    dbc.Row([
        dbc.Col([
            html.Div(
                dbc.Accordion(
        [
            dbc.AccordionItem(
                [
         dbc.Row([
              dbc.Col(html.Label("Number of variable genes to use in PCA:")),
               dbc.Col(dcc.Input(id="variable_genes-input-2", value=2000, type="number", min=1, step=1))
            ]),
            dbc.Row([
                dbc.Col(html.Label("Number of PCA dimensions to use in Clustering:")),
                dbc.Col(dcc.Input(id="pca_dims-input-2", value=50, type="number", min=1, step=1))
            ]),
            dbc.Row([
                dbc.Col(html.Label("Normalisation method:")),
                dbc.Col(dcc.Dropdown(id="normalisation-dropdown-2", options=[{"label": "Z-score", "value": "zscore"}, {"label": "LogNormalize", "value": "lognormalize"}], value="zscore"))
            ]),
        dbc.Row([
                dbc.Col(html.Label("What to calculate?")),
                dbc.Col(dcc.Checklist(id="calculation-checklist-2", options=[{"label": "UMAP", "value": "umap"}, {"label": "t-SNE", "value": "tsne"}])),
            ])
    ],
                title="Advanced Clustering Settings",
            ),
        ],
    )
)
        ])
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(html.Button("Train RF Model and Classify Target", id="train-button-2", className="btn btn-primary"), width=4),
        dbc.Col(dbc.Progress(id="train-progress-2", striped=True, animated=True, style={"margin-top": "10px", "display": "none"})),
        dbc.Col(html.P(id="classification-status-2", className="text-danger")),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.H4("Data visualisation"),
            dbc.Accordion([
                dbc.AccordionItem(
                    [
                        dbc.Row([
                            dbc.Col(html.Pre(id="classification-report-2", className="mt-2", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"})),
                            dbc.Col(dcc.Graph(id="cm-plot-2")),
                            #dbc.Col(dcc.Graph(id="roc-auc-plot"))
                        ]),
                    ],
                    title="Advanced Classifier Results",
                    id="classifier-plots-2",
                ),
                dbc.AccordionItem(
                    [
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="heatmap-plot-2")),
                        ]),
                    ],
                    title="Advanced Target Data Analyses",
                    id="target-plots-2",
                ),
            ], id="classifier-plots-container-2", className="mb-4"),
            dcc.Graph(id="spatial-plot-2", style={'display': 'none'}),
            dcc.Graph(id="umap-plot-2", style={'display': 'none', 'width': '100%', 'height': '800px'}),
            dcc.Graph(id="tsne-plot-2", style={'display': 'none', 'width': '100%', 'height': '800px'}),
        ])
    ]),
    dcc.Store(id="analysis-results"),  # Store computed results
    dbc.Row([
        html.H3("Class Comparison and Analysis"),
        dcc.Store(id="classes-store"),  # Store the classes in a hidden state
        dcc.Dropdown(id="class-selector", placeholder="Select a class to view", className="mb-4"),
        dbc.Button("Update Classes", id="update-classes-btn", n_clicks=0, className="mb-4"),
        dbc.Button("Run Analysis", id="run-analysis-btn", n_clicks=0, className="mb-4"),
        html.Div([
            html.H4("DEG Heatmap"),
            dcc.Graph(id="heatmap-graph"),
        ]),
        html.Div([
            html.H4("GSEA Plot"),
            html.Iframe(
                    src=None,
                    style={'width': '50%', 'height': '300px'},
                    id="gsea-image"
                ),
        ]),
    ])

])

# Callbacks
@app.callback(
    [
        Output("card1-button", "color"),
        Output("card1-button", "children"),
        Output("card2-button", "color"),
        Output("card2-button", "children"),
        Output("card3-button", "color"),
        Output("card3-button", "children"),
        Output("card-error-message", "children"),
        Output("selected-button", "data"),
    ],
    [Input("card1-button", "n_clicks"), Input("card2-button", "n_clicks"), Input("card3-button", "n_clicks")],
    State("selected-button", "data"),
    prevent_initial_call=True,
)
def toggle_ref_datasets(card1_clicks, card2_clicks, card3_clicks, selected_button):
    # Identify which button triggered the callback
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Logic for toggling buttons
    if triggered_id == "card1-button":
        if selected_button == "card1-button":  # Unselect card1
            return "primary", "Select", "primary", "Select", "primary", "Select", None, None
        elif selected_button in ["card2-button", "card3-button"]:  # Switching to card1
            return (
                "success",
                "Unselect",
                "primary",
                "Select",
                "primary",
                "Select",
                None,
                "card1-button",
            )
        elif selected_button is None:  # Selecting card1
            return "success", "Unselect", "primary", "Select", "primary", "Select", None, "card1-button"

    elif triggered_id == "card2-button":
        if selected_button == "card2-button":  # Unselect card2
            return "primary", "Select", "primary", "Select", "primary", "Select", None, None
        elif selected_button in ["card1-button", "card3-button"]:  # Switching to card2
            return (
                "primary",
                "Select",
                "success",
                "Unselect",
                "primary",
                "Select",
                None,
                "card2-button",
            )
        elif selected_button is None:  # Selecting card2
            return "primary", "Select", "success", "Unselect", "primary", "Select", None, "card2-button"

    elif triggered_id == "card3-button":
        if selected_button == "card3-button":  # Unselect card3
            return "primary", "Select", "primary", "Select", "primary", "Select", None, None
        elif selected_button in ["card1-button", "card2-button"]:  # Switching to card3
            return (
                "primary",
                "Select",
                "primary",
                "Select",
                "success",
                "Unselect",
                None,
                "card3-button",
            )
        elif selected_button is None:  # Selecting card3
            return "primary", "Select", "primary", "Select", "success", "Unselect", None, "card3-button"

    return "primary", "Select", "primary", "Select", "primary", "Select", None, None  # Default state

@app.callback(
    Output("advanced-section", "children"),
    Output("advanced-section", "style"),
    [Input("advanced1-link", "n_clicks"),
     Input("advanced2-link", "n_clicks"),
     Input("advanced3-link", "n_clicks")]
)
def show_advanced_section(n_clicks1, n_clicks2, n_clicks3):
    # Determine which "Advanced" link was clicked
    triggered_id = dash.callback_context.triggered_id
    n_clicks = 0  # Default value for n_clicks
    if triggered_id == "advanced1-link":
        n_clicks = n_clicks1
    elif triggered_id == "advanced2-link":
        n_clicks = n_clicks2
    elif triggered_id == "advanced3-link":
        n_clicks = n_clicks3

    # Hide section if n_clicks is even, show if odd
    if n_clicks % 2 == 0:
        return None, {"display": "none", "overflow": "hidden", "height": "0px", "transition": "height 0.5s ease-in-out"}
    else:
        # Show content based on the clicked link
        advanced_text = ""
        if triggered_id == "advanced1-link":
            advanced_text = "52,789 cells from 3 HCC and 4 iCCA patients; all before treatment"
        elif triggered_id == "advanced2-link":
            advanced_text = "112,506 cells from 46 total HCC and iCCA patients; before/after treatment"
        elif triggered_id == "advanced3-link":
            advanced_text = "57,567 cells from 9 HCC and 2 iCCA patients; all after treatment"

        # Return the text and style to make the section visible with swipe down effect
        return html.Div([html.P(advanced_text)]), {"display": "block", "height": "auto", "overflow": "visible", "transition": "height 0.5s ease-in-out"}

@app.callback(
    [Output("rf-accordion", "style"),
     Output("nn-accordion", "style")],
    [Input("model-selection", "value")]
)
def update_accordion_visibility(selected_model):
    if selected_model == "RF":
        return {"display": "block"}, {"display": "none"}
    elif selected_model == "NN":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"} # Default to showing RF

@app.callback(
    [Output("upload-scrna", "style", allow_duplicate=True),
     Output("upload-h5ad", "style"),
     Output("upload-coordinates", "style"),
     Output("upload-image", "style"),
     Output("lowres-scale-factor", "style")],
    [Input("spatial-data-check", "value")],
    prevent_initial_call=True
)
def update_upload_visibility(data_type):
    if data_type == "scrna":
        #must set the entire style block, else if just set display only, it will render as one line
        return {"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}
    elif data_type == "spatial":
        return {"display": "none"}, {"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "block"}, {"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "block"}, {"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "block"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}

@app.callback(
    Output("lowres-scale-factor-store", "data"),
    Input("lowres-scale-factor", "value")
)
def update_lowres_scale_factor_settings(lowres_scale_factor):
    # Create dictionary to store the settings
    return {"lowres_scale_factor": lowres_scale_factor}
    
@app.callback(
    Output("model-settings-store", "data"),
    [Input("model-selection", "value"),
     Input("n_estimators-input", "value"),
     Input("max_depth_RF-input", "value"),
     Input("min_samples_split-input", "value"),
     Input("min_samples_leaf-input", "value"),
     Input("hidden_layers-input", "value"),
     Input("neurons_per_layer-input", "value"),
     Input("activation_function-input", "value")]
)
def update_model_settings(model_type, n_estimators, max_depth, min_samples_split, min_samples_leaf, hidden_layers, neurons_per_layer, activation_function):
    # Create dictionary to store the settings
    if model_type == "RF":
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }
    elif model_type == "NN":
        return {
            "hidden_layers": hidden_layers,
            "neurons_per_layer": neurons_per_layer,
            "activation_function": activation_function
        }

#upload scrna data
@app.callback(
    Output("target-upload-status", "children", allow_duplicate = True),
    Output("upload-scrna", "children"),
    Output("upload-scrna", "style", allow_duplicate=True),
    Output("upload-scrna", "disabled"),
    Output("target-data-store", "data", allow_duplicate=True),
    Input("upload-scrna", "contents"),
    State("upload-scrna", "filename"), #file must be in the same directory as this script
    prevent_initial_call = True,
)
def upload_target(contents, filename):
    if contents:
        target_data = sc.read_h5ad(filename)
        target_data_dict = encode_anndata(target_data)
        return f"Uploaded Target File: {filename}", "Dataset uploaded successfully!", {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "solid",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
            "backgroundColor": "green",  # Change background to green
            "color": "darkgreen"
        }, True, {"target_data": target_data_dict} # Disable the upload box"
    return "No target file uploaded."

#Upload spatial data
@app.callback(
    Output("target-upload-status", "children", allow_duplicate = True),
    Output("upload-h5ad", "children"),
    Output("upload-h5ad", "style", allow_duplicate=True),
    Output("upload-h5ad", "disabled"),
    Output("target-data-store", "data", allow_duplicate=True),
    Input("upload-h5ad", "contents"),
    State("upload-h5ad", "filename"),
    prevent_initial_call = True,
)
def upload_h5ad(contents, filename):
    if contents:
        target_data = sc.read_h5ad(filename)
        target_data_dict = encode_anndata(target_data)
        return f"Uploaded Target File: {filename}", "h5ad dataset uploaded successfully!", {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "solid",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
            "backgroundColor": "green",  # Change background to green
            "color": "darkgreen"
        }, True, {"target_data": target_data_dict}   # Disable the upload box, do not put orient="records"!!!
    return "No target file uploaded."

@app.callback(
    Output("target-upload-status", "children", allow_duplicate = True),
    Output("upload-coordinates", "children"),
    Output("upload-coordinates", "style", allow_duplicate=True),
    Output("upload-coordinates", "disabled"),
    Output("coordinates-store", "data", allow_duplicate=True),
    Input("upload-coordinates", "contents"),
    State("upload-coordinates", "filename"),
    prevent_initial_call = True,
)
def upload_coordinates(contents, filename):
    if contents:
        target_coordinates = pd.read_csv(filename, index_col=0)
        target_coordinates_dict = target_coordinates.to_dict() 
        return f"Uploaded Target File: {filename}", "Coordinates dataset uploaded successfully!", {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "solid",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
            "backgroundColor": "green",  # Change background to green
            "color": "darkgreen"
        }, True, {"target_coordinates": target_coordinates_dict}  # Disable the upload box
    return "No target file uploaded."

@app.callback(
    Output("target-upload-status", "children", allow_duplicate=True),
    Output("upload-image", "children"),
    Output("upload-image", "style", allow_duplicate=True),
    Output("upload-image", "disabled"),
    Output("image-store", "data", allow_duplicate=True),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    prevent_initial_call=True,
)
def upload_image(contents, filename):
    if contents:
        # Decode the Base64 image
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        pil_image = Image.open(io.BytesIO(decoded))
        flipped_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
        buffer = io.BytesIO()
        flipped_image.save(buffer, format="PNG")  # Ensure format is set
        buffer.seek(0)
        target_image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        # Return success message and style
        return (
            f"Uploaded Target File: {filename}",
            "Image uploaded successfully!",
            {
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "solid",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
                "backgroundColor": "green",  # Change background to green
                "color": "darkgreen",
            },
            True,  # Disable the upload box
            {"pil_image": target_image_base64},  # Store image as Base64
        )

@app.callback(
    Output("target-upload-status", "children"),
    Output("target-progress", "value"),
    Output("target-progress", "style"),
    Input("upload-scrna", "contents"),
    State("upload-scrna", "filename"),
    prevent_initial_call=True
)
def upload_target(contents, filename):
    if not contents:
        return "No file uploaded.", 0, {"display": "none"}
    try:
        # Simulate progress
        for i in range(1, 11):
            time.sleep(0.3)
        decoded = base64.b64decode(contents.split(",")[1])
        with open(filename, "wb") as f:
            f.write(decoded)
        data = sc.read_h5ad(filename)
        return f"Uploaded {filename}.", 100, {"display": "block"}
    except Exception as e:
        return f"Error reading file: {e}", 0, {"display": "block"}
    
@app.callback(
    Output("target-data-settings-store", "data"),
    [Input("variable_genes-input", "value"),
     Input("pca_dims-input", "value"),
     Input("normalisation-dropdown", "value"),
     Input("calculation-checklist", "value")]
)
def update_model_settings(variable_genes, pca_dims, normalisation, calculation):
    # Create dictionary to store the settings
    return {
        "variable_genes": variable_genes,
        "pca_dims": pca_dims,
        "normalisation": normalisation,
        "calculation": calculation
    }

@app.callback(
    Output("marker-gene-store", "data"),
    Input("marker_gene-input", "value")
)
def update_model_settings(marker_gene):
    # Create dictionary to store the settings
    return {
        "marker_gene": marker_gene,
    }


@app.callback(
    Output("train-progress", "value"),
    Output("train-progress", "style"),
    Output("classification-status", "children"),
    Output("heatmap-plot", "figure"),
    Output("classification-report", "children"),
    Output("cm-plot", "figure"),
    Output("spatial-plot", "figure"),
    Output("spatial-plot", "style"),
    Output("umap-plot-store", "data"),
    Output("tsne-plot-store", "data"),
    Output("umap-plot", "style"),
    Output("tsne-plot", "style"),
    Output("umap-plot", "figure", allow_duplicate=True),
    Output("tsne-plot", "figure", allow_duplicate=True),
    Output("target-data-sub-store", "data", allow_duplicate=True),
    Input("train-button", "n_clicks"),
    State("model-selection", "value"),
    State("selected-button", "data"),
    State("spatial-data-check", "value"),
    State("lowres-scale-factor-store", "data"),
    State("model-settings-store", "data"),
    State("target-data-settings-store", "data"),
    State("target-data-store", "data"),
    State("coordinates-store", "data"),
    State("image-store", "data"),
    prevent_initial_call=True
)
def train_model(n_clicks, selected_model, selected_button, spatial_data_check, lowres_scale_factor_dict, model_settings, target_data_settings, target_data_store, coordinates_store, image_store):
    try:
        if selected_button == "card1-button":
            reference_data = sc.read_h5ad("GSE189903_filtered.h5ad")
        elif selected_button == "card2-button":
            reference_data = sc.read_h5ad("GSE151530_filtered.h5ad")
        elif selected_button == "card3-button":
            reference_data = sc.read_h5ad("GSE125449_filtered.h5ad")
        elif selected_button == None:
            return "Please select a reference dataset."
        target_data_dict = target_data_store["target_data"]
        target_data = decode_dict(target_data_dict)
        if not target_data:
            return "Please upload a target dataset."
        if n_clicks is None:
            return "Click the 'Train RF Model and Classify Target' button."
        if model_settings is None:
            return "Model settings are missing."

        # Find common genes
        common_genes = reference_data.var_names.intersection(target_data.var_names)
        reference_data_sub = reference_data[:, common_genes]
        target_data_sub = target_data[:, common_genes]

        # Prepare features and labels
        X_ref = reference_data_sub.X.toarray()
        y_ref = reference_data_sub.obs["Type"]  # Replace 'Class' with actual label column

        X_target = target_data_sub.X.toarray()

        # Normalize
        X_ref_normalized = normalize(X_ref, axis=1)
        X_target_normalized = normalize(X_target, axis=1)

        # Train Random Forest model
        if selected_model == "RF":
            X_train, X_test, y_train, y_test = train_test_split(
                X_ref_normalized, y_ref, test_size=0.2, random_state=42, stratify=y_ref
            )
            clf = RandomForestClassifier(**model_settings)  # Use the parameters from dcc.Store
            clf.fit(X_train, y_train)
            y_test_labels = y_test #Standardise to this variable name
            #Predict on test dataset
            predicted_classes_test = clf.predict(X_test) #Standardise to this variable name
            report = classification_report(y_test_labels, predicted_classes_test, target_names=clf.classes_) #keep the variable name "report" consistent

            #Predict on target dataset
            predicted_classes = clf.predict(X_target_normalized)

        elif selected_model == "NN":
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_ref)  
            if not isinstance(reference_data.X, csr_matrix):
                X = csr_matrix(reference_data.X)
            else:
                X = reference_data.X
            y = reference_data.obs.apply(lambda row: f"{row['Type']}", axis=1).values
            n_cells = X.shape[0]
            n_sample = int(n_cells * 100 / 100)
            sampled_indices = np.random.choice(n_cells, size=n_sample, replace=False)
            
            X = X[sampled_indices, :]
            Y = y[sampled_indices]
            var_names = reference_data.var_names
            # Normalize the data
            X_normalized = normalize(X, axis=1, norm='l2')

            # --- Train-Test Split ---
            print("Splitting data into training and testing sets...")
            X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42, stratify=y)

            # One-hot encode the labels for neural network
            unique_labels = np.unique(y_train)
            y_train_encoded = to_categorical([np.where(unique_labels == label)[0][0] for label in y_train], num_classes=len(unique_labels))
            y_test_encoded = to_categorical([np.where(unique_labels == label)[0][0] for label in y_test], num_classes=len(unique_labels))

            # --- Neural Network Configuration ---
            num_epochs = 3 #TO CUSTOMISE
            num_layers = 3
            layer_nodes = []
            nodes = [128, 64, 32]
            for i in range(num_layers):
                layer_nodes.append(nodes[i])

            # --- Neural Network Model ---
            print("Building and training the neural network...")
            model = Sequential()

            # Add input layer and first hidden layer
            model.add(Dense(layer_nodes[0], activation='relu', input_shape=(X_train.shape[1],)))

            # Add additional hidden layers
            for nodes in layer_nodes[1:]:
                model.add(Dense(nodes, activation='relu'))

            # Add output layer
            model.add(Dense(len(unique_labels), activation='softmax'))

            model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

            # Train the model
            history = model.fit(X_train, y_train_encoded, epochs=num_epochs, batch_size=32, validation_split=0.2, verbose=1)

            # Evaluate the model
            loss, accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)

            # Predict on the test set
            y_pred_prob = model.predict(X_test)
            y_pred = np.argmax(y_pred_prob, axis=1)
            predicted_classes_test = y_pred
            y_test_labels = np.argmax(y_test_encoded, axis=1)

            # Predict on target data
            if not isinstance(target_data.X, csr_matrix):
                X_target_NN = csr_matrix(target_data.X)
            else:
                X_target_NN = target_data.X
            # Normalize the data
            X_target_normalized_NN = normalize(X_target_NN, axis=1, norm='l2')
            # Predicted classes for multi-class classification
            predicted_probs = model.predict(X_target_normalized_NN)  # Predict probabilities for each class
            predicted_indices = np.argmax(predicted_probs, axis=1)  # Get the index of the max probability
            predicted_classes = label_encoder.inverse_transform(predicted_indices)  # Map indices to class names

            # Calculate accuracy and display classification report
            report = classification_report(y_test_labels, predicted_classes_test, target_names=unique_labels)

        target_data_sub.obs['Predicted_Class'] = predicted_classes
        class_counts = target_data_sub.obs['Predicted_Class'].value_counts()
        print(class_counts)

        # Heatmap
        # Create gene expression DataFrame
        gene_expression = pd.DataFrame(
            X_target_normalized,
            index=target_data_sub.obs_names,  # Rows: Cells
            columns=target_data_sub.var_names  # Columns: Genes
        )
        gene_expression["Predicted_Class"] = predicted_classes

        # Calculate mean expression for each class
        class_means = gene_expression.groupby("Predicted_Class").mean()

        # Identify the 10 most differentially expressed genes per class
        selected_genes = []
        gene_class_map = []  # To track which genes belong to which class

        for cls in class_means.index:
            diff_expr = class_means.loc[cls] - class_means.mean()
            top_genes = diff_expr.abs().nlargest(10).index.tolist()
            selected_genes.extend(top_genes)  # List of just gene names
            gene_class_map.extend([(gene, cls) for gene in top_genes])  # List of (gene, class) pairs

        # Convert gene_class_map to a DataFrame
        selected_genes_df = pd.DataFrame(gene_class_map, columns=["Gene", "Class"])

        # Ensure selected genes are valid and align with the data
        valid_genes = [gene for gene in selected_genes_df["Gene"] if gene in gene_expression.columns]
        selected_genes_df = selected_genes_df[selected_genes_df["Gene"].isin(valid_genes)]

        # Subset and normalize the data for selected genes
        heatmap_data = gene_expression[valid_genes].assign(Predicted_Class=gene_expression["Predicted_Class"])

        # Extract the gene expression data (numerical data) and store the non-numeric columns (e.g., 'Predicted_Class')
        gene_expression_data = heatmap_data.drop(columns=["Predicted_Class"])  # Remove non-numeric columns
        class_labels = heatmap_data["Predicted_Class"]  # Store class labels separately

        # Normalize the numeric gene expression data
        heatmap_data_normalized = pd.DataFrame(
            normalize(gene_expression_data, axis=0),  # Normalize only numeric columns
            columns=gene_expression_data.columns,     # Maintain original column names (genes)
            index=gene_expression_data.index          # Keep the original index
        )

        # Re-add the 'Predicted_Class' column after normalization
        heatmap_data_normalized["Predicted_Class"] = class_labels

        # Reorder the genes by their associated class
        heatmap_data_ordered = heatmap_data_normalized

        # Prepare data for combined heatmap
        long_format_data = heatmap_data_ordered.melt(
            id_vars=["Predicted_Class"],
            var_name="Gene",
            value_name="Expression",
        )

        pivot_table = long_format_data.pivot_table(
            index="Gene",
            columns="Predicted_Class",
            values="Expression",
            aggfunc="mean"
        )


        # Plot heatmap with ordered genes
        heatmap_figure = px.imshow(
            pivot_table,
            labels={"x": "Predicted Class", "y": "Genes", "color": "Expression"},
            color_continuous_scale="Viridis",
            title="Combined Heatmap of Differentially Expressed Genes by Class",
        )

        # Update the layout
        heatmap_figure.update_layout(
            height=800,
            width=1200,
            yaxis={
                "categoryorder": "array",
                "categoryarray": valid_genes,  # Align genes by their class order
            },
        )

        #Confusion matrix
        classes = sorted(set(y_test_labels).union(set(predicted_classes_test)))
        cm = confusion_matrix(y_test_labels, predicted_classes_test)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_figure = go.Figure(data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale='Blues',
            colorbar=dict(title='Count'),
            hoverongaps=False
        ))

        cm_figure.update_layout(
            title='Confusion Matrix',
            xaxis=dict(title='Predicted label'),
            yaxis=dict(title='True label'),
            autosize=True
        )

        # Run PCA
        if not np.issubdtype(target_data_sub.X.dtype, np.floating):
            target_data_sub.X = target_data_sub.X.astype(np.float64)
        sc.pp.highly_variable_genes(target_data_sub, n_top_genes = target_data_settings["variable_genes"])
        pca = PCA(n_components=target_data_settings["pca_dims"], random_state=42)
        X_pca = pca.fit_transform(X_target_normalized)

        # Run t-SNE
        if target_data_settings["calculation"] == ['umap']:
            reduction = umap.UMAP(n_components=2, random_state=42)
            X_reduction = reduction.fit_transform(X_pca)
        elif target_data_settings["calculation"] == ['tsne']:
            reduction = TSNE(n_components=2, random_state=42, perplexity=30)
            X_reduction = reduction.fit_transform(X_pca)
        else:
            print(target_data_settings["calculation"])
            return 0, {"display": "block"}, "No reduction selected", None, None, {"display": "none"}, {}


        # Add t-SNE coordinates to the AnnData object
        target_data.obsm['X_reduction'] = X_reduction

        # If predicted_classes is a numpy array, convert it to pandas Categorical if necessary
        if isinstance(predicted_classes, np.ndarray):
            predicted_classes = pd.Categorical(predicted_classes)

        # Get the class names
        class_labels = predicted_classes.categories  # Get class names
        class_codes = predicted_classes.codes  # Convert to numerical codes for coloring

        # Initialize an empty list to store traces
        traces = []

        # Create a separate trace for each class
        for class_code, class_label in enumerate(class_labels):
            mask = (class_codes == class_code)  # Filter points belonging to the current class
            trace = go.Scatter(
                x=X_reduction[mask, 0],
                y=X_reduction[mask, 1],
                mode='markers',
                marker=dict(
                    color=class_code,  # Assign a unique color for the class
                    colorscale='electric',  # Choose a colorscale
                    showscale=False,  # Do not display the color bar
                    size=3,
                ),
                name=class_label,  # Use the class name in the legend
                text=[class_label] * mask.sum(),  # Add class name as hover text
                hoverinfo='text',  # Display class name when hovering
            )
            traces.append(trace)

        # Create the layout for the plot
        if target_data_settings["calculation"] == ['umap']:
            layout = go.Layout(
                title="UMAP Visualization of Predicted Classes",
                xaxis=dict(title='UMAP1'),
                yaxis=dict(title='UMAP2'),
                showlegend=True,  # Show legend
                legend=dict(title='Cell Types', traceorder='normal'),  # Title for the legend
                height=800
            )
        elif target_data_settings["calculation"] == ['tsne']:
            layout = go.Layout(
                title="t-SNE Visualization of Predicted Classes",
                xaxis=dict(title='t-SNE1'),
                yaxis=dict(title='t-SNE2'),
                showlegend=True,  # Show legend
                legend=dict(title='Cell Types', traceorder='normal'),  # Title for the legend
                height=800
            )

        # Create the figure
        figure = go.Figure(data=traces, layout=layout)

        if spatial_data_check == "spatial":
            target_coordinates_dict = coordinates_store["target_coordinates"]
            target_coordinates = pd.DataFrame(target_coordinates_dict)
            content_string = image_store["pil_image"]
            decoded_data = base64.b64decode(content_string)
            pil_image = Image.open(io.BytesIO(decoded_data))

            # Map each predicted class to a unique color
            unique_classes = target_data_sub.obs['Predicted_Class'].unique()
            print("unique_classes:", unique_classes)
            class_colors = {cls: color for cls, color in zip(unique_classes, px.colors.qualitative.Set1)}
            print("class_colors:", class_colors)

            # Add colors to the target coordinates DataFrame
            target_coordinates['scaled_x'] = target_coordinates['imagecol'] * lowres_scale_factor_dict["lowres_scale_factor"]
            target_coordinates['scaled_y'] = target_coordinates['imagerow'] * lowres_scale_factor_dict["lowres_scale_factor"]
            print("1", target_data_sub.obs['Predicted_Class'])
            target_coordinates['class'] = target_data_sub.obs['Predicted_Class']
            target_coordinates['color'] = target_coordinates['class'].map(class_colors)
            #target_coordinates['color'] = target_coordinates['class'].apply(lambda x: class_colors[x])
            print(target_coordinates[['class', 'color']])

            # Create scatter plot for classified spatial data
            spatial_scatter = go.Scatter(
                x=target_coordinates['scaled_x'],
                y=target_coordinates['scaled_y'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=target_coordinates['color'],  # Use the mapped class colors
                    showscale=False  # Disable colorbar since we have discrete classes
                ),
                text=target_coordinates['class'],  # Display class labels on hover
                name='Spots'
            )

            # Layout for the spatial plot
            print(type(pil_image), pil_image)
            spatial_layout = go.Layout(
                images=[dict(
                    source=f"data:image/png;base64,{content_string}",
                    #source=pil_image,
                    x=0,
                    y=0,
                    xref="x",
                    yref="y",
                    sizex=pil_image.width, #without * lowres_scale_factor_dict["lowres_scale_factor"]
                    sizey=pil_image.height,
                    xanchor="left",
                    yanchor="bottom",
                    opacity=0.5,
                    layer="below"
                )],
                xaxis=dict(
                    scaleanchor="y",
                    showgrid=False,
                    zeroline=False,
                    visible=False
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    visible=False
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                title="Spatial Plot of Predicted Classes"
            )

            # Combine scatter and layout into a figure
            spatial_figure = go.Figure(data=[spatial_scatter], layout=spatial_layout)

            target_data_sub_dict = encode_anndata(target_data_sub)

            # Return figure with additional metadata if necessary
            if target_data_settings["calculation"] == ['umap']:
                return 100, {"display": "block"}, "Model trained successfully.", heatmap_figure, report, cm_figure, spatial_figure, {"display": "block"}, figure, None, {"display": "block"}, {"display": "none"}, figure, None, {'target_data_sub': target_data_sub_dict}
            elif target_data_settings["calculation"] == ['tsne']:
                return 100, {"display": "block"}, "Model trained successfully.", heatmap_figure, report, cm_figure, spatial_figure, {"display": "block"}, None, figure, {'display': 'none'}, {"display": "block"}, None, figure, {'target_data_sub': target_data_sub_dict}
        target_data_sub_dict = encode_anndata(target_data_sub)
        if target_data_settings["calculation"] == ['umap']:
            return 100, {"display": "block"}, "Model trained successfully.", heatmap_figure, report, cm_figure, None, {"display": "none"}, figure, None, {"display": "block"}, {"display": "none"}, figure, None, {'target_data_sub': target_data_sub_dict}
        elif target_data_settings["calculation"] == ['tsne']:
            return 100, {"display": "block"}, "Model trained successfully.", heatmap_figure, report, cm_figure, None, {"display": "none"}, None, figure, {"display": "none"}, {"display": "block"}, None, figure, {'target_data_sub': target_data_sub_dict}

    except Exception as e:
        full_traceback = traceback.format_exc()
        return 0, {"display": "block"}, f"Error: {e}\nDetails:\n{full_traceback}", None, None, None, None, {"display": "none"}, None, None, {"display": "none"}, {"display": "none"}, None, None, {'target_data_sub': None}

@app.callback(
    Output("umap-plot", "figure", allow_duplicate=True),
    Output("tsne-plot", "figure", allow_duplicate=True),
    Output("marker-gene-status", "children"),
    Output("return-button", "style", allow_duplicate=True),
    Input("marker-button", "n_clicks"),
    State("marker_gene-input", "value"),
    State("umap-plot", "figure"),
    State("tsne-plot", "figure"),
    State("target-data-store", "data"),
    prevent_initial_call=True,
)
def update_umap_plot(n_clicks, gene_name, current_umap, current_tsne, target_data_store):
    target_data = target_data_store["target_data"]
    print("Checking...")
    if not gene_name:
        return current_umap, current_tsne, "Please enter a valid gene name.", {'display': 'none'}
    
    if gene_name not in target_data.var_names:
        print(target_data.var_names)
        return current_umap, current_tsne, f"Gene '{gene_name}' not found in the dataset.", {'display': 'none'}
    
    # Update the marker color in the current figure
    print("Calculating...")

    # Determine which figure is being used (UMAP or t-SNE)
    if current_umap is not None:
        current_figure = current_umap
    elif current_tsne is not None:
        current_figure = current_tsne

    # Retrieve the gene expression values
    gene_expression = target_data[:, gene_name].X.toarray().flatten() if issparse(target_data.X) else target_data[:, gene_name].X.flatten()

    # Update the color of each trace
    for trace in current_figure["data"]:
        trace["marker"]["color"] = gene_expression  # Update the color to reflect the gene expression
        trace["marker"]["showscale"] = True
        # Ensure the colorbar exists and set properties
        if "colorbar" not in trace["marker"]:
            trace["marker"]["colorbar"] = {}

        # Define the colorbar properties
        trace["marker"]["colorbar"].update({
            "title": f"Expression Level of {gene_name}",
            "titleside": "right",  # Position the title on the right side of the colorbar
            "ticks": "outside",    # Show ticks on the outside of the colorbar
            "ticklen": 5,          # Set the length of the ticks
            "tickvals": [min(gene_expression), max(gene_expression)],  # Set tick values based on the range of the gene expression
            "ticktext": [f'{min(gene_expression):.2f}', f'{max(gene_expression):.2f}']  # Show min and max gene expression values on the ticks
        })

    # Update the title of the plot to reflect the gene expression being shown
    current_figure["layout"]["title"]["text"] = f"Expression Levels of {gene_name}"

    # Update the layout to make the colorbar visible and adjust legend
    current_figure["layout"].update(
        showlegend=False,  # Hide the legend to avoid clutter
        coloraxis_colorbar_title=f"Expression Level of {gene_name}",  # Set the colorbar title for the layout
        coloraxis_colorbar_ticks="outside",  # Ensure ticks are visible outside the colorbar
        coloraxis_colorbar_ticklen=5,  # Set the tick length on the colorbar
        coloraxis_colorbar_tickvals=[min(gene_expression), max(gene_expression)],  # Set tick values
        coloraxis_colorbar_ticktext=[f'{min(gene_expression):.2f}', f'{max(gene_expression):.2f}']  # Min and max gene expression values on ticks
    )

    print("Done!")

    if current_umap is not None:
        return current_figure, None, "", {'display': 'block'}
    elif current_tsne is not None:
        return None, current_figure, "", {'display': 'block'}

@app.callback(
    Output("umap-plot", "figure", allow_duplicate=True),
    Output("tsne-plot", "figure", allow_duplicate=True),
    Output("return-button", "style", allow_duplicate=True),
    Input("return-button", "n_clicks"),
    State("umap-plot-store", "data"),  # Assuming the original figure is stored in dcc.Store
    State("tsne-plot-store", "data"),
    prevent_initial_call=True,
)
def reset_to_original_plot(n_clicks, original_umap, original_tsne):
    print("Resetting...")
    if original_umap is not None:
        return original_umap, None, {'display': 'none'}
    elif original_tsne is not None:
        return None, original_tsne, {'display': 'none'}

#---SECOND DATASET---
@app.callback(
    Output("second-reference-store", "data"),  # Save selected value to dcc.Store
    Output("second-reference-output-div", "children"),  # Optional: Display selected value
    [
        Input("second-GSE189903", "n_clicks"),
        Input("second-GSE151530", "n_clicks"),
        Input("second-GSE125449", "n_clicks"),
    ],
    [State("second-reference-store", "data")],
)
def update_selected_value(n1, n2, n3, current_data):
    # Map each item to its corresponding dataset value
    options = {
        "second-GSE189903": "GSE189903",
        "second-GSE151530": "GSE151530",
        "second-GSE125449": "GSE125449",
    }
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_data, "No selection yet"
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id in options:
        selected_value = options[triggered_id]
        return selected_value, f"Selected: {selected_value}"
    return current_data, "No selection yet"

@app.callback(
    [Output("rf-accordion-2", "style"),
     Output("nn-accordion-2", "style")],
    [Input("second-model-selection", "value")]
)
def update_accordion_visibility(selected_model):
    if selected_model == "RF":
        return {"display": "block"}, {"display": "none"}
    elif selected_model == "NN":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}  # Default to showing RF

@app.callback(
    [Output("upload-scrna-2", "style", allow_duplicate=True),
     Output("upload-h5ad-2", "style"),
     Output("upload-coordinates-2", "style"),
     Output("upload-image-2", "style"),
     Output("lowres-scale-factor-2", "style")],
    [Input("spatial-data-check-2", "value")],
    prevent_initial_call=True
)
def update_upload_visibility(data_type):
    if data_type == "scrna":
        #must set the entire style block, else if just set display only, it will render as one line
        return {"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}
    elif data_type == "spatial":
        return {"display": "none"}, {"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "block"}, {"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "block"}, {"width": "100%", "height": "60px", "lineHeight": "60px",
                       "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                       "textAlign": "center", "margin": "10px", "display": "block"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}

@app.callback(
    Output("second-lowres-scale-factor-store", "data"),
    Input("lowres-scale-factor-2", "value")
)
def update_lowres_scale_factor_settings(lowres_scale_factor):
    # Create dictionary to store the settings
    return {"lowres_scale_factor": lowres_scale_factor}
    
@app.callback(
    Output("second-model-settings-store", "data"),
    [Input("second-model-selection", "value"),
     Input("n_estimators-input-2", "value"),
     Input("max_depth_RF-input-2", "value"),
     Input("min_samples_split-input-2", "value"),
     Input("min_samples_leaf-input-2", "value"),
     Input("hidden_layers-input-2", "value"),
     Input("neurons_per_layer-input-2", "value"),
     Input("activation_function-input-2", "value")]
)
def update_model_settings(model_type, n_estimators, max_depth, min_samples_split, min_samples_leaf, hidden_layers, neurons_per_layer, activation_function):
    # Create dictionary to store the settings
    if model_type == "RF":
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }
    elif model_type == "NN":
        return {
            "hidden_layers": hidden_layers,
            "neurons_per_layer": neurons_per_layer,
            "activation_function": activation_function
        }

#upload scrna data
@app.callback(
    Output("target-upload-status-2", "children", allow_duplicate = True),
    Output("upload-scrna-2", "children"),
    Output("upload-scrna-2", "style", allow_duplicate=True),
    Output("upload-scrna-2", "disabled"),
    Output("second-data-store", "data", allow_duplicate=True),
    Input("upload-scrna-2", "contents"),
    State("upload-scrna-2", "filename"), #file must be in the same directory as this script
    prevent_initial_call = True,
)
def upload_target(contents, filename):
    if contents:
        second_data = sc.read_h5ad(filename)
        return f"Uploaded Target File: {filename}", "Dataset uploaded successfully!", {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "solid",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
            "backgroundColor": "green",  # Change background to green
            "color": "darkgreen"
        }, True, {"second_data": second_data}  # Disable the upload box
    return "No target file uploaded."

#Upload spatial data
@app.callback(
    Output("target-upload-status-2", "children", allow_duplicate = True),
    Output("upload-h5ad-2", "children"),
    Output("upload-h5ad-2", "style", allow_duplicate=True),
    Output("upload-h5ad-2", "disabled"),
    Output("second-data-store", "data", allow_duplicate=True),
    Input("upload-h5ad-2", "contents"),
    State("upload-h5ad-2", "filename"),
    prevent_initial_call = True,
)
def upload_h5ad(contents, filename):
    if contents:
        second_data = sc.read_h5ad(filename)
        second_data_dict = encode_anndata(second_data)
        return f"Uploaded Target File: {filename}", "h5ad dataset uploaded successfully!", {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "solid",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
            "backgroundColor": "green",  # Change background to green
            "color": "darkgreen"
        }, True, {"second_data": second_data_dict}  # Disable the upload box
    return "No target file uploaded."

@app.callback(
    Output("target-upload-status-2", "children", allow_duplicate = True),
    Output("upload-coordinates-2", "children"),
    Output("upload-coordinates-2", "style", allow_duplicate=True),
    Output("upload-coordinates-2", "disabled"),
    Output("second-coordinates-store", "data", allow_duplicate=True),
    Input("upload-coordinates-2", "contents"),
    State("upload-coordinates-2", "filename"),
    prevent_initial_call = True,
)
def upload_coordinates(contents, filename):
    if contents:
        target_coordinates_2 = pd.read_csv(filename, index_col=0)
        target_coordinates_dict_2 = target_coordinates_2.to_dict() 
        return f"Uploaded Target File: {filename}", "Coordinates dataset uploaded successfully!", {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "solid",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
            "backgroundColor": "green",  # Change background to green
            "color": "darkgreen"
        }, True, {"target_coordinates_2": target_coordinates_dict_2}  # Disable the upload box
    return "No target file uploaded."

@app.callback(
    Output("target-upload-status-2", "children", allow_duplicate = True),
    Output("upload-image-2", "children"),
    Output("upload-image-2", "style", allow_duplicate=True),
    Output("upload-image-2", "disabled"),
    Output("second-image-store", "data", allow_duplicate=True),
    Input("upload-image-2", "contents"),
    State("upload-image-2", "filename"),
    prevent_initial_call = True,
)
def upload_image(contents, filename):
    if contents:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        pil_image = Image.open(io.BytesIO(decoded))
        flipped_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
        buffer = io.BytesIO()
        flipped_image.save(buffer, format="PNG")  # Ensure format is set
        buffer.seek(0)
        target_image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        return f"Uploaded Target File: {filename}", "Image uploaded successfully!", {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "solid",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
            "backgroundColor": "green",  # Change background to green
            "color": "darkgreen"
        }, True, {"pil_image_2": target_image_base64}  # Disable the upload box
    return "No target file uploaded."

@app.callback(
    Output("target-upload-status-2", "children"),
    Output("target-progress-2", "value"),
    Output("target-progress-2", "style"),
    Input("upload-scrna-2", "contents"),
    State("upload-scrna-2", "filename"),
    prevent_initial_call=True
)
def upload_target(contents, filename):
    if not contents:
        return "No file uploaded.", 0, {"display": "none"}
    try:
        # Simulate progress
        for i in range(1, 11):
            time.sleep(0.3)
        decoded = base64.b64decode(contents.split(",")[1])
        with open(filename, "wb") as f:
            f.write(decoded)
        data = sc.read_h5ad(filename)
        return f"Uploaded {filename}.", 100, {"display": "block"}
    except Exception as e:
        return f"Error reading file: {e}", 0, {"display": "block"}
    
@app.callback(
    Output("second-target-data-settings-store", "data"),
    [Input("variable_genes-input-2", "value"),
     Input("pca_dims-input-2", "value"),
     Input("normalisation-dropdown-2", "value"),
     Input("calculation-checklist-2", "value")]
)
def update_model_settings(variable_genes, pca_dims, normalisation, calculation):
    # Create dictionary to store the settings
    return {
        "variable_genes": variable_genes,
        "pca_dims": pca_dims,
        "normalisation": normalisation,
        "calculation": calculation
    }

@app.callback(
    Output("train-progress-2", "value"),
    Output("train-progress-2", "style"),
    Output("classification-status-2", "children"),
    Output("heatmap-plot-2", "figure"),
    Output("classification-report-2", "children"),
    Output("cm-plot-2", "figure"),
    Output("spatial-plot-2", "figure"),
    Output("spatial-plot-2", "style"),
    Output("umap-plot-2", "style"),
    Output("tsne-plot-2", "style"),
    Output("umap-plot-2", "figure", allow_duplicate=True),
    Output("tsne-plot-2", "figure", allow_duplicate=True),
    Output("second-data-sub-store", "data", allow_duplicate=True),
    Input("train-button-2", "n_clicks"),
    State("second-model-selection", "value"),
    State("second-reference-store", "data"),
    State("second-model-settings-store", "data"),
    State("second-target-data-settings-store", "data"),
    State("spatial-data-check-2", "value"),
    State("second-lowres-scale-factor-store", "data"),
    State("second-data-store", "data"),
    State("second-coordinates-store", "data"),
    State("second-image-store", "data"),
    prevent_initial_call = True,
)
def analyse_second_dataset(n_clicks, selected_model, selected_data, model_settings, target_data_settings, spatial_data_check, lowres_scale_factor_dict, second_data_store, second_coordinates_store, second_image_store):
    try:
        if selected_data == "GSE189903":
            reference_data = sc.read_h5ad("GSE189903_2_subset.h5ad")
        elif selected_data == "GSE151530":
            reference_data = sc.read_h5ad("GSE151530_1_subset.h5ad")
        elif selected_data == "GSE125449":
            reference_data = sc.read_h5ad("GSE151530_1.h5ad")
        elif selected_data == None:
            return "Please select a reference dataset."
        target_data_dict = second_data_store["second_data"] #Note the local variable name!!!
        target_data = decode_dict(target_data_dict)
        if not target_data:
            return "Please upload a target dataset."
        if n_clicks is None:
            return "Click the 'Train RF Model and Classify Target' button."
        if model_settings is None:
            return "Model settings are missing."

        # Multiple people worked on this part so the processing is a bit funny
        # Find common genes
        common_genes = reference_data.var_names.intersection(target_data.var_names)
        reference_data_sub = reference_data[:, common_genes]
        target_data_sub = target_data[:, common_genes]

        # Prepare features and labels
        X_ref = reference_data_sub.X.toarray()
        y_ref = reference_data_sub.obs["Type"]  # Replace 'Class' with actual label column

        X_target = target_data_sub.X.toarray()

        # Normalize
        X_ref_normalized = normalize(X_ref, axis=1)
        X_target_normalized = normalize(X_target, axis=1)

        # Train Random Forest model
        if selected_model == "RF":
            X_train, X_test, y_train, y_test = train_test_split(
                X_ref_normalized, y_ref, test_size=0.2, random_state=42, stratify=y_ref
            )
            clf = RandomForestClassifier(**model_settings)  # Use the parameters from dcc.Store
            clf.fit(X_train, y_train)
            y_test_labels = y_test #Standardise to this variable name
            #Predict on test dataset
            predicted_classes_test = clf.predict(X_test) #Standardise to this variable name
            report = classification_report(y_test_labels, predicted_classes_test, target_names=clf.classes_) #keep the variable name "report" consistent

            #Predict on target dataset
            predicted_classes = clf.predict(X_target_normalized)

        elif selected_model == "NN":
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_ref)  
            if not isinstance(reference_data.X, csr_matrix):
                X = csr_matrix(reference_data.X)
            else:
                X = reference_data.X
            y = reference_data.obs.apply(lambda row: f"{row['Type']}", axis=1).values
            n_cells = X.shape[0]
            n_sample = int(n_cells * 100 / 100)
            sampled_indices = np.random.choice(n_cells, size=n_sample, replace=False)
            
            X = X[sampled_indices, :]
            Y = y[sampled_indices]
            var_names = reference_data.var_names
            # Normalize the data
            X_normalized = normalize(X, axis=1, norm='l2')

            # --- Train-Test Split ---
            print("Splitting data into training and testing sets...")
            X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42, stratify=y)

            # One-hot encode the labels for neural network
            unique_labels = np.unique(y_train)
            y_train_encoded = to_categorical([np.where(unique_labels == label)[0][0] for label in y_train], num_classes=len(unique_labels))
            y_test_encoded = to_categorical([np.where(unique_labels == label)[0][0] for label in y_test], num_classes=len(unique_labels))

            # --- Neural Network Configuration ---
            num_epochs = 3 #TO CUSTOMISE
            num_layers = 3
            layer_nodes = []
            nodes = [128, 64, 32]
            for i in range(num_layers):
                layer_nodes.append(nodes[i])

            # --- Neural Network Model ---
            print("Building and training the neural network...")
            model = Sequential()

            # Add input layer and first hidden layer
            model.add(Dense(layer_nodes[0], activation='relu', input_shape=(X_train.shape[1],)))

            # Add additional hidden layers
            for nodes in layer_nodes[1:]:
                model.add(Dense(nodes, activation='relu'))

            # Add output layer
            model.add(Dense(len(unique_labels), activation='softmax'))

            model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

            # Train the model
            history = model.fit(X_train, y_train_encoded, epochs=num_epochs, batch_size=32, validation_split=0.2, verbose=1)

            # Evaluate the model
            loss, accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)

            # Predict on the test set
            y_pred_prob = model.predict(X_test)
            y_pred = np.argmax(y_pred_prob, axis=1)
            predicted_classes_test = y_pred
            y_test_labels = np.argmax(y_test_encoded, axis=1)

            # Predict on target data
            if not isinstance(target_data.X, csr_matrix):
                X_target_NN = csr_matrix(target_data.X)
            else:
                X_target_NN = target_data.X
            # Normalize the data
            X_target_normalized_NN = normalize(X_target_NN, axis=1, norm='l2')
            # Predicted classes for multi-class classification
            predicted_probs = model.predict(X_target_normalized_NN)  # Predict probabilities for each class
            predicted_indices = np.argmax(predicted_probs, axis=1)  # Get the index of the max probability
            predicted_classes = label_encoder.inverse_transform(predicted_indices)  # Map indices to class names

            # Calculate accuracy and display classification report
            report = classification_report(y_test_labels, predicted_classes_test, target_names=unique_labels)

        target_data_sub.obs['Predicted_Class'] = predicted_classes
        class_counts = target_data_sub.obs['Predicted_Class'].value_counts()
        print(class_counts)

        # Heatmap
        # Create gene expression DataFrame
        gene_expression = pd.DataFrame(
            X_target_normalized,
            index=target_data_sub.obs_names,  # Rows: Cells
            columns=target_data_sub.var_names  # Columns: Genes
        )
        gene_expression["Predicted_Class"] = predicted_classes

        # Calculate mean expression for each class
        class_means = gene_expression.groupby("Predicted_Class").mean()

        # Identify the 10 most differentially expressed genes per class
        selected_genes = []
        gene_class_map = []  # To track which genes belong to which class

        for cls in class_means.index:
            diff_expr = class_means.loc[cls] - class_means.mean()
            top_genes = diff_expr.abs().nlargest(10).index.tolist()
            selected_genes.extend(top_genes)  # List of just gene names
            gene_class_map.extend([(gene, cls) for gene in top_genes])  # List of (gene, class) pairs

        # Convert gene_class_map to a DataFrame
        selected_genes_df = pd.DataFrame(gene_class_map, columns=["Gene", "Class"])

        # Ensure selected genes are valid and align with the data
        valid_genes = [gene for gene in selected_genes_df["Gene"] if gene in gene_expression.columns]
        selected_genes_df = selected_genes_df[selected_genes_df["Gene"].isin(valid_genes)]

        # Subset and normalize the data for selected genes
        heatmap_data = gene_expression[valid_genes].assign(Predicted_Class=gene_expression["Predicted_Class"])

        # Extract the gene expression data (numerical data) and store the non-numeric columns (e.g., 'Predicted_Class')
        gene_expression_data = heatmap_data.drop(columns=["Predicted_Class"])  # Remove non-numeric columns
        class_labels = heatmap_data["Predicted_Class"]  # Store class labels separately

        # Normalize the numeric gene expression data
        heatmap_data_normalized = pd.DataFrame(
            normalize(gene_expression_data, axis=0),  # Normalize only numeric columns
            columns=gene_expression_data.columns,     # Maintain original column names (genes)
            index=gene_expression_data.index          # Keep the original index
        )

        # Re-add the 'Predicted_Class' column after normalization
        heatmap_data_normalized["Predicted_Class"] = class_labels

        # Reorder the genes by their associated class
        heatmap_data_ordered = heatmap_data_normalized

        # Prepare data for combined heatmap
        long_format_data = heatmap_data_ordered.melt(
            id_vars=["Predicted_Class"],
            var_name="Gene",
            value_name="Expression",
        )

        pivot_table = long_format_data.pivot_table(
            index="Gene",
            columns="Predicted_Class",
            values="Expression",
            aggfunc="mean"
        )

        # Plot heatmap with ordered genes
        heatmap_figure = px.imshow(
            pivot_table,
            labels={"x": "Predicted Class", "y": "Genes", "color": "Expression"},
            color_continuous_scale="Viridis",
            title="Combined Heatmap of Differentially Expressed Genes by Class",
        )

        # Update the layout
        heatmap_figure.update_layout(
            height=800,
            width=1200,
            yaxis={
                "categoryorder": "array",
                "categoryarray": valid_genes,  # Align genes by their class order
            },
        )

        #Confusion matrix
        #print(type(y_ref), y_ref, type(predicted_classes), predicted_classes)
        classes = sorted(set(y_test_labels).union(set(predicted_classes_test)))
        cm = confusion_matrix(y_test_labels, predicted_classes_test)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_figure = go.Figure(data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale='Blues',
            colorbar=dict(title='Count'),
            hoverongaps=False
        ))

        cm_figure.update_layout(
            title='Confusion Matrix',
            xaxis=dict(title='Predicted label'),
            yaxis=dict(title='True label'),
            autosize=True
        )

        # Run PCA
        if not np.issubdtype(target_data_sub.X.dtype, np.floating):
            target_data_sub.X = target_data_sub.X.astype(np.float64)
        sc.pp.highly_variable_genes(target_data_sub, n_top_genes = target_data_settings["variable_genes"])
        pca = PCA(n_components=target_data_settings["pca_dims"], random_state=42)
        X_pca = pca.fit_transform(X_target_normalized)

        # Run t-SNE
        if target_data_settings["calculation"] == ['umap']:
            reduction = umap.UMAP(n_components=2, random_state=42)
            X_reduction = reduction.fit_transform(X_pca)
        elif target_data_settings["calculation"] == ['tsne']:
            reduction = TSNE(n_components=2, random_state=42, perplexity=30)
            X_reduction = reduction.fit_transform(X_pca)
        else:
            print(target_data_settings["calculation"])
            return 0, {"display": "block"}, "No reduction selected", None, None, {"display": "none"}, {}


        # Add t-SNE coordinates to the AnnData object
        target_data.obsm['X_reduction'] = X_reduction

        # If predicted_classes is a numpy array, convert it to pandas Categorical if necessary
        if isinstance(predicted_classes, np.ndarray):
            predicted_classes = pd.Categorical(predicted_classes)

        # Get the class names
        class_labels = predicted_classes.categories  # Get class names
        class_codes = predicted_classes.codes  # Convert to numerical codes for coloring

        # Initialize an empty list to store traces
        traces = []

        # Create a separate trace for each class
        for class_code, class_label in enumerate(class_labels):
            mask = (class_codes == class_code)  # Filter points belonging to the current class
            trace = go.Scatter(
                x=X_reduction[mask, 0],
                y=X_reduction[mask, 1],
                mode='markers',
                marker=dict(
                    color=class_code,  # Assign a unique color for the class
                    colorscale='electric',  # Choose a colorscale
                    showscale=False,  # Do not display the color bar
                    size=3,
                ),
                name=class_label,  # Use the class name in the legend
                text=[class_label] * mask.sum(),  # Add class name as hover text
                hoverinfo='text',  # Display class name when hovering
            )
            traces.append(trace)

        # Create the layout for the plot
        if target_data_settings["calculation"] == ['umap']:
            layout = go.Layout(
                title="UMAP Visualization of Predicted Classes",
                xaxis=dict(title='UMAP1'),
                yaxis=dict(title='UMAP2'),
                showlegend=True,  # Show legend
                legend=dict(title='Cell Types', traceorder='normal'),  # Title for the legend
                height=800
            )
        elif target_data_settings["calculation"] == ['tsne']:
            layout = go.Layout(
                title="t-SNE Visualization of Predicted Classes",
                xaxis=dict(title='t-SNE1'),
                yaxis=dict(title='t-SNE2'),
                showlegend=True,  # Show legend
                legend=dict(title='Cell Types', traceorder='normal'),  # Title for the legend
                height=800
            )

        # Create the figure
        figure = go.Figure(data=traces, layout=layout)

        if spatial_data_check == "spatial":
            target_coordinates_dict = second_coordinates_store["target_coordinates_2"]
            target_coordinates = pd.DataFrame(target_coordinates_dict)
            content_string = second_image_store["pil_image_2"]
            decoded_data = base64.b64decode(content_string)
            pil_image = Image.open(io.BytesIO(decoded_data))
            # Map each predicted class to a unique color
            unique_classes = target_data_sub.obs['Predicted_Class'].unique()
            print("unique_classes:", unique_classes)
            class_colors = {cls: color for cls, color in zip(unique_classes, px.colors.qualitative.Set1)}
            print("class_colors:", class_colors)

            # Add colors to the target coordinates DataFrame
            target_coordinates['scaled_x'] = target_coordinates['imagecol'] * lowres_scale_factor_dict["lowres_scale_factor"]
            target_coordinates['scaled_y'] = target_coordinates['imagerow'] * lowres_scale_factor_dict["lowres_scale_factor"]
            print("1", target_data_sub.obs['Predicted_Class'])
            target_coordinates['class'] = target_data_sub.obs['Predicted_Class']
            target_coordinates['color'] = target_coordinates['class'].map(class_colors)
            #target_coordinates['color'] = target_coordinates['class'].apply(lambda x: class_colors[x])
            print(target_coordinates[['class', 'color']])

            # Create scatter plot for classified spatial data
            spatial_scatter = go.Scatter(
                x=target_coordinates['scaled_x'],
                y=target_coordinates['scaled_y'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=target_coordinates['color'],  # Use the mapped class colors
                    showscale=False  # Disable colorbar since we have discrete classes
                ),
                text=target_coordinates['class'],  # Display class labels on hover
                name='Spots'
            )

            # Layout for the spatial plot
            spatial_layout = go.Layout(
                images=[dict(
                    #source=f"data:image/png;base64,{target_image_base64}",
                    source=pil_image,
                    x=0,
                    y=0,
                    xref="x",
                    yref="y",
                    sizex=pil_image.width,
                    sizey=pil_image.width,
                    xanchor="left",
                    yanchor="bottom",
                    opacity=0.5,
                    layer="below"
                )],
                xaxis=dict(
                    scaleanchor="y",
                    showgrid=False,
                    zeroline=False,
                    visible=False
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    visible=False
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                title="Spatial Plot of Predicted Classes"
            )

            # Combine scatter and layout into a figure
            spatial_figure = go.Figure(data=[spatial_scatter], layout=spatial_layout)

            target_data_sub_dict = encode_anndata(target_data_sub)

            # Return figure with additional metadata if necessary
            if target_data_settings["calculation"] == ['umap']:
                return 100, {"display": "block"}, "Model trained successfully.", heatmap_figure, report, cm_figure, spatial_figure, {"display": "block"}, {'display': 'block'}, {'display': 'none'}, figure, None, {'second_data_sub': target_data_sub_dict}
            elif target_data_settings["calculation"] == ['tsne']:
                return 100, {"display": "block"}, "Model trained successfully.", heatmap_figure, report, cm_figure, spatial_figure, {"display": "block"}, {'display': 'none'}, {'display': 'block'}, None, figure, {'second_data_sub': target_data_sub_dict}
        target_data_sub_dict = encode_anndata(target_data_sub)
        if target_data_settings["calculation"] == ['umap']:
            return 100, {"display": "block"}, "Model trained successfully.", heatmap_figure, report, cm_figure, None, {"display": "none"}, {'display': 'block'}, {'display': 'none'}, figure, None, {'second_data_sub': target_data_sub_dict}
        elif target_data_settings["calculation"] == ['tsne']:
            return 100, {"display": "block"}, "Model trained successfully.", heatmap_figure, report, cm_figure, None, {"display": "none"}, {'display': 'none'}, {'display': 'block'}, None, figure, {'second_data_sub': target_data_sub_dict}

    except Exception as e:
        full_traceback = traceback.format_exc()
        return 0, {"display": "block"}, f"Error: {e}\nDetails:\n{full_traceback}", None, None, None, None, {"display": "none"}, {"display": "none"}, {"display": "none"}, None, None, {'second_data_sub': None}

#DEG and GSEA Analysis
@app.callback(
    Output("classes-store", "data"),
    Input("update-classes-btn", "n_clicks"),
    State("target-data-sub-store", "data"),
    prevent_initial_call=True,
)
def compute_classes(n_clicks, target_data_sub_store):
    target_data_sub_dict = target_data_sub_store["target_data_sub"]
    target_data_sub = decode_dict(target_data_sub_dict)
    if n_clicks == 0:
        return []
    # Dynamically compute classes (mock example here)
    classes = target_data_sub.obs["Predicted_Class"].unique().tolist()
    return classes

@app.callback(
    Output("class-selector", "options"),
    Input("classes-store", "data"),
)
def update_dropdown_options(classes):
    if not classes:
        return []
    return [{"label": c, "value": c} for c in classes]

# Unified callback to compute everything
@app.callback(
    [Output("heatmap-graph", "figure"),
     Output("gsea-image", "src")],
    Input("run-analysis-btn", "n_clicks"),
    State("class-selector", "value"),
    State("target-data-sub-store", "data"),
    State("second-data-sub-store", "data"),
    prevent_initial_call=True,
)
def compute_deg_and_gsea(n_clicks, selected_class, target_data_sub_store, second_data_sub_store):
    target_data_sub_dict = target_data_sub_store["target_data_sub"] #Note the local variable name!
    second_data_sub_dict = second_data_sub_store["second_data_sub"]
    target_data_sub = decode_dict(target_data_sub_dict)
    second_data_sub = decode_dict(second_data_sub_dict)
    # Step 1: Subset T cells
    target_t_cells = target_data_sub[target_data_sub.obs["Predicted_Class"] == selected_class]
    second_t_cells = second_data_sub[second_data_sub.obs["Predicted_Class"] == selected_class]

    # Step 2: Extract gene expression matrices
    target_expr = pd.DataFrame(
        target_t_cells.X.toarray(), 
        columns=target_t_cells.var_names, 
        index=target_t_cells.obs_names
    )
    second_expr = pd.DataFrame(
        second_t_cells.X.toarray(), 
        columns=second_t_cells.var_names, 
        index=second_t_cells.obs_names
    )

    # Step 3: Find shared genes
    shared_genes = target_expr.columns.intersection(second_expr.columns)
    target_expr = target_expr[shared_genes]
    second_expr = second_expr[shared_genes]
    print(shared_genes)
    print("target_expr:")
    print(target_expr)
    print(type(target_expr))

    # Step 4: Calculate differential expression
    p_values = []
    log_fold_changes = []

    for gene in shared_genes:
        # Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(target_expr[gene], second_expr[gene], alternative="two-sided")
        log_fc = np.log2(target_expr[gene].mean() + 1) - np.log2(second_expr[gene].mean() + 1)

        p_values.append(p_value)
        log_fold_changes.append(log_fc)

    # Create a results DataFrame
    results = pd.DataFrame({
        "gene": shared_genes,
        "p_value": p_values,
        "log_fold_change": log_fold_changes
    })
    print(results)

    # Filter significant genes and sort
    results = results.sort_values("p_value").query("p_value < 0.05")
    top_genes_target = results.nlargest(30, "log_fold_change")
    top_genes_second = results.nsmallest(30, "log_fold_change")
    selected_genes = pd.concat([top_genes_target, top_genes_second])
    print(selected_genes)

    # Step 5: Create heatmap data
    selected_gene_names = selected_genes["gene"]
    target_expr_filtered = target_expr[selected_gene_names]
    second_expr_filtered = second_expr[selected_gene_names]

    # Combine data
    heatmap_data = pd.concat([target_expr_filtered, second_expr_filtered])

    # Create labels
    cell_labels = ["Target" for _ in range(len(target_expr))] + ["Second" for _ in range(len(second_expr))]

    # Step 6: Plot heatmap
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values.T,
            x=heatmap_data.index,
            y=heatmap_data.columns,
            colorscale="Viridis",  # Colorscale based on expression values
            colorbar=dict(title="Expression"),
        )
    )

    # Add a color bar at the bottom to show the two cell types
    heatmap_fig.add_trace(
        go.Bar(
            x=heatmap_data.index,
            y=[1] * len(cell_labels),
            marker=dict(
                color=[1 if label == "Target" else 2 for label in cell_labels],  # Color based on cell type
                colorscale="Viridis",
                showscale=False
            ),
            width=1,
            name="Cell Type",
            yaxis="y2"
        )
    )

    # Update layout to remove cell names and show a color bar for cell types
    heatmap_fig.update_layout(
        title="Differentially Expressed Genes Heatmap",
        xaxis=dict(
            title="Cells",
            tickvals=[],  # Remove the cell names from the x-axis
            ticktext=[]
        ),
        yaxis=dict(title="Genes"),
        yaxis2=dict(
            overlaying="y",
            range=[0, 1],
            showticklabels=False,
        ),
        legend=dict(title="Cell Origin"),
        coloraxis=dict(colorscale="Viridis"),  # For consistent coloring in the heatmap
    )

    # Step 7: Perform GSEA
    # Prepare data for GSEA
    gene_ranking = results.set_index("gene")["log_fold_change"]
    gene_ranking = gene_ranking.sort_values(ascending=False)
    print("gene_ranking:", gene_ranking)

    # Run GSEA
    gsea_results = gp.prerank(
        rnk=gene_ranking,
        gene_sets="KEGG_2019_Human",  # Example gene set; modify as needed
        outdir=None,
        seed=42,
        min_size=1,  # Try reducing this
        max_size=500  # Try increasing this
    )

    # Get the first significant enrichment result
    top_pathway = gsea_results.res2d.sort_values("NOM p-val").iloc[0]
    enrichment_score = gsea_results.results[top_pathway["Term"]]["es"]

    # Step 8: Plot enrichment plot using GSEA
    fig = gp.plot.gseaplot(gsea_results.ranking, ofname='gsea_enrichment_plot.pdf', **gsea_results.results[top_pathway["Term"]])
    if not os.path.exists('assets'):
        os.makedirs('assets')
    os.rename('gsea_enrichment_plot.pdf', 'assets/gsea_enrichment_plot.pdf')
    return heatmap_fig, app.get_asset_url('gsea_enrichment_plot.pdf')

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
