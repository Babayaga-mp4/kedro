from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import load_dataset, collect_dataset, data_viz

def create_pipeline(**kwargs) -> Pipeline:
     return pipeline(
        [
            node(collect_dataset, inputs=["Customer Data", "Towers and Complaints",
                                       "Network Logs", "CDRs", "IMEI info", "Other Data Sources"], outputs="Raw_Data",name="Data_Ingestion"),

            node(load_dataset, inputs="Raw_Data", outputs = ["Pre-Processed Data"], name = 'Filtering'),


            # node(data_viz, inputs='Feature Importance', outputs = ["Visualization"]),

            ],
         namespace = "Data Ingestion" ,
         inputs = {"Customer Data", "Towers and Complaints",
                                       "Network Logs", "CDRs", "IMEI info", "Other Data Sources"} ,
         outputs = {
             "Pre-Processed Data" ,
             # "Visualization" ,
         }
     )