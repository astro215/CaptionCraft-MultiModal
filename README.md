# CaptionCraft-MultiModal

## Introduction
CaptionCraft-MultiModal is a project focused on creating captions for multimedia content using multimodal approaches. This project leverages advanced techniques to provide accurate and context-aware captions for various types of media.

## Training Overview
The training process involves several key steps:

```mermaid
    flowchart TB
        subgraph Input
            I[Image] --> R[ResNet50]
            I --> Y[YOLOv8]
            Y --> B[BERT]
        end
        subgraph Feature_Extraction
            R --> IE[Image Embeddings<br/>2048-dim]
            B --> TE[Text Embeddings<br/>768-dim]
        end
        subgraph Fusion_Approaches
            direction LR
            subgraph Early_Fusion
                IE --> CAT[Concatenate<br/>2816-dim]
                TE --> CAT
                CAT --> EF[FC Layer + ReLU<br/>512/768/1024-dim]
            end
            subgraph Late_Fusion
                IE --> IFC[Image FC + ReLU<br/>512/768/1024-dim]
                TE --> TFC[Text FC + ReLU<br/>512/768/1024-dim]
                IFC --> ADD{Concate}
                TFC --> ADD
                ADD --> LF[FC Layer<br/>512/768/1024-dim]
            end
        end
        EF --> T5E[T5 Encoder]
        LF --> T5L[T5 Encoder]
        
        T5E --> CE[Caption<br/>Early Fusion]
        T5L --> CL[Caption<br/>Late Fusion]
        classDef default fill:#2d2d2d,stroke:#666,stroke-width:2px,color:#fff
        classDef modelNode fill:#4a235a,stroke:#666,stroke-width:2px,color:#fff
        classDef embedNode fill:#1a5276,stroke:#666,stroke-width:2px,color:#fff
        classDef processNode fill:#145a32,stroke:#666,stroke-width:2px,color:#fff
        classDef fusionNode fill:#7d6608,stroke:#666,stroke-width:2px,color:#fff
        class R,Y,B modelNode
        class IE,TE embedNode
        class EF,IFC,TFC,LF processNode
        class CAT,ADD fusionNode
```

The training module includes detailed instructions and scripts to prepare and train models on your own datasets. For more information, refer to the training [README](https://github.com/astro215/CaptionCraft-MultiModal/blob/main/training/README.md).

## Tech Stack

- Jupyter Notebook: For data analysis and experimentation.
- Python: The programming language used for model development.
- PyTorch: The deep learning framework used to build and train the models.
- Streamlit: For deploying the model as an interactive web application.
- YOLO: For object detection in images.
- ResNet: For generating image embeddings.
- BERT: For generating text embeddings


## Setup and Installation
To set up the project, follow these steps:
1. Clone the repository:
   ```
   git clone https://github.com/astro215/CaptionCraft-MultiModal.git
   ```
2. Navigate to the project directory:
   ```
   cd CaptionCraft-MultiModal
   ```
3. Install the required dependencies (if any):
   ```
   pip install -r requirements.txt
   ```



