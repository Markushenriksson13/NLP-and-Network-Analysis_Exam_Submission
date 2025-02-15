
## 🔬Breaking Bad NLP & Network Analysis 🎯
 > YEAH, SCIENCE Mr. White!

Welcome to this Breaking Bad Network Analysis project! 
This repository contains various notebooks and data files for analyzing the Network of Breaking Bad series 
The project uses LLM (TogetherAPI) for data processing and SetFit (Local) machine learning for text classification of characters.

![BreakingBadNetwork](https://github.com/Markushenriksson13/NLP-and-Network-Analysis_Exam_Submission/blob/main/breaking_bad_network.png?raw=true)

### ℹ️ This repository consists of 3 different notebooks - They can all 3 be ran independently
#### ✅ Main Network analysis and classification notebook (imports already pre-saved data and models generated from the other two notebooks)
```bash
├── M2_Main_Network_Analysis_and_Text_Classification.ipynb
```
##### ⚙️ Data fetching and LLM-processing notebook
```bash
├── M2_LLM_Data_Fetch_and_Processing_(JSON_Creation).ipynb
```
##### 🧮 SetFit Model training notebook for Text Classification 
```bash
└── M2_Model_Train.ipynb
```                                      


Notebook descriptions:

### 🖊️ ⚙️ M2_LLM_Data_Fetch_and_Processing_(JSON_Creation).ipynb 📨
This notebook leverages Large Language Models (LLMs) - to fetch the dialogue (subtitles) of the TV series *Breaking Bad* from the Movie and TV-show wiki: *Fandom.com* and extract relationships between characters, locations, events & season number.

#### 1. Data Acquisition

- **Scraping**: Subtitles are scraped from the Breaking Bad Fandom Wiki for all seasons and saved as individual text files.
- **Cleaning**: The scraped subtitles are cleaned to remove timestamps and other unnecessary elements, leaving only the dialogue.

#### 2. Context and Prompt Creation

- **Wikipedia Summary**: The notebook utilizes the Wikipedia article "List of characters in the Breaking Bad franchise" to create a summarized context of key characters and their relationships using the LLM: `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`. This summary is used as background knowledge for the LLM processing of the subtitles (dialogue in the TV-show).
- **LLM Prompts**: System prompts are  defined for the LLM. These prompts guide its analysis, ensuring that the extracted information follows a predefined JSON schema for representing relationships.

#### 3. LLM Processing, Extraction & Output

- **Episode Analysis**: The LLM (`Qwen/Qwen2.5-72B-Instruct-Turbo`) iterates through each episode's subtitle file. The content of the subtitles, along with details like episode name and season number, are fed to the LLM as prompts.
- **Entity and Relationship Extraction**: The LLM analyzes the script and extracts entities (characters, locations, events). It then identifies relationships between these entities using a set of predefined relationship types (e.g., "friend of," "enemy of," "works with").
- **JSON Structuring**: The extracted information is structured into a JSON format, for easy storage and further analysis...

- **JSON Output**: All the LLM-processed episode data is saved into a single JSON file named "breaking_bad_analysisV2.json".
- **Summary**: A summary indicating the number of processed episodes is displayed to the user along with failed episodes (if it unlikely would occur).

### 🧮 M2_Model_Train.ipynb
This notebook trains a SetFit model to classify relationships between characters in the TV series Breaking Bad.
#### Functionality

- Reads data from a JSON file (`breaking_bad_analysisV2.json`) generated in a previous notebook (`M2_LLM_Data_Fetch_and_Processing.ipynb`).
- Prepares the data for classification by splitting it into training and testing sets (80/20 split).
- Trains a SetFit model using `sentence-transformers/paraphrase-mpnet-base-v2` as the base model.
- Evaluates the trained model using a classification report and shows example predictions.
- Saves the trained model to the `saved_model` folder.

#### Requirements for this notebook

- Google Colab environment (recommended)
- Highly recommended!: GPU for training

### ✅ M2_Main_Network_Analysis_and_Text_Classification.ipynb 🛣️ ⛓️

#### Functionality

The notebook performs the following tasks:

1. **Network Analysis**: 
    - Extracts characters and their relationships from episode data stored in a JSON file (`breaking_bad_analysisV2.json`).
    - Builds a network graph representing characters and their interactions.
    - Calculates centrality measures like degree, betweenness, and eigenvector centrality to identify key characters.
    - Visualizes the network graph using interactive Plotly plots, highlighting character importance and communities.
    - Identifies and clusters communities (groups of closely related characters).
    - Illustrates character relationships with relation types.

2. **Text Classification**:
    - Leverages a pre-trained SetFit model to predict relationship types between pairs of characters.
    - Provides a Gradio interface to compare relationship between any two selected characters.


## Projectstructure 📂 

```bash
.
├── saved_model/                             # Directory for saved models
├── subtitles/                               # Subtitles for the series
├── .gitattributes                           # Git attributes for LFS
├── breaking_bad_analysisV2.json            # JSON data for analysis
├── Exam Overview.pdf                       # PowerPoint PDF presentation
├── breaking_bad_network.png                 # Network visualization
├── requirements.txt                         # Project dependencies
├── wiki_breaking_bad_characters.txt         # Character information from Wikipedia
├── M2_LLM_Data_Fetch_and_Processing_(JSON_Creation).ipynb # Data fetching and processing notebook
├── M2_Main_Network_Analysis_and_Text_Classification.ipynb # Main analysis and classification notebook
└── M2_Model_Train.ipynb                      # Model training notebook
```

## Installation 📦 

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Markushenriksson13/NLP-and-Network-Analysis_Exam_Submission.git
   ```
2. **Open/Run in your desired Jupyter Enviroment**
   - If ran in Colab make sure to replace #'s in the given notebook

##
