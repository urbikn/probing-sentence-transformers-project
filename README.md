# How Much Effort is Needed to Transmit the Tense: Probing Sentence Transformer Representations for Linguistic Properties
Transformer models dominate the NLP field, excelling in various tasks, yet comprehending the information they capture remains a challenge. Probing tasks offer a method to unveil the type of information that is encoded in these models. The goal of this project is to investigate the degree of linguistic information embedded in sentence representations. In line with prior research, the study aims to probe sentence transformers in comparison to a BiLSTM baseline across different simple linguistic tasks. We probe with a conventional probing classifier and a more informative MDL probe. We find that most linguistic information is encoded in the beginner and middle layers of the sentence transformers that encode more information compared to the BiLSTM on 8 of 10 probing tasks. While that is noteworthy, our examination of the final representations of the sentence transformer models showed that they encode less linguistic information compared to the BiLSTM. Additionally, our findings indicate that the conventional probing classifier falls short of the MDL probe in scrutinizing the encoding of simple linguistic information. This observation prompts a reconsideration of probing tools used that are used to analyze properties in sentence representations.

## Project structure
The repository has the following structure:
- `data/` - directory where all `.txt` dataset files are stored
    - `original_pov_data/` - contains original data from McCoy et al. (2020)
    - `new_pov_data/` - contains data transformed from McCoy et al. (2020) using `prep_data.py` script
    - `probing_data/` - contains original data from Conneau et al. (2018) 
- `plots/` - here you will find plots of the results which are included in the final report
- `results/` - contains results of the probing stored as `.json` files
    - `results_analysis.ipynb` - notebook used for analyzing the results (plots, tables, etc.)
- `src/` - contain all the source code of the project
    - `bilstm/` - folder containing the code for loading and using BiLSTM model
    - `dataset.py` - handles the operations of loading the datasets and the embeddings
    - `embeddings.py` - deals with embedding operations
    - `pca.py` - contains code to perform PCA reduction 
    - `probe.py` - training and evaluating probing models
- `prep_data.py` - script used for preparing the data into a unified format
- `main.py` - Main script used for running the entire analysis

## Reproduction
If you'd like to reproduce the results of this project, kindly follow the steps outlined below:
1) **Clone the repository**: Start by cloning this repository to your local machine.
2) **Install dependencies**: Install the ncessary dependencies
```
pip install -r requirements.txt
```
3) **Generate model embeddings**: Run the following script that downloads the models and extracts the sentence embeddings:
```
python main.py -e --training_data data/probing_data/ --embedding_data .embeddings
```
Extract the embeddings from the models using the data from the data/probing_data folder and save them to disk in the .embeddings folder
3.1) **Apply PCA (optional)**: Run the following script that applies PCA. Note that the path to the original embeddings is declared directly within the script:

```
python main.py -r 100 --embedding_data .embeddings
```
Reduce the dimensionality of the embeddings in the .embeddings folder to 100 dimensions
4) **Run probes**: Initiate the probing process by running the following script:
```
python main.py
    -p 
    --model mpnet
    --task subj_number
    --seed 42
    --training_data data/probing_data/subj_number.txt
    --embedding_data .embeddings/subj_number.pt
    --save_report ./reports/probing/subj_number-mpnet.json
```
Run the probing experiment on the `subj_number` task/dataset using the embeddings from the `mpnet` model
and save the results in the `./reports/probing/subj_number-mpnet.json` file.

**Note:** all information can be found by running `python main.py --help`