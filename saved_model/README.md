---
base_model: sentence-transformers/paraphrase-mpnet-base-v2
library_name: setfit
metrics:
- accuracy
pipeline_tag: text-classification
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: Walter Jr. (Flynn White) - Mineral Show
- text: Walter White - 308 Negra Arroyo Lane, Albuquerque, New Mexico, 87104
- text: Daughter - Janitor
- text: Marie Schrader - Hank's and Marie's House
- text: Hank Schrader - Tuco's last-known address
inference: true
---

# SetFit with sentence-transformers/paraphrase-mpnet-base-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 93 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label                      | Examples                                                                                                                                                               |
|:---------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| teaches                    | <ul><li>'Walter White - Chemistry Lesson'</li></ul>                                                                                                                    |
| works in                   | <ul><li>'Walter White - Classroom'</li></ul>                                                                                                                           |
| occurs in                  | <ul><li>'Chemistry Lesson - Classroom'</li><li>"Discussion about Georgia O'Keeffe's paintings - Season_3"</li><li>"Walter Jr.'s Driving Test - Season 3"</li></ul>     |
| is part of                 | <ul><li>'Chemistry Lesson - Season 1'</li></ul>                                                                                                                        |
| works with                 | <ul><li>'DEA - APD'</li><li>'Janitor - School'</li><li>'Hank Schrader - Steve Gomez'</li></ul>                                                                         |
| investigates               | <ul><li>'DEA - Lab Equipment Theft'</li><li>'DEA - Marijuana Possession'</li></ul>                                                                                     |
| arrested for               | <ul><li>'Janitor - Marijuana Possession'</li></ul>                                                                                                                     |
| fired from                 | <ul><li>'Janitor - School'</li></ul>                                                                                                                                   |
| conducts                   | <ul><li>'School - Background Check'</li><li>'School - Hiring Policies Review'</li></ul>                                                                                |
| witnesses                  | <ul><li>'Daughter - Janitor'</li><li>'Daughter - Marijuana Possession'</li><li>"Walter White - Jane's Overdose"</li></ul>                                              |
| works at                   | <ul><li>'Walter White - School'</li></ul>                                                                                                                              |
| participates in            | <ul><li>'Walter White - Lab Equipment Theft'</li><li>'Hank Schrader - Operation Icebreaker'</li><li>'Steve Gomez - Operation Icebreaker'</li></ul>                     |
| appears in                 | <ul><li>'Walter White - Season_1'</li><li>'Steve Gomez - Stakeout'</li><li>'Hank Schrader - Stakeout'</li></ul>                                                        |
| is central to              | <ul><li>'Krazy-8 - Operation Icebreaker'</li><li>"Crazy Handful of Nothin' - Season 1"</li><li>'Pilot Episode - Season 1'</li></ul>                                    |
| related to                 | <ul><li>'Krazy-8 - Emilio Koyama'</li><li>'Walter White - Elliott'</li><li>'Walter White - Skyler White'</li></ul>                                                     |
| ratted out by              | <ul><li>'Emilio Koyama - Krazy-8'</li></ul>                                                                                                                            |
| missing in                 | <ul><li>'Krazy-8 - Boonies'</li></ul>                                                                                                                                  |
| presumed dead in           | <ul><li>'Krazy-8 - Boonies'</li></ul>                                                                                                                                  |
| suspected revenge against  | <ul><li>'Emilio Koyama - Krazy-8'</li></ul>                                                                                                                            |
| visits                     | <ul><li>'Walter White - Ditch'</li><li>'Jesse Pinkman - Ditch'</li><li>'Jesse Pinkman - Walter White'</li></ul>                                                        |
| owns                       | <ul><li>'Walter White - RV'</li><li>'Jesse Pinkman - RV'</li><li>'Heisenberg - Blue Meth'</li></ul>                                                                    |
| spills                     | <ul><li>'Walter White - Coffee Mug'</li></ul>                                                                                                                          |
| uses                       | <ul><li>'Walter White - Sony GPS'</li><li>'Walter White - Kelly Criterion'</li></ul>                                                                                   |
| causes                     | <ul><li>'Walter White - Bodies'</li><li>'Jesse Pinkman - Bodies'</li><li>'Technician - PETICT Scan'</li></ul>                                                          |
| part of                    | <ul><li>'RV - Season_1'</li><li>'Ditch - Season_1'</li><li>'Coffee Mug - Season_1'</li></ul>                                                                           |
| lives in                   | <ul><li>'Walter White - Skyler White'</li><li>'Walter White - 308 Negra Arroyo Lane, Albuquerque, New Mexico, 87104'</li><li>"Jesse Pinkman - Jesse's House"</li></ul> |
| father of                  | <ul><li>'Walter White - Walter Jr.'</li><li>'Walter White - Walter Jr.'</li><li>'Walter White - Walter Jr.'</li></ul>                                                  |
| mother of                  | <ul><li>'Skyler White - Walter Jr.'</li><li>'Skyler White - Walter Jr.'</li></ul>                                                                                      |
| occurs at                  | <ul><li>'Job Interview - Job Interview Location'</li></ul>                                                                                                             |
| friend of                  | <ul><li>'Jesse Pinkman - Badger Mayhew'</li><li>'Badger Mayhew - Jesse Pinkman'</li><li>'Skinny Pete - Jesse Pinkman'</li></ul>                                        |
| married to                 | <ul><li>'Walter White - Skyler White'</li><li>'Skyler White - Walter White'</li><li>'Walter White - Skyler White'</li></ul>                                            |
| competes with              | <ul><li>'Kleinman - Technician'</li></ul>                                                                                                                              |
| sends                      | <ul><li>'Saul Goodman - Walter White'</li></ul>                                                                                                                        |
| asks                       | <ul><li>'Dude (Potential Buyer) - Seller (Potential Dealer)'</li></ul>                                                                                                 |
| suspects                   | <ul><li>'Dude (Potential Buyer) - Brown Van'</li><li>'Dude (Potential Buyer) - Duke City Flowers'</li></ul>                                                            |
| denies                     | <ul><li>'Seller (Potential Dealer) - Dude (Potential Buyer)'</li></ul>                                                                                                 |
| suggests                   | <ul><li>'Dude (Potential Buyer) - Seller (Potential Dealer)'</li></ul>                                                                                                 |
| proposes                   | <ul><li>'Dude (Potential Buyer) - Garbage Truck'</li></ul>                                                                                                             |
| demands                    | <ul><li>'Dude (Potential Buyer) - Seller (Potential Dealer)'</li></ul>                                                                                                 |
| refuses                    | <ul><li>'Seller (Potential Dealer) - Dude (Potential Buyer)'</li></ul>                                                                                                 |
| son of                     | <ul><li>'Walter Jr. - Walter White'</li><li>'Walter Jr. - Walter White'</li><li>'Walter Jr. - Skyler White'</li></ul>                                                  |
| brother-in-law of          | <ul><li>'Hank Schrader - Walter White'</li></ul>                                                                                                                       |
| uncle of                   | <ul><li>'Hank Schrader - Walter Jr.'</li><li>'Hector Salamanca - Tuco Salamanca'</li></ul>                                                                             |
| kills                      | <ul><li>'Hank Schrader - Drug Dealer'</li><li>'Tuco Salamanca - Dog Paulson'</li><li>'Tuco Salamanca - Mexican national'</li></ul>                                     |
| killed by                  | <ul><li>'Drug Dealer - Hank Schrader'</li></ul>                                                                                                                        |
| worries about              | <ul><li>'Walter Jr. - Hank Schrader'</li><li>'Walter White - Hank Schrader'</li><li>'Walter White - Skyler White'</li></ul>                                            |
| enemy of                   | <ul><li>'Jesse Pinkman - Tuco Salamanca'</li><li>'Tuco Salamanca - Jesse Pinkman'</li><li>'Tuco Salamanca - Hank Schrader'</li></ul>                                   |
| develops                   | <ul><li>'Season 2 - Cooking Meth'</li><li>'Season 2 - Money Dispute'</li><li>'Season 2 - Settling Down'</li></ul>                                                      |
| raids                      | <ul><li>"Hank Schrader - Tuco's headquarters"</li><li>"Hank Schrader - Tuco's last-known address"</li><li>"Hank Schrader - Tuco's meth-hag girlfriend's den"</li></ul> |
| suspected of               | <ul><li>"Tuco Salamanca - Krazy-8's disappearance"</li></ul>                                                                                                           |
| calls                      | <ul><li>'Jesse Pinkman - Skinny Pete'</li><li>'Walter White - Saul Goodman'</li><li>'Walter White - Cab driver'</li></ul>                                              |
| runs through               | <ul><li>'Blue Meth - New Mexico'</li></ul>                                                                                                                             |
| travels to                 | <ul><li>'Blue Meth - Michoacan'</li></ul>                                                                                                                              |
| wants to                   | <ul><li>'Cartel - Heisenberg'</li></ul>                                                                                                                                |
| runs hot because           | <ul><li>'Cartel - Blue Meth'</li></ul>                                                                                                                                 |
| is like                    | <ul><li>'New Mexico - Mexico'</li></ul>                                                                                                                                |
| introduces                 | <ul><li>'Walter White (Heisenberg) - Seven Thirty-Seven'</li><li>'Jesse Pinkman - Two fine ladies'</li></ul>                                                           |
| paints                     | <ul><li>"Georgia O'Keeffe - Door"</li></ul>                                                                                                                            |
| discuss                    | <ul><li>"Characters - Discussion about Georgia O'Keeffe's paintings"</li></ul>                                                                                         |
| pulls over                 | <ul><li>'Officer - Walter White'</li></ul>                                                                                                                             |
| was in                     | <ul><li>'Walter White - Debris Field'</li></ul>                                                                                                                        |
| cites                      | <ul><li>'Officer - Walter White'</li></ul>                                                                                                                             |
| meets                      | <ul><li>'Stan - Walt'</li></ul>                                                                                                                                        |
| gives                      | <ul><li>'Stan - House Tour'</li></ul>                                                                                                                                  |
| offers                     | <ul><li>'Customer - Drug Offer'</li></ul>                                                                                                                              |
| treats                     | <ul><li>'Medical Team - Gunshot Victim'</li></ul>                                                                                                                      |
| moves to                   | <ul><li>'Medical Team - Operating Room 1'</li><li>'Gunshot Victim - Operating Room 1'</li></ul>                                                                        |
| buys                       | <ul><li>'Jesse Pinkman - RV'</li></ul>                                                                                                                                 |
| receives reports from      | <ul><li>'KOB - Callers'</li></ul>                                                                                                                                      |
| provides information about | <ul><li>'FAA - Boeing 737'</li><li>'FAA - King Air 350'</li></ul>                                                                                                      |
| operates out of            | <ul><li>'King Air 350 - St. George, Utah'</li></ul>                                                                                                                    |
| is bound for               | <ul><li>'King Air 350 - Amarillo, Texas'</li></ul>                                                                                                                     |
| experiences                | <ul><li>'Neighbourhood - Falling debris'</li></ul>                                                                                                                     |
| divorces                   | <ul><li>'Skyler White - Walter White'</li></ul>                                                                                                                        |
| is                         | <ul><li>'Walter White - Thief'</li></ul>                                                                                                                               |
| tells                      | <ul><li>'Walter White - Hank Schrader'</li></ul>                                                                                                                       |
| concerned about            | <ul><li>'Walter White - Hank Schrader'</li></ul>                                                                                                                       |
| wants to visit             | <ul><li>'Hank Schrader - Factory Farm'</li></ul>                                                                                                                       |
| stalls                     | <ul><li>'Walter White - Hank Schrader'</li></ul>                                                                                                                       |
| brings bomb to             | <ul><li>'Jesse Pinkman - Hospital'</li></ul>                                                                                                                           |
| wants to talk to           | <ul><li>'Detective Kalanchoe - Jesse Pinkman'</li><li>'Detective Munn - Jesse Pinkman'</li></ul>                                                                       |
| fears for                  | <ul><li>"Walter White - Walter White's family"</li></ul>                                                                                                               |
| attacks                    | <ul><li>'Assassin - Hank Schrader'</li><li>'Nephews - Hank Schrader'</li></ul>                                                                                         |
| dies                       | <ul><li>'Gunmen - Scene'</li><li>'Gunmen - Hospital'</li></ul>                                                                                                         |
| wants to kill              | <ul><li>'Nephews - Walter White'</li></ul>                                                                                                                             |
| wants to talk about        | <ul><li>'Skyler White - Car Wash'</li></ul>                                                                                                                            |
| wants to go to             | <ul><li>'Skyler White - Police'</li></ul>                                                                                                                              |
| wants to tell              | <ul><li>'Skyler White - Drug Dealing'</li></ul>                                                                                                                        |
| talks to                   | <ul><li>'Walter White - Glenn'</li></ul>                                                                                                                               |
| near                       | <ul><li>'Pavilion parking lot on University - Airport'</li></ul>                                                                                                       |
| business partner           | <ul><li>'Walter White - Jesse Pinkman'</li></ul>                                                                                                                       |
| advises                    | <ul><li>'Mike Ehrmantraut - Walter White'</li></ul>                                                                                                                    |
| rides                      | <ul><li>'The Kid - Dirt Bike'</li></ul>                                                                                                                                |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("Daughter - Janitor")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 3   | 4.8274 | 11  |

| Label                      | Training Sample Count |
|:---------------------------|:----------------------|
| advises                    | 1                     |
| appears in                 | 38                    |
| arrested for               | 1                     |
| asks                       | 1                     |
| attacks                    | 2                     |
| brings bomb to             | 1                     |
| brother-in-law of          | 1                     |
| business partner           | 1                     |
| buys                       | 1                     |
| calls                      | 3                     |
| causes                     | 13                    |
| cites                      | 1                     |
| competes with              | 1                     |
| concerned about            | 1                     |
| conducts                   | 2                     |
| demands                    | 1                     |
| denies                     | 1                     |
| develops                   | 9                     |
| dies                       | 2                     |
| discuss                    | 1                     |
| divorces                   | 1                     |
| enemy of                   | 10                    |
| experiences                | 1                     |
| father of                  | 5                     |
| fears for                  | 1                     |
| fired from                 | 1                     |
| friend of                  | 3                     |
| gives                      | 1                     |
| introduces                 | 2                     |
| investigates               | 2                     |
| is                         | 1                     |
| is bound for               | 1                     |
| is central to              | 25                    |
| is like                    | 1                     |
| is part of                 | 1                     |
| killed by                  | 1                     |
| kills                      | 3                     |
| lives in                   | 19                    |
| married to                 | 7                     |
| meets                      | 1                     |
| missing in                 | 1                     |
| mother of                  | 2                     |
| moves to                   | 2                     |
| near                       | 1                     |
| occurs at                  | 1                     |
| occurs in                  | 5                     |
| offers                     | 1                     |
| operates out of            | 1                     |
| owns                       | 15                    |
| paints                     | 1                     |
| part of                    | 25                    |
| participates in            | 60                    |
| presumed dead in           | 1                     |
| proposes                   | 1                     |
| provides information about | 2                     |
| pulls over                 | 1                     |
| raids                      | 3                     |
| ratted out by              | 1                     |
| receives reports from      | 1                     |
| refuses                    | 1                     |
| related to                 | 39                    |
| rides                      | 1                     |
| runs hot because           | 1                     |
| runs through               | 1                     |
| sends                      | 1                     |
| son of                     | 3                     |
| spills                     | 1                     |
| stalls                     | 1                     |
| suggests                   | 1                     |
| suspected of               | 1                     |
| suspected revenge against  | 1                     |
| suspects                   | 2                     |
| talks to                   | 1                     |
| teaches                    | 1                     |
| tells                      | 1                     |
| travels to                 | 1                     |
| treats                     | 1                     |
| uncle of                   | 2                     |
| uses                       | 2                     |
| visits                     | 33                    |
| wants to                   | 1                     |
| wants to go to             | 1                     |
| wants to kill              | 1                     |
| wants to talk about        | 1                     |
| wants to talk to           | 2                     |
| wants to tell              | 1                     |
| wants to visit             | 1                     |
| was in                     | 1                     |
| witnesses                  | 19                    |
| works at                   | 1                     |
| works in                   | 1                     |
| works with                 | 51                    |
| worries about              | 4                     |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 20
- body_learning_rate: (2e-05, 2e-05)
- head_learning_rate: 2e-05
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0008 | 1    | 0.2835        | -               |
| 0.0421 | 50   | 0.2538        | -               |
| 0.0842 | 100  | 0.2063        | -               |
| 0.1263 | 150  | 0.1595        | -               |
| 0.1684 | 200  | 0.1253        | -               |
| 0.2104 | 250  | 0.1003        | -               |
| 0.2525 | 300  | 0.1005        | -               |
| 0.2946 | 350  | 0.0961        | -               |
| 0.3367 | 400  | 0.0877        | -               |
| 0.3788 | 450  | 0.0845        | -               |
| 0.4209 | 500  | 0.0822        | -               |
| 0.4630 | 550  | 0.0691        | -               |
| 0.5051 | 600  | 0.0676        | -               |
| 0.5471 | 650  | 0.0612        | -               |
| 0.5892 | 700  | 0.0497        | -               |
| 0.6313 | 750  | 0.0685        | -               |
| 0.6734 | 800  | 0.0623        | -               |
| 0.7155 | 850  | 0.0527        | -               |
| 0.7576 | 900  | 0.0604        | -               |
| 0.7997 | 950  | 0.051         | -               |
| 0.8418 | 1000 | 0.0572        | -               |
| 0.8838 | 1050 | 0.0442        | -               |
| 0.9259 | 1100 | 0.0488        | -               |
| 0.9680 | 1150 | 0.0532        | -               |

### Framework Versions
- Python: 3.10.12
- SetFit: 1.1.0
- Sentence Transformers: 3.2.1
- Transformers: 4.44.2
- PyTorch: 2.5.0+cu121
- Datasets: 3.0.2
- Tokenizers: 0.19.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->