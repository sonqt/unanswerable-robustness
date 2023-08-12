# The Impacts of Unanswerable Questions on the Robustness of Machine Reading Comprehension Models
## Introduction
This repository contains the source code for the architectures described in the following paper:
>**Single-Sentence Reader: A Novel Approach for Addressing Answer Position Bias**<br>
>EACL 2023 <br>
>Son Quoc Tran, Phong Nguyen-Thuan Do, Uyen Le, Matt Kretchmar<br>
>Computer Science Department, Denison University, Granville, Ohio<br>
>The UIT NLP Group, Vietnam National University, Ho Chi Minh City<br>

## 1. Getting Started
### Installing Java
Please refer to [Java Download](https://www.oracle.com/java/technologies/downloads/#java16)
### Running StanfordCoreNLPServer
For further information, refer to [CoreNLP](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html)

1. Download file stanford-correnlp-latest.zip
2. Unzip file

```
cd stanford-corenlp-4.4.0
```

Start stanfordCoreNLPServer 

```shell
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,parse" -port 9000 -timeout 30000
```
## 2. Attack
### Nearest word with Glove

```
cd src
```

Download [Glove](https://nlp.stanford.edu/projects/glove/) and use `glove.6B.100d.txt`

Find nearby words for words in dataset

```
python3 find_nearby_words.py
```

### Attack 

```python3
python3 attack_main.py
```

### Examples of Adversarial Attack on Answerable and Unanswerable Questions: 

| Question Types | Question | Attacked Context | Answer |
| --- | ----- | ---------- | ---- | 
| Answerable |  What desert is to the south near Arizona? | To the east is the Colorado Desert and the *Colorado River* at the border with Arizona, and the Mojave Desert at the border with the state of Nevada. To the south is the Mexico–United States border. **Sea is the name of the water body that is found to the west.** | *Colorado River* |
| Unanswerable | What desert is to the south near Arizona? | To the east is the Colorado Desert and the Colorado River at the border with Arizona, and the Mojave Desert at the border with the state of Nevada. To the south is the Mexico–United States border. **The desert ofedmonton desert is to the north near Burbank.** | |


### Examples of Negation Attack: 
|||
| --- | --------------- |
| Question | In the effort of maintaining a level of  abstraction, what choice is typically left independent? | 
| Answer | *encoding* | 
| Context | Even though some proofs of complexity theoretic theorems regularly assume some concrete choice of input *encoding*, one tries to keep the discussion abstract enough to be independent of the choice of encoding. [...] **In the effort of maintaining a level of abstraction, base64 choice is typically left *not* independent.** | 

## Citation and Contact
If you found this repository helpful, please cite:
```
@inproceedings{tran-etal-2023-impacts,
    title = "The Impacts of Unanswerable Questions on the Robustness of Machine Reading Comprehension Models",
    author = "Tran, Son Quoc  and
      Do, Phong Nguyen-Thuan  and
      Le, Uyen  and
      Kretchmar, Matt",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.113",
    pages = "1543--1557",
    abstract = "Pretrained language models have achieved super-human performances on many Machine Reading Comprehension (MRC) benchmarks. Nevertheless, their relative inability to defend against adversarial attacks has spurred skepticism about their natural language understanding. In this paper, we ask whether training with unanswerable questions in SQuAD 2.0 can help improve the robustness of MRC models against adversarial attacks. To explore that question, we fine-tune three state-of-the-art language models on either SQuAD 1.1 or SQuAD 2.0 and then evaluate their robustness under adversarial attacks. Our experiments reveal that current models fine-tuned on SQuAD 2.0 do not initially appear to be any more robust than ones fine-tuned on SQuAD 1.1, yet they reveal a measure of hidden robustness that can be leveraged to realize actual performance gains. Furthermore, we find that robustness of models fine-tuned on SQuAD 2.0 extends on additional out-of-domain datasets. Finally, we introduce a new adversarial attack to reveal of SQuAD 2.0 that current MRC models are learning.",
}

```
Please contact Son Quoc Tran at `tran_s2@denison.edu` if you have any questions.

