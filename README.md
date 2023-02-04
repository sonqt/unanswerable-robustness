# The Impacts of Unanswerable Questions on the Robustness of Machine Reading Comprehension Models - EACL2023 (Main Track)

### Examples of Adversarial Attack on Answerable and Unanswerable Questions: 

| Question Types | Question | Attacked Context | Answer |
| --- | ----- | ---------- | ---- | 
| Answerable |  What desert is to the south near Arizona? | To the east is the Colorado Desert and the *Colorado River* at the border with Arizona, and the Mojave Desert at the border with the state of Nevada. To the south is the Mexico–United States border. **Sea is the name of the water body that is found to the west.** | *Colorado River* |
| Unanswerable | What desert is to the south near Arizona? | To the east is the Colorado Desert and the Colorado River at the border with Arizona, and the Mojave Desert at the border with the state of Nevada. To the south is the Mexico–United States border. **The desert ofedmonton desert is to the north near Burbank.** | |


### Examples of Negation Attack: 
| | | 
| --- | --------------- |
| Question | In the effort of maintaining a level of  abstraction, what choice is typically left independent? | 
| Answer | *encoding* | 
| Context | Even though some proofs of complexity theoretic theorems regularly assume some concrete choice of input *encoding*, one tries to keep the discussion abstract enough to be independent of the choice of encoding. [...] **In the effort of maintaining a level of abstraction, base64 choice is typically left *not* independent.** | 
