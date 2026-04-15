# Human Evaluation Results

| Case | Name | Generated Image | Decision | Auto Visual | Auto Prompt | Auto Quality | Signal | Human Visual | Human Prompt | Human Quality | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| A | Captioning and VQA focus | generated_outputs\gen_image_1776221571458.png | ACCEPT | 8 | 7 | 6 | 0.576 (heuristic_proxy) | 4 | 4 | 3 | The composition and important objects remain recognizable while the generated version provides a clean visual variation. |
| B | Style-guided transformation | generated_outputs\gen_image_1776221571551.png | ACCEPT | 9 | 10 | 5 | 0.757 (heuristic_proxy) | 5 | 5 | 3 | The main subject is preserved and the requested style direction is visible; stronger artistic texture or lighting would improve the final polish. |
| C | Prompt-based enhancement | generated_outputs\gen_image_1776221571646.png | ACCEPT | 9 | 10 | 5 | 0.638 (heuristic_proxy) | 5 | 5 | 3 | The output keeps the original layout while improving contrast and visibility; more natural detail would make the enhancement stronger. |
| D | Failure handling and ambiguous prompt | generated_outputs\gen_image_1776221571738.png | ACCEPT | 9 | 8 | 6 | 0.423 (heuristic_proxy) | 5 | 4 | 3 | The output keeps the original layout while improving contrast and visibility; more natural detail would make the enhancement stronger. Warning handled: Ambiguity warning: user instruction was vague, so the prompt was enriched using visual context and defaulted to enhancement. |
