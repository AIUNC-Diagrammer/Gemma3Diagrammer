Natural-Language to Mermaid Diagram Generator (Fine-Tuned Gemma)

This project fine-tunes Gemma-3-1B-IT, a lightweight instruction-tuned LLM, to generate MermaidJS diagrams from natural-language descriptions. The goal is to automate the creation of flowcharts, state diagrams, and structural diagrams for tools like draw.io, enabling developers to produce editable diagrams from plain text.

Overview

We created a synthetic dataset of 300 AI-generated examples, each pairing:

a human-readable prompt, and

the corresponding Mermaid diagram.

Because no public dataset exists for this task, synthetic generation allowed us to cover diverse diagram structures while controlling quality. All samples were converted into a conversational format compatible with Gemma’s chat template and split into training (80%) and test (20%).

Training

We used Supervised Fine-Tuning (SFT) with the TRL library:

- 6 epochs

- learning rate: 5e-5

- batch size: 1

- constant LR schedule

- AdamW fused optimizer

- FP16/BF16 mixed precision (T4/T1000 GPUs)

- Training and validation metrics were tracked in TensorBoard, and checkpoints were evaluated at each epoch.

Usage

After training, the model generates Mermaid diagrams directly from natural-language input:

prompt = "Show a login process with authentication and error handling."
output = model.generate(prompt)
print(output)


The model outputs clean, render-ready Mermaid code.

Results

Despite the small dataset and model size, the fine-tuned Gemma model produces syntactically valid and semantically coherent diagrams, demonstrating that compact LLMs can learn structured code generation from synthetic data and remain deployable on modest hardware.
