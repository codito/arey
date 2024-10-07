---
# A play file is a markdown format file with settings specified in yaml
# frontmatter (this section within `---`).
#
# You can open this file in an editor, arey will watch for any changes and apply
# them immediately. It will process the content section in markdown and attempt
# generate a completion.

# Model settings
model: TODO # must be defined in config.yml
#settings: # settings update reloads the model for non ollama models
#  n_threads: 11      # default: cpu_count/2
#  n_gpu_layers: 18   # default: 0, run on cpu
profile: # profile update applies to every inference
  temperature: 0.7
  repeat_penalty: 1.176
  top_k: 40
  top_p: 0.1
  max_tokens: 300 # use `num_predict: 300` for ollama
  stop: ["<|im_end|>"] # list of stop words
output:
  format: plain # also supported 'markdown' for highlighting
---

You're a philosopher from the same time period as Socrates, Plato, Seneca etc.
You believe strongly in the Stoic philosophy. Answer the questions
asked as truthfully as possible.

Why do we believe that life is short?
