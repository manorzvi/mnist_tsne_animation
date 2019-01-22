# mnist_tsne_animation
1. Visualization of MNIST network LLA (last layer activations) by project it onto lower dimension space using
   modified version of T-SNE (t-distributed stochastic neighbor embedding) algorithm.
2. There are 2 modes of animations:
      2.1 Algorithm's training progress as series of 2D intermediate embeddings.
      2.2 Algorithm's final results, change through time, while the inputs slightly changes.

# tSNE Problem:
Small changes in the Inputs cause major changes in the 2D embedding.

# Prior:
We suggest to add a prior to tSNE cost function, in order to decrease embedding's movements through time.

## To run:
1. train mnist model by run: mnist_tsne.py
2. create tSNE embeddings by: tsne_animation.generate_tSNE(metric='regular', location)
   <br>2.1 Specify metric for the modified tSNE algorithm: 'regular', 'init', 'prior'. 'regular' is the default
   <br>2.2 Generate arrays of sequential plots by: tsne_animation.generate_arrays(location)
      <br>2.2.1 Specify location to read embeddings from (same location you gave at generate_tSNE)
3. animate final tSNE results by: tsne_animation._anim()
4. create tSNE embeddings while record intermediate positions by: tsne_animation._generate_intermediate_positions(metric, origin, alpha)
   <br>4.1 this time metric, origin are lists
   <br.4.2 this methods generate embeddings and arrays for animation
5. animate different metrices of tSNE on diffrent inputs by: tsne_animation._animate_intemediate_positions()

5. Dont forget to warship the Flying Spaghetti Monster.
