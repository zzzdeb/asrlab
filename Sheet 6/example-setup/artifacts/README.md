6.3.
  1. Num arcs: 417, Num phonemes: 571
  2. O(nWords ^ time);
  3. reduces search space. 90% of search effort in first two phonemes.
  4. In digit recognition, the search space is so low that we can just go
     without tree representation.

6.8.
  Some word combinations are very unlikely. This will have an negative effect
  on the score once the language model is applied. We can then remove such
  hypothesis right away without starting a whole tree search for those.
