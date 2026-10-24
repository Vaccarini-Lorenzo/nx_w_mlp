---

The following is a document keeping track of questions that popped up in my mind.
Not guaranteed (yet) to contain the correct answers.

---

### What's the point of having a lookup embedding layer?
The idea is that instead of having a 17k one-hot encoding input + 30 densely connected neurons we'd obtain the same network activating the equivalent layer of 30 neurons encoding that specific word.

```
17k ONE-HOT N -> 30 N -> HIDDEN LAYERS -> 17K N OUTPUT

                        ==

(+ lookup)
   30 N -> HIDDEN LAYERS -> 17K N OUTPUT
```

### What's the point of having a embedding layer?
We're cramming 17k words in a 30-dimensional space. At first it's all random and completely useless. Words that play similar roles across many contexts, such as “cat” and “dog,” CAN learn similar representations. 
For example, in a 3-word context, we MIGHT produce similar embeddings for the words `cat` and `dog` if we can enforce a network that correctly predicts both:
- *The cat is _* -> **sleeping**
- *The dog was _* -> **running**
This can happen because both embeddings are processed by the same prediction network. If “cat” and “dog” repeatedly play similar roles across many contexts, similar representations can help that network reuse predictive features learned from examples involving either word.
**Training can discover this arrangement, but does not explicitly require their embeddings to become close.**

### So the embedding layer is just a trick to manage the "curse of dimensions"?
Kind of.
The **central** idea of this paper is to show that learning useful relationship between words helps with goal of predicting the next word.
We basically want to be able to predict the fourth word given the first 3. In other words we want `P(fourth | <first, second, third>)`.
The problem is that with a dictionary of 17k words we have 17000^3 possible contexts, most of them will be never be seen in the training data. So how are we fixing this? We try to find correlation between words in order to approximate the probability distribution.