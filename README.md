# dropout
Python implementation of "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (Hinton et al.))

Read the paper [here!](https://jmlr.org/papers/v15/srivastava14a.html)

## Paper implementation overview

Hinton et al. describe dropout as the idea of randomly dropping $(1 - p)$ percent of units in the network, to prevent networks from learning too much from specificities of the dataset (co-adapting), where $p$ is the probability (from 0 to 1) that a neuron/unit is kept in the network. This also boasts training efficiency benefits and forces more neurons to "strengthen" when learning through optimization

The idea behind this intuition can be easily explained by a simple (and horribly unrealistic) biology example: imagine putting all of your braincells in a "pie", and randomly cutting out $(1 - p)$ percent of said pie over and over again while trying to do a task that the full amount of braincells would perform. The idea is that you are strengthening the remaining braincells (kind of like isolating a muscle during a workout).

**Dropout Layer:**
The actual implementation of such an approach is remarkably simple:

In a Layer system based on classes, we can simply implement it as:

```Python
class Dropout:
    def __init__(self, p=0.5, training_mode=True):
        self.p = p
        self.training = training_mode

    def __call__(self, x):
        if self.training:
            # create a random boolean matrix, and
            self.mask = torch.rand(x.shape) < self.p
            self.out = x * self.mask / self.p
        else:
            # (f"training mode enabled: {self.training}")
            self.out = x
        return self.out

    def parameters(self):
        return []
```

The authors of the paper regularize outputs by multiplying weights by $p$, but in production, it is more typical to implement inverse dropout (`self.out = x * self.mask / self.p`). The devision by `self.p` returns the shape of the output to what the shape of the input was, which is important since the shape of the values passed through the layer are bound to change when dropping out random neurons.
