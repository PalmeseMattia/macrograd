from engine import Layer

if __name__ == "__main__":
    training_set = [
        [1,1,1],
        [1,0,0],
        [0,1,0],
        [0,0,0]
    ]

    l = Layer(2, 2)
    out = Layer(2, 1)
    
    def forward(data) :
        x = out(l(data))
        return x

    for t in training_set:
        result = forward([t[0], t[1]])
        print(f"Input: {t[0], t[1]} Output: {result.data}")

        result.backward()
