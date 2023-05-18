from models.hmm.model import HiddenMarkovModel


obs = [
    [],
    ['man'],
    ['The', 'man', 'saw', 'the', 'dog', 'with', 'the', 'telescope', '.']
]

states = [
    [],
    ['p'],
    ['o', 'p', 'o', 'o', 'a', 'o', 'a', 'o', 'o']
]


if __name__ == '__main__':

    model = HiddenMarkovModel(2)
    model.fit(obs, states, smoothing='one-count')
    decoded = model.predict(obs)
    print(decoded)
