"""A small character-level next-character generation demo."""

import math
import random


SENTENCES = ["I like cats.", "I like dogs.", "I like noodles."]
REPETITIONS = 12
BLOCK_SIZE = 40
FEATURE_SIZE = BLOCK_SIZE
EPOCHS = 20
LEARNING_RATE = 0.001
GENERATED_LENGTH = 120
SEED = 7


def softmax(logits):
    peak = max(logits)
    exp_values = [math.exp(value - peak) for value in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def make_examples(text, char_to_id):
    examples = []
    for start in range(len(text) - BLOCK_SIZE):
        window = text[start : start + BLOCK_SIZE]
        features = [char_to_id[char] for char in window[-FEATURE_SIZE:]]
        target = char_to_id[text[start + BLOCK_SIZE]]
        examples.append((features, target))
    return examples


def train(examples, vocabulary_size, rng):
    weights = [
        [[rng.uniform(-0.01, 0.01) for _ in range(vocabulary_size)]
         for _ in range(vocabulary_size)]
        for _ in range(FEATURE_SIZE)
    ]
    biases = [0.0] * vocabulary_size

    for _ in range(EPOCHS):
        rng.shuffle(examples)
        for features, target in examples:
            logits = biases.copy()
            for position, feature in enumerate(features):
                feature_weights = weights[position][feature]
                for output_id in range(vocabulary_size):
                    logits[output_id] += feature_weights[output_id]

            probabilities = softmax(logits)
            for output_id, probability in enumerate(probabilities):
                gradient = probability - (output_id == target)
                biases[output_id] -= LEARNING_RATE * gradient
                for position, feature in enumerate(features):
                    weights[position][feature][output_id] -= (
                        LEARNING_RATE * gradient
                    )

    return weights, biases


def training_loss(examples, weights, biases, vocabulary_size):
    total_loss = 0.0
    for features, target in examples:
        logits = biases.copy()
        for position, feature in enumerate(features):
            feature_weights = weights[position][feature]
            for output_id in range(vocabulary_size):
                logits[output_id] += feature_weights[output_id]
        probabilities = softmax(logits)
        total_loss -= math.log(max(probabilities[target], 1e-12))
    return total_loss / len(examples)


def generate(prompt, text, char_to_id, id_to_char, weights, biases, temperature, rng):
    prompt_start = text.find(prompt, BLOCK_SIZE)
    if prompt_start < 0:
        raise ValueError("The generation prompt must occur in the training text.")
    context = text[prompt_start - BLOCK_SIZE : prompt_start] + prompt
    result = prompt

    for _ in range(GENERATED_LENGTH):
        features = [char_to_id[char] for char in context[-FEATURE_SIZE:]]
        logits = biases.copy()
        for position, feature in enumerate(features):
            feature_weights = weights[position][feature]
            for output_id in range(len(id_to_char)):
                logits[output_id] += feature_weights[output_id]

        probabilities = softmax([value / temperature for value in logits])
        next_id = rng.choices(range(len(id_to_char)), weights=probabilities, k=1)[0]
        next_char = id_to_char[next_id]
        result += next_char
        context = (context + next_char)[-BLOCK_SIZE:]

    return result


def main():
    text = "\n".join(SENTENCES * REPETITIONS)
    vocabulary = sorted(set(text))
    char_to_id = {char: index for index, char in enumerate(vocabulary)}
    id_to_char = {index: char for char, index in char_to_id.items()}
    examples = make_examples(text, char_to_id)

    weights, biases = train(examples, len(vocabulary), random.Random(SEED))
    loss = training_loss(examples, weights, biases, len(vocabulary))

    print(f"Corpus length: {len(text)} characters")
    print(f"Vocabulary: {vocabulary!r}")
    print(f"Training samples: {len(examples)}")
    print(f"Final loss: {loss:.4f}")

    for temperature in (0.1, 0.5, 0.7, 1.0):
        sample = generate(
            "I like ", text, char_to_id, id_to_char, weights, biases,
            temperature, random.Random(SEED + int(temperature * 100)),
        )
        print(f"\n--- Temperature {temperature:.1f} ---")
        print(sample)


if __name__ == "__main__":
    main()