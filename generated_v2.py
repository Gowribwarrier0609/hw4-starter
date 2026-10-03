"""Train a tiny character-level LSTM and sample at several temperatures."""

import math
import random


SENTENCES = [
    "I like cats.",
    "I like dogs.",
    "I like noodles.",
    "I like music.",
    "I like books.",
    "I like apples.",
    "I like walks.",
    "I like tea.",
    "I like cookies.",
    "I like pizza.",
]
PROMPT = "I like "
BLOCK_SIZE = 40
HIDDEN_SIZE = 12
GATE_COUNT = 4
EPOCHS = 150
LEARNING_RATE = 0.025
GENERATED_LENGTH = 120
TOP_COUNT = 5
SEED = 7


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def softmax(logits):
    peak = max(logits)
    values = [math.exp(value - peak) for value in logits]
    total = sum(values)
    return [value / total for value in values]


class CharacterLSTM:
    def __init__(self, vocabulary_size, rng):
        self.vocabulary_size = vocabulary_size
        self.input_weights = [
            [[rng.uniform(-0.06, 0.06) for _ in range(vocabulary_size)]
             for _ in range(HIDDEN_SIZE)]
            for _ in range(GATE_COUNT)
        ]
        self.recurrent_weights = [
            [[rng.uniform(-0.06, 0.06) for _ in range(HIDDEN_SIZE)]
             for _ in range(HIDDEN_SIZE)]
            for _ in range(GATE_COUNT)
        ]
        self.gate_biases = [[0.0] * HIDDEN_SIZE for _ in range(GATE_COUNT)]
        self.gate_biases[0] = [0.5] * HIDDEN_SIZE
        self.output_weights = [
            [rng.uniform(-0.06, 0.06) for _ in range(HIDDEN_SIZE)]
            for _ in range(vocabulary_size)
        ]
        self.output_biases = [0.0] * vocabulary_size

    def forward(self, character_ids):
        hidden = [0.0] * HIDDEN_SIZE
        cell = [0.0] * HIDDEN_SIZE
        states = []

        for character_id in character_ids:
            previous_hidden = hidden
            previous_cell = cell
            gates = []
            for gate_id in range(GATE_COUNT):
                values = []
                for hidden_id in range(HIDDEN_SIZE):
                    value = self.gate_biases[gate_id][hidden_id]
                    value += self.input_weights[gate_id][hidden_id][character_id]
                    value += sum(
                        self.recurrent_weights[gate_id][hidden_id][index]
                        * previous_hidden[index]
                        for index in range(HIDDEN_SIZE)
                    )
                    values.append(value)
                if gate_id == 3:
                    gates.append([math.tanh(value) for value in values])
                else:
                    gates.append([sigmoid(value) for value in values])

            forget_gate, input_gate, output_gate, candidate = gates
            cell = [
                forget_gate[index] * previous_cell[index]
                + input_gate[index] * candidate[index]
                for index in range(HIDDEN_SIZE)
            ]
            tanh_cell = [math.tanh(value) for value in cell]
            hidden = [
                output_gate[index] * tanh_cell[index]
                for index in range(HIDDEN_SIZE)
            ]
            states.append((
                character_id, previous_hidden, previous_cell, gates, cell,
                tanh_cell, hidden,
            ))

        logits = [
            self.output_biases[character_id]
            + sum(
                self.output_weights[character_id][index] * hidden[index]
                for index in range(HIDDEN_SIZE)
            )
            for character_id in range(self.vocabulary_size)
        ]
        return states, softmax(logits)

    def train_example(self, character_ids, target_id):
        states, probabilities = self.forward(character_ids)
        output_gradient = probabilities.copy()
        output_gradient[target_id] -= 1.0
        final_hidden = states[-1][6]

        output_weight_gradient = [
            [output_gradient[character_id] * final_hidden[index]
             for index in range(HIDDEN_SIZE)]
            for character_id in range(self.vocabulary_size)
        ]
        hidden_gradient = [
            sum(
                self.output_weights[character_id][index]
                * output_gradient[character_id]
                for character_id in range(self.vocabulary_size)
            )
            for index in range(HIDDEN_SIZE)
        ]
        cell_gradient = [0.0] * HIDDEN_SIZE
        input_gradient = [
            [[0.0] * self.vocabulary_size for _ in range(HIDDEN_SIZE)]
            for _ in range(GATE_COUNT)
        ]
        recurrent_gradient = [
            [[0.0] * HIDDEN_SIZE for _ in range(HIDDEN_SIZE)]
            for _ in range(GATE_COUNT)
        ]
        bias_gradient = [[0.0] * HIDDEN_SIZE for _ in range(GATE_COUNT)]

        for character_id, previous_hidden, previous_cell, gates, cell, tanh_cell, hidden in reversed(states):
            forget_gate, input_gate, output_gate, candidate = gates
            gate_gradient = [[0.0] * HIDDEN_SIZE for _ in range(GATE_COUNT)]
            previous_cell_gradient = [0.0] * HIDDEN_SIZE

            for index in range(HIDDEN_SIZE):
                output_derivative = hidden_gradient[index] * tanh_cell[index]
                total_cell_gradient = (
                    cell_gradient[index]
                    + hidden_gradient[index] * output_gate[index]
                    * (1.0 - tanh_cell[index] ** 2)
                )
                forget_derivative = total_cell_gradient * previous_cell[index]
                input_derivative = total_cell_gradient * candidate[index]
                candidate_derivative = total_cell_gradient * input_gate[index]

                gate_gradient[0][index] = (
                    forget_derivative * forget_gate[index] * (1.0 - forget_gate[index])
                )
                gate_gradient[1][index] = (
                    input_derivative * input_gate[index] * (1.0 - input_gate[index])
                )
                gate_gradient[2][index] = (
                    output_derivative * output_gate[index] * (1.0 - output_gate[index])
                )
                gate_gradient[3][index] = (
                    candidate_derivative * (1.0 - candidate[index] ** 2)
                )
                previous_cell_gradient[index] = total_cell_gradient * forget_gate[index]

            previous_hidden_gradient = [0.0] * HIDDEN_SIZE
            for gate_id in range(GATE_COUNT):
                for hidden_id, gradient in enumerate(gate_gradient[gate_id]):
                    bias_gradient[gate_id][hidden_id] += gradient
                    input_gradient[gate_id][hidden_id][character_id] += gradient
                    for previous_id in range(HIDDEN_SIZE):
                        recurrent_gradient[gate_id][hidden_id][previous_id] += (
                            gradient * previous_hidden[previous_id]
                        )
                        previous_hidden_gradient[previous_id] += (
                            self.recurrent_weights[gate_id][hidden_id][previous_id]
                            * gradient
                        )

            hidden_gradient = previous_hidden_gradient
            cell_gradient = previous_cell_gradient

        for character_id in range(self.vocabulary_size):
            self.output_biases[character_id] -= (
                LEARNING_RATE * output_gradient[character_id]
            )
            for index in range(HIDDEN_SIZE):
                self.output_weights[character_id][index] -= (
                    LEARNING_RATE * output_weight_gradient[character_id][index]
                )

        for gate_id in range(GATE_COUNT):
            for hidden_id in range(HIDDEN_SIZE):
                self.gate_biases[gate_id][hidden_id] -= (
                    LEARNING_RATE * bias_gradient[gate_id][hidden_id]
                )
                for character_id in range(self.vocabulary_size):
                    self.input_weights[gate_id][hidden_id][character_id] -= (
                        LEARNING_RATE * input_gradient[gate_id][hidden_id][character_id]
                    )
                for previous_id in range(HIDDEN_SIZE):
                    self.recurrent_weights[gate_id][hidden_id][previous_id] -= (
                        LEARNING_RATE * recurrent_gradient[gate_id][hidden_id][previous_id]
                    )

    def loss(self, examples):
        total_loss = 0.0
        for character_ids, target_id in examples:
            _, probabilities = self.forward(character_ids)
            total_loss -= math.log(max(probabilities[target_id], 1e-12))
        return total_loss / len(examples)


def make_examples(text, char_to_id):
    return [
        ([char_to_id[character] for character in text[start : start + BLOCK_SIZE]],
         char_to_id[text[start + BLOCK_SIZE]])
        for start in range(len(text) - BLOCK_SIZE)
    ]


def starting_context(prompt, char_to_id):
    if len(prompt) > BLOCK_SIZE:
        raise ValueError(f"Prompt must be no longer than {BLOCK_SIZE} characters.")
    return [char_to_id[character] for character in prompt.rjust(BLOCK_SIZE)]


def train_model(examples, vocabulary_size, rng):
    model = CharacterLSTM(vocabulary_size, rng)
    for _ in range(EPOCHS):
        rng.shuffle(examples)
        for character_ids, target_id in examples:
            model.train_example(character_ids, target_id)
    return model


def generate(model, prompt, char_to_id, id_to_char, temperature, rng):
    context = starting_context(prompt, char_to_id)
    output = prompt
    for _ in range(GENERATED_LENGTH):
        _, probabilities = model.forward(context)
        scaled_probabilities = softmax(
            [math.log(max(probability, 1e-12)) / temperature
             for probability in probabilities]
        )
        next_id = rng.choices(
            range(len(id_to_char)), weights=scaled_probabilities, k=1
        )[0]
        output += id_to_char[next_id]
        context = context[1:] + [next_id]
    return output


def main():
    text = "\n".join(SENTENCES)
    vocabulary = sorted(set(text))
    char_to_id = {character: index for index, character in enumerate(vocabulary)}
    id_to_char = {index: character for character, index in char_to_id.items()}

    padded_text = PROMPT.rjust(BLOCK_SIZE) + text[len(PROMPT):]
    examples = make_examples(padded_text, char_to_id)
    prompt_context = starting_context(PROMPT, char_to_id)
    examples.extend(
        (prompt_context, char_to_id[sentence[len(PROMPT)]])
        for sentence in SENTENCES
    )

    model = train_model(examples, len(vocabulary), random.Random(SEED))
    final_loss = model.loss(examples)

    print(f"Corpus length: {len(text)} characters")
    print(f"Vocabulary: {vocabulary!r}")
    print(f"Training samples: {len(examples)}")
    print(f"Final loss: {final_loss:.4f}")

    for temperature in (0.1, 0.5, 0.7, 1.0):
        _, probabilities = model.forward(prompt_context)
        scaled_probabilities = softmax(
            [math.log(max(probability, 1e-12)) / temperature
             for probability in probabilities]
        )
        likely_ids = sorted(
            range(len(vocabulary)),
            key=lambda index: scaled_probabilities[index],
            reverse=True,
        )[:TOP_COUNT]
        print(
            f"\nMost likely next characters for {PROMPT!r} "
            f"at temperature {temperature:.1f}:"
        )
        for character_id in likely_ids:
            print(f"  {id_to_char[character_id]!r}: {scaled_probabilities[character_id]:.3f}")

        sample = generate(
            model, PROMPT, char_to_id, id_to_char, temperature,
            random.Random(SEED + int(temperature * 100)),
        )
        print(f"\n--- Temperature {temperature:.1f} ---")
        print(sample)


if __name__ == "__main__":
    main()